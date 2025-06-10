#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

const int BLOCK_DIM = 1024;
const int MAX_BLOCKS = 65535;
const float ERR_THRES = 1e-3;

#define cuda_check_errors(msg) \
    do {\
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) \
        {\
            printf("Fatal Error For:%s =>(%s at %s:%d)\n", \
                    msg, \
                    cudaGetErrorString(__err),\
                     __FILE__, \
                    __LINE__); \
        }\
    } while(0);


void print_device_prop()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Name: %s\n", prop.name);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("\n\n");
}


int generate_float_matrix(float *h_in, int N)
{
    for (int i = 0; i < N; i++)
    {
        //float val = ((float ) rand() / RAND_MAX) * 2.0 + 1.0;
        //float val = (float)(rand() % (9 - 1 + 1)) + 1;
        //*(h_in + i) = val;
        *(h_in + i) = (i % 3) + 5;
        //*(h_in + i) = 1.00f;
    }
    return 0;
}

int hostk_basic(float *h_in, float *h_out, int N)
{
    *h_out = 0.0f;
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        //*h_out += *(h_in + i);
        sum += *(h_in + i);
    }

    *h_out = (float)sum;

    return 0;
}

int print_matrix(float *h_in, int N)
{
    printf("\n");
    
    for (int i = 0; i < N; i++)
    {
        printf("%f|  ", *(h_in + i));

        if (i % 5 == 0)
            printf("\n");
    }

    printf("\n");

    return 0;
}


int host_main(int N)
{
    printf("Executing Host Main Basic:%d\n", N);
    
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_in;
    float  h_out = 0.0f;
    int alloc_size = N * sizeof(float);

    /* allocate host memory */
    h_in = (float *)malloc(alloc_size);

    generate_float_matrix(h_in, N);
    //print_matrix(h_in, N);

    cudaEventRecord(begin, 0);

    hostk_basic(h_in, &h_out, N);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Host Main Basic Elapsed Time: %f\n", elapsed_time);

    //printf("Reduced Sum:%f\n", h_out);

    /* free host memory */
    free(h_in);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return 0;
}

int verify_results(float *h_in, float h_out, int N)
{
    float v_out = 0.0f;
    hostk_basic(h_in, &v_out, N); 

    if (fabs(h_out - v_out) > ERR_THRES)
    {
        printf("Wrong Result Actual:%f Vs Expected:%f, Diff:%f\n", h_out, v_out, (fabs(h_out - v_out)));
    }

    return 0;
}


__global__ void devicek_basic(float *gm_in, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int gx = (blockDim.x * blockIdx.x) + lx;
    float local_sum = 0.0f;

    if (gx < N)
    {
        local_sum += (*(gm_in + gx));
    }

    atomicAdd(gm_out, local_sum);
}

/*
1. entire grid load an element from global memory to shared memory
2. Each block performs block wide reduction 
3. The intra-block results are written into first locations of shared memory
4. The first thread of each block, then perform atomic reduction
*/
__global__ void devicek_tiled(float *gm_in, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int total_threads = (blockDim.x * gridDim.x);
    double val = 0.0f;

    __shared__ float sm_in[BLOCK_DIM];

    /* SM Usage Analysis
    Step 1)Used by all threads to store 4 bytes of value
    Step 2)Used by all threads to read 4 bytes of value
    Step 3) Used by all threads to store 4 bytes of value
    */

    /* load data from GMEM TO SMEM, using grid stride loop */
    for (int i = gx; i < N; i += total_threads)
        val += (double)gm_in[i];

    /* write grid wide reduction results to shared memory */
    sm_in[lx] = (float)val;

    /* ensure all vaalues are completely written in shared memory */
    __syncthreads();

    /* block wide reduction, losing half of threads each iteration 
        sm = 0  1   2   3   4   5   6   7
        tid(0) 0 + 4
        tid(1) 1 + 5
        tid(2) 2 + 6
        tid(3) 3 + 7
        -------------------
        tid(0) = tid(0) + tid(2)
        tid(1) = tid(1) + tid(3)
        -----------------------
        tid(0) = tid(0) + tid(1)

    */
    for (int i = (BLOCK_DIM / 2) ; i > 0; i >>= 1)
    {
        if (lx < i)
            sm_in[lx] += sm_in[lx + i];

        __syncthreads();
    }

    //__syncthreads(); //only 1st thread in each blocks updates so not necessary

    /* Now only the first thread from each block atomically updates the values*/
    if (lx == 0)
        atomicAdd(gm_out, sm_in[0]);
}


/*
1. First load data from global memory to local variable
2. Then grid wide reduction
3. The intra-warp reduction suing warp shuffle
4. Now to do block wide or inter-warp reduction, data must wrriten to shared memory per warp
5. Then again perform warp shuffle on the share memory
6. then final atomic update

*/
__global__ void devicek_warp_shuffle(float *gm_in, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int wx = lx / warpSize; // 35/32 = 1,   65 / 32 = 2, 99/32 = 3
    int wlx = lx % warpSize; // 35 % 32 = 3, 65 % 32 = 1, 99%32 = 3
    float val = 0.0;
    int warps_per_block = blockDim.x / warpSize; // 256/32 = 8
    int total_threads = (blockDim.x * gridDim.x);
    int warp_participants = 0xFFFFFFFF;

    /* 1 result per warp, BLOCK_DIM / 32 warps per block i.e 256/32 = 8 results per block */
    __shared__ float sm_in[8];

    for (int i = gx; i < N; i += total_threads)
        val += gm_in[i];

    /* warp shuffle */
    for (unsigned int i = warpSize / 2; i > 0; i>>=1)
        val += __shfl_down_sync(warp_participants, val, i); //pass value down to lower lanes

    /* write results to shared memory */
    if (wlx == 0)
        sm_in[wx] = val;

    __syncthreads();

    /* only the 1st warp of each block participates in final warp shuffle 
       from last reduction, 1 result per warp was written into shared memory
       so we have 256/32 = 8 warps, so 1 result per warp = 8 result per block
       so only first 8 threads of the very first warp is needed.
    */
    if (wx == 0)
    {
        if (wlx < warps_per_block)
            val = sm_in[wlx];
        else
            val = 0.0;

        __syncthreads();

        /* now warp shuffle */
        for (unsigned int i = warpSize / 2; i > 0; i >>= 1)
            val += __shfl_down_sync(warp_participants, val, i);

        /* at this point there is 1 result per block, perfom final atomic globla update */
        if (wlx == 0)
            atomicAdd(gm_out, val);
    }
}


int device_main(int N, int mode, bool verif)
{
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_in;
    float h_out = 0.0f;
    
    float *d_in;
    float *d_out;

    int alloc_size = N * sizeof(float);

    /* allocate host memory */
    h_in = (float *)malloc(alloc_size);

    generate_float_matrix(h_in, N);

    cudaEventRecord(begin, 0);
    
    /* alloc device memory */
    cudaMalloc(&d_in, alloc_size);
    cuda_check_errors("DIN_ALC");

    cudaMalloc(&d_out, sizeof(float));
    cuda_check_errors("DOUT_ALC");

    cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_DIN");

    cudaMemset(d_out, 0, sizeof(float));
    cuda_check_errors("HOUT_MSET");

    dim3 block_shape(BLOCK_DIM);
    int  num_blocks = ((N + BLOCK_DIM - 1) / BLOCK_DIM);

    if (verif)
        printf("N:%d, BLOCK_DIM:%d, Num_Blocks:%d\n", N, BLOCK_DIM, num_blocks);

    dim3 grid_shape(num_blocks);

    if (mode == 0)
        devicek_basic<<<grid_shape, block_shape>>>(d_in, d_out, N);
    else if (mode == 1)
        devicek_tiled<<<grid_shape, block_shape>>>(d_in, d_out, N);
    else if (mode == 2)
        devicek_tiled<<<grid_shape, block_shape>>>(d_in, d_out, N);

    cuda_check_errors("klaunch");

    cudaDeviceSynchronize();
    cuda_check_errors("dev_sync");

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_HOUT");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device Mode: %d Elapsed Time:%f\n", mode, elapsed_time);

    if (verif)
        verify_results(h_in, h_out, N);

    cudaFree(d_out);
    cudaFree(d_in);

    free(h_in);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int main(int argc, char *argv[]) 
{
    if (argc < 6) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [Num_Elements] {verify_results} {run_tests}\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[3]);
    bool verify_results = (strcmp(argv[4], "TRUE") == 0);
    bool run_tests = (strcmp(argv[5], "TRUE") == 0); 

    printf("N:%d, Verify Results:%d, Tests:%d\n", N, verify_results, run_tests);

    if (run_tests)
    {
        printf("Tests For Reductions\n");

        int NAR[18] = {222, 777, 999,   2222, 7777, 9999,   22222, 77777, 99999, 
                       222222, 777777, 999999,  2222222, 7777777, 9999999,
                       22222222, 77777777, 99999999};

        for (int i = 0; i < 18; i++)
        {
            //host_main  (NAR[i]);
            device_main(NAR[i], 0, verify_results);
            device_main(NAR[i], 1, verify_results);
            device_main(NAR[i], 2, verify_results);
            printf("\n");
        }
    }
    else
    {
        if (strcmp(argv[1], "HOST") == 0) 
        {
            if (strcmp(argv[2], "BASIC") == 0) 
            {
                host_main(N);
            } 
            else 
            {
                //host_main_tiled(X, Y, Z, verify_results);
            }
        } 
        else if (strcmp(argv[1], "DEVICE") == 0) 
        {
            print_device_prop();

            if (strcmp(argv[2], "BASIC") == 0) 
            {
                device_main(N, 0, verify_results);
            } 
            else if (strcmp(argv[2], "TILED") == 0) 
            {
                device_main(N, 1, verify_results);
            }
            else if (strcmp(argv[2], "WS") == 0)
            {
                device_main(N, 2, verify_results);
            } 
        } 
        else
        {
            fprintf(stderr, "Unknown mode: %s\n", argv[1]);
            return 1;
        }
    }

    return 0;
}