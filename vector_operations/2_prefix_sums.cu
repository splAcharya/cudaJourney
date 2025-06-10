#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

const int BLOCK_DIM = 1024;
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
        float val = ((float ) rand() / RAND_MAX) * 2.0 + 1.0;
        //*(h_in + i) = val;
        *(h_in + i) = i % 9;//
        //*(h_in + i) = 1.00//f;
    }                       //
    return 0;
}

int print_matrix(float *h_in, int N)
{
    printf("\n");
    
    for (int i = 0; i < N; i++)
    {
        printf("%f|  ", *(h_in + i));

        if (i && i % 7 == 0)
            printf("\n");
    }

    printf("\n");

    return 0;
}

void hostk_basic(float *h_in, float *h_out, int N)
{
    h_out[0] = h_in[0];

    for (int i = 1; i < N; i++)
        h_out[i] = h_out[i - 1] + h_in[i];
}


int host_main(int N)
{
    printf("Executing Host Main Basic:%d\n", N);
    
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_in, *h_out;
    int alloc_size = N * sizeof(float);

    /* allocate host memory */
    h_in = (float *)malloc(alloc_size);
    h_out = (float *)malloc(alloc_size);

    generate_float_matrix(h_in, N);
    //print_matrix(h_in, N);

    cudaEventRecord(begin, 0);

    hostk_basic(h_in, h_out, N);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Host Main Basic Elapsed Time: %f\n", elapsed_time);

    //print_matrix(h_out, N);
    //printf("Reduced Sum:%f\n", h_out);

    /* free host memory */
    free(h_out);
    free(h_in);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return 0;
}

int verify_results(float *h_in, float *h_out, int N)
{
    float *h_v = (float *)malloc(N * sizeof(float)); 
    hostk_basic(h_in, h_v, N); 

    //print_matrix(h_in, N);
    //print_matrix(h_out, N);
    //print_matrix(h_v, N);

    printf("Last Element==> Actual:%f, Expected:%f\n", h_out[N - 1], h_v[N - 1]);
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_out[i] - h_v[i]) > ERR_THRES)
        {
            printf("Wrong Result @:%d => Actual:%f Vs Expected:%f\n", i, h_out[i], h_v[i]);
            break;
        }
    }

    free(h_v);

    return 0;
}

int hostk_simulate_partial(float *gm_in, float *gm_buf, int N)
{
    //printf("Partial Simulation\n");

    int num_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    for (int i = 0; i < num_blocks; i++)
    {
        gm_buf[i * BLOCK_DIM + 0] = gm_in[i * BLOCK_DIM + 0]; 
        
        //printf("Starting: %i\n", i);

        for (int j = 1; j < BLOCK_DIM; j++)
        {
            int idx = (i * BLOCK_DIM + j);
            //printf("Flat IDx: %d\n", idx);
            if (idx < N)
                gm_buf[idx] = gm_in[idx] + gm_buf[idx - 1];
        }
    }

    //print_matrix(gm_in, N);
    //print_matrix(gm_buf, N);

    return 0;
}


int verify_partial_results(float *h_in, float *h_buffer, int N)
{
    float *h_v = (float *)malloc(N * sizeof(float)); 

    hostk_simulate_partial(h_in, h_v, N); 

    //print_matrix(h_in, N);
    //print_matrix(h_buffer, N);
    //print_matrix(h_v, N);

    //printf("Last Element==> Actual:%f, Expected:%f\n", h_out[N - 1], h_v[N - 1]);
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_buffer[i] - h_v[i]) > ERR_THRES)
        {
            printf("Wrong Partial Result @:%d => Actual:%f Vs Expected:%f\n", i, h_buffer[i], h_v[i]);
            break;
        }
    }

    free(h_v);

    return 0;
}

/* super naive approach each thread does it position's prefix sum 
    so the N-1th element would go through the entire array
    Simple optimization here to try get coalesed acces
    would be for each thread to start from the right and move to left

    0   1   2   3   4
    10  20  30  40  50
    
    cycle 1
    0   1   2   3   4

    cycle 2
    x   10  20  30  40

    cycle 3
    x   x   10  20  30

    and so on..
*/
__global__ void devicek_basic(float *gm_in, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int gx = (blockDim.x * blockIdx.x)  + lx;

    if (gx < N)
    {
        float val = 0.0f;
        //printf("Bef-->GX:%d-->Gmem:%f--->Val:%f\n", gx, gm_in[gx], val);
        for (int i = gx; i >= 0; i--)
        {
            val += (float)gm_in[i];
        }
        //printf("GX:%d-->Gmem:%f--->Val:%f\n", gx, gm_in[gx], val);
        gm_out[gx] = (float)val;
    }
}

/*
    
    10  20      30  40      50  60      70  80
    10  30      60  100     150 210     280 360

    0   1       2   3       4   5       6   7
    10  20      30  40      50  60      70  80

    Bring Down To SMEM 
    Block-0     Block-1     Block-2     BLock-3
    10  20      30  40      50   60     70  80
    wait for all threads to completed __syncthreads()

    starting elements at thread0 to blockDim - 1, add stride element to your left
    stride = 1
    10  30      30  70      50  110     70  150
    stride 2 ///stop

    Transfer last element from each block To GLobal Memory
    30  70  110 150
    ===> so each thread from a block accesses these.. could the buffer
    be a constant memory ..if this is broadcast .... TODO actually 
    need to understand the constant memory broadcast better
    
    To Each Shared Memory starting from block 1 (ignore lbokc 0) add eme up
    i = 1
                
    Block-0     BLock-1     BLock-2     BLock-3
    10  30      60  100     120 180     180  260 
    x           x           
                            150 210     250  330
                                x
                                        280  360

*/

__global__ void devicek_step2(float *gm_buffer, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (blockDim.x * bx) + lx;
    float block_sum = 0.0;

    /* load from GMEM */
    if (gx < N)
        block_sum += gm_buffer[gx];

    //printf("[Gx:%d, lx:%d] ---> GM:%f, SM:%f\n", gx, lx, gm_out[gx], sm_out[lx]);
   
    /* grid wide reduction */
    if (bx > 0)
    { 
        for (unsigned int stride = 0; stride < bx; stride += 1)
        {
            int stride_step = (stride * blockDim.x) + (blockDim.x - 1);
            block_sum += gm_buffer[stride_step];
        }
    }

    //printf("[Gx:%d, lx:%d] ---> BSUM:%f\n", gx, lx, block_sum);

    /* write results back to G-MEM */
    if (gx < N)
        gm_out[gx] = block_sum;
}


__global__ void devicek_step1(float *gm_in, float *gm_buffer, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (blockDim.x * bx) + lx;
    
    __shared__ float sm_in[BLOCK_DIM];

    /* load to SMEM */
    sm_in[lx] = (gx < N) ? gm_in[gx] : 0.00;    
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        float val = sm_in[lx];
       
        if (lx > 0 && stride <= lx)
            val += sm_in[lx - stride];
        
        __syncthreads();

        sm_in[lx] = val;

        __syncthreads();
    }

    /* transfer to output block */
    if (gx < N)
        gm_buffer[gx] = sm_in[lx];
}


__global__ void devicek_wef_step2(const float *gm_buffer, float *gm_out, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (blockDim.x * bx) + lx;
    float sum = 0.0;

    //load from GMEM to local var
    if (gx < N)
        sum += gm_buffer[gx];

    for (unsigned int stride = 1; stride <= bx; stride += 1)
    {
        if (bx >= stride)
        {
            int load_bx = (bx - stride); //block to read
            int load_lx = (blockDim.x - 1); 
            int load_gx = (load_bx * blockDim.x) + load_lx; //specfic item to read
            sum += gm_buffer[load_gx];
        }
    } 

    /* update output */
    if (gx < N)
        gm_out[gx] = sum;
}


__global__ void devicek_wef_step1(const float *gm_in, float *gm_buffer, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (blockDim.x * bx) + lx;

    __shared__ float sm_in[BLOCK_DIM + 1];

    //load from GMEM TO SMEM
    if (gx < N)
        sm_in[lx] = gm_in[gx];
    else
        sm_in[lx] = 0.0;

    __syncthreads(); //block wide barrier, all threads must reach this

    //up sweep==>propagate results from left to right
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int right_idx = ((lx + 1) * stride * 2) - 1;
        int left_idx  = right_idx - stride;

        if (right_idx < blockDim.x)
            sm_in[right_idx] += sm_in[left_idx];
        
        __syncthreads(); //block-wide barrier, all threads must reach here
    }

    //down-sweep, propgate result from right to left
    if (lx == blockDim.x - 1) //set root to 0
        sm_in[lx] = 0.0;

    __syncthreads(); //bock wide barier, all threads must reach here

    int start_stride = blockDim.x / 2;
    for (int stride = start_stride; stride > 0; stride /= 2)
    {
        int right_idx = ((lx + 1) * stride * 2) - 1;
        int left_idx = right_idx - stride;

        if (right_idx < blockDim.x)
        {
            float temp = sm_in[right_idx];
            sm_in[right_idx] += sm_in[left_idx];
            sm_in[left_idx] = temp;
        }

        __syncthreads(); //block-wide barrier synchthreads;
    }

    /*
        The steps so far produce a exclusive block level prefix sum.
        If we exten by 1 more element this will end up being a suffix sum.
        so 1 of the threads needs to do extra work. a

        BLock Left Exlcusive Prefix Sum

        GMEM    0   1   2   3   4   5   6   7   
        SMEM    0   0   1   3   6   10  15  21
        
        if (lx < blockdim.x - 1)    val = SMEM(lx + 1)
        else    val = SMEM[lx] + GMEM[gx] 
        
        VAL     0   1   3   6   10  15  21  28
        BUFFER  0   1   3   6   10  15  21  28
    */
    float cur_val = 0.0;
    if (lx == blockDim.x - 1)
        cur_val = sm_in[lx] + gm_in[gx];
    else
        cur_val = sm_in[lx + 1];

    //store intra block results in buffer
    if (gx < N)
        gm_buffer[gx] = cur_val;
} 


int device_main(int N, int mode, bool verif)
{
   printf("Executing Device Main (Mode:%d) For N:%d\n", mode, N);

   cudaEvent_t begin, end;
   cudaEventCreate(&begin);
   cudaEventCreate(&end);

   float *h_in, *h_out, *h_buffer=NULL;
   float *d_in, *d_out, *d_buffer=NULL;

   int alloc_size = N * sizeof(float);

   /* allocate host memory */
   h_in = (float *)malloc(alloc_size);
   h_out = (float *)malloc(alloc_size);
   
   generate_float_matrix(h_in, N);
   //print_matrix(h_in, N);

   cudaEventRecord(begin, 0);
   
   /* alloc device memory */
   cudaMalloc(&d_in, alloc_size);
   cuda_check_errors("DIN_ALC");

   cudaMalloc(&d_out, alloc_size);
   cuda_check_errors("DOUT_ALC");

   cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);
   cuda_check_errors("H2D_DIN");

   cudaMemset(d_out, 0, alloc_size);
   cuda_check_errors("DOUT_MSET");

   dim3 block_shape(BLOCK_DIM);
   int  num_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;

   if (verif)
        printf("N:%d, BLOCK_DIM:%d, num_blocks:%d \n", N, BLOCK_DIM, num_blocks);    

   dim3 grid_shape(num_blocks);
   
   if (mode == 0)
   {
        devicek_basic<<<grid_shape, block_shape>>>(d_in, d_out, N);
        cuda_check_errors("klaunch");
        cudaDeviceSynchronize();
        cuda_check_errors("device_sync");
   }
   else if (mode == 1)
   {
        if (verif)
        {
            h_buffer = (float *)malloc(alloc_size);
        }

        cudaMalloc(&d_buffer, alloc_size);
        cuda_check_errors("DBUF_ALC");

        cudaMemset(d_buffer, 0, alloc_size);
        cuda_check_errors("DBUF_MSET0");

        devicek_step1<<<grid_shape, block_shape>>>(d_in, d_buffer, N);
        cuda_check_errors("klaunch_s1");
        cudaDeviceSynchronize();
        cuda_check_errors("device_sync");

        devicek_step2<<<grid_shape, block_shape>>>(d_buffer, d_out, N);
        cuda_check_errors("klaunch_s2");
        cudaDeviceSynchronize();
        cuda_check_errors("device_sync");
        
        if (verif)
        {
            cudaMemcpy(h_buffer, d_buffer, alloc_size, cudaMemcpyDeviceToHost);
            cuda_check_errors("D2H_DBUF");
        }    
   }
   else if (mode == 2)
   {
        if (verif)
        {
            h_buffer = (float *)malloc(alloc_size);
        }

        cudaMalloc(&d_buffer, alloc_size);
        cuda_check_errors("DBUF_ALC");

        cudaMemset(d_buffer, 0, alloc_size);
        cuda_check_errors("DBUF_MSET0");

        devicek_wef_step1<<<grid_shape, block_shape>>>(d_in, d_buffer, N);
        cuda_check_errors("klaunch_s1");
        cudaDeviceSynchronize();
        cuda_check_errors("device_sync");

        devicek_wef_step2<<<grid_shape, block_shape>>>(d_buffer, d_out, N);
        cuda_check_errors("klaunch_s2");
        cudaDeviceSynchronize();
        cuda_check_errors("device_sync");
        
        if (verif)
        {
            cudaMemcpy(h_buffer, d_buffer, alloc_size, cudaMemcpyDeviceToHost);
            cuda_check_errors("D2H_DBUF");
        }    
   }


   cudaMemcpy(h_out, d_out, alloc_size, cudaMemcpyDeviceToHost);
   cuda_check_errors("D2H_HOUT");

   cudaEventRecord(end, 0);
   cudaEventSynchronize(end);
   float elapsed_time = 0.0f;
   cudaEventElapsedTime(&elapsed_time, begin, end);
   printf("Device Mode: %d Elapsed Time:%f\n", mode, elapsed_time);

   if (verif)
   {
        //print_matrix(h_in, N);
        //print_matrix(h_buffer, N);
        //print_matrix(h_out, N);

        if (mode == 1)
            verify_partial_results(h_in, h_buffer, N);
        
        verify_results(h_in, h_out, N);
   }

   if (mode == 1 && d_buffer)
    cudaFree(d_buffer);
   
   cudaFree(d_out);
   cudaFree(d_in);

   if (verif && mode == 1 && h_buffer)
    free(h_buffer);

   free(h_out);
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
        printf("Tests For Prefix Sums\n");

        int NAR[15] = {2, 7, 9, 22, 77, 99, 222, 777, 999, 2222, 7777, 9999, 22222, 77777, 99999}; 

        for (int i = 0; i < 15; i++)
        {
            host_main  (NAR[i]);
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
            if (verify_results)
                print_device_prop();

            if (strcmp(argv[2], "BASIC") == 0) 
            {
                device_main(N, 0, verify_results);
            } 
            else if (strcmp(argv[2], "TILED") == 0) 
            {
                device_main(N, 1, verify_results);
            }
            else if (strcmp(argv[2], "TILEDWEF") == 0)
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