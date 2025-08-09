#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cuda_runtime.h>
#include <time.h>

#define cuda_check_errors(msg) \
    do{\
        cudaError_t __err = cudaGetLastError();\
        if (__err != cudaSuccess){\
            printf("Fatal Error:%s (%s at %s:%d):", msg, \
                cudaGetErrorString(__err), \
                __FILE__, \
                __LINE__);\
        }\
    }while(0)


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

const int BLOCK_DIM = 16; /* 16 * 16 = 256*/
const double ERR_THRES = 1e-3;

void generate_double_matrix(size_t M, size_t N, float *output_matrix) 
{
    for (int y = 0; y < M; y++){
        
        for (int x = 0; x < N; x++){
            
            int flat_idx = (y * N) + x;
            
            float val = ((float) rand() / RAND_MAX) * 2.0 + 1.0;
            *(output_matrix + flat_idx) = val;
        }
    }
}

__host__ __device__
void print_matrix(const float *matrix, size_t M, size_t N)
{
    for (int y = 0; y < M; y++){
        for (int x= 0; x < N; x++){
            printf("%f  ", matrix[ (y * N) + x]);
        }
        printf("\n");
    }
    printf("\n");
}

void hostk_matmul_basic(const float * A, const float * B, float * C, size_t M, size_t N, size_t P)
{
    for (int y = 0; y < M; y ++)
    {
        for (int x = 0; x < P; x++)
        {
            float rsum = 0.0;
            
            /* For every, column in A and every row in B */
            for (int i = 0; i < N; i++)
            {
                /* A = fix row, change col*, A has N element Per Row */
                size_t idx_a = (y * N) + i;

                /* B = change row, fix col, B has P elements Per Row */
                size_t idx_b = (i * P) + x;

                rsum += (A[idx_a] * B[idx_b]);
            }

            /* Update Output Matrix C With The Cumulative Sum of Products 
               C has P elements per row
            */
           size_t idx_c = (y * P) + x;
           C[idx_c] = rsum;
        }
    }
}

void verify_results(const float *h_a, const float *h_b, const float *h_c, size_t M, size_t N, size_t P)
{
    float *h_v = (float *)calloc(M * P, sizeof(float));
    
    hostk_matmul_basic(h_a, h_b, h_v, M, N, P);

    for (int i = 0; i < (M * P); i++)
    {
        if (fabs(h_v[i] - h_c[i]) > ERR_THRES)
        {
            printf("Wrong Result @idx:%d, Expected:%f, Got:%f\n", 
                    i, h_v[i], h_c[i]);

            exit(EXIT_FAILURE);
        }
    }
}

void host_exec_main(const float *h_a, const float *h_b, float *h_c, size_t M, size_t N, size_t P, int verify, int mode)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    if (mode == 1)
        hostk_matmul_basic(h_a, h_b, h_c, M, N, P);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Exec (Mode:%d) Elapsed Time:%f ms\n", mode, elapsed_time);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}

__global__ 
void devicek_matmul_basic(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N, size_t P)
{
    
    int tid_gl_y = threadIdx.y + (blockIdx.y * blockDim.y);
    int tid_gl_x = threadIdx.x + (blockIdx.x * blockDim.x);
    
    if ((tid_gl_y < M) && (tid_gl_x < P)){

        float rsum = 0.0;
        
        for (int i = 0; i < N; i++){
       
            /* Matrix A has N elements per row*/
            int flat_idx_A = (tid_gl_y * N + i);
            
            /* Matrix B has P elements per row */
            int flat_idx_B = (i * P + tid_gl_x);
            
            rsum += (gm_A[flat_idx_A] * gm_B[flat_idx_B]);
        }
       
        int flat_idx_C = (tid_gl_y * P) + tid_gl_x;

        gm_C[flat_idx_C] = rsum;
    }
}



__global__
void devicek_matmul_tiled(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N, size_t P)
{
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int gy = (blockDim.y * blockIdx.y) + ly;

    //printf("bx:%d, by:%d, lx:%d, ly:%d, gx:%d, gy:%d\n", 
    //        blockIdx.x, blockIdx.y, lx,    ly,    gx,    gy);

    //each thread block loads an entire matrix tile into shared memory
    __shared__ float sm_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sm_b[BLOCK_DIM][BLOCK_DIM+1];

    float rsum = 0.0f;

    //for each tiles from mat a and b perfom matrix multiplication
    for (int tileIdx = 0; tileIdx < N; tileIdx += BLOCK_DIM)
    {
        //load tile into a, same row, col right to left
        int row_idx_a = gy;
        int col_idx_a = (tileIdx + lx);
        if (row_idx_a < M  && col_idx_a < N)
            sm_a[ly][lx] = gm_A[ (row_idx_a * N) + col_idx_a];
        else
            sm_a[ly][lx] = 0.0f;
            
        //load tile into b, different row, same column
        int row_idx_b = (tileIdx + ly);
        int col_idx_b = gx;
        if (row_idx_b < N && col_idx_b < P)
            sm_b[ly][lx] = gm_B[ (row_idx_b * P) + col_idx_b];
        else
            sm_b[ly][lx] = 0.0f;
            

        //block wide barrier
        __syncthreads(); //wait for all threads to load data into shared mem

        //matmul
        for (int i = 0; i < BLOCK_DIM; i++)
            rsum += (sm_a[ly][i] * sm_b[i][lx]);

        //block-wide barrier
        __syncthreads();  //wait for all threads before loading next tile
    }
    
    if (gy < M && gx < P)
        gm_C[(gy * P) + gx] = rsum;
}


__global__
void devicek_matmul_warp_shuffle(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N, size_t P)
{
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int gy = (blockDim.y * blockIdx.y) + ly;
    const unsigned int participants = 0xffffffffu;

    //printf("bx:%d, by:%d, lx:%d, ly:%d, gx:%d, gy:%d\n", 
    //        blockIdx.x, blockIdx.y, lx,    ly,    gx,    gy);

    //each thread block loads an entire matrix tile into shared memory
    __shared__ float sm_b[BLOCK_DIM][BLOCK_DIM+1];

    float rsum = 0.0f;

    //for each tiles from mat a and b perfom matrix multiplication
    for (int tileIdx = 0; tileIdx < N; tileIdx += BLOCK_DIM)
    {
        //load tile into a, same row, col right to left
        int row_idx_a = gy;
        int col_idx_a = (tileIdx + lx);
        float a_reg = 0.0f;
        if (row_idx_a < M  && col_idx_a < N)
            a_reg = gm_A[ (row_idx_a * N) + col_idx_a];
        else
            a_reg = 0.0f;
            
        //load tile into b, different row, same column
        int row_idx_b = (tileIdx + ly);
        int col_idx_b = gx;
        if (row_idx_b < N && col_idx_b < P)
            sm_b[ly][lx] = gm_B[ (row_idx_b * P) + col_idx_b];
        else
            sm_b[ly][lx] = 0.0f;

        //block wide barrier
        __syncthreads(); //wait for all threads to load data into shared mem

        //matmul
        for (int i = 0; i < BLOCK_DIM; i++)
        {
            float a_bcast = __shfl_sync(participants, a_reg, i);
            rsum += a_bcast * sm_b[i][lx];
        }

        //block-wide barrier
        __syncthreads();  //ensure current tiles are fully read before next write
    }
    
    if (gy < M && gx < P)
        gm_C[(gy * P) + gx] = rsum;
}


void device_exec_main(const float *h_a, const float *h_b, float *h_c, size_t M, size_t N, size_t P, int verify, int mode)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *d_a, *d_b, *d_c;
    size_t alc_sz_a = (M * N * sizeof(float));
    size_t alc_sz_b = (N * P * sizeof(float));
    size_t alc_sz_c = (M * P * sizeof(float));

    cudaMalloc(&d_a, alc_sz_a);
    cuda_check_errors("dev_alc_a");

    cudaMalloc(&d_b, alc_sz_b);
    cuda_check_errors("dev_alc_b");

    cudaMalloc(&d_c, alc_sz_c);
    cuda_check_errors("dev_alc_c");

    cudaMemcpy(d_a, h_a, alc_sz_a, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_A");

    cudaMemcpy(d_b, h_b, alc_sz_b, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_B");

    cudaMemset(d_c, 0, alc_sz_c);
    cuda_check_errors("MSET_C");
   
    dim3 block_shape(BLOCK_DIM, BLOCK_DIM);
    size_t num_blocks_y = (M + BLOCK_DIM - 1) / BLOCK_DIM;
    size_t num_blocks_x = (P +  BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_shape(num_blocks_x, num_blocks_y);

    //printf("NBY:%d, NBX:%d\n", num_blocks_y, num_blocks_y);

    if (mode == 2)
        devicek_matmul_basic<<<grid_shape, block_shape>>>(d_a, d_b, d_c, M, N, P); 
    else if (mode == 3)
        devicek_matmul_tiled<<<grid_shape, block_shape>>>(d_a, d_b, d_c, M, N, P);
    else if (mode == 4) 
        devicek_matmul_warp_shuffle<<<grid_shape, block_shape>>>(d_a, d_b, d_c, M, N, P);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, alc_sz_c, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_C");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Exec (Mode:%d) Elapsed Time:%f\n", mode, elapsed_time);

    if (verify)
    {
        verify_results(h_a, h_b, h_c, M, N, P);
    }
        
    //print_matrix(h_a, M, N);
    //print_matrix(h_b, N, P);
    //print_matrix(h_c, M, P);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}

/*
1. Generate Test Case ( N Cases if Scaling Test)
2. Execute on a specific variation or compare all
3. Can use preprocessor directive to switch between coperative groups Or Pinned Memory or 
streaming if possible ...
*/
int main(int argc, char *argv[]) 
{
    // argv[0] executable
    // argv[1] mode
    // argv[2] result verification
    // argv[3] M
    // argv[4] N
    // argv[5] P
    // argv[6] Scaling
    if (argc != 7) 
    {
        fprintf(stderr, "Usage: %s [Mode] [Verify Results] [M] [N] [P] [Scaling Tests] \n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);
    int verify_results = atoi(argv[2]);
    size_t M = atoi(argv[3]);
    size_t N = atoi(argv[4]);
    size_t P = atoi(argv[5]);
    int scaling = atoi(argv[6]);
    const int num_tests = 5;

    size_t scaling_sizes[num_tests][3] = {{33, 55, 77}, {99, 77, 99}, {999, 777, 999}, {3333, 5555, 7777}, {9999, 7777, 9999}};

    printf("Data Parameter(M:%zu, N:%zu, P:%zu)=>MatA(M:%zu,N:%zu), MatB(N:%zu,P:%zu), MatC(M:%zu, P:%zu)\n", M, N, P, M, N, N, P, M, P);

    print_device_prop();

    //generate inputs
    float *h_a, *h_b, *h_c;

    if (scaling == 1)
    {
        for (int i = 0; i < num_tests; i++)
        {
            size_t cM = scaling_sizes[i][0];
            size_t cN = scaling_sizes[i][1];
            size_t cP = scaling_sizes[i][2];
           
            printf("MatA(M:%zu,N:%zu), MatB(N:%zu,P:%zu), MatC(M:%zu, P:%zu)\n", 
                    cM, cN, cN, cP, cM, cP);                    

            printf("Doing Paged Alloc\n");

            h_a = (float *)malloc(cM * cN * sizeof(float));
            h_b = (float *)malloc(cN * cP * sizeof(float));
            h_c = (float *)calloc(cM * cP, sizeof(float));

            //fill in data for matrices
            generate_double_matrix(cM, cN, h_a);
            generate_double_matrix(cN, cP, h_b); 

            if (i < 3)
                host_exec_main(h_a, h_b, h_c, cM, cN, cP, verify_results, 1);

            device_exec_main(h_a, h_b, h_c, cM, cN, cP, verify_results, 2);
          
            device_exec_main(h_a, h_b, h_c, cM, cN, cP, verify_results, 3);
          
            device_exec_main(h_a, h_b, h_c, cM, cN, cP, verify_results, 4);

            printf("Freeing Paged Alloc\n\n");
            
            //free input matrices
            free(h_c);
            free(h_b);
            free(h_a);
        }
    }
    else
    {
        printf("MatA(M:%zu,N:%zu), MatB(N:%zu,P:%zu), MatC(M:%zu, P:%zu)\n", 
                M, N, N, P, M, P);

        //allocate matrices
        if (mode > 4)
        {
            printf("Doing Page Locked Alloc\n");

            cudaMallocHost((void **)&h_a, (M * N * sizeof(float)));
            cuda_check_errors("PL_ALC_HA");

            cudaMallocHost((void **)&h_b, (N * P * sizeof(float)));
            cuda_check_errors("PL_ALC_HB");

            cudaMallocHost((void **)&h_c, (M * P * sizeof(float)));
            cuda_check_errors("PL_ALC_C");

            memset(h_c, 0, (M * P * sizeof(float)));
        }
        else
        {
            printf("Doing Paged Alloc\n");

            h_a = (float *)malloc(M * N * sizeof(float));
            h_b = (float *)malloc(N * P * sizeof(float));
            h_c = (float *)calloc(M * P, sizeof(float));
        }

        //fill in data for matrices
        generate_double_matrix(M, N, h_a);
        generate_double_matrix(N, P, h_b);

        //device mode basic
        if (mode == 1)
            host_exec_main(h_a, h_b, h_c, M, N, P, verify_results, 1);
        //device mode basic
        else if (mode == 2 )
            device_exec_main(h_a, h_b, h_c, M, N, P, verify_results, 2);
        //device mode tiled
        else if (mode == 3 )
            device_exec_main(h_a, h_b, h_c, M, N, P, verify_results, 3);
        else if (mode == 4)
            device_exec_main(h_a, h_b, h_c, M, N, P, verify_results, 4);
        ////device mode tiled pinned 
        //else if (mode == 5)
        //    device_exec_main(h_a, h_b, h_c, M, N, P, verify_results, 5);
        //else
        //{

        //}
        
        //free input matrices
        if (mode > 4)
        {
            printf("Freeing Page Locked Alloc\n");

            cudaFreeHost(h_c);
            cuda_check_errors("FR_PL_C");

            cudaFreeHost(h_b);
            cuda_check_errors("FR_PL_B");

            cudaFreeHost(h_a);
            cuda_check_errors("FR_PL_A");
        }
        else
        {
            printf("Freeing Paged Alloc\n");
            
            free(h_c);
            free(h_b);
            free(h_a);
        }
    }

    return 0;
}

