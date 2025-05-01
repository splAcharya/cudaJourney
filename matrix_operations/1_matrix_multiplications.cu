#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cuda_runtime.h>

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

void generate_float_matrix(size_t M, size_t N, float *output_matrix) 
{
    
    for (int y = 0; y < M; y++){
        
        for (int x = 0; x < N; x++){
            
            int flat_idx = (y * N) + x;
            
            float val = ((float) rand() / RAND_MAX) * 2.0 + 1.0;
            *(output_matrix + flat_idx) = val;
            
            *(output_matrix + flat_idx) = flat_idx % 200;
            //*(output_matrix + flat_idx) = 1.0f;
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
    /* for every row, col of the output matrix
       A = (M, N), B= (N, P), C = (M, P)
    */
    for (int y = 0; y < M; y ++)
    {
        for (int x = 0; x < P; x++)
        {
            float rsum = 0.0f;
            
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

// Function declarations
void host_main_basic(size_t M, size_t N, size_t P) 
{
    printf("Executing host_main(M:%zu, N:%zu, P:%zu)\n", M, N, P);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    size_t alloc_size_h_a = M * N * sizeof(float);
    size_t alloc_size_h_b = N * P * sizeof(float);
    size_t alloc_size_h_c = M * P * sizeof(float);

    h_A = (float *)malloc(alloc_size_h_a);
    h_B = (float *)malloc(alloc_size_h_b);
    h_C = (float *)malloc(alloc_size_h_c);

    generate_float_matrix(M, N, h_A);
    generate_float_matrix(N, P, h_B);

    cudaEventRecord(start, 0);

    hostk_matmul_basic(h_A, h_B, h_C, M, N, P);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Basic Elapsed Time: %f ms\n", elapsed_time);

    //print_matrix(h_A, M, N);
    //print_matrix(h_B, N, P);
    //print_matrix(h_C, M, P);
    
    free(h_A);
    free(h_B);
    free(h_C);
}


void verify_results(const float *A, const float *B, const float *C, size_t M, size_t N, size_t P)
{
    size_t alloc_size_v = M * P * sizeof(float);
    float* V = (float *)malloc(alloc_size_v);
    
    /* get results from simplest matrix mul algo */
    hostk_matmul_basic(A, B, V, M, N, P);
   
    //print_matrix(C, M, P);
    //printf("\n");
    //print_matrix(V, M, P);
    
    for (int y = 0; y < M; y++)
    {
        for (int x = 0; x < P; x++)
        {
            int flat_idx = (y * P) + x;
            
            if (fabsf(C[flat_idx] - V[flat_idx]) > 1e-3)
            // if (h_C[flat_idx] != h_V[flat_idx])
            {
                printf("Wrong Result From Host Tiled @(y,x):(%i,%i)=>flat:%i!\n", y, x, flat_idx);
                printf("Host Val:%f, Verification Val:%f\n", C[flat_idx], V[flat_idx]);
                exit(1);
            }
        }
    } 

    printf("Correct Result It Semms Within [1e-3] Tolerance \n");

    free(V);
}


void hostk_matmul_tiled(const float * A, const float *B, float *C, size_t M, size_t N, size_t P)
{
    printf("In Host Tiled(M:%zu, N:%zu, P:%zu)\n", M, N, P);
 
    /* define tiles, set the size same as the block size in device
    to simulate device like kernels */
    float sm_A[BLOCK_DIM][BLOCK_DIM];
    float sm_B[BLOCK_DIM][BLOCK_DIM];
    float sm_C[BLOCK_DIM][BLOCK_DIM];
   
    size_t num_tiles = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    printf("Num Tiles:%zu, BlockDim:%u\n", num_tiles, BLOCK_DIM);

    //print_matrix(A, M, N);
    //print_matrix(B, N, P);

    /* for each tile of the output matrix */
    for (int y = 0; y < M; y += BLOCK_DIM)
    {
        for (int x = 0; x < P; x += BLOCK_DIM)
        {
            /* Init given output tile */
            memset(sm_C, 0, (sizeof(float) * BLOCK_DIM * BLOCK_DIM));

            for (int tileIdx = 0; tileIdx < N; tileIdx += BLOCK_DIM)
            {
                /* Init Shared memory matrices */
                memset(sm_A, 0, (sizeof(float) * BLOCK_DIM * BLOCK_DIM));
                memset(sm_B, 0, (sizeof(float) * BLOCK_DIM * BLOCK_DIM));

                //printf("This SM Portion Init \n");
                //print_matrix((const float *)sm_A, BLOCK_DIM, BLOCK_DIM);
                //print_matrix((const float *)sm_B, BLOCK_DIM, BLOCK_DIM);

                /* Load values for matix A(M x N) same row count as  C (M X P) */
                for (int ty = 0; ( (ty < BLOCK_DIM) && ( (y + ty) < M ) ); ty ++)
                {
                    /* The column count however should be the tile idx */
                    for (int tx = 0; ( (tx < BLOCK_DIM) && ( (tileIdx + tx) < N)); tx++)
                    {
                        int row_index = (y + ty) * N; //each row has N elements
                        int col_index = (tileIdx + tx);
                        sm_A[ty][tx] = A[row_index + col_index];
                    }
                }

                /* Load values for matrix B(N x P) different row count as C (M x P),
                   however it should be alighted to tile iterator */
                for (int ty = 0; ((ty < BLOCK_DIM) && ( (tileIdx + ty) < N) ); ty++)
                {
                    /* the col position should be same as the output matrix */
                    for (int tx = 0; ( (tx < BLOCK_DIM) && ( (x + tx) < P) );  tx++)
                    {
                        int row_index = (tileIdx + ty) * P; //each row has P elements
                        int col_index = (x + tx);
                        sm_B[ty][tx] = B[row_index + col_index];
                    }
                }

                /* Perform Matrix Multiplication */
                //printf("This SM Portion \n");
                //print_matrix((const float *)sm_A, BLOCK_DIM, BLOCK_DIM);
                //print_matrix((const float *)sm_B, BLOCK_DIM, BLOCK_DIM);
                
                /* The values written in output matrix C is  tile by tile left to right in x direction */
                for (int ty = 0; ( (ty < BLOCK_DIM) && ((y + ty) < M) );  ty++)
                {
                    for (int tx = 0; ( (tx < BLOCK_DIM) && ( (x + tx) < P) ); tx++)
                    {
                        float rsum = 0.0f;
                        for (int k = 0; ((k < BLOCK_DIM) && ((tileIdx + k) < N )); k++)
                        {
                            rsum += (sm_A[ty][k] * sm_B[k][tx]);
                            //printf("Eh: A:%f, B:%f Rsum:%f\n", sm_A[ty][k], sm_B[k][tx], rsum);   
                        }
                        //printf("  Rsum: %f", rsum);
                        
                        sm_C[ty][tx] += rsum;
                    }
                    //printf("\n");
                }
                //printf("\n");

                //print_matrix((const float *)sm_C, BLOCK_DIM, BLOCK_DIM);
            }

            /* copy values from shared tile to global memory */
            for (int ty = 0; ((ty < BLOCK_DIM) && ((y + ty) < M)) ; ty++)
            {
                for (int tx = 0; ((tx < BLOCK_DIM) && ((x + tx) < P)) ; tx++)
                {
                    int row_index = (y + ty) * P; //P elements per row
                    int col_index = (x + tx);
                    C[row_index + col_index] = sm_C[ty][tx];
                }
            }
        }
    }
}


void host_main_tiled(bool verif, size_t M, size_t N, size_t P)
{
    printf("Executing host_main_tiled(M:%zu, N:%zu, P:%zu)\n", M, N, P);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    size_t alloc_size_h_a = M * N * sizeof(float);
    size_t alloc_size_h_b = N * P * sizeof(float);
    size_t alloc_size_h_c = M * P * sizeof(float);

    h_A = (float *)malloc(alloc_size_h_a);
    h_B = (float *)malloc(alloc_size_h_b);
    h_C = (float *)malloc(alloc_size_h_c);
    memset(h_C, 0, alloc_size_h_c);

    generate_float_matrix(M, N, h_A);
    generate_float_matrix(N, P, h_B);

    cudaEventRecord(start, 0);
    hostk_matmul_tiled(h_A, h_B, h_C, M, N, P);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Tiled Elapsed Time:%f ms\n", elapsed_time);

    /* verify against traditional */
    if (verif)
    {
        verify_results(h_A, h_B, h_C, M, N, P);
    } 

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);
}

/* simple device kernel, each thread is responsible for an element in the output matrix,
    so each threads is responsible for processing and entire row and entire column
*/
__global__ 
void devicek_matmul_basic(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N, size_t P)
{
    
    int tid_gl_y = threadIdx.y + (blockIdx.y * blockDim.y);
    int tid_gl_x = threadIdx.x + (blockIdx.x * blockDim.x);
    
    if ((tid_gl_y < M) && (tid_gl_x < P)){

        float rsum = 0.0f;
        
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


void device_main_basic(bool verif, size_t M, size_t N, size_t P) 
{
    printf("Executing device_main_basic(M:%zu, N:%zu, P:%zu) \n", M, N, P);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t alloc_size_a = M * N * sizeof(float);
    size_t alloc_size_b = N * P * sizeof(float);
    size_t alloc_size_c = M * P * sizeof(float);

    h_A = (float *)malloc(alloc_size_a);
    h_B = (float *)malloc(alloc_size_b);
    h_C = (float *)malloc(alloc_size_c);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(1);
    }

    generate_float_matrix(M, N, h_A);
    generate_float_matrix(N, P, h_B);

    cudaEventRecord(start, 0);

    cudaMalloc(&d_A, alloc_size_a);
    cuda_check_errors("alloc A in device");
    
    cudaMalloc(&d_B, alloc_size_b);
    cuda_check_errors("alloc B in device");

    cudaMalloc(&d_C, alloc_size_c);
    cuda_check_errors("alloc C in device");

    //copy data
    cudaMemcpy(d_A, h_A, alloc_size_a, cudaMemcpyHostToDevice);
    cuda_check_errors("mecmpy A H2D");
   
    cudaMemcpy(d_B, h_B, alloc_size_b, cudaMemcpyHostToDevice);
    cuda_check_errors("memcpy B H2D");

    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    size_t num_blocks_y = (M + block_size.y - 1) / block_size.y;
    size_t num_blocks_x = (P + block_size.x - 1) / block_size.x;
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_matmul_basic<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, P);
    cuda_check_errors("klaunch");
    cudaDeviceSynchronize();
    cuda_check_errors("dev synch");
    cudaMemcpy(h_C, d_C, alloc_size_c, cudaMemcpyDeviceToHost);
    cuda_check_errors("memcpy C D2H");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Basic Elapsed Time: %f ms\n", elapsed_time);

    //print_matrix(h_A, M, N);
    //print_matrix(h_B, N, P);
    //print_matrix(h_C, M, P);

    //verify
    if (verif)
    {
        verify_results(h_A, h_B, h_C, M, N, P);
    } 

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}


/*
Tile Matrix Multiplication Using Device
1. The tile size will be same as the defined/max BLOCK_SIZE of the device.
2. The threads within same block can access a shared memory.
3. Each thread will access a element from global memory (from both Mat A and B).
   Then write that result in the shared memory.
4. Once the shared memory is populate, each thread perfom matrix multiplication
   for 1 element of the output matrix. Swapping tiles in and out.
*/
__global__ 
void devicek_matmul_tiled(const float* gm_A, const float* gm_B, float* gm_C, size_t M, size_t N, size_t P)
{
    //printf("M:%d\n", M);
    //printf("N:%d\n", N);
    //printf("P:%d\n", P);

    size_t tid_gl_y = (blockDim.y * blockIdx.y) + threadIdx.y;
    size_t tid_gl_x = (blockDim.x * blockIdx.x) + threadIdx.x;
    size_t tid_lc_y = threadIdx.y;
    size_t tid_lc_x = threadIdx.x;

    __shared__ float sm_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sm_B[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sm_C[BLOCK_DIM][BLOCK_DIM];


    /* NOTE: Previously, the most liekly issue was with the whole idea
    of loading into shared memory... if all the computations happen
    under the check (tid_gl_y < M) && (tid_gl_x < P)... however
    the idea is for a block of thread to first load value into the shared 
    memory, for thread block that are not alined to input dimnesions,
    we cannos imply ignore threads beyond the M and P boundary, 
    i.e for 3 x 3 matrix, the first 2 x 2 block works fine since
    all the threads from 0..2 read from global memory and then write
    into shared memory, however, threads 2...4 for the next block
    only thread 2..3 will load vaalues the thread 3..4 will not do
    anyhting thus the shared memory will not fully be read properly..
    simply hostk_tiling cannot be mapped to device... need a little 
    more thought :) .... this took a whole day... LMFAO !!!
   
            YOU NEED A FULL FREAKING BLOCK OF THREADS TO F****KING
            LOAD VALUES INTO THE TILE FIRST...ISN'T IT DUMBASS!!! :)
    */

    /* init shared memory portion of the thread */
    sm_C[tid_lc_y][tid_lc_x] = 0.0f;

    /* wait for all threads to finih shared memory init */
    //__syncthreads();

    for (int tileIdx = 0; tileIdx < N; tileIdx += BLOCK_DIM)
    {
        int row_idx_a = (tid_gl_y * N); //N elements per row
        int col_idx_a = (tileIdx + tid_lc_x);   
        int idx_a = row_idx_a + col_idx_a;
        
        if ((tid_gl_y < M) && (col_idx_a < N))
            sm_A[tid_lc_y][tid_lc_x] = gm_A[idx_a];
        else
            sm_A[tid_lc_y][tid_lc_x] = 0.0f;

        int row_idx_b = (tileIdx + tid_lc_y) * P; //P elements per row
        int col_idx_b = (tid_gl_x);
        int idx_b = row_idx_b + col_idx_b;

        if ( ((tileIdx + tid_lc_y) < N) && ( col_idx_b < P) )
            sm_B[tid_lc_y][tid_lc_x] = gm_B[idx_b];
        else
            sm_B[tid_lc_y][tid_lc_x] = 0.0f;

        __syncthreads();

        float rsum = 0.0f;
        for (int ti = 0; ti < BLOCK_DIM; ti++)
            rsum += (sm_A[tid_lc_y][ti] * sm_B[ti][tid_lc_x]);

        sm_C[tid_lc_y][tid_lc_x] += rsum;
    }

    __syncthreads();

    if (tid_gl_y < M && tid_gl_x < P)
    {
        int row_idx_c = (tid_gl_y * P); //P elements per row
        int col_idx_c = (tid_gl_x);
        int idx_c = row_idx_c + col_idx_c;
        gm_C[idx_c] = sm_C[tid_lc_y][tid_lc_x];
    }
}

void device_main_tiled(bool verif, size_t M, size_t N, size_t P) 
{
    printf("Executing device_main_tiled(M:%zu, N:%zu, P:%zu) \n", M, N, P);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    size_t alloc_size_a = M * N * sizeof(float);
    size_t alloc_size_b = N * P * sizeof(float);
    size_t alloc_size_c = M * P * sizeof(float);

    h_A = (float *)malloc(alloc_size_a);
    h_B = (float *)malloc(alloc_size_b);
    h_C = (float *)malloc(alloc_size_c);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(1);
    }

    generate_float_matrix(M, N, h_A);
    generate_float_matrix(N, P, h_B);

    //print_matrix(h_A, M, N);
    //print_matrix(h_B, N, P);

    cudaEventRecord(start, 0);

    /* Allocate memory in device */
    cudaMalloc(&d_A, alloc_size_a);
    cuda_check_errors("alloc A in device");

    cudaMalloc(&d_B, alloc_size_b);
    cuda_check_errors("alloc B in device");

    cudaMalloc(&d_C, alloc_size_c);
    cuda_check_errors("alloc C in device");

    /* copy data from host to device */
    cudaMemcpy(d_A, h_A, alloc_size_a, cudaMemcpyHostToDevice);
    cuda_check_errors("h2d A");

    cudaMemcpy(d_B, h_B, alloc_size_b, cudaMemcpyHostToDevice);
    cuda_check_errors("h2d B");

    /* Laucnh kernel */
    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    size_t num_blocks_y = (M + block_size.y - 1) / block_size.y;
    size_t num_blocks_x = (P + block_size.x - 1) / block_size.x;
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_matmul_tiled<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, P);
    cuda_check_errors("k-launch");

    /* Make Host Wait for GPU i.e finish all GPU operatiosn so far */
    cudaDeviceSynchronize();
    cuda_check_errors("dev sync");
    /* copy results back to host */
    cudaMemcpy(h_C, d_C, alloc_size_c, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device tiled Elapsed Time: %f ms\n", elapsed_time);
    
    //verify
    if (verif)
    {
        verify_results(h_A, h_B, h_C, M, N, P);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
}


int main(int argc, char *argv[]) 
{
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <mode> [submode] {result_verif} [M] [N] [P] \n", argv[0]);
        return 1;
    }

    bool verif = (strcmp(argv[3], "TRUE") == 0);
    printf("Result Verification ? : %d\n", verif);
    size_t M = atoi(argv[4]);
    size_t N = atoi(argv[5]);
    size_t P = atoi(argv[6]);
    printf("Data Parameter(M:%zu, N:%zu, P:%zu)=>MatA(M:%zu,N:%zu), MatB(N:%zu,P:%zu), MatC(M:%zu, P:%zu)\n", M, N, P, M, N, N, P, M, P);

    if (M < BLOCK_DIM || N < BLOCK_DIM || P < BLOCK_DIM)
    {
        printf("Matrix Size is smaller than Block Size, skipping tiled matrix multiplication\n");
        return 1;
    }

    if (strcmp(argv[1], "HOST") == 0) {
        if (argc < 4){
            fprintf(stderr, "Usage: %s HOST <submode> <result_verif>\n", argv[0]);
            return 1;
        }
        if (strcmp(argv[2], "BASIC") == 0) {
            host_main_basic(M, N, P);
        } else {
            host_main_tiled(verif, M, N, P);
        }
    } else if (strcmp(argv[1], "DEVICE") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s DEVICE <submode> <result_verif>\n", argv[0]);
            return 1;
        }

        print_device_prop();

        if (strcmp(argv[2], "BASIC") == 0) {
            device_main_basic(verif, M, N, P);
        } else {
            device_main_tiled(verif, M, N, P);
        }
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
