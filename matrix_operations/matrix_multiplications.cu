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

const int DIM_SIZE =  95;
const int BLOCK_DIM = 16; /* 16 * 16 = 256*/

void generate_float_matrix(size_t M, size_t N, float *output_matrix) {
    
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

void print_matrix(const float *matrix, size_t M, size_t N){
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
void host_main_basic(size_t M, size_t N, size_t P) {
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
    printf("Host Elapsed Time: %f ms\n", elapsed_time);

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

    if (M < BLOCK_DIM || N < BLOCK_DIM || P < BLOCK_DIM)
    {
        printf("Matrix Size is smaller than Block Size, skipping tiled matrix multiplication\n");
        return;
    }

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
__global__ void devicek_matmul_basic(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N, size_t P)
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


void device_main_basic(bool verif, size_t M, size_t N, size_t P) {
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
   The goal is to acheive some level of data level parallelism

   1. Bring data from global memory to shared memory, 
   2. The data in shared memory should be acessible to all threads within the tile/block.
      So now do tiled matrix mltiplication.
   3. Copy results over to output memory
  
   Iterate for all tiles 
		   Each thread block must do all respective
		   sub matrix computation i.e 
		   each thread block must, 
		   	a) load tiles/blocks from matrix A left to right
			b) load tiles/blocks from matrix B top to bttom
			c) perfrom submatrix multiplication and accumate that
			   for all tiles [left to right] x [top to bottom]

		  Each thread is responsible for 1 output element, now in tiled
		  region, each thread block is responsible for each output
		  tile.
		  Given a thread's (x,y) position, 
		  Each thread must load and element from left to right for given
		  row/y positiion of matrix A

*/
__global__ void devicek_matmul_tiled(const float* gm_A, const float* gm_B, float* gm_C, size_t Y_DIM, size_t X_DIM)
{
    /*
        Steps
        1. for each tile, load data from GM TO SM
        2. Perform Matrix Multiplication
        3. Write Output Back to GLobal Matrix
    */

    int tid_gl_x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int tid_gl_y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int tid_lc_x = threadIdx.x;
    int tid_lc_y = threadIdx.y;

    __shared__ float sm_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sm_B[BLOCK_DIM][BLOCK_DIM];
    float running_partial_sum = 0.0f;

    size_t num_tiles = (X_DIM + BLOCK_DIM - 1) / BLOCK_DIM;

    /*
    
           Tile    0       1
        ---------------------------------
           col     0   1   2   3

    Tile|ROw    
    0   |   0      0   1   2   3    
        |   1      4   5   6   7
    1   |   2      8   9   10  11
        |   3      12  13  14  15

    Tile Jump
    0   0   1
    1   2   3

    Inter Tile Jump: TileIdx * num_elements per tile
    Inter Tile: Inter Tile + offset
    */

    if ((tid_gl_y < Y_DIM) && (tid_gl_x < X_DIM))
    {
        //An entire thread block loads its portion of data into shared memory 
        //   starting from 0th tile to Nth tile. Each thread
        // for its given its (y,x) position loads its data
        running_partial_sum = 0.0f;
       
        //num_tiles = 1;
        for (int i  = 0; i < num_tiles; i++)
        { 
            //for given thread'x (y,x) position load the specific item within current tile
            // i.e for matrix A, it will be the same y position, but the x postition will be
            // tile pos + offset of the sepcifc element withint the tile, whihc is 
            //exactly the thread offset ==> tile_pos + threadIdx.x
            //now for Matrix B, tile elements are (tile_size * x_dim apart)

            int tile_pos_a = (i * BLOCK_DIM);
            int a_idx =  (tid_gl_y * X_DIM) +  (tile_pos_a + threadIdx.x);

            int tile_pos_b = (i  * BLOCK_DIM * X_DIM);
            int b_idx =  (tile_pos_b + (threadIdx.y * X_DIM)) + tid_gl_x;

            sm_A[tid_lc_y][tid_lc_x] = gm_A[a_idx];
            sm_B[tid_lc_y][tid_lc_x] = gm_B[b_idx];

            //printf("TGLY:%u, TGLX:%u, LCLY:%u, LGLX:%u, GM_A_IDX:%u, GM_B_IDX:%u, GM_A:%f, GM_B:%f\n", 
            //        tid_gl_y, tid_gl_x, tid_lc_y, tid_lc_x, a_idx, b_idx, gm_A[a_idx], gm_B[b_idx]);

            //printf("TGLY:%u, TGLX:%u, SM_A:%f, SM_B:%f,  SM_C:%f\n", 
            //       tid_gl_y, tid_gl_x, sm_A[tid_lc_y][tid_lc_x], 
            //       sm_B[tid_lc_y][tid_lc_x], sm_C[tid_lc_y][tid_lc_x]);

            //wait for shared memory populate to finish
            __syncthreads();

            //now that each thread had loaded its portion of data into shared memory, 
            //perfom matrix muplication on the sub matrix, each thread only does
            //its portion of matrix multiplication i.e only 1 element of the output matrix
            float rsum = 0.0f;
            for (int idx = 0; idx < BLOCK_DIM; idx++)
            {
                //printf("TGLY:%u, TGLX:%u, SM_A:%f, SM_B:%f,  SM_C:%f\n", 
                //   tid_gl_y, tid_gl_x, sm_A[tid_lc_y][tid_lc_x], 
                //   sm_B[tid_lc_y][tid_lc_x], sm_C[tid_lc_y][tid_lc_x]);

                rsum += (sm_A[tid_lc_y][idx] * sm_B[idx][tid_lc_x]);
            }

            //udpate shared memory
            running_partial_sum += rsum;
        }

        //wait for full tiles to completel yupdate shared memory
        __syncthreads();

        //now that all the partial multiplications are completed,
        //update the global mmemory output
        int c_idx = tid_gl_y* X_DIM + tid_gl_x;
        gm_C[c_idx] = running_partial_sum;
        //printf("Outputidx:%u, Val:%f\n", c_idx, gm_C[c_idx]);
    }
}

void device_main_tiled(bool verif) {
    printf("Executing device_main_tiled()\n");
    // Add the device_main_opt implementation here

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t alloc_size = DIM_SIZE * DIM_SIZE * sizeof(float);

    h_A = (float *)malloc(alloc_size);
    h_B = (float *)malloc(alloc_size);
    h_C = (float *)malloc(alloc_size);

    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_A);
    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_B);

    cudaEventRecord(start, 0);

    /* Allocate memory in device */
    cudaMalloc(&d_A, alloc_size);
    cuda_check_errors("alloc A in device");

    cudaMalloc(&d_B, alloc_size);
    cuda_check_errors("alloc B in device");

    cudaMalloc(&d_C, alloc_size);
    cuda_check_errors("alloc C in device");

    /* copy data from host to device */
    cudaMemcpy(d_A, h_A, alloc_size, cudaMemcpyHostToDevice);
    cuda_check_errors("h2d A");

    cudaMemcpy(d_B, h_B, alloc_size, cudaMemcpyHostToDevice);
    cuda_check_errors("h2d B");

    /* Laucnh kernel */
    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    size_t num_blocks_y = (DIM_SIZE + block_size.y - 1) / block_size.y;
    size_t num_blocks_x = (DIM_SIZE + block_size.x - 1) / block_size.x;
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_matmul_tiled<<<grid_size, block_size>>>(d_A, d_B, d_C, DIM_SIZE, DIM_SIZE);
    cuda_check_errors("klancuh");
    /* Make Host Wait for GPU i.e finish all GPU operatiosn so far */
    cudaDeviceSynchronize();
    cuda_check_errors("dev sync");
    /* copy results back to host */
    cudaMemcpy(h_C, d_C, alloc_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device tiled Elapsed Time: %f ms\n", elapsed_time);
    
    //verify
    if (verif)
    {
        float *h_V = (float *)malloc(alloc_size);
        hostk_matmul_basic(h_A, h_B, h_V, DIM_SIZE, DIM_SIZE, DIM_SIZE);

        for (int y =0; y < DIM_SIZE; y++){
            for (int x = 0; x < DIM_SIZE; x++){
                int flat_idx = y * DIM_SIZE + x;
                if (fabsf(h_C[flat_idx] - h_V[flat_idx]) > 1e-3)
                //if (h_C[flat_idx] != h_V[flat_idx])
                {
                    printf("Wrong Result From Device @(y,x):(%i,%i)=>flat:%i!\n", y, x, flat_idx);
                    printf("Result Val:%f, Verif Val:%f\n", h_C[flat_idx], h_V[flat_idx]);
                    exit(1);
                }
            }
        }

        free(h_V);
        printf("Correct Results it seems\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
}

int main(int argc, char *argv[]) {
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
            device_main_tiled(verif);
        }
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
