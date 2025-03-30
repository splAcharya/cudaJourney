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
const int BLOCK_DIM = 15; /* 16 * 16 = 256*/

void generate_float_matrix(size_t M, size_t N, float *output_matrix) {
    for (int y = 0; y < M; y++){
        for (int x = 0; x < N; x++){
            int flat_idx = y * M + x;
            float val = ((float) rand() / RAND_MAX) * 2.0 + 1.0;
            *(output_matrix + flat_idx) = val;
            
            //*(output_matrix + flat_idx) = flat_idx % 200;
            //*(output_matrix + flat_idx) = 1.0f;
        }
    }
}

void print_matrix(const float *matrix, size_t M, size_t N){
    for (int y = 0; y < M; y++){
        for (int x= 0; x < N; x++){
            printf("%f  ", matrix[y * M + x]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void hostk_matmul_basic(const float * A, const float * B, float * C, size_t M, size_t N){
    
    for (int y = 0; y < M; y++){
        
        for (int x = 0; x < N; x++){
            
            /* entire row X entire column */
            float cur_sum = 0.0f;
            for (int i = 0; i < M; i++)
                cur_sum += (A[y * M + i] * B[i * N + x]);

            C[y * N + x] = cur_sum;  
        }
    }
}

// Function declarations
void host_main() {
    printf("Executing host_main()\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    size_t alloc_size = 0;

    alloc_size = DIM_SIZE * DIM_SIZE * sizeof(float);
    h_A = (float *)malloc(alloc_size);
    h_B = (float *)malloc(alloc_size);
    h_C = (float *)malloc(alloc_size);

    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_A);
    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_B);

    cudaEventRecord(start, 0);

    hostk_matmul_basic(h_A, h_B, h_C, DIM_SIZE, DIM_SIZE);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Elapsed Time: %f ms\n", elapsed_time);

    //print_matrix(h_A, DIM_SIZE, DIM_SIZE);
    //print_matrix(h_B, DIM_SIZE, DIM_SIZE);
    //print_matrix(h_C, DIM_SIZE, DIM_SIZE);
    
    free(h_A);
    free(h_B);
    free(h_C);
}


/*
A00  A01       B00  B01
A10  A11       B10  B11

C00 = A00 * B00   +    A01 * B10

*/
void hostk_matmul_tiled_cm(const float *A, const float *B, float *C, size_t y_dim, size_t x_dim)
{
    //printf("Submat, y_dim:%zu, x_dim:%zu\n", y_dim, x_dim);
    /* FOr each output element */
    //float lc[y_dim][x_dim];

    for (int y = 0; y < y_dim; y++){
        for (int x = 0; x < x_dim; x++){

            /* iterate entire row for A, entire column for B*/
            float rsum = 0.0f;
            for (int i = 0; i < y_dim; i++){
                size_t flat_idx_a = (y * x_dim) + i;
                size_t flat_idx_b = (i * x_dim) + x;
                //printf("Submat, flat_idx_a:%zu, flat_idx_b:%zu\n", flat_idx_a, flat_idx_b);
                rsum += (A[flat_idx_a] * B[flat_idx_b]);
            }
           
            size_t flat_idx_c = y * x_dim + x;
            //printf("Submat, flat_idx_c:%zu\n", flat_idx_c);
            C[flat_idx_c] += rsum;
            //lc[y][x] = rsum;
        }
    }

    //float lv[y_dim][x_dim];
    //hostk_matmul_basic(A, B, (float*)lv, y_dim, x_dim);

    //for (int y = 0; y < y_dim; y++){
    //    for (int x = 0; x < x_dim; x++){
    //        if (lc[y][x] != lv[y][x]){
    //            printf("Wrong SubMultiplicatoion @(y:%d,x:%d), verif:%f, res:%f\n", y, x, lv[y][x], lc[y][x]);
    //            exit(1);
    //        }
    //    }
    //}

    //print_matrix(A, y_dim, x_dim);
    //print_matrix(B, y_dim, x_dim);
    //print_matrix(C, y_dim, x_dim);
    //printf("------Sub Mat----\n");
}



/* think from the output matrix perspective */
//if (tile == 0)
//    sm_C[ty][tx] = 0.0f; //;
/* dry run             //
    In the matrix below,

    FOr Matix A, tiles iterater from let to right, T0 --T1
    T0 ==> {0,0} to flat idx
    0   y   +   x 
    1   y   +   (x   +   1)
    4   {(y  + 1)* DIM_SIZE} + x
    5   (y + 1) * DIM_SIZE  + x + 1

    T1 ==> (0, 1) to flat idx
    2   y   +   (x  * BLOCK_DIM)
    3   y   +   (x  * BLOCK_DIM + 1)
    6   {(y  + 1) * DIM_SIZE} + (x * BLOCK_DIM)
    7   {(y + 1 ) * DIM_SIZE} + (x * BLOCK_DIM + 1)

    Generalize >>> {(y + ty) * DIM_SIZE} + {(x * BLOCK_DIM) + tx}

    T0 ==> {x0, y0}
    ty  tx
    0   0   0+0 * 4 + 0 * 2 + 0 = 0
    0   1   0+0 * 4 + 0 * 2 + 1 = 1
    1   0   0+1 * 4 + 0 * 2 + 0 = 4
    1   1   0+1 * 4 + 0 * 2 + 1 = 5
    T2  ==> {x0,y2}
    ty  tx
    0   0   2 + 0 * 4 + 0 * 2 + 0 = 8
    0   1   2 + 0 * 4 + 0 * 2 + 1 = 9
    1   0   2 + 1 * 4 + 0 * 2 + 0 = 12
    1   1   2 + 1 * 4 + 0 * 2 + 1 = 13

    Now for matrix B, we go from top to bottom . T0 ==> T2
    same generalization should apply as long as y value is 
    inmcremented accordiny.

   -------------------
   |    T0  |    T1  | 
   | -------|--------|
   | 0   1  | 2   3  |
   | 4   5  | 6   7  |
   | ----------------|
   | 8   9  | 10  11 |
   | 12  13 | 14  15 |
   | ----------------|
   |    T2  |   T3   |
   -------------------
*/
void hostk_matmul_tiled(const float * A, const float *B, float *C, size_t y_dim, size_t x_dim){

    printf("Host Tiled: In hostk_matmul_tiled\n");
    float sm_A[BLOCK_DIM][BLOCK_DIM];
    float sm_B[BLOCK_DIM][BLOCK_DIM];
    float sm_C[BLOCK_DIM][BLOCK_DIM];

    int num_tiles = y_dim / BLOCK_DIM;
    printf("hostk_matmul_tiled=> DATA_SIZE:%d, BLOCK_SIZE:%d, num_tiles:%d \n", DIM_SIZE, BLOCK_DIM ,num_tiles);
   // printf("Host Tiled: In hostk_matmul_tiled, Allocated Local Memory\n");

    /* iterate for each tile of otput matrix */
    for (int y = 0; y < y_dim; y += BLOCK_DIM){
        for (int x = 0; x < x_dim; x+= BLOCK_DIM){

            for (int ty = 0; ty < BLOCK_DIM; ty ++){
                for (int tx = 0; tx < BLOCK_DIM; tx++){
                    sm_A[ty][tx] = 0.0f;
                    sm_B[ty][tx] = 0.0f;
                    sm_C[ty][tx] = 0.0f;
                }
            }

            for (int tile = 0; tile < num_tiles; tile ++){

                for (int ty = 0; ty < BLOCK_DIM; ty++){
                    for (int tx = 0; tx < BLOCK_DIM; tx++){

                        int flat_idx_a = ((y + ty) * DIM_SIZE) + ((tile * BLOCK_DIM) + tx);
                        int flat_idx_b = ((tile * BLOCK_DIM + ty) * DIM_SIZE) + (x + tx);
                        //int flat_idx_b = (((tile * BLOCK_DIM) + ty) * DIM_SIZE) + (x  + tx);
                        //printf("flat_idx=>A:%d, B:%d\n", flat_idx_a, flat_idx_b);

                        sm_A[ty][tx] = A[flat_idx_a];
                        sm_B[ty][tx] = B[flat_idx_b]; 
                    }
                }
                
                hostk_matmul_tiled_cm((const float *)sm_A, (const float *)sm_B, (float *)sm_C, BLOCK_DIM, BLOCK_DIM);
            }
            
            for (int ty = 0; ty < BLOCK_DIM; ty ++){
                for (int tx = 0; tx < BLOCK_DIM; tx++){
                    int flat_idx_c = ((y + ty) * DIM_SIZE) + (x + tx);
                    //printf("flat_idx_c(y:%d,x:%d): %d\n", y, x, flat_idx_c);
                    C[flat_idx_c] = sm_C[ty][tx];
                }
            }
        }
    }
}


void host_main_tiled(bool verif){
    printf("Executing host_main_tiled()\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C;
    size_t alloc_size = 0;

    printf("Host Tiled: About to Begin Host Kernel\n");

    alloc_size = DIM_SIZE * DIM_SIZE * sizeof(float);
    h_A = (float *)malloc(alloc_size);
    h_B = (float *)malloc(alloc_size);
    h_C = (float *)malloc(alloc_size);
    
    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_A);
    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_B);

    //printf("Host Tiled: About to Begin Host Kernel\n");

    cudaEventRecord(start, 0);

    hostk_matmul_tiled(h_A, h_B, h_C, DIM_SIZE, DIM_SIZE);
    //print_matrix(h_C, DIM_SIZE, DIM_SIZE);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Tiled Elapsed Time:%f ms\n", elapsed_time);

    /* verify against traditional */
    if (verif)
    {
        float *h_V = (float *)malloc(alloc_size);
        hostk_matmul_basic(h_A, h_B, h_V, DIM_SIZE, DIM_SIZE);
        //print_matrix(h_V, DIM_SIZE, DIM_SIZE);

        for (int y =0; y < DIM_SIZE; y++){
            for (int x = 0; x < DIM_SIZE; x++){
                int flat_idx = y * DIM_SIZE + x;
                if (fabsf(h_C[flat_idx] - h_V[flat_idx]) > 1e-3)
               // if (h_C[flat_idx] != h_V[flat_idx])
                {
                    printf("Wrong Result From Host Tiled @(y,x):(%i,%i)=>flat:%i!\n", y, x, flat_idx);
                    printf("Host Val:%f, Verification Val:%f\n", h_C[flat_idx], h_V[flat_idx]);
                    exit(1);
                }
            }
        }
        free(h_V);
    } 

    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* simple device kernel, each thread is responsible for an element in the output matrix,
    so each threads is responsible for processing and entire row and entire column
*/
__global__ void devicek_matmul_basic(const float *gm_A, const float *gm_B, float *gm_C, size_t M, size_t N)
{
    
    int tid_gl_y = threadIdx.y + (blockIdx.y * blockDim.y);
    int tid_gl_x = threadIdx.x + (blockIdx.x * blockDim.x);
    
    if ((tid_gl_y < M) && (tid_gl_x < N)){
        float rsum = 0.0f;
        for (int i = 0; i < M; i++){
            int flat_idx_A = (tid_gl_y * M + i);
            int flat_idx_B = (i * M + tid_gl_x);
            rsum += (gm_A[flat_idx_A] * gm_B[flat_idx_B]);
        }
        gm_C[tid_gl_y * M + tid_gl_x] = rsum;
    }
}

void device_main_basic(bool verif) {
    printf("Executing device_main_basic()\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C, *h_V;
    float *d_A, *d_B, *d_C;
    size_t alloc_size = 0;

    alloc_size = DIM_SIZE * DIM_SIZE * sizeof(float);
    h_A = (float *)malloc(alloc_size);
    h_B = (float *)malloc(alloc_size);
    h_C = (float *)malloc(alloc_size);

    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_A);
    generate_float_matrix(DIM_SIZE, DIM_SIZE, h_B);

    cudaEventRecord(start, 0);

    cudaMalloc(&d_A, alloc_size);
    cuda_check_errors("alloc A in device");
    
    cudaMalloc(&d_B, alloc_size);
    cuda_check_errors("alloc B in device");

    cudaMalloc(&d_C, alloc_size);
    cuda_check_errors("alloc C in device");

    //copy data
    cudaMemcpy(d_A, h_A, alloc_size, cudaMemcpyHostToDevice);
    cuda_check_errors("mecmpy A H2D");
   
    cudaMemcpy(d_B, h_B, alloc_size, cudaMemcpyHostToDevice);
    cuda_check_errors("memcpy B H2D");

    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    int num_blocks_x = (DIM_SIZE + block_size.x - 1) / block_size.x;
    int num_blocks_y = (DIM_SIZE + block_size.y - 1) / block_size.y; 
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_matmul_basic<<<grid_size, block_size>>>(d_A, d_B, d_C, DIM_SIZE, DIM_SIZE);
    cuda_check_errors("klaunch");
    cudaDeviceSynchronize();
    cuda_check_errors("dev synch");
    cudaMemcpy(h_C, d_C, alloc_size, cudaMemcpyDeviceToHost);
    cuda_check_errors("memcpy C D2H");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Basic Elapsed Time: %f ms\n", elapsed_time);

    //verify
    if (verif)
    {
        h_V = (float *)malloc(alloc_size);
        hostk_matmul_basic(h_A, h_B, h_V, DIM_SIZE, DIM_SIZE);

        for (int y =0; y < DIM_SIZE; y++){
            for (int x = 0; x < DIM_SIZE; x++){
                int flat_idx = y * DIM_SIZE + x;
                if (fabsf(h_C[flat_idx] - h_V[flat_idx]) > 1e-3)
                //if (h_C[flat_idx] != h_V[flat_idx])
                {
                    printf("Wrong Result From Device @(y,x):(%i,%i)=>flat:%i!\n", y, x, flat_idx);
                    printf("Host Val:%f, DeVIce Val:%f\n", h_C[flat_idx], h_V[flat_idx]);
                    exit(1);
                }
            }
        }
        
        free(h_V);
        printf("Correct Results it seems\n");
    } 

    //print_matrix(h_A, DIM_SIZE, DIM_SIZE);
    //print_matrix(h_B, DIM_SIZE, DIM_SIZE);
    //print_matrix(h_C, DIM_SIZE, DIM_SIZE);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 

    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
        hostk_matmul_basic(h_A, h_B, h_V, DIM_SIZE, DIM_SIZE);

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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mode> [submode] {result_verif}\n", argv[0]);
        return 1;
    }

    bool verif = (strcmp(argv[3], "TRUE") == 0);
    printf("Result Verification ? : %d\n", verif);

    if (strcmp(argv[1], "HOST") == 0) {
        if (argc < 4){
            fprintf(stderr, "Usage: %s HOST <submode> <result_verif>\n", argv[0]);
            return 1;
        }
        if (strcmp(argv[2], "BASIC") == 0) {
            host_main();
        } else {
            host_main_tiled(verif);
        }
    } else if (strcmp(argv[1], "DEVICE") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s DEVICE <submode> <result_verif>\n", argv[0]);
            return 1;
        }

        print_device_prop();

        if (strcmp(argv[2], "BASIC") == 0) {
            device_main_basic(verif);
        } else {
            device_main_tiled(verif);
        }
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
