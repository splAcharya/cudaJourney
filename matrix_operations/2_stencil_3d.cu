#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cuda_runtime.h>

const int VERIF_THRESHOLD = 1e-3;
const int BLOCK_DIM = 8; /* 8 x 8 x 8 =>512 */
const int STENCIL_DIM = 5;
const int STENCIL_RADIUS = STENCIL_DIM / 2;

#define cuda_check_errors(msg) \
    do{ \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) \
        { \
            printf("Fatal Error For :%s =>(%s at %s:%d)\n", \
                    msg, \
                    cudaGetErrorString(__err), \
                    __FILE__,  \
                    __LINE__); \
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



/*
Layout

z = 0
  |0   1   2
--------------    
0 |0   1   2
0 |3   4   5
0 |6   7   8

z = 1
  |0   1   2
--------------    
0 |9   10  11
0 |12  13  14
0 |15  16  17

z = 2
  |0   1   2
--------------    
0 |18  19  20
0 |21  22  23
0 |24  25  26

linear idx = { z * (Y * X)} + {y * X} + x;

Since data in memory will be laid out in row major order.
*/
int generate_float_matrix(int X, int Y, int Z, float *output_matrix)
{   
    for (int z = 0; z < Z; z++)
    {
        for (int y = 0; y < Y; y ++)
        {
            for (int x = 0; x < X; x++)
            {
                int layer_idx = z * (Y * X);
                int row_idx = y * X;
                int col_idx = x;
                int flat_idx = layer_idx + row_idx + col_idx;
                
                float val = ((float ) rand() / RAND_MAX) * 2.0 + 1.0;
                //*(output_matrix + flat_idx) = val;
                *(output_matrix + flat_idx) = flat_idx % 9;
            }
        }
    }
    return 0;
}


void print_3d_matrix(int X, int Y, int Z, float *matrix)
{
    printf("Matrix Tile By Tile\n");
    
    for (int z = 0; z < Z; z++)
    {
        for (int y = 0; y < Y; y++)
        {
            for (int x = 0; x < X; x++)
            {
                int layer_idx = z * (Y * X);
                int row_idx = y * X;
                int col_idx = x;
                int flat_idx = layer_idx + row_idx + col_idx;
                printf("%f |", matrix[flat_idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void hostk_basic(int X, int Y, int Z, float *h_in, float *h_out)
{
    for (int z = 0; z < Z; z++)
    {
        for (int y = 0; y < Y; y++)
        {
            for (int x = 0; x < X; x++)
            {
                float rsum = 0.0f;
                //for (int mz = -1; mz <= 1; mz++)
                for (int mz = -STENCIL_RADIUS; mz <= STENCIL_RADIUS; mz++)
                {
                    //for (int my = -1; my <= 1; my++)
                    for (int my = -STENCIL_RADIUS; my <= STENCIL_RADIUS; my++)
                    {
                        //for (int mx = -1; mx <= 1; mx++)
                        for (int mx = -STENCIL_RADIUS; mx <= STENCIL_RADIUS; mx++)
                        {
                            if ((z + mz >= 0 && z + mz < Z) &&
                                (y + my >= 0 && y + my < Y) &&
                                (x + mx >= 0 && x + mx < X))
                            {
                                int layer_idx = (z + mz) * Y * X;
                                int row_idx = (y + my) * X;
                                int col_idx = x + mx;
                                int flat_idx = layer_idx + row_idx + col_idx;
                                rsum += h_in[flat_idx];
                            }
                        }
                    }
                }
                
                int layer_idx = z * (Y * X);
                int row_idx = y * X;
                int col_idx = x;
                int flat_idx = layer_idx + row_idx + col_idx;
                h_out[flat_idx] = rsum / (STENCIL_DIM * STENCIL_DIM * STENCIL_DIM);
            }
        }
    }
}


int host_main_basic(int X, int Y, int Z)
{
    printf("Executing Host Basic {X:%d, Y:%d, Z:%d}\n", X, Y, Z);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_input, *h_output;
    size_t matrix_size = (Z * Y * X * sizeof(float));
    
    h_input = (float *)malloc(matrix_size);
    h_output = (float *)malloc(matrix_size);
    
    generate_float_matrix(X, Y, Z, h_input);
    //print_3d_matrix(X, Y, Z, h_input);

    cudaEventRecord(begin, 0);

    hostk_basic(X, Y, Z, h_input,h_output);

    cudaEventRecord(end, 0);

    //print_3d_matrix(X, Y, Z, h_output);

    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Host Basic Elapsed Time: %f ms\n", elapsed_time);

    free(h_input);
    free(h_output);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}

int verify_results(int X, int Y, int Z, float *h_in, float *h_out)
{
    int matrix_size = (X * Y * Z * sizeof(float));
    float *h_verif = (float *)malloc(matrix_size);
    hostk_basic(X, Y, Z, h_in, h_verif);

    //print_3d_matrix(X, Y , Z, h_in);
    //print_3d_matrix(X, Y , Z, h_out);
    //print_3d_matrix(X, Y , Z, h_verif);

    for (int z = 0; z < Z; z++)
    {
        for (int y = 0; y < Y; y++)
        {
            for (int x = 0; x < X; x++)
            {
                int layer_idx = z * (Y * X);
                int row_idx = y * X;
                int col_idx = x;
                int flat_idx = layer_idx + row_idx + col_idx;
                float actual = h_out[flat_idx];
                float expected = h_verif[flat_idx];

                if (fabs(actual - expected) > VERIF_THRESHOLD)
                {
                    printf("Wrong Result @(z:%d, y:%d, x:%d)=>[A:%f, E:%f]\n", 
                            z, y, x, actual, expected);
                    exit(1);
                }
            }
        }
    }

    free(h_verif);
    return 0;
}


__global__ void devicek_basic(int X, int Y, int Z, const float *gm_input, float *gm_output)
{
    int ly = threadIdx.y;
    int lx = threadIdx.x;
    int lz = threadIdx.z;
    int gy = (blockDim.y * blockIdx.y) + ly;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int gz = (blockDim.z * blockIdx.z) + lz;

    if (gy < Y && gx < X && gz < Z)
    {
        float rsum = 0.0f;
        //for (int mz = -1; mz <= 1; mz++)
        for (int mz = -STENCIL_RADIUS; mz <= STENCIL_RADIUS; mz++)
        {
            //for (int my = -1; my <= 1; my++)
            for (int my = -STENCIL_RADIUS; my <= STENCIL_RADIUS; my++)
            {
                //for (int mx = -1; mx <= 1; mx++)
                for (int mx = -STENCIL_RADIUS; mx <= STENCIL_RADIUS; mx++)
                {
                    if (gz + mz >= 0 && gz + mz < Z &&
                        gy + my >= 0 && gy + my < Y &&
                        gx + mx >= 0 && gx + mx < X)
                    {
                        int layer_idx = (gz + mz) * Y * X;
                        int row_idx = (gy + my) * X;
                        int col_idx = (gx + mx);
                        int flat_idx = layer_idx + row_idx + col_idx;
                        rsum += (gm_input[flat_idx]);
                    }
                }
            }
        }
        
        int layer_idx = (gz) * Y * X;
        int row_idx = (gy) * X;
        int col_idx = (gx);
        int flat_idx = layer_idx + row_idx + col_idx;
        gm_output[flat_idx] = rsum / (STENCIL_DIM * STENCIL_DIM * STENCIL_DIM);

    }
}

int device_main_basic(int X, int Y, int Z, bool verif)
{
    printf("Executing Device Basic {X:%d, Y:%d, Z:%d}\n", X, Y, Z);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_input, *h_output;
    float *d_input, *d_output;
    size_t matrix_size  = (Y * X * Z * sizeof(float));

    h_input = (float *)malloc(matrix_size);
    h_output = (float *)malloc(matrix_size);

    generate_float_matrix(X, Y, Z, h_input);

    cudaEventRecord(begin, 0);

    cudaMalloc(&d_input, matrix_size);
    cuda_check_errors("D_IN_ALC");
    
    cudaMalloc(&d_output, matrix_size);
    cuda_check_errors("D_OUT_ALC");
    
    cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_D_IN");

    dim3 block_shape(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    int num_blocks_y = (Y +  BLOCK_DIM - 1) / BLOCK_DIM;
    int num_blocks_x = (X + BLOCK_DIM - 1) / BLOCK_DIM;
    int num_blocks_z = (Z + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_shape(num_blocks_x, num_blocks_y, num_blocks_z);
    devicek_basic<<<grid_shape, block_shape>>>(X, Y, Z, d_input, d_output);
    cuda_check_errors("k_launch");

    cudaDeviceSynchronize();
    cuda_check_errors("dev_sync");

    cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_D_OUT");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device basic elapsed time:%f\n", elapsed_time);
    
    if (verif)
    {
        verify_results(X, Y, Z, h_input, h_output);
    }

    cudaFree(d_output);
    cudaFree(d_input);

    free(h_output);
    free(h_input);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


__global__ void devicek_tiled(int X, int Y, int Z, const float *gm_in, float *gm_out)
{
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lz = threadIdx.z;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int gy = (blockDim.y * blockIdx.y) + ly;
    int gz = (blockDim.z * blockIdx.z) + lz;
    int sm_dim_z = BLOCK_DIM + (2 * STENCIL_RADIUS);
    int sm_dim_y = BLOCK_DIM + (2 * STENCIL_RADIUS);
    int sm_dim_x = BLOCK_DIM + (2 * STENCIL_RADIUS);
    int total_elements = sm_dim_z * sm_dim_y * sm_dim_x;
    int block_stride = blockDim.z * blockDim.y * blockDim.x;
    int fbi = (lz * (blockDim.y * blockDim.x)) + (ly * blockDim.x) + lx;

    //__shared__ float sm_in[sm_dim_z][sm_dim_y][sm_dim_x];

    __shared__ float sm_in[BLOCK_DIM + (2 * STENCIL_RADIUS) ][BLOCK_DIM + (2 * STENCIL_RADIUS)][BLOCK_DIM + (2 * STENCIL_RADIUS)];

    /* Block stride loop */
    for (int fbsi = fbi; fbsi < total_elements; fbsi += block_stride)
    {
        int sm_load_z   = fbsi / (sm_dim_y * sm_dim_x);
        int sm_load_rem = fbsi % (sm_dim_y * sm_dim_x);
        int sm_load_y   = sm_load_rem / sm_dim_x;
        int sm_load_x   = sm_load_rem % sm_dim_x;
        
        int gz_load_z = (blockDim.z * blockIdx.z) + sm_load_z - STENCIL_RADIUS;
        int gy_load_y = (blockDim.y * blockIdx.y) + sm_load_y - STENCIL_RADIUS;
        int gx_load_x = (blockDim.x * blockIdx.x) + sm_load_x - STENCIL_RADIUS;

        if ((gz_load_z >= 0 && gz_load_z < Z) &&
            (gy_load_y >= 0 && gy_load_y < Y) &&
            (gx_load_x >= 0 && gx_load_x < X))
        {
            int flat_idx = (gz_load_z * (Y * X)) + (gy_load_y * X) + gx_load_x;
            sm_in[sm_load_z][sm_load_y][sm_load_x] = gm_in[flat_idx];
        }
        else
        {
            sm_in[sm_load_z][sm_load_y][sm_load_x] = 0.0f;
        }
    }

    __syncthreads();

    /* stencil operation */
    if (gz < Z && gy < Y && gx < X)
    {
        float rsum = 0.0f;
        //for (int mz = -1; mz <= 1; mz++)
        for (int mz = -STENCIL_RADIUS; mz <= STENCIL_RADIUS; mz++)
        {
            //for (int my = -1; my <= 1; my++)
            for (int my = -STENCIL_RADIUS; my <= STENCIL_RADIUS; my++)
            {
                //for (int mx = -1; mx <= 1; mx++)
                for (int mx = -STENCIL_RADIUS; mx <= STENCIL_RADIUS; mx++)
                {
                    rsum += sm_in[lz + mz + STENCIL_RADIUS][ly + my + STENCIL_RADIUS][lx + mx + STENCIL_RADIUS];
                }
            }
        }

        int flat_idx = (gz * (Y * X)) + (gy * X) + gx;
        gm_out[flat_idx] = rsum / (STENCIL_DIM * STENCIL_DIM * STENCIL_DIM);
    }
}


int device_main_tiled(int X, int Y, int Z, bool verif)
{
    printf("Executing Device Tiled {X:%d, Y:%d, Z:%d}\n", X, Y, Z);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_input, *h_output;
    float *d_input, *d_output;
    size_t matrix_size  = (Y * X * Z * sizeof(float));

    h_input = (float *)malloc(matrix_size);
    h_output = (float *)malloc(matrix_size);

    generate_float_matrix(X, Y, Z, h_input);

    cudaEventRecord(begin, 0);

    cudaMalloc(&d_input, matrix_size);
    cuda_check_errors("D_IN_ALC");
    
    cudaMalloc(&d_output, matrix_size);
    cuda_check_errors("D_OUT_ALC");
    
    cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_D_IN");

    dim3 block_shape(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    int num_blocks_y = (Y +  BLOCK_DIM - 1) / BLOCK_DIM;
    int num_blocks_x = (X + BLOCK_DIM - 1) / BLOCK_DIM;
    int num_blocks_z = (Z + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_shape(num_blocks_x, num_blocks_y, num_blocks_z);
    devicek_tiled<<<grid_shape, block_shape>>>(X, Y, Z, d_input, d_output);
    cuda_check_errors("k_launch");

    cudaDeviceSynchronize();
    cuda_check_errors("dev_sync");

    cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_D_OUT");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device tiled elapsed time:%f\n", elapsed_time);
    
    if (verif)
    {
        verify_results(X, Y, Z, h_input, h_output);
    }

    cudaFree(d_output);
    cudaFree(d_input);

    free(h_output);
    free(h_input);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}

int main(int argc, char *argv[]) 
{
    if (argc < 8) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [X] [Y] [Z] {verify_results} {run_tests}\n", argv[0]);
        return 1;
    }

    int X = atoi(argv[3]);
    int Y = atoi(argv[4]);
    int Z = atoi(argv[5]);
    
    bool verify_results = (strcmp(argv[6], "TRUE") == 0);
    bool run_tests = (strcmp(argv[7], "TRUE") == 0); 

    printf("X:%u, Y:%u, Z:%u, Verify Results:%d, Tests:%d\n", X, Y, Z, verify_results, run_tests);

    if (X < BLOCK_DIM || Y < BLOCK_DIM || Z < BLOCK_DIM)
    {
        printf("Matrix Size is smaller than Block Size, Exiting\n");
        return 1;
    }

    if (run_tests)
    {
        printf("Tests For Matrix Dims\n");

        host_main_basic(3, 3, 3);
        device_main_basic(3, 3, 3, verify_results);
        device_main_tiled(3, 3, 3, verify_results);
        printf("\n");

        host_main_basic(5, 5, 5);
        device_main_basic(5, 5, 5, verify_results);
        device_main_tiled(5, 5, 5, verify_results);
        printf("\n");
        
        host_main_basic(17, 17, 17);
        device_main_basic(17, 17, 17, verify_results);
        device_main_tiled(17, 17, 17, verify_results);
        printf("\n");
        
        host_main_basic(13, 97, 59);
        device_main_basic(13, 97, 59, verify_results);
        device_main_tiled(13, 97, 59, verify_results);
        printf("\n");

        host_main_basic(113, 197, 159);
        device_main_basic(113, 197, 159, verify_results);
        device_main_tiled(113, 197, 159, verify_results);
        printf("\n");

        host_main_basic(313, 397, 359);
        device_main_basic(313, 397, 359, verify_results);
        device_main_tiled(313, 397, 359, verify_results);
        printf("\n");

        host_main_basic(513, 597, 559);
        device_main_basic(513, 597, 559, verify_results);
        device_main_tiled(513, 597, 559, verify_results);
        printf("\n");

        host_main_basic(713, 797, 759);
        device_main_basic(713, 797, 759, verify_results);
        device_main_tiled(713, 797, 759, verify_results);
        printf("\n");
    }
    else
    {
        if (strcmp(argv[1], "HOST") == 0) 
        {
            if (strcmp(argv[2], "BASIC") == 0) 
            {
                host_main_basic(X, Y, Z);
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
                device_main_basic(X, Y, Z, verify_results);
            } 
            else 
            {
                device_main_tiled(X, Y, Z, verify_results);
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