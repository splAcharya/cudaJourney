#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

typedef unsigned int uint;

const uint BLOCK_DIM = 256;
const uint MAX_BLOCKS = 65535;
const float ERR_THRESHOLD = 1e-3;
const uint NUM_STREAMS = 8;

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
    
    printf("---------- Device Properties ----------\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Grid Size: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Thread Dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Clock Rate: %d MHz\n", prop.clockRate / 1000);
    
    // Properties related to asynchronous operations:
    printf("Device Overlap (async copy support): %s\n",
           prop.deviceOverlap ? "Yes" : "No");
    printf("Concurrent Kernels: %s\n",
           prop.concurrentKernels ? "Yes" : "No");
    printf("Kernel Execution Timeout Enabled: %s\n",
           prop.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("Can Map Host Memory: %s\n",
           prop.canMapHostMemory ? "Yes" : "No");
    printf("Async Engine Count: %d\n", prop.asyncEngineCount);
    printf("Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");
    printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("---------------------------------------\n\n");
}

//void print_device_prop()
//{
//    cudaDeviceProp prop;
//    cudaGetDeviceProperties(&prop, 0);
//    printf("Device Name: %s\n", prop.name);
//    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
//    printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
//    printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
//    printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
//    printf("Number of SMs: %d\n", prop.multiProcessorCount);
//    printf("\n\n");
//}

int generate_vector(float *h_in, uint N)
{
    uint seed = (uint)time(NULL);
    srand(seed);

    for (uint i = 0; i < N; i++)
    {
        float min_val = 10.0;
        float max_val = 20.0;
        float rval = min_val + ((float)rand() / (float)RAND_MAX) * (max_val - min_val);
        *(h_in + i) = rval; 
    }      
    
    return 0;
}

void hostk_basic(const float *gm_a, const float *gm_b, float *gm_c, uint N)
{
    for (uint i = 0; i < N; i++)
        gm_c[i] = gm_a[i] + gm_b[i];
}

int host_main(uint N)
{
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float *h_a, *h_b, *h_c;
    uint alc_sz = N * sizeof(float);

    h_a = (float *)malloc(alc_sz);
    h_b = (float *)malloc(alc_sz);
    h_c = (float *)malloc(alc_sz);

    generate_vector(h_a, N);
    generate_vector(h_b, N);
    //memset(h_c, 0, alc_sz);

    cudaEventRecord(begin, 0);
    
    hostk_basic(h_a, h_b, h_c, N);
    
    cudaEventRecord(end, 0);
    
    cudaEventSynchronize(end);
    
    float elapsed_time = 0.0f;
    
    cudaEventElapsedTime(&elapsed_time, begin, end);
    
    printf("Hostk Elapsed Time:%f\n", elapsed_time);    

    free(h_c);
    free(h_b);
    free(h_a);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int verify_results(const float *gm_a, const float *gm_b, float *gm_c, uint N)
{
    float *gm_v = (float *)malloc(N * sizeof(float));
    hostk_basic(gm_a, gm_b, gm_v, N);

    for (uint i = 0; i < N; i++)
    {
        if (fabs(gm_c[i] - gm_v[i]) > ERR_THRESHOLD)
        {
            printf("Wrong Result @:%d, Actual:%f, Expected:%f\n", i, gm_c[i], gm_v[i]);
            break;
        }
    }

    free(gm_v);
    return 0;
}

__global__ void devicek_basic(const float *gm_a, const float *gm_b, float *gm_c, uint N)
{
    int bx = blockIdx.x;
    int lx = threadIdx.x;
    int gx = (bx * blockDim.x) + lx;

    if (gx < N)
        gm_c[gx] = gm_a[gx] + gm_b[gx];
}


int device_main(uint mode, uint N, bool verify)
{
    printf("Executing Device Main(Mode:%d) For N:%d\n", mode, N);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaStream_t streams_ar[NUM_STREAMS];

    float *h_a, *h_b, *h_c;
    uint alc_sz = N * sizeof(float);

    if (mode == 0)
    {
        h_a = (float *)malloc(alc_sz);
        h_b = (float *)malloc(alc_sz);
        h_c = (float *)malloc(alc_sz);
    }
    else if (mode == 1 || mode == 2)
    {
        cudaMallocHost((void **)&h_a, alc_sz);
        cuda_check_errors("PL_ALC_A");

        cudaMallocHost((void **)&h_b, alc_sz);
        cuda_check_errors("PL_ALC_B");

        cudaMallocHost((void **)&h_c, alc_sz);
        cuda_check_errors("PL_ALC_C");
    }

    generate_vector(h_a, N);
    generate_vector(h_b, N);
    //memset(h_c, 0, alc_sz);

    float *d_a, *d_b, *d_c;

    cudaEventRecord(begin, 0);
    
    cudaMalloc(&d_a, alc_sz);
    cuda_check_errors("ALC_A");
        
    cudaMalloc(&d_b, alc_sz);
    cuda_check_errors("ALC_B");

    cudaMalloc(&d_c, alc_sz);
    cuda_check_errors("ALC_C");

    if (mode == 0 || mode == 1)
    {
        cudaMemcpy(d_a, h_a, alc_sz, cudaMemcpyHostToDevice);
        cuda_check_errors("H2D_A");

        cudaMemcpy(d_b, h_b, alc_sz, cudaMemcpyHostToDevice);
        cuda_check_errors("H2D_B");

        dim3 block_shape(BLOCK_DIM);
        uint num_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 grid_shape(num_blocks);

        devicek_basic<<<grid_shape, block_shape>>>(d_a, d_b, d_c, N);
        cuda_check_errors("KLAUNCH");

        cudaDeviceSynchronize();
        cuda_check_errors("DEV_SYNC");

        cudaMemcpy(h_c, d_c, alc_sz, cudaMemcpyDeviceToHost);
        cuda_check_errors("D2H_C");
    }
    else if(mode == 2)
    {
        for (int i = 0; i < NUM_STREAMS; i++)
            cudaStreamCreate(&streams_ar[i]);

        uint segment_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;

        for (uint i = 0; i < NUM_STREAMS; i++)
        {
            uint segment_start = i * segment_size;
            uint segment_end = (i + 1) * segment_size;

            if (i == NUM_STREAMS - 1)
            {
                uint remaining_data = N - segment_start;
                segment_end = segment_start + remaining_data;
            }

            uint cur_segment_size = segment_end - segment_start;
            dim3 block_shape(BLOCK_DIM);
            uint num_blocks = (cur_segment_size + BLOCK_DIM - 1) / BLOCK_DIM;
            dim3 grid_shape(num_blocks);

            uint copy_size = cur_segment_size * sizeof(float);

            cudaMemcpyAsync(&d_a[segment_start], &h_a[segment_start], 
                            copy_size, cudaMemcpyHostToDevice, streams_ar[i]);
            cuda_check_errors("AH2D_A");

            cudaMemcpyAsync(&d_b[segment_start], &h_b[segment_start], 
                            copy_size, cudaMemcpyHostToDevice, streams_ar[i]);
            cuda_check_errors("AH2D_B");

            devicek_basic<<<grid_shape, block_shape, 0, streams_ar[i]>>>(&d_a[segment_start], &d_b[segment_start], 
                                                                         &d_c[segment_start], cur_segment_size);

            cuda_check_errors("KLAUNCH");

            cudaMemcpyAsync(&h_c[segment_start], &d_c[segment_start], 
                            copy_size, cudaMemcpyDeviceToHost, streams_ar[i]);
            cuda_check_errors("AD2H_A");
        }

        //cudaDeviceSynchronize();
        //cuda_check_errors("DEV_SYNC");

        for (uint i = 0; i < NUM_STREAMS; i++)
        {
            cudaStreamSynchronize(streams_ar[i]);
            cuda_check_errors("STREAM_SYNC");
        }
    }   
    
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Devicek(Mode:%d) Elapsed Time:%f\n", mode, elapsed_time);    

    if (verify)
    {
        verify_results(h_a, h_b, h_c, N);
    }

    if (mode == 2)
    {
        for (uint i = 0; i < NUM_STREAMS; i++)
        {
            cudaStreamDestroy(streams_ar[i]);
        }
    }

    if (mode == 0 || mode == 1)
    {
        cudaFree(d_c);
        cudaFree(d_b);
        cudaFree(d_a);
    }

    if (mode == 1 || mode == 2)
    {
        cudaFreeHost(h_c);
        cuda_check_errors("FH_C");

        cudaFreeHost(h_b);
        cuda_check_errors("FH_B");

        cudaFreeHost(h_a);
        cuda_check_errors("FH_A");
    }
    else
    {
        free(h_c);
        free(h_b);
        free(h_a);
    }
    
    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int main(int argc, char *argv[]) 
{
    if (argc < 5) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [N] {verify_results}\n", argv[0]);
        return 1;
    }

    uint N = (uint)atoi(argv[3]);
    bool verify_results = (strcmp(argv[4], "TRUE") == 0);

    printf("N:%d, Verify Results:%d\n", N, verify_results);
 
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
            device_main(0, N, verify_results);
        }
        else if (strcmp(argv[2], "PINNED") == 0) 
        {
            device_main(1, N, verify_results);
        }
        else if (strcmp(argv[2], "STREAM") == 0) 
        {
            device_main(2, N, verify_results);
        } 
    } 
    else
    {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}