#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

const int BLOCK_DIM = 1024;
const int MAX_BLOCKS = 65535;

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

int generate_unsigned_char_matrix(unsigned char *h_in, int N)
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    for (int i = 0; i < N; i++)
    {
        unsigned char value = (unsigned char)(rand() % 256);
        *(h_in + i) = value; 
    }      
    return 0;
}

int print_matrix(unsigned char *h_in, int N)
{
    printf("\n");
    
    for (int i = 0; i < N; i++)
    {
        printf("%d|  ", *(h_in + i));

        if (i && i % 7 == 0)
            printf("\n");
    }

    printf("\n");

    return 0;
}

int print_bins(unsigned int *h_in, int N)
{
    printf("\n");
    
    for (int i = 0; i < N; i++)
    {
        printf("|i:%d->val:%d|  ", i, *(h_in + i));

        if (i && i % 7 == 0)
            printf("\n");
    }

    printf("\n");

    return 0;
}

/* serially load from */
void hostk_basic(const unsigned char *gm_in, unsigned int *gm_bins, int M, int N)
{
    for (int y = 0; y < M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            unsigned char bin_idx = gm_in[ y * N + x];
            gm_bins[bin_idx] += 1;
        }
    }
}

int host_main(int M, int N)
{
    printf("Executing Host Main Basic(M:%d x N:%d)\n", M, N);
    
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned char *h_in;
    unsigned int  *h_out;
    int alloc_size_input = M * N * sizeof(unsigned char);
    int alloc_size_bins = 256 * sizeof(unsigned int);

    /* allocate host memory */
    h_in = (unsigned char *)malloc(alloc_size_input);
    h_out = (unsigned int *)malloc(alloc_size_bins);

    /* generate inputs*/
    generate_unsigned_char_matrix(h_in, M * N);

    /* clear output storage: bins*/
    memset(h_out, 0, alloc_size_bins);

    cudaEventRecord(begin, 0);
    hostk_basic(h_in, h_out, M , N);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Host Main Basic Elapsed Time: %f\n", elapsed_time);

    //print_matrix(h_in, M * N);
    //print_bins(h_out, 256);

    /* free host allocation */
    free(h_out);
    free(h_in);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaDeviceReset();

    return 0;
}


void verify_results(const unsigned char *h_in, unsigned int *h_bins, int M, int N)
{
    unsigned int *h_vbins = (unsigned int *)malloc(256 * sizeof(unsigned int));

    hostk_basic(h_in, h_vbins, M , N);

    //print_bins(h_bins, 5);
    //print_bins(h_vbins, 5);

    for (int i = 0; i < 256; i++)
    {
        if (h_bins[i] != h_vbins[i])
        {
            printf("Wrong Results@i:%d=> Actual:%d Vs Expected:%d \n", i, h_bins[i], h_vbins[i]);
            break;
        }
    }

    free(h_vbins);
}

__global__ void devicek_basic(const unsigned char *gm_in, unsigned int *gm_bins, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (bx * blockDim.x) + lx;

    if (gx < N)
    {
        unsigned char bin_idx = gm_in[gx];

        if (bin_idx >= 0 && bin_idx < 256)
            atomicAdd( &gm_bins[bin_idx], 1);
    }
}

__global__ void devicek_smem(const unsigned char *gm_in, unsigned int *gm_bins, int N)
{
    int lx = threadIdx.x;
    int bx = blockIdx.x;
    int gx = (bx * blockDim.x) + lx;

    __shared__ unsigned int sm_bins[256];

    if (lx < 256)
        sm_bins[lx] = 0;

    __syncthreads();

    if (gx < N)
    {
        unsigned char val = gm_in[gx];
        atomicAdd(&sm_bins[val], 1);
    }

    __syncthreads();

    if (lx < 256)
    {
        unsigned int val = sm_bins[lx];
        atomicAdd(&gm_bins[lx], val);
    }
}


int device_main(int mode, int M, int N, bool verif)
{
    printf("Executing Device Main(Mode:%d) (M:%d x N:%d)\n", mode, M, N);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned char *h_in;
    unsigned int *h_bins;
    unsigned char *d_in;
    unsigned int *d_bins;

    unsigned int alloc_size_in = M * N * sizeof(unsigned char);
    unsigned int alloc_size_bins = 256 * sizeof(unsigned int);

    /* allocate host memory */
    h_in = (unsigned char *)malloc(alloc_size_in);
    h_bins = (unsigned int *)malloc(alloc_size_bins);

    /* generate input matrix */
    generate_unsigned_char_matrix(h_in, M * N);

    /* Determine where a single grid can handle input 
       or multiple grids need to launched */
    unsigned int ds_total = M * N;
    unsigned int required_blocks = (ds_total + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned int ds_per_tile = ds_total;
    unsigned int num_tiles = 1;
    unsigned int num_blocks = required_blocks;

    if (required_blocks > MAX_BLOCKS)
    {
        num_blocks = MAX_BLOCKS;
        ds_per_tile = MAX_BLOCKS * BLOCK_DIM; //each grid can launch BLOCK_DIM * NUM_BLOCKS Threads
        num_tiles  = (ds_total + ds_per_tile - 1) / ds_per_tile; 
    }

    if (verif)
    {
        printf("DS_TOTAL:%d, REQ_BLOCKS:%d,  NUM_TILES:%d, DS_PER_TILE:%d, NUM_BLOCKS:%d\n",            
                ds_total, required_blocks, num_tiles, ds_per_tile, num_blocks);
    }

    unsigned int alc_sz_in_dev = ds_per_tile * sizeof(unsigned char);
    unsigned int alc_sz_bins_dev = 256 * sizeof(unsigned int);

    cudaEventRecord(begin, 0);

    /* allocate device memory */
    cudaMalloc(&d_in, alc_sz_in_dev);
    cuda_check_errors("DIN_ALC");

    cudaMalloc(&d_bins, alc_sz_bins_dev);
    cuda_check_errors("DBINS_ALC");

    cudaMemset(d_bins, 0, alc_sz_bins_dev);
    cuda_check_errors("DBINS_MSET0");

    for (int i = 0; i < num_tiles; i++)
    {
        unsigned int start_pos = i * ds_per_tile;
        unsigned int copy_size = sizeof(unsigned char) * ds_per_tile;
        unsigned int num_elements = ds_per_tile;

        /* Last Tile */
        if (i == num_tiles - 1)
        {
            copy_size = (ds_total - start_pos) * sizeof(unsigned char);
            num_elements = copy_size / sizeof(unsigned char);
            num_blocks = (num_elements + BLOCK_DIM - 1) / BLOCK_DIM;
        }

        dim3 block_shape(BLOCK_DIM);
        dim3 grid_shape(num_blocks);

        if (verif)
            printf("Klaunch TileNum:%d/%d\n, Start_Pos:%d, Copy_Size:%d, Num_Elem:%d, Num_Blocks:%d\n", 
                    i+1, num_tiles, start_pos, copy_size, num_elements, num_blocks);

        /* copy current tile to device */
        cudaMemset(d_in, 0, copy_size); 

        cudaMemcpy(d_in, &h_in[start_pos], copy_size, cudaMemcpyHostToDevice);
        cuda_check_errors("DIN_H2D");
      
        if (mode == 0)
        {
            devicek_basic<<<grid_shape, block_shape>>>(d_in, d_bins, num_elements);
            cuda_check_errors("KLAUNCH_BASIC");
            
            cudaDeviceSynchronize();
            cuda_check_errors("DEV_SYNC_BASIC");
        }
        else if (mode == 1)
        {
            devicek_smem<<<grid_shape, block_shape>>>(d_in, d_bins, num_elements);
            cuda_check_errors("KLAUNCH_SMEM");

            cudaDeviceSynchronize();
            cuda_check_errors("DEV_SYNCH_SMEM");
        }
    }

    /* copy results back to host*/
    cudaMemcpy(h_bins, d_bins, alloc_size_bins, cudaMemcpyDeviceToHost);
    cuda_check_errors("HBINS_H2D");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device Mode:%d Elapsed Time:%f \n", mode, elapsed_time);

    if (verif)
    {
        verify_results(h_in, h_bins, M , N);
    }

    cudaFree(d_bins);
    cudaFree(d_in);

    free(h_bins);
    free(h_in);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}

int main(int argc, char *argv[]) 
{
    if (argc < 5) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [M] [N] {verify_results} {run_tests}\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    bool verify_results = (strcmp(argv[5], "TRUE") == 0);

    printf("M:%d, N:%d, Verify Results:%d\n", M, N, verify_results);
 
    if (1)
    {
        if (strcmp(argv[1], "HOST") == 0) 
        {
            if (strcmp(argv[2], "BASIC") == 0) 
            {
                host_main(M, N);
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
                device_main(0, M, N, verify_results);
            } 
            else if (strcmp(argv[2], "SMEM") == 0) 
            {
                device_main(1, M, N, verify_results);
            }
            //else if (strcmp(argv[2], "TILEDWEF") == 0)
            //{
            //    device_main(N, 2, verify_results);
            //}
        } 
        else
        {
            fprintf(stderr, "Unknown mode: %s\n", argv[1]);
            return 1;
        }
    }

    return 0;
}