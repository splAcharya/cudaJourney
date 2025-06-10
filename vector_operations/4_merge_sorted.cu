#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

const int BLOCK_DIM = 32;
const int MAX_BLOCKS = 65535;
const int ELEM_SCALER = 16;
const int DS_PER_BLOCK = BLOCK_DIM * ELEM_SCALER; 

//-------------------------------------------------------------------------------------------------------------
//
//TODO: Profile This Behaviour and Understand occupany...this is fun one to understand occupancy!
// 
//NOTE: It seems higher ELEM_SCALER benefits basic kernel 
//and lower elem scaler benefits TILED
//
//### **1. Basic Kernel (Global Memory Only)**
//- **Higher ELEM_SCALER** means each thread/block processes more elements.
//- This reduces the number of blocks, increases work per thread/block, and improves global memory coalescing.
//- Fewer blocks means less scheduling overhead and better occupancy, as each block does more useful work.
//- The kernel is memory-bound, so maximizing per-block throughput helps.
//
//**Result:**  
//Higher ELEM_SCALER = Fewer, busier blocks = Better for global memory merge.
//
//---
//
//### **2. Tiled Kernel (Shared Memory)**
//- **Lower ELEM_SCALER** means each block processes a smaller tile.
//- Shared memory per block is limited (e.g., 48KB on your GPU). If ELEM_SCALER is too high, you exceed shared memory limits or reduce occupancy (fewer blocks can run concurrently).
//- Smaller tiles fit comfortably in shared memory, allowing more blocks to run in parallel and maximizing occupancy.
//- If the tile is too large, you may run out of shared memory or launch too few blocks, underutilizing the GPU.
//
//**Result:**  
//Lower ELEM_SCALER = More blocks in flight, better shared memory utilization, higher occupancy.
//
//
//-------------------------------------------------------------------------------------------------------------

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


int generate_vector(int *h_in, int N)
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    int min = 1;
    int max = (rand() % (7 - min + 1)) + min;
    int random_value = 0;

    for (int i = 0; i < N; i++)
    {
        random_value = (rand() % (max - min + 1)) + min;
        *(h_in + i) = random_value; 
        min = random_value;
        max = random_value + 10;
    }      
    return 0;
}


int print_matrix(int *h_in, int N)
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


/*
    2   5   8   10   15  19
            a
    1   4   7   11   13  17
            b
    
    1   2   4   5   7      
                    c   
*/
int hostk_merge_sorted(int *h_A, int *h_B, int *h_C, int M, int N)
{
    int pos_a = 0, pos_b = 0;
    int pos_c = 0;
    
    while ((pos_a < M) && (pos_b < N))
    {
        if (h_A[pos_a] < h_B[pos_b])
        {
            h_C[pos_c] = h_A[pos_a];
            pos_a++;
        }
        else
        {
            h_C[pos_c] = h_B[pos_b];
            pos_b++;
        }
        pos_c++;   
    }

    if (pos_a < M)
    {
        while (pos_a < M)
        {
            h_C[pos_c] = h_A[pos_a];
            pos_a++;
            pos_c++;  
        }
    }
    
    if (pos_b < N)
    {
        while (pos_b < N)
        {
            h_C[pos_c] = h_B[pos_b];
            pos_b++;
            pos_c++;  
        }
    }

    return 0;
}


int host_main(int M, int N)
{
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_A, *h_B, *h_C;
    int alc_size_a = M * sizeof(int);
    int alc_size_b = N * sizeof(int);
    int alc_size_c = (M + N) * sizeof(int);

    h_A = (int *)malloc(alc_size_a);
    h_B = (int *)malloc(alc_size_b);
    h_C = (int *)malloc(alc_size_c);

    generate_vector(h_A, M);
    
    generate_vector(h_B, N);

    cudaEventRecord(begin, 0);

    hostk_merge_sorted(h_A, h_B, h_C, M, N);

    //print_matrix(h_A, M);
    //print_matrix(h_B, N);
    //print_matrix(h_C, (M + N));

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Hostk Elapsed Time:%f\n", elapsed_time);

    free(h_C);
    free(h_B);
    free(h_A);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}

int verify_results(int *h_A, int *h_B, int *h_C, int M, int N)
{
    int alc_sz_v = (M + N) * sizeof(int);
    int *h_V = (int *)malloc(alc_sz_v);

    hostk_merge_sorted(h_A, h_B, h_V, M, N);

    for (int i = 0; i < (M + N); i++)
    {
        if (h_V[i] != h_C[i])
        {
            printf("Wrong Result @:%i, h_V:%i, h_C:%i\n", i, h_V[i], h_C[i]);
            break;
        }
    }

    //print_matrix(h_V, (M+N));

    //print_matrix(h_C, (M+N));

    free(h_V);

    return 0;
}

__device__ void device_merge_sequential(
    const int *gm_A, 
    const int *gm_B, 
    int *gm_C,
    int k_start, int k_end, 
    int i_start, int j_start,
    int i_end,   int j_end)
{
    int pos_a = i_start;
    int pos_b = j_start;
    int pos_c = k_start;

    while ((pos_a < i_end) && (pos_b < j_end))
    {
        if (gm_A[pos_a] < gm_B[pos_b])
        {
            gm_C[pos_c] = gm_A[pos_a];
            pos_a++;
        }
        else
        {
            gm_C[pos_c] = gm_B[pos_b];
            pos_b++;
        }

        pos_c++;
    }

    if (pos_a < i_end)
    {
        while (pos_a < i_end)
        {
            gm_C[pos_c] = gm_A[pos_a];
            pos_c++;
            pos_a++;
        }
    }

    if (pos_b < j_end)
    {
        while (pos_b < j_end)
        {
            gm_C[pos_c] = gm_B[pos_b];
            pos_c++;
            pos_b++;
        }
    }
}


__device__ void device_compute_corank(
    int gx, const int *gm_A, const int *gm_B, 
    int M, int N,  int k, int *i, int *j)
{
    int low  = max(0, k - N);
    int high = min(k, M);
    int mid = 0;
    int curr_i = 0; 
    int curr_j = 0;

    //if (gx == 1 && k == 15)
    //    printf("Gx:%d ==> low:%d, High:%d \n", gx, low, high);

    /* binary search */
    while (low < high)
    {
        mid = (low + high) / 2;
        curr_i = mid; 
        curr_j = k - curr_i;
        bool i_in_bounds = (curr_i > 0 && curr_i < M);
        bool j_in_bounds = (curr_j > 0 && curr_j < N);
        bool ij_in_bounds = i_in_bounds && j_in_bounds;

        //if (gx == 1 && k == 15)
        //    printf("gx:%d ==> ci:%d, cj:%d, mid:%d\n", gx, curr_i, curr_j, mid);

        /* condition 1: A[i - 1] <= B[j] ==> If fail, 
           too many elements from A, high = mid - 1*/
        if (ij_in_bounds && (gm_A[curr_i - 1] > gm_B[curr_j]))
            high = mid;

        /* condition 2: B[j - 1] <= A[i] ==> If fail, 
           too few elements from A, low = mid + 1 */
        else if (ij_in_bounds && (gm_B[curr_j - 1] > gm_A[curr_i]))
            low = mid + 1;

        /* found co-ranks, break out*/
        else
        {
            low = mid;
            break;
        }
    }

    *i = low;
    *j = k - low;
}


/*
    1. Determine Block Level Corank Start and End
    2. Load block Level Data from GMEM to SMEM

*/
__global__ void devicek_merge_sorted_tiled(const int *gm_A, const int *gm_B, int *gm_C, int M, int N)
{
    int bx = blockIdx.x;
    int lx = threadIdx.x;
    int gx = (blockDim.x * bx) + lx;
    int ds_total = M + N;
    int total_threads = blockDim.x * gridDim.x;

    __shared__  int blk_ks; 
    __shared__  int blk_ke; 
    __shared__  int blk_is; 
    __shared__  int blk_js; 
    __shared__  int blk_ie; 
    __shared__  int blk_je; 

    blk_ks = (bx * DS_PER_BLOCK);
    blk_ke = ((bx + 1) * DS_PER_BLOCK);    
    blk_is = 0;
    blk_ie = 0;
    blk_js = 0;
    blk_je = 0;
    
    if (bx == gridDim.x - 1)
    {
        int ds_used = (gridDim.x - 1) * DS_PER_BLOCK;
        int ds_remain = ds_total - ds_used;
        blk_ke = blk_ks + ds_remain;
       
        //if (lx == 0)
        //    printf("Bx:%d, Gx:%d, gridDim.x:%d, ds_total:%d, ds_used:%d, ds_remain:%d, blk_ks:%d, blk_ks:%d\n",
        //            bx,     gx,   gridDim.x,  ds_total,   ds_used,    ds_remain,      blk_ks, blk_ke);
    }

    /* only 1 thread per block needs to compute  block level co-rank*/
    if (lx == 0)
    {
        /* compute block level co-ranks */
        if (bx != 0)
            device_compute_corank(gx, gm_A, gm_B, M, N, blk_ks, &blk_is, &blk_js);

        device_compute_corank(gx, gm_A, gm_B, M, N, blk_ke, &blk_ie, &blk_je);
    }

    __syncthreads();

    /* load data from GMEM TO SMEM 
        assume 2 threads per block and we have 4 blocks

               10  20  30  40  50  60      70  80  90  100  110  120
    Cycle1:    T1  T2  T3                  T4  T5  T6
    Cycle2:                T1  T2  T3                   T4   T5   T6   
    */
    int blk_m = blk_ie - blk_is;
    int blk_n = blk_je - blk_js;

    __shared__ int sm_A[DS_PER_BLOCK];
    __shared__ int sm_B[DS_PER_BLOCK];
    __shared__ int sm_C[DS_PER_BLOCK];

    //if (lx == 0)
    //    printf("gx:%d, blk_i[s:%d,e:%d], blk_j[s:%d,e:%d], blk_m:%d, blk_n:%d\n",
    //            gx, blk_is, blk_ie, blk_js, blk_je, blk_m, blk_n);

    if (lx < (blk_m + blk_n))
    {
        /* load data from GMEM TO SMEM in a block stride loop */
        for (int i = lx; i < blk_m; i+= blockDim.x) 
            sm_A[i] = gm_A[blk_is + i];

        /* load data from GMEM TO SMEM in a block stride loop */
        for (int j = lx; j < blk_n; j+= blockDim.x)
            sm_B[j] = gm_B[blk_js + j];     
    }

    /* block wide barrier, wait for SMEM writes to complete */
    __syncthreads();

    /* block level merge */
    if (lx < (blk_m + blk_n))
    {
        int ds_per_thread = (DS_PER_BLOCK + BLOCK_DIM - 1) / BLOCK_DIM;
        int ks = lx * ds_per_thread;
        int ke = (lx + 1) * ds_per_thread;
        int is = 0, ie = 0;
        int js = 0, je = 0;
        
        if (lx == blockDim.x - 1)
        {
            int ds_total = blk_m + blk_n;
            int ds_used = (blockDim.x - 1) * ds_per_thread;
            int ds_remain = ds_total - ds_used;
            ke = ks + ds_remain;
            
            //printf("Gx:%d, lx:%d, ds_total:%d, ds_per_thread:%d, ds_used:%d, ds_remain:%d\n",
            //        gx, lx, ds_total, ds_per_thread, ds_used, ds_remain);
        }

        /* compute co-ranks*/
        if (lx != 0)
            device_compute_corank(gx, sm_A, sm_B, blk_m, blk_n, ks, &is, &js);

        device_compute_corank(gx, sm_A, sm_B, blk_m, blk_n, ke, &ie, &je);

        /* merge data */
        device_merge_sequential(sm_A, sm_B, sm_C, ks, ke, is, js, ie, je);
    }

    /* block wide barrier, wait for SM writes to complete */
    __syncthreads();

    for (int i = lx; i < (blk_m + blk_n); i += blockDim.x)
        gm_C[blk_ks + i] = sm_C[i];
}

/*
1. partition output array among threads
2. determine start and end co-ranks
3. sequential merge
*/
__global__ void devicek_merge_sorted_basic(const int *gm_A, const int *gm_B, int *gm_C, int M, int N)
{
    int bx = blockIdx.x;
    int lx = threadIdx.x;
    int gx = (blockDim.x * bx) + lx;
    int ds_total = M + N;    
    int total_threads = blockDim.x * gridDim.x;

    /*
        total_blocks = gridDim.x;
    */
    
    if (gx < (M + N))
    {
        //printf("Gx:%d, TotalDs:%d, Total_threads:%d, Ds_Per_Threads:%d\n", 
        //        gx, total_ds, total_threads, ds_per_thread);
    
        int ds_per_thread = (DS_PER_BLOCK + BLOCK_DIM - 1) / BLOCK_DIM;
        int k_start = gx * ds_per_thread;
        int k_end   = (gx + 1) * ds_per_thread;
        int i_start = 0, i_end = 0;
        int j_start = 0, j_end = 0;

        /*
            Each block processed DS_PER_BLOCK elements
            last block needs to procesed remaing elements
        */
        if (bx == gridDim.x - 1)
        {
            int ds_used = (gridDim.x - 1) * DS_PER_BLOCK;
            int ds_remain = ds_total - ds_used;
            k_end = k_start + ds_remain;
            
            //if (lx == 0)
            //    printf("Gx:%d, bx:%d, k[s:%d,e:%d], i[s:%d,e:%d], j[s:%d,e:%d]\n", 
            //            gx, bx, k_start, k_end, i_start, i_end, j_start, j_end);
        }

        if (gx != 0)
            device_compute_corank(gx, gm_A, gm_B, M, N, k_start, &i_start, &j_start); 

        device_compute_corank(gx, gm_A, gm_B, M, N, k_end, &i_end, &j_end); 

        //printf("Gx:%d, k[s:%d,e:%d], i[s:%d,e:%d], j[s:%d,e:%d]\n", 
        //        gx, k_start, k_end, i_start, i_end, j_start, j_end);

        device_merge_sequential(gm_A, gm_B, gm_C, k_start, k_end, 
                                 i_start, j_start, i_end, j_end);
    }
}

int device_main(int mode, int M, int N, bool verif)
{
    //M = 8;
    //N = 7;

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    int alc_sz_a = M * sizeof(int);
    int alc_sz_b = N * sizeof(int);
    int alc_sz_c = (M + N) * sizeof(int);

    h_A = (int *)malloc(alc_sz_a);
    h_B = (int *)malloc(alc_sz_b);
    h_C = (int *)malloc(alc_sz_c);

    generate_vector(h_A, M);
    generate_vector(h_B, N);
    
    /* init matrix */
    //h_A[0] = 5;
    //h_A[1] = 10;
    //h_A[2] = 15;
    //h_A[3] = 20;
    //h_A[4] = 25;
    //h_A[5] = 30;
    //h_A[6] = 35;
    //h_A[7] = 40;

    //h_B[0] = 1;
    //h_B[1] = 2;
    //h_B[2] = 3;
    //h_B[3] = 4;
    //h_B[4] = 50;
    //h_B[5] = 60;
    //h_B[6] = 70;
    
    memset(h_C, 0, alc_sz_c);

    //print_matrix(h_A, M);
    //print_matrix(h_B, N);

    cudaEventRecord(begin, 0);

    cudaMalloc(&d_A, alc_sz_a);
    cuda_check_errors("D_ALC_A");

    cudaMemcpy(d_A, h_A, alc_sz_a, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_MCPY_A");

    cudaMalloc(&d_B, alc_sz_b);
    cuda_check_errors("D_ALC_B");

    cudaMemcpy(d_B, h_B, alc_sz_b, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_MCPY_B");

    cudaMalloc(&d_C, alc_sz_c);
    cuda_check_errors("D_ALC_C");

    cudaMemset(d_C, 0, alc_sz_c);
    cuda_check_errors("D_MSET_C");

    /*
        Each Block Will Process DS_PER_BLOCK = BLOCK_DIM * ELEM_SCALER
        So total_data_size = ((M + N) + DS_PER_BLOCK - 1) / DS_PER_BLOCK;
        NUM_BLOCKS = TOTAL_DATA_SIZE / BLOCK_DIM;
    */

    dim3 block_shape(BLOCK_DIM);  //at least 1 complete warp per SM
    int total_ds = ((M + N) + DS_PER_BLOCK - 1) / DS_PER_BLOCK;
    dim3 grid_shape(total_ds);
    //int num_blocks = (total_ds + BLOCK_DIM - 1) / BLOCK_DIM;
    //dim3 grid_shape(num_blocks);

    printf("(M+N:%d), BLKDIM:%d, DSPBLK:%d, ELESCL:%d, TOTALDS:%d\n", 
            (M + N), BLOCK_DIM, DS_PER_BLOCK, ELEM_SCALER, total_ds);

    if (mode == 0)
    {
        devicek_merge_sorted_basic<<<grid_shape, block_shape>>>(d_A, d_B, d_C, M, N);
        cuda_check_errors("KLAUNCH_BASIC");
    }
    else if (mode == 1)
    {
        devicek_merge_sorted_tiled<<<grid_shape, block_shape>>>(d_A, d_B, d_C, M, N); 
        cuda_check_errors("KLAUNCH_TILED");
    }

    cudaDeviceSynchronize();
    cuda_check_errors("DEV_SYNC");

    cudaMemcpy(h_C, d_C, alc_sz_c, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_MCPY_C");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Devicek (Mode:%d) Elapsed Time:%f\n", mode, elapsed_time);

    //print_matrix(h_C, (M+N));

    if (verif)
        verify_results(h_A, h_B, h_C, M , N);

    free(h_C);
    free(h_B);
    free(h_A);

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
        else if (strcmp(argv[2], "TILED") == 0) 
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

    return 0;
}