#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

const unsigned int BLOCK_DIM = 32;
const unsigned int MAX_BLOCKS = 65535;
const float ERR_THRESHOLD = 1e-2;
const float MAX_PRIVATE_FRONTIER_SCALER = 1.5;
const unsigned int MAX_PRIVATE_FRONTIER_SZ = BLOCK_DIM * MAX_PRIVATE_FRONTIER_SCALER;

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


int generate_csr_matrix(
    unsigned int *h_in_src, unsigned int M, 
    unsigned int *h_in_dst, unsigned int NNZ)
{
    unsigned int *row_counts = (unsigned int *)calloc(M, sizeof(unsigned int));
    unsigned int *row_ids    = (unsigned int *)malloc(NNZ * sizeof(unsigned int));

    // Step 1: Assign each nonzero to a row and record it
    for (int i = 0; i < NNZ; i++)
    {
        unsigned int r = rand() % M;
        row_counts[r]++;
        row_ids[i] = r;
    }

    // Step 2: Build row pointer (prefix sum)
    h_in_src[0] = 0;
    for (unsigned int i = 0; i < M; i++)
        h_in_src[i+1] = h_in_src[i] + row_counts[i];

    // Step 3: Fill in col/val arrays in row-major order
    unsigned int *row_offsets = (unsigned int *)malloc(M * sizeof(unsigned int));
    for (unsigned int i = 0; i < M; i++)
        row_offsets[i] = h_in_src[i];

    for (unsigned int i = 0; i < NNZ; i++) 
    {
        unsigned int r = row_ids[i];
        unsigned int idx = row_offsets[r]++;
        h_in_dst[idx]  = rand() % M;
        //h_in_src[idx] = rand() % M;
        //h_in_dst[idx] = (float)(rand() % 50) / 10.0f;
    }

    free(row_counts);
    free(row_offsets);
    free(row_ids);

    return 0;
}

void print_int_matrix(unsigned int * mat, unsigned int M)
{
    for (unsigned int i = 0; i < M; i++)
    {
        printf("%d |", mat[i]);
    }
    printf("\n");
}

int hostk_vertex_bfs(
    unsigned int M, unsigned int start, const unsigned int *gm_src, 
    const unsigned int *gm_dst, unsigned int *gm_dists, 
    unsigned int *gm_frntr, unsigned int *gm_nxt_frntr)
{
    //Init the frontier with starting vertex
    for (unsigned int i = 0; i < M; i++)
        gm_dists[i] = UINT_MAX; //-1 indicates unvisited

    gm_dists[start] = 0;

    gm_frntr[0] =  start;
    unsigned int  frntr_sz = 1; //first frontier is size 1, just for start
    unsigned int level = 0;

    //as long as there are frontiers to be explored
    while (frntr_sz > 0)
    {
        unsigned int nxt_frntr_sz = 0;

        //printf("frntr_sz:%d\n", frntr_sz);

        for (unsigned int i = 0; i < frntr_sz; i++)
        {
            unsigned int u = gm_frntr[i]; //for the current frontier

            for (int j = gm_src[u]; j < gm_src[u + 1]; j++)
            {
                unsigned int v = gm_dst[j];

                if (gm_dists[v] == UINT_MAX)
                {
                    gm_dists[v] = level + 1;
                    gm_nxt_frntr[nxt_frntr_sz] = v;
                    nxt_frntr_sz++;
                }
            }
        }

        //update frontier to next frontier
        unsigned int *temp = gm_frntr;
        gm_frntr = gm_nxt_frntr;
        gm_nxt_frntr = temp;
        frntr_sz = nxt_frntr_sz;
        level++;
    }

    return 0;
}


int host_main_bfs(int M, int NNZ, bool verify)
{
    printf("Executing Host Main Bfs M:%d, NNZ:%d\n", M, NNZ);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned int *h_in_src, *h_in_dst, *h_in_dist;
    unsigned int *h_frontier, *h_nfrontier;
    unsigned int alc_sz_src    = (M + 1) * sizeof(unsigned int);
    unsigned int alc_sz_dst    = NNZ     * sizeof(unsigned int);
    unsigned int alc_sz_dist   = M       * sizeof(unsigned int);
    unsigned int alc_sz_frntr  = M       * sizeof(unsigned int);
    unsigned int alc_sz_nfrntr = M       * sizeof(unsigned int);

    h_in_src    = (unsigned int *)malloc(alc_sz_src);
    h_in_dst    = (unsigned int *)malloc(alc_sz_dst);
    h_in_dist   = (unsigned int *)malloc(alc_sz_dist);
    h_frontier  = (unsigned int *)malloc(alc_sz_frntr);
    h_nfrontier = (unsigned int *)malloc(alc_sz_nfrntr);

    generate_csr_matrix(h_in_src, M, h_in_dst, NNZ);

    cudaEventRecord(begin, 0);

    hostk_vertex_bfs(M, 0, h_in_src, h_in_dst, h_in_dist, 
                     h_frontier, h_nfrontier);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Hostk Elapsed time: %f ms\n", elapsed_time);

    free(h_nfrontier);
    free(h_frontier);
    free(h_in_dist);
    free(h_in_dst);
    free(h_in_src);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


void verify_results(
    const unsigned int *gm_src, const unsigned int *gm_dst, 
    const unsigned int *gm_dist, unsigned int *gm_frntr, 
    unsigned int *gm_nxt_frntr, unsigned int M, unsigned int start)
{
    unsigned int *h_vdist = (unsigned int *)malloc(M * sizeof(unsigned int));

    hostk_vertex_bfs(M, start, gm_src, gm_dst, h_vdist, gm_frntr, gm_nxt_frntr); 

    for(unsigned int i = 0; i < M; i++)
    {
        if (h_vdist[i] != gm_dist[i])
        {
            printf("Wrong Result @:%d, Expected:%d, Actual:%d \n", 
                   i, h_vdist[i], gm_dist[i]);
            break;
        }
    }

    free(h_vdist);
}

__global__ void devicek_explore_frontier_bfs(
    const unsigned int *gm_src, const unsigned int *gm_dst, unsigned int *gm_dists,
    const unsigned int *gm_frontier, const unsigned int gm_frontier_sz,
    unsigned int *gm_next_frontier, unsigned int *gm_next_frontier_size, 
    unsigned int level)
{
    unsigned int bx = blockIdx.x;
    unsigned int lx = threadIdx.x;
    unsigned int gx = (bx * blockDim.x) + lx;

    if (gx < gm_frontier_sz)
    {
        unsigned int u = gm_frontier[gx]; //get source vertex
        unsigned int start_pos = gm_src[u];
        unsigned int end_pos = gm_src[u + 1];

        for (unsigned int i = start_pos; i < end_pos; i++)
        {
            unsigned int v = gm_dst[i]; //destination vertex

            //in the current array of frontiers (vertex) that 
            //are being explore parallely by threads, multiple
            //threads could visit the same the neighbor i.e dst
            //vertex, so atomic operation is used to ensure
            //only 1 update takes place to mark a destination
            //as being visited
            //if not visited (-1), atomically swap level + 1
            if (atomicCAS(&gm_dists[v], UINT_MAX, level + 1) == UINT_MAX)
            {
                //atomic add return old value at the address 
                //before the addition
                unsigned int pos = atomicAdd(gm_next_frontier_size, 1);
                gm_next_frontier[pos] = v;
            }
        }
    }
}



__global__ void devicek_explore_frontier_bfs_tiled(
    const unsigned int *gm_src, const unsigned int *gm_dst, unsigned int *gm_dists,
    const unsigned int *gm_frontier, const unsigned int gm_frontier_sz,
    unsigned int *gm_next_frontier, unsigned int *gm_next_frontier_size, 
    unsigned int level, unsigned int M)
{
    unsigned int bx = blockIdx.x;
    unsigned int lx = threadIdx.x;
    unsigned int gx = (bx * blockDim.x) + lx;

    __shared__ unsigned int sm_next_frontier[MAX_PRIVATE_FRONTIER_SZ];
    __shared__ unsigned int sm_next_frontier_size;
    __shared__ unsigned int sm_merge_start_pos;
    
    if (lx == 0)
    {
        sm_next_frontier_size = 0;  
        sm_merge_start_pos = 0;
    }    

    __syncthreads();

    if (gx < gm_frontier_sz)
    {
        unsigned int u = gm_frontier[gx];
        unsigned int start_pos = gm_src[u];
        unsigned int end_pos = gm_src[u + 1];

        for (unsigned int i = start_pos; i < end_pos; i++)
        {
            unsigned int v = gm_dst[i];

            if (atomicCAS(&gm_dists[v], UINT_MAX, level + 1) == UINT_MAX)
            {
                unsigned int pos = atomicAdd(&sm_next_frontier_size, 1);
                
                if (pos < MAX_PRIVATE_FRONTIER_SZ)
                {
                    sm_next_frontier[pos] = v;
                }
                else
                {
                    //either start merging directly in global
                    //or implement some flush mechanism
                } 
            }
        }
    }

    __syncthreads();

    //merge to global --> reserve memory
    if (lx == 0 && gx < M)
        sm_merge_start_pos = atomicAdd(gm_next_frontier_size, sm_next_frontier_size);

    __syncthreads();
    
    //block stride merge
    if (gx < M)
    {
        for (unsigned int i = lx; i < sm_next_frontier_size; i += blockDim.x)
            gm_next_frontier[sm_merge_start_pos + i] = sm_next_frontier[i]; 
    }
}

int device_main_bfs(unsigned int M, unsigned int NNZ, bool verify, unsigned int mode)
{
    printf("Executing Device Main Bfs(Mode:%d) M:%d, NNZ:%d\n", mode, M, NNZ);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned int *h_in_src, *h_in_dst, *h_in_dist;
    unsigned int *h_frontier, *h_nfrontier;
    unsigned int alc_sz_src    =  (M + 1)  * sizeof(unsigned int);
    unsigned int alc_sz_dst    =  NNZ      * sizeof(unsigned int);
    unsigned int alc_sz_dist   =  M        * sizeof(unsigned int);
    unsigned int alc_sz_frntr  =  M        * sizeof(unsigned int);
    unsigned int alc_sz_nfrntr =  M        * sizeof(unsigned int);

    h_in_src    = (unsigned int *)malloc(alc_sz_src);
    h_in_dst    = (unsigned int *)malloc(alc_sz_dst);
    h_in_dist   = (unsigned int *)malloc(alc_sz_dist);
    h_frontier  = (unsigned int *)malloc(alc_sz_frntr);
    h_nfrontier = (unsigned int *)malloc(alc_sz_nfrntr);

    generate_csr_matrix(h_in_src, M, h_in_dst, NNZ);

    cudaEventRecord(begin, 0);

    ///dont' need most dist, frontier, nfrontier ...only for very perfhaps

    //---AAACTUALLLYYY YOU CAN COMPARE THE DISTANCE MATRIX

    unsigned int *d_in_src, *d_in_dst, *d_in_dist;
    unsigned int *d_in_frntr, *d_in_nxt_frntr;
    unsigned int start  = 0;

    cudaMalloc(&d_in_src, alc_sz_src);
    cuda_check_errors("ALC_SRC");

    cudaMalloc(&d_in_dst, alc_sz_dst);
    cuda_check_errors("ALC_DST");

    cudaMalloc(&d_in_dist, alc_sz_dist);
    cuda_check_errors("ALC_DIST");

    cudaMalloc(&d_in_frntr, alc_sz_frntr);
    cuda_check_errors("ALC_FRNTR");

    cudaMalloc(&d_in_nxt_frntr, alc_sz_nfrntr);
    cuda_check_errors("ALC_NFRNTR");

    cudaMemcpy(d_in_src, h_in_src, alc_sz_src, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_SRC");

    cudaMemcpy(d_in_dst, h_in_dst, alc_sz_dst, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_DST");

    for(int i = 0; i < M; i++)
        h_in_dist[i] = UINT_MAX;
    h_in_dist[start] = 0;
    
    cudaMemcpy(d_in_dist, h_in_dist, alc_sz_dist, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_DIST");

    cudaMemset(d_in_frntr, 0, alc_sz_frntr);
    cuda_check_errors("MSET_FRNTR");

    cudaMemset(d_in_nxt_frntr, 0, alc_sz_nfrntr);
    cuda_check_errors("MSET_NFRNTR");

    //intial frontier size is 0
    unsigned int frntr_sz = 1;
    unsigned int *d_nxt_frntr_sz;
    cudaMalloc(&d_nxt_frntr_sz, sizeof(unsigned int));
    cuda_check_errors("ALC_NFRNTRSZ");

    //initial frontier is the start index
    h_frontier[0] = start; 
    unsigned int level = 0;
    cudaMemcpy(d_in_frntr, h_frontier, alc_sz_frntr, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_FRNTRSZ");

    //for each frontier to explroe launch kernel
    while (frntr_sz > 0)
    {
        cudaMemset(d_nxt_frntr_sz, 0, sizeof(unsigned int));
        cuda_check_errors("MSET_NFRNTRSZ");

        dim3 block_shape(BLOCK_DIM);
        unsigned int num_blocks = (frntr_sz + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 grid_shape(num_blocks);

        //explore current frontier
        if (mode == 0)
        {
            devicek_explore_frontier_bfs<<<grid_shape, block_shape>>>(d_in_src, d_in_dst, d_in_dist, d_in_frntr, 
                                                                      frntr_sz, d_in_nxt_frntr, d_nxt_frntr_sz, level);
        }
        else if (mode == 1)
        {
            devicek_explore_frontier_bfs_tiled<<<grid_shape, block_shape>>>(d_in_src, d_in_dst, d_in_dist, d_in_frntr, 
                                                                            frntr_sz, d_in_nxt_frntr, d_nxt_frntr_sz, level, M);
        }

        cudaDeviceSynchronize();
        cuda_check_errors("DEV_SYNC");

        //update next frontier size
        cudaMemcpy(&frntr_sz, d_nxt_frntr_sz, sizeof(int), cudaMemcpyDeviceToHost);
        cuda_check_errors("D2H_NFRNTRSZ");

        //update next fotnier array
        unsigned int *temp = d_in_frntr;
        d_in_frntr = d_in_nxt_frntr;
        d_in_nxt_frntr = temp;
        level++;
    }
    
    //copy distance
    cudaMemcpy(h_in_dist, d_in_dist, alc_sz_dist, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_DIST");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Devicek Elapsed time: %f ms\n", elapsed_time);

    if (verify)
    {
        verify_results(h_in_src, h_in_dst, h_in_dist, h_frontier, h_nfrontier, M, start);
    }

    cudaFree(d_nxt_frntr_sz);
    cudaFree(d_in_nxt_frntr);
    cudaFree(d_in_frntr);
    cudaFree(d_in_dist);
    cudaFree(d_in_dst);
    cudaFree(d_in_src);

    free(h_nfrontier);
    free(h_frontier);
    free(h_in_dist);
    free(h_in_dst);
    free(h_in_src);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}

int main(int argc, char *argv[]) 
{
    if (argc < 5) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [M] [NNZ_PCT] {verify_results}\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[3]);

    float NNZ_PCT = (float)atoi(argv[4]);

    int NNZ = (M * M) * (NNZ_PCT / 100);
 
    int graph_mode = (strcmp(argv[5], "BFS") == 0) ? 0 :
                    (strcmp(argv[5], "DFS") == 0) ? 1 : 2;

    bool verify_results = (strcmp(argv[6], "TRUE") == 0);

    printf("M:%d, MxM:%d, NNZ_PCT:%f, NNZ:%d, MODE:%d, verify_results:%d\n", 
            M,   (M*M),   NNZ_PCT,    NNZ,    graph_mode, verify_results);
 
    if (strcmp(argv[1], "HOST") == 0) 
    {
        if (strcmp(argv[2], "BASIC") == 0) 
        {
            host_main_bfs(M, NNZ, verify_results);
        } 
    } 
    else if (strcmp(argv[1], "DEVICE") == 0) 
    {
        if (verify_results)
            print_device_prop();

        if (strcmp(argv[2], "BASIC") == 0) 
        {
            device_main_bfs(M, NNZ, verify_results, 0);
            //if (spmv_mode == 0)
            //    device_main_coo(M, N, NNZ, verify_results);
            //else if (spmv_mode == 1)
            //    device_main_csr(M, N, NNZ, verify_results);
            //else if 
        } 
        else if (strcmp(argv[2], "TILED") == 0) 
        {
            device_main_bfs(M, NNZ, verify_results, 1);
            //device_main(1, M, N, verify_results);
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