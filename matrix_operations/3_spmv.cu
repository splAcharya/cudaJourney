#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

const int BLOCK_DIM = 32;
const int MAX_BLOCKS = 65535;
const float ERR_THRESHOLD = 1e-2;

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


int generate_vector(
    float *h_in,
    int N)
{
    int seed = (int)time(NULL);
    srand(seed);
    
    for (int i = 0; i < N; i++)
        *(h_in + i) = (float)(rand() % 100) / 10.0f;

    return 0;
}

int generate_coo_matrix(
    int *h_in_row, 
    int *h_in_col, 
    float *h_in_val, 
    int M, int N, int NNZ)
{    
    int seed = (int)time(NULL);
    srand(seed);
    
    for (int i = 0; i < NNZ; i++)
    {
        *(h_in_row + i) = rand() % M;
        *(h_in_col + i) = rand() % N;
        *(h_in_val + i) = (float)(rand() % 50) / 10.0f;
    }

    return 0;
}

int generate_csr_matrix(
    int *h_in_row, 
    int *h_in_col, 
    float *h_in_val, 
    int M, int N, int NNZ)
{
    int *row_counts = (int *)calloc(M, sizeof(int));
    int *row_ids = (int *)malloc(NNZ * sizeof(int));

    // Step 1: Assign each nonzero to a row and record it
    for (int i = 0; i < NNZ; i++) {
        int r = rand() % M;
        row_counts[r]++;
        row_ids[i] = r;
    }

    // Step 2: Build row pointer (prefix sum)
    h_in_row[0] = 0;
    for (int i = 0; i < M; i++)
        h_in_row[i+1] = h_in_row[i] + row_counts[i];

    // Step 3: Fill in col/val arrays in row-major order
    int *row_offsets = (int *)malloc(M * sizeof(int));
    for (int i = 0; i < M; i++)
        row_offsets[i] = h_in_row[i];

    for (int i = 0; i < NNZ; i++) {
        int r = row_ids[i];
        int idx = row_offsets[r]++;
        h_in_col[idx] = rand() % N;
        h_in_val[idx] = (float)(rand() % 50) / 10.0f;
    }

    free(row_counts);
    free(row_offsets);
    free(row_ids);

    return 0;
}


void print_vector_int(const int *h_in, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d| ", *(h_in + i));

        if ((i+1) % 7 == 0)
            printf("\n");
    }
    printf("\n");
}

void print_vector_float(const float *h_in, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%.2f| ", *(h_in + i));

        if ((i+1) % 7 == 0)
            printf("\n");
    }
    printf("\n");
}

/*
        A               X                A.x                    
    1   2   3    *   5  6  7    =     [ (1*5)+(2*6)+(3*7)    ]
    4   5   6                         [ (4*5)+(5*6)+(6*7)    ] 
    7   8   9                         [ (7*5)+(8*6)+(9*7)    ]    
    10  11  12                        [ (10*5)+(11*6)+(12*7) ]

    M * N        *   N          =     M * 1

*/
int hostk_coo(
    const int   *h_in_row, 
    const int   *h_in_col, 
    const float *h_in_val, 
    const float *h_in_vec,
    float *h_out_vec,
    int NNZ)
{
    for (int i = 0; i < NNZ; i++)
    {
        int row = h_in_row[i];
        int col = h_in_col[i];
        h_out_vec[row] += h_in_val[i] * h_in_vec[col];
    }
    
    return 0;
}


int host_main_coo(int M, int N, int NNZ, bool verify)
{
    printf("Executing Host Main COO, M:%d, N:%d, NNZ:%d\n", M, N, NNZ);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_in_row, *h_in_col;
    float *h_in_val;
    float *h_in_vec;
    float *h_out_vec;
    int alc_sz_row = NNZ * sizeof(int);
    int alc_sz_col = NNZ * sizeof(int);
    int alc_sz_val = NNZ * sizeof(float);
    int alc_sz_in_vec = N * sizeof(float);
    int alc_sz_out_vec = M * sizeof(float);

    h_in_row = (int *)malloc(alc_sz_row);
    h_in_col = (int *)malloc(alc_sz_col);
    h_in_val = (float *)malloc(alc_sz_val);
    h_in_vec = (float *)malloc(alc_sz_in_vec);
    h_out_vec = (float *)malloc(alc_sz_out_vec);

    generate_coo_matrix(h_in_row, h_in_col, h_in_val, M, N, NNZ);

    generate_vector(h_in_vec, N);

    memset(h_out_vec, 0, alc_sz_out_vec);

    cudaEventRecord(begin, 0);

    hostk_coo(h_in_row, h_in_col, h_in_val, 
              h_in_vec, h_out_vec, NNZ);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Hostk COO Elapsed Time:%f ms\n", elapsed_time);

    free(h_out_vec);
    free(h_in_vec);
    free(h_in_val);
    free(h_in_col);
    free(h_in_row);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int hostk_csr(const int *gm_in_rowptr, const int *gm_in_col, const float *gm_in_val, 
              const float *gm_in_vec, float *gm_out_vec, 
              int M, int N, int NNZ)
{
    for (int i = 0; i < M; i++)
    {
        int start_pos = *(gm_in_rowptr + i);
        int end_pos = *(gm_in_rowptr + i + 1);

        for (int j = start_pos; j < end_pos; j++)
        {
            int     row = i;
            int     col = *(gm_in_col + j);
            float m_val = *(gm_in_val + j);
            float v_val = *(gm_in_vec + col);
            float result = m_val * v_val;
            *(gm_out_vec + row) += result;
        }
    }   
    
    return 0;
}


int host_main_csr(int M, int N, int NNZ, bool verify)
{
    printf("Executing Host Main CSR, M:%d, N:%d, NNZ:%d\n", M, N, NNZ);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_in_row;
    int *h_in_col;
    float *h_in_val;
    float *h_in_vec;
    float *h_out_vec;
    int alc_sz_row = (M + 1) * sizeof(int);
    int alc_sz_col = NNZ     * sizeof(int);
    int alc_sz_val = NNZ     * sizeof(float);
    int alc_sz_in_vec = N    * sizeof(float);
    int alc_sz_out_vec = M   * sizeof(float);

    h_in_row = (int *)malloc(alc_sz_row);
    h_in_col = (int *)malloc(alc_sz_col);
    h_in_val = (float *)malloc(alc_sz_val);
    h_in_vec = (float *)malloc(alc_sz_in_vec);
    h_out_vec = (float *)malloc(alc_sz_out_vec);

    generate_csr_matrix(h_in_row, h_in_col, h_in_val, M, N, NNZ);

    generate_vector(h_in_vec, N);

    memset(h_out_vec, 0, alc_sz_out_vec);

    cudaEventRecord(begin, 0);

    hostk_csr(h_in_row, h_in_col, h_in_val, 
              h_in_vec, h_out_vec, M, N, NNZ);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Hostk CSR Elapsed Time:%f ms\n", elapsed_time);

    //print_vector_int(h_in_row, M);
    //print_vector_int(h_in_col, NNZ);
    //print_vector_float(h_in_val, NNZ);
    //print_vector_float(h_in_vec, N);
    //print_vector_float(h_out_vec, M);

    free(h_out_vec);
    free(h_in_vec);
    free(h_in_val);
    free(h_in_col);
    free(h_in_row);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}



int verify_results_coo(
    const int   *h_in_row,  const int *h_in_col,
    const float *h_in_val,  const float *h_in_vec, 
    const float *h_out_vec, int M, int N, int NNZ)
{
    int alc_sz_vecv = M * sizeof(float);
    float *h_out_vecv = (float *)malloc(alc_sz_vecv);
    memset(h_out_vecv, 0, alc_sz_vecv);

    hostk_coo(h_in_row, h_in_col, h_in_val, h_in_vec, h_out_vecv, NNZ);

    //print_vector_int(h_in_row, NNZ);
    //print_vector_int(h_in_col, NNZ);
    //print_vector_float(h_in_val, NNZ);
    //print_vector_float(h_in_vec, N);
    //print_vector_float(h_out_vec, M);
    //print_vector_float(h_out_vecv, M);

    for (int i = 0; i < M; i++)
    {
        float diff = fabs(h_out_vec[i] - h_out_vecv[i]);

        if (diff > ERR_THRESHOLD)
        {
            printf("Wrong Result@:%d, ACTUAL:%f, EXPECTED:%f, Diff:%f\n", 
                   i, h_out_vec[i], h_out_vecv[i], diff);  
            break; 
        }
    }

    free(h_out_vecv);

    return 0;
}


__global__ void devicek_coo(
    const int *gm_in_row, const int *gm_in_col, 
    const float *gm_in_val, float *gm_in_vec, 
    float *gm_out_vec, int NNZ)
{
    int bx = blockIdx.x;
    int lx = threadIdx.x;
    int gx = (bx * blockDim.x) + lx;

    if (gx < NNZ)
    {
        int row = gm_in_row[gx];
        int col = gm_in_col[gx];
        float val = gm_in_val[gx];
        float vec_val = gm_in_vec[col];
        float result = val * vec_val;
        atomicAdd(&gm_out_vec[row], result);
    }
}

int device_main_coo(int M, int N, int NNZ, bool verify)
{
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_in_row, *h_in_col;
    float *h_in_val;
    float *h_in_vec;
    float *h_out_vec;
    int alc_sz_row = NNZ * sizeof(int);
    int alc_sz_col = NNZ * sizeof(int);
    int alc_sz_val = NNZ * sizeof(float);
    int alc_sz_in_vec = N * sizeof(float);
    int alc_sz_out_vec = M * sizeof(float);

    h_in_row = (int *)malloc(alc_sz_row);
    h_in_col = (int *)malloc(alc_sz_col);
    h_in_val = (float *)malloc(alc_sz_val);
    h_in_vec = (float *)malloc(alc_sz_in_vec);
    h_out_vec = (float *)malloc(alc_sz_out_vec);

    generate_coo_matrix(h_in_row, h_in_col, h_in_val, M, N, NNZ);

    generate_vector(h_in_vec, N);

    memset(h_out_vec, 0, M);

    cudaEventRecord(begin, 0);

    int *d_in_row, *d_in_col;
    float *d_in_val;
    float *d_in_vec;
    float *d_out_vec;

    cudaMalloc(&d_in_row, alc_sz_row);
    cuda_check_errors("ALC_ROW");

    cudaMalloc(&d_in_col, alc_sz_col);
    cuda_check_errors("ALC_COL");

    cudaMalloc(&d_in_val, alc_sz_val);
    cuda_check_errors("ALC_VAL");

    cudaMalloc(&d_in_vec, alc_sz_in_vec);
    cuda_check_errors("ALC_IVEC");

    cudaMalloc(&d_out_vec, alc_sz_out_vec);
    cuda_check_errors("ALC_OVEC");

    cudaMemcpy(d_in_row, h_in_row, alc_sz_row, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_ROW");

    cudaMemcpy(d_in_col, h_in_col, alc_sz_col, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_COL");

    cudaMemcpy(d_in_val, h_in_val, alc_sz_val, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_VAL");

    cudaMemcpy(d_in_vec, h_in_vec, alc_sz_in_vec, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_VEC");

    cudaMemset(d_out_vec, 0, alc_sz_out_vec);
    cuda_check_errors("MSET_DOV");

    dim3 block_shape(BLOCK_DIM);
    int num_blocks = (NNZ +  BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_shape(num_blocks); 

    devicek_coo<<<grid_shape, block_shape>>>(d_in_row, d_in_col, d_in_val, d_in_vec, d_out_vec, NNZ);
    cuda_check_errors("KLAUNCH");

    cudaDeviceSynchronize();
    cuda_check_errors("DEV_SYNC");

    cudaMemcpy(h_out_vec, d_out_vec, alc_sz_out_vec, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_OVEC");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Event Elapsed Time:%f ms\n", elapsed_time);

    if (verify)
        verify_results_coo(h_in_row, h_in_col, h_in_val, h_in_vec, h_out_vec, M, N, NNZ);

    free(h_out_vec);
    free(h_in_vec);
    free(h_in_val);
    free(h_in_col);
    free(h_in_row);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int verify_results_csr(
    const int   *h_in_row,  const int *h_in_col,
    const float *h_in_val,  const float *h_in_vec, 
    const float *h_out_vec, int M, int N, int NNZ)
{
    int alc_sz_vecv = M * sizeof(float);
    float *h_out_vecv = (float *)malloc(alc_sz_vecv);
    memset(h_out_vecv, 0, alc_sz_vecv);

    hostk_csr(h_in_row, h_in_col, h_in_val, 
              h_in_vec, h_out_vecv, M, N, NNZ);

    //print_vector_int(h_in_row, M+1);
    //print_vector_int(h_in_col, NNZ);
    //print_vector_float(h_in_val, NNZ);
    //print_vector_float(h_in_vec, N);
    //print_vector_float(h_out_vec, M);
    //print_vector_float(h_out_vecv, M);

    for (int i = 0; i < M; i++)
    {
        float diff = fabs(h_out_vec[i] - h_out_vecv[i]);

        if (diff > ERR_THRESHOLD)
        {
            printf("Wrong Result@:%d, ACTUAL:%f, EXPECTED:%f, Diff:%f\n", 
                   i, h_out_vec[i], h_out_vecv[i], diff);  
            break; 
        }
    }

    free(h_out_vecv);

    return 0;
}

__global__ void devicek_csr(const int *gm_in_row, const int *gm_in_col, const float *gm_in_val, 
                            const float *gm_in_vec, float *gm_out_vec, int M, int N, int NNZ)
{
    int bx = blockIdx.x;
    int lx = threadIdx.x;
    int gx = (blockDim.x * bx) + lx;

    if (gx < M)
    {
        int start_pos = gm_in_row[gx];
        int end_pos = gm_in_row[gx + 1];

        for (int i = start_pos; i < end_pos; i++)
        {
            int row = gx;
            int col = gm_in_col[i];
            float m_val = gm_in_val[i] ;
            float v_val = gm_in_vec[col];
            float result = m_val * v_val;
            gm_out_vec[row] += result;  
        }
    }
}

int device_main_csr(int M, int N, int NNZ, bool verify)
{
    printf("Executing Device Main CSR, M:%d, N:%d, NNZ:%d\n", M, N, NNZ);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *h_in_row;
    int *h_in_col;
    float *h_in_val;
    float *h_in_vec;
    float *h_out_vec;
    int alc_sz_row = (M + 1) * sizeof(int);
    int alc_sz_col = NNZ     * sizeof(int);
    int alc_sz_val = NNZ     * sizeof(float);
    int alc_sz_in_vec = N    * sizeof(float);
    int alc_sz_out_vec = M   * sizeof(float);

    h_in_row = (int *)malloc(alc_sz_row);
    h_in_col = (int *)malloc(alc_sz_col);
    h_in_val = (float *)malloc(alc_sz_val);
    h_in_vec = (float *)malloc(alc_sz_in_vec);
    h_out_vec = (float *)malloc(alc_sz_out_vec);

    generate_csr_matrix(h_in_row, h_in_col, h_in_val, M, N, NNZ);

    generate_vector(h_in_vec, N);

    memset(h_out_vec, 0, alc_sz_out_vec);

    cudaEventRecord(begin, 0);

    int *d_in_row;
    int *d_in_col;
    float *d_in_val;
    float *d_in_vec;
    float *d_out_vec;

    cudaMalloc(&d_in_row, alc_sz_row);
    cuda_check_errors("ALC_ROW");

    cudaMalloc(&d_in_col, alc_sz_col);
    cuda_check_errors("ALC_COL");

    cudaMalloc(&d_in_val, alc_sz_val);
    cuda_check_errors("ALC_VAL");

    cudaMalloc(&d_in_vec, alc_sz_in_vec);
    cuda_check_errors("ALC_IVEC");

    cudaMalloc(&d_out_vec, alc_sz_out_vec);
    cuda_check_errors("ALC_OVEC");

    cudaMemset(d_out_vec, 0, alc_sz_out_vec);
    cuda_check_errors("MSET_OVEC");

    cudaMemcpy(d_in_row, h_in_row, alc_sz_row, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_ROW");

    cudaMemcpy(d_in_col, h_in_col, alc_sz_col, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_COL");

    cudaMemcpy(d_in_val, h_in_val, alc_sz_val, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_VAL");

    cudaMemcpy(d_in_vec, h_in_vec, alc_sz_in_vec, cudaMemcpyHostToDevice);
    cuda_check_errors("H2D_VEC");

    dim3 block_shape(BLOCK_DIM);
    int num_blocks = (M + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_shape(num_blocks);

    devicek_csr<<<grid_shape, block_shape>>>(d_in_row, d_in_col, d_in_val, 
                                             d_in_vec, d_out_vec, M, N, NNZ);
    cuda_check_errors("KLAUNCH_CSR");
    
    cudaDeviceSynchronize();
    cuda_check_errors("DEV_SYNCH");

    cudaMemcpy(h_out_vec, d_out_vec, alc_sz_out_vec, cudaMemcpyDeviceToHost);
    cuda_check_errors("D2H_MCPY");

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Devicek CSR Elapsed Time:%f ms\n", elapsed_time);

    if (verify)
        verify_results_csr(h_in_row, h_in_col, h_in_val, h_in_vec, h_out_vec, M, N, NNZ);
    //print_vector_int(h_in_row, M);
    //print_vector_int(h_in_col, NNZ);
    //print_vector_float(h_in_val, NNZ);
    //print_vector_float(h_in_vec, N);
    //print_vector_float(h_out_vec, M);

    cudaFree(d_out_vec);
    cudaFree(d_in_vec);
    cudaFree(d_in_val);
    cudaFree(d_in_col);
    cudaFree(d_in_row);

    free(h_out_vec);
    free(h_in_vec);
    free(h_in_val);
    free(h_in_col);
    free(h_in_row);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int main(int argc, char *argv[]) 
{
    if (argc < 6) 
    {
        fprintf(stderr, "Usage: %s <mode> [submode] [M] [N] [NNZ_PCT] {verify_results}\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    float NNZ_PCT = (float)atoi(argv[5]);
    int NNZ = (M * N) * (NNZ_PCT / 100);
    int spmv_mode = (strcmp(argv[6], "COO") == 0) ? 0 : 1;
    bool verify_results = (strcmp(argv[7], "TRUE") == 0);

    printf("M:%d, N:%d,  MxN:%d, NNZ_PCT:%f, NNZ:%d, MODE:%d, Verify Results:%d\n", 
            M, N, (M*N), NNZ_PCT, NNZ, spmv_mode, verify_results);
 
    if (strcmp(argv[1], "HOST") == 0) 
    {
        if (strcmp(argv[2], "BASIC") == 0) 
        {
            if (spmv_mode == 0)
                host_main_coo(M, N, NNZ, verify_results);
            else if (spmv_mode == 1)
                host_main_csr(M, N, NNZ, verify_results);
        } 
    } 
    else if (strcmp(argv[1], "DEVICE") == 0) 
    {
        if (verify_results)
            print_device_prop();

        if (strcmp(argv[2], "BASIC") == 0) 
        {
            if (spmv_mode == 0)
                device_main_coo(M, N, NNZ, verify_results);
            else if (spmv_mode == 1)
                device_main_csr(M, N, NNZ, verify_results);
        } 
        else if (strcmp(argv[2], "TILED") == 0) 
        {
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
