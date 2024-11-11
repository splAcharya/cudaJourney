#include <stdio.h>

const size_t DSIZE = 16384; //matrix side demension
const int block_size = 256; //CUDA maximum is 1024



//row sums kernel
// assume block_size = 2 i,e num threads
// DSIZE = 4
// A ==> 4 * 4 = 16
// A = 0    1   2   3   4   5   6   7   8   9   10  .....
//      0    1   2   3   
//      4   5   6   7   
//      8   9   10  11
//      12  13  14  15
//tid = 0 => A[(gl_idx * 4) + i] => A[0], A[1], A[2], A[3]
//tid = 3 => A[(3 * 4) + i] => A[12], A[13]...
//
// row major format
///block size 4,  DS = 4
//cycle    0        1       2       3
// t0 ==> A[0]      A[1]    A[]
// t1 ==> A[4]      A[5]
// t2 ==> A[8]      A[9]
// t3 ==> A[12]     A[13]
//vs
//column major format
// block size 3, DS = 4
//cycle    0       1       2        3
// t0 ==> A[0]    A[4]     
// t1 ==> A[1]    A[5]
// t2 ==> A[2]    A[6]
// t3 ==> A[3]    A[7]
//
// Summary:
// Asuuming a cache line in 4 bytes
// In ROw major format, each cycle threads tries to accces data 
// that almost require multiple cache line access
// whereas in column format, each cycle thread tries to access data
// that almost serviced in 1 cache line directly
// thus far better memory through put 

__global__ void row_sums(const float *A, float *sums, size_t ds)
{
    int gl_tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int lc_tid = threadIdx.x;

    if (gl_tid < ds)
    {
        float sum = 0.0f;
        for (int i = 0; i < ds; i++)
            sum += A[(gl_tid * ds) + i];
        sums[gl_tid] = sum;
    }
}

// coulms nums kernel
// assume block_size = 2 i,e num threads
// DSIZE = 4
// A ==> 4 * 4 = 16
// A = 0    1   2   3   4   5   6   7   8   9   10  .....
//      0    1   2   3   
//      4   5   6   7   
//      8   9   10  11
//      12  13  14  15
// tid = 0 => A[0], A[4], A[8], A[12] => A[0 + (4 * i)]
// tid = 1 => A[1], A[5], A[9], A[13] => A[1 + (4 * i)]
// ==> bascially A[gl_tid + (ds * i)] 
__global__ void column_sums(const float *A, float *sums, size_t ds)
{
    int gl_tid = threadIdx.x + (blockDim.x * blockIdx.x);
    int lc_tid = threadIdx.x;

    if (gl_tid < ds)
    {
        float sum = 0.0f;
        for (int i = 0; i < ds; i++)
        {
            sum += A[(gl_tid + (ds * i))];
        }
        sums[gl_tid] = sum;
    }
}

int main()
{
    float *h_A, *h_sums;
    float *d_A, *d_sums;

    h_A = new float[DSIZE * DSIZE];
    h_sums = new float[DSIZE];

    //populate host data
    for (int i = 0; i < (DSIZE * DSIZE); i++)
        h_A[i] = 1.0f;

    //for verificaiotn
    //h_A[0] = 2.0f;
    //h_A[2] = 2.5f;

    // allocate global memory on device
    cudaMalloc(&d_A, (DSIZE * DSIZE * sizeof(float)));
    cudaMalloc(&d_sums, (DSIZE * sizeof(float))); 

    //copy data from host to device's global memory
    cudaMemcpy(d_A, h_A, (DSIZE * DSIZE * sizeof(float)), cudaMemcpyHostToDevice);

    //initialize output array on device's global memory
    cudaMemset(d_sums, 0, (DSIZE * sizeof(float)));

    //launch kernel
    int num_blocks = DSIZE / block_size;
    row_sums<<<num_blocks, block_size>>>(d_A, d_sums, DSIZE);
    //column_sums<<<num_blocks, block_size>>>(d_A, d_sums, DSIZE);

    //copy results from device to host
    cudaMemcpy(h_sums, d_sums, (DSIZE * sizeof(float)), cudaMemcpyDeviceToHost);

    //inject error
   // h_sums[4] = 0.0f;

    //verify  kernel
    //for( int i = 0; i < DSIZE; i++)
    //{
    //    //verify row major
    //    // A = 0    1   2 
    //    //      3   4   5
    //    //      6   7   8
    //    // ==> A[0], A[1], A[2], ==> 0, 1, 2
    //    // ==> A[3], A[4], A[5]  ===> 3, 4, 5 ==> 
    //    // ==> A[i + j]
    //    ///columns major
    //    // A[0], A[3], A[6] == 0, 0 + ds, 0 + 2*ds
    //    // A[1], A[4], A[7] ==>1, 1 + ds, 1 + 2*ds
    //    // ==> A[i + (j * ds)] =>  
    //    float sum = 0.0f;
    //    for (int j = 0; j < DSIZE; j++)
    //    {
    //        sum += h_A[(i*DSIZE) + j];
    //        //sum += h_A[i + (j * DSIZE)];
    //    }

    //    printf("sum:%f, h_sums:%f\n", sum, h_sums[i]);
    //    
    //    if (h_sums[i] != sum)
    //        printf("Error!\n");
    //}

    return 0;
}