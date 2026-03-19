#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;


#define cudaCheckErrors(msg)                          \
    do{                                               \
    cudaError_t __err = cudaGetLastError();           \
    if (__err != cudaSuccess){                        \
    printf("Fatal Error:%s (%s at %s:%d)\n",          \
                    msg,                              \
                    cudaGetErrorString(__err),        \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(1);                                  \
        }                                             \
    }while(0);


# define convert_to_gray(b, g, r) (0.299f * (r) + 0.587 * (g) + 0.114f * (b))

constexpr const char* kInputImagePath = "inputs/manhattan_traffic.jpg";
constexpr const char* kHostOutputPath = "outputs/grayscale/host_manhattan_gray.jpg";
constexpr const char* kDeviceBasicOutputPath = "outputs/grayscale/device_basic_manhattan_gray.jpg";
constexpr const char* kDeviceSmemOutputPath = "outputs/grayscale/device_smem_manhattan_gray.jpg";

int read_image(unsigned char **rgb_image, int *width, int *height)
{
    Mat in_image = imread(kInputImagePath, IMREAD_COLOR);
    
    if (in_image.empty()){
        printf("Failed TO Load Image\n");
        return -1;
    }

    *width  = in_image.cols;
    *height = in_image.rows;
    int size = in_image.total() * in_image.elemSize();
    *rgb_image = (unsigned char*)malloc(size);
    memcpy(*rgb_image, in_image.data, size);

    return 0;
}

int host_main()
{
    unsigned char *h_rgb, *h_gray;
    int width, height;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    read_image(&h_rgb, &width, &height);

    h_gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    cudaEventRecord(start, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            unsigned char r = h_rgb[3 * idx];
            unsigned char g = h_rgb[3 * idx + 1];
            unsigned char b = h_rgb[3 * idx + 2];
            h_gray[idx] = convert_to_gray(b, g, r);
        }        
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat gray_image(height, width, CV_8UC1, h_gray);
    cv::imwrite(kHostOutputPath, gray_image);

    free(h_gray);

    return 0; 
}


/*
In the basic kernel shows little coalese acessed, with both load and stores ettifcienty around 30%
Lets look at the access patterns,

    Load Pattern:
    The input array is a row major 3D matrix, B,G,R channel, so every 3 element is a single pixel.

        0      1        2      3        4       5
    0,0,0   0,0,1   0,0,2   0,1,0   0,1,1   0,1,2   

    6        7          8      9       10       11
    1,0,0   1,0,1   1,0,2   1,1,0   1,1,1   1,1,2

    Cycle   0  
    T0      0
    T1      3
    T2      6
    T3      9

    In cycle 0, 0th, 3rd, 6th, 9th elements are accessed,
    these elemetns are 8 + 8 = 16bytes apart. Assume a cache line size of
    8bytes, only 1 request can be serviced. Thus the low load efficiency.

    It would be great if all threads can process and element each. However,
    untimately pixel values need to be added together. So some thread ultimately
    needs to process the 3 channels at once. One way would be to fetch elements 
    at once and then store in local memory. Threads within a block can share the 
    local memory.  

        0      1        2      3        4       5
    0,0,0   0,0,1   0,0,2   0,1,0   0,1,1   0,1,2   

    6        7          8      9       10       11
    1,0,0   1,0,1   1,0,2   1,1,0   1,1,1   1,1,2

    Cycle   0
    T0      0    ==> T0 = SMEM[0] + SMEM[1] + SMEM[2]
    T1      1    
    T2      2 
    -------------
    T3      3  ==> T3 = SMEM[3] + SMEM[4] + SMEM[5] 
    T4      4
    T5      5

    So in each block of 256 threads, 
    only 256/3 ==> ~86 threads will have the results.

    The number of threads launched should be same as the number of pixels
    width * height * 3;

    assume a 30 * 50 * 3 image = 4500 number of threads
    There is apossiblity that there won't be enough
    threads to process an entire input image.
    So we will need to do  a grid stride loop.

    Assuem block_size of 256,
    4500 / 256 = 17.57 blocks are needed


    TODO: Still Needs To Be Worked On.. :(

*/
__global__ void devicek_rgbtogray_smem(
    const unsigned char *gm_rgb, 
    size_t width, 
    size_t height, 
    unsigned char *gm_gray)
{
    size_t tid_glx = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread block will be of size 256,
    // i.e 256  / 3, rounded down = 85 pixels so unliatemly 85 thread
    // 85 * 3 = 255 elements, the 256th thread from every block will 
    // have to be ignored.
    // The next thread block should request global memory from index - 1,
    
    //Even though only block_size - 1 is required, 
    //the +1 can be helpful for bank conflicts
    __shared__ unsigned char sm_ar[256]; 
    
    /* Every 3rd thread gets to store data. */
   if ( (tid_glx < (width * height * 3)) && (threadIdx.x < blockDim.x - 1)) 
   {
        //other than the 0th block, every other block must load data from gl_idx - 1
        if (blockIdx.x == 0)
            sm_ar[threadIdx.x] = gm_rgb[tid_glx];
        else
            sm_ar[threadIdx.x] = gm_rgb[tid_glx - 1];

        //wait for all threads to complete share memory storage
        __syncthreads();

        //now 1st of every 3 threads processed grayscale conversion
        if  (threadIdx.x % 3 == 0)
        {
            sm_ar[threadIdx.x]  = convert_to_gray(sm_ar[threadIdx.x + 0],
                                                  sm_ar[threadIdx.x + 1], 
                                                  sm_ar[threadIdx.x + 2]);
        }

        //now wait for threads to complete the conversion
        __syncthreads();

        //share memory should look like this now
        // sm_ar = res, x, x, res, x, x, res, x, x, res, x, x
        // first 85 threads will have to acess every 3 items from shared memory
        // thread idx     Acess Idx
        //  0               0
        //  1               3
        //  2               6
        //  3               9
        //  4               12
        // so looks like local thread idx * 3 
        
        //first 85 threads of each block
        if (threadIdx.x < 85)
        {
            //first block processed the first 85 elements
            //every other block processes [global idx - 85] i.e 
            // block 0 ==> [0  .... 84] ==> global id [0... 84] 
            // block 1 ==> [85 .... 169] ==> global id [256 ...340]
            // block 2 ==> [170 .... 254] ==> global id [512 ... 596]

            if (threadIdx.x < 85)
            {
                gm_gray[blockIdx.x * 85 + threadIdx.x] = sm_ar[threadIdx.x * 3];
            }
        }
   }
}


int  device_main_smem() {

    const int BLOCK_SIZE = 256;
    unsigned char *h_rgb, *h_gray;
    unsigned char *d_rgb, *d_gray;
    int width, height;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    read_image(&h_rgb, &width, &height);
    h_gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    
    cudaMalloc(&d_rgb, rgb_size);
    cudaCheckErrors("device rgb alloc");

    cudaMalloc(&d_gray, gray_size);
    cudaCheckErrors("device gray alloc");
    
    int num_blocks = (height * width * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE,1, 1);
    dim3 grid_size(num_blocks, 1, 1);
    cudaEventRecord(start, 0);
    
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Host to device copy");
    devicek_rgbtogray_smem<<<grid_size, block_size>>>(d_rgb, width, height, d_gray);
    cudaDeviceSynchronize();
    cudaCheckErrors("Device Synchronize");
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Device To Host Copy");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device SMEM Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat gray_image(height, width, CV_8UC1, h_gray);
    cv::imwrite(kDeviceSmemOutputPath, gray_image);

    free(h_gray);
    free(h_rgb);
    cudaFree(d_gray);
    cudaFree(d_rgb);

    return 0; 
}

/*
     serially the input is expect to be in the following format
     0  1  2  3  4  5  6  7  8  
     B  G  R  B  G  R  B  G  R  

     9  10  11  12  13  14  15  16  17
     B  G   R   B   G    R  B   G   R

     which translates to 
     0      1       2       3     4     5
    0,0,0  0,0,1  0,0,2  0,1,0  0,1,1  0,1,2

    6        7      8       9     10     11
    1,0,0  1,0,1  1,0,2  1,1,0  1,1,1  1,1,2

      12    13      14     15     16    17
    3,0,0  3,0,1  3,0,2  3,1,0  3,1,1  3,1,2 

*/
__global__ void devicek_rgbtogray_basic(const unsigned char *gm_rgb, size_t width, size_t height, unsigned char *gm_gray)
{
    size_t gl_tidx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gl_tidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gl_tidx < width && gl_tidy < height)
    {
        //each thread will process 1 entire pixel
        size_t flat_rgb_index = (gl_tidy * width + gl_tidx) * 3;
        size_t flat_gray_index = (gl_tidy * width + gl_tidx);

        unsigned char b = gm_rgb[flat_rgb_index + 0];
        unsigned char g = gm_rgb[flat_rgb_index + 1];
        unsigned char r = gm_rgb[flat_rgb_index + 2];
        
        //convert
        unsigned char gray_val = 0.21f * r + 0.72f * g + 0.07f * b;
        gm_gray[flat_gray_index] = gray_val;
    }   
}

int  device_main_basic() {

    const int BLOCK_SIZE = 16;
    unsigned char *h_rgb, *h_gray;
    unsigned char *d_rgb, *d_gray;
    int width, height;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    read_image(&h_rgb, &width, &height);
    h_gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    
    cudaMalloc(&d_rgb, rgb_size);
    cudaCheckErrors("device rgb alloc");

    cudaMalloc(&d_gray, gray_size);
    cudaCheckErrors("device gray alloc");
    
    int num_blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_size(num_blocks_x, num_blocks_y);
    cudaEventRecord(start, 0);
    
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Host to device copy");
    devicek_rgbtogray_basic<<<grid_size, block_size>>>(d_rgb, width, height, d_gray);
    cudaDeviceSynchronize();
    cudaCheckErrors("Device Synchronize");
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Device To Host Copy");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Basic Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat gray_image(height, width, CV_8UC1, h_gray);
    cv::imwrite(kDeviceBasicOutputPath, gray_image);

    free(h_gray);
    free(h_rgb);
    cudaFree(d_gray);
    cudaFree(d_rgb);

    return 0; 
}

void print_device_prop()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl; 
    printf("\n\n");
}

int main(int argc, char *argv[])
{
    for (int i = 0 ; i < argc; i++)
        printf("Parameter:%i ==> %s\n", i, argv[i]);
    
    if (argc <= 3)
    {
        if (!strcmp("HOST", argv[1]))
            host_main();
        else if (!strcmp("DEVICE", argv[1]))
        {
            print_device_prop();
            
            if (!strcmp("BASIC", argv[2]))
            {
                device_main_basic();
            }
            else if (!strcmp("SMEM", argv[2]))
            {
                device_main_smem();
            }
        }
    }

    return 0;
}
