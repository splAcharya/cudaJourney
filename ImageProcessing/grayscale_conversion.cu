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


int read_image(unsigned char **rgb_image, int *width, int *height)
{
    Mat in_image = imread("manhattan_traffic.jpg", IMREAD_COLOR);
    
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
            h_gray[idx] = 0.299f * r + 0.587 * g + 0.114f * b;
        }        
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat gray_image(height, width, CV_8UC1, h_gray);
    cv::imwrite("h_mh_gray.jpg", gray_image);

    free(h_gray);

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

    assuming and imput image with 2x3 size
    indexing ocmputation 
    xdirection, y direction
    pixel at x,y is
    
    each row is at a multiple of width i.e 

    6th element (1, 0, 0)  (x, y, z)
        2D ==> x * width + y ==> 1 * 6  + 0 ==> 6
    however in 3D grid, elemtns are Z dims apart
    7the element (1,0,1) translates to 
        3D ==> (x * width + y) * channels => (1 * 3 + 0) *3 = 9

*/

__global__ void devicek_rgbtogray(const unsigned char *gm_rgb, size_t width, size_t height, unsigned char *gm_gray)
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


int  device_main() {

#define  BLOCK_SIZE 16

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    
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
    devicek_rgbtogray<<<grid_size, block_size>>>(d_rgb, width, height, d_gray);
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Device To Host Copy");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat gray_image(height, width, CV_8UC1, h_gray);
    cv::imwrite("d_mh_gray.jpg", gray_image);

    free(h_gray);
    free(h_rgb);
    cudaFree(d_gray);
    cudaFree(d_rgb);

    return 0; 
}


int main(int argc, char *argv[])
{
    for (int i = 0 ; i < argc; i++)
        printf("Parameter:%i ==> %s\n", i, argv[i]);
    
    if (argc == 2)
    {
        if (!strcmp("HOST", argv[1]))
            host_main();
        else
            device_main();
    }

    return 0;
}