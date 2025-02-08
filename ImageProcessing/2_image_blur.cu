#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;


#define cuda_check_errors(msg) \
    do{ \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            printf("Fatal Error:%s=>(%s at %s:%d)\n", msg, \
                                        cudaGetErrorString(__err), \
                                        __FILE__, \
                                        __LINE__); \
            exit(1); \
        } \
    }while(0)

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


void padded_image(unsigned char **padded_img, int *width, int *height, int *channels) {
    // Read the image
    cv::Mat image = cv::imread("manhattan_traffic.jpg");
    
    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        exit(1);
    }

    // Padding size
    int top, bottom, left, right;
    top = bottom = left = right = 1;

    // Create a new image with padding
    cv::Mat padded_image = cv::Mat::zeros(image.rows + top + bottom, image.cols + left + right, image.type());

    // Copy the original image to the center of the new padded image
    image.copyTo(padded_image(cv::Rect(left, top, image.cols, image.rows)));

    // Get the dimensions of the padded image
    *width = padded_image.cols;
    *height = padded_image.rows;
    *channels = padded_image.channels();

    // Return the padded image data as unsigned char*
    unsigned char* paddedImageData = new unsigned char[(*width) * (*height) * (*channels)];
    std::memcpy(paddedImageData, padded_image.data, (*width) * (*height) * (*channels));
    *padded_img = paddedImageData;
}

__global__ void devicek_blur_basic(
    const unsigned char * gm_original, 
    const int width, 
    const int height, 
    const int channels, 
    unsigned char * gm_blurred){

    //basic kernel, assign each element a thread, then apply blurring
    size_t tid_glx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid_gly = blockDim.y * blockIdx.y + threadIdx.y;

    if ((tid_glx > 0) && (tid_glx < width - 1) && 
        (tid_gly > 0) && (tid_gly < height - 1))
    {
        int b = 0, g = 0, r = 0;

        for (int ky =-1; ky <= 1; ky++){            
            for (int kx = -1; kx <= 1; kx++){
                size_t flat_nbr_idx = ((tid_gly+ky) * width + (tid_glx+kx)) * channels;
                b += gm_original[flat_nbr_idx + 0];
                g += gm_original[flat_nbr_idx + 1];
                r += gm_original[flat_nbr_idx + 2];
            }
        }

        size_t flat_idx = (tid_gly * width + tid_glx) * channels;
        gm_blurred[flat_idx + 0] = (unsigned char)(b / (3 * 3));
        gm_blurred[flat_idx + 1] = (unsigned char)(g / (3 * 3));
        gm_blurred[flat_idx + 2] = (unsigned char)(r / (3 * 3));
    }
}


int device_main_basic(){
   const int BLOCK_SIZE = 16;
   const int KERNEL_SIZE = 3;
   unsigned char *h_orginal, *h_blurred;
   unsigned char *d_orignal, *d_blurred;
   int width, height, channels;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

    padded_image(&h_orginal, &width, &height, &channels);
    size_t image_size = width * height * channels * sizeof(unsigned char);
    h_blurred = (unsigned char*)malloc(image_size);
    
    cudaMalloc(&d_orignal, image_size);
    cuda_check_errors("device orignal alloc");

    cudaMalloc(&d_blurred, image_size);
    cuda_check_errors("device blurred alloc");
    
    int num_blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(num_blocks_x, num_blocks_y);
    cudaEventRecord(start, 0);
    
    cudaMemcpy(d_orignal, h_orginal, image_size, cudaMemcpyHostToDevice);
    cuda_check_errors("Host to device copy");
    devicek_blur_basic<<<grid_size, block_size>>>(d_orignal, width, height, channels, d_blurred);
    cudaDeviceSynchronize();
    cuda_check_errors("Device Synchronize");
    cudaMemcpy(h_blurred, d_blurred, image_size, cudaMemcpyDeviceToHost);
    cuda_check_errors("Device To Host Copy");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Device Basic Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat blurred_image(height, width, CV_8UC3, h_blurred);
    cv::imwrite("dbsic_mh_blurred.jpg", blurred_image);

    free(h_orginal);
    free(h_blurred);
    cudaFree(d_orignal);
    cudaFree(d_blurred);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0; 
}


/*
    Blurring a Image matrix

    0,0     0,1     0,2     0,3     0,4     
    1,0     1,1     1,2     1,3     1,4
    2,0     2,1     2,2     2,3     2,4
    3,0     3,1     3,2     3,3     3,4  

    Simple way is to first pad the image left right top and down
    or set some boundary conditions, however, padding would keep
    concept simpler. 
    Then after padding, take surrouning pixesl in every direction
    then add them up

*/

int host_main()
{
    const int kernel_size = 3;
    unsigned char *h_original, *h_blurred;
    int width, height, channels;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    padded_image(&h_original, &width, &height, &channels);

    h_blurred = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));

    cudaEventRecord(start, 0);

    for (int y = 1; y < height - 1; y++){
        for (int x = 1; x < width - 1; x++){
            for (int z = 0; z < channels; z++){

                int sum = 0;                
                for (int ky = -1; ky <= 1; ky++){
                    for (int kx = -1; kx <= 1; kx++){
                        int flat_nbr_idx = (((y + ky) * width) + (x + kx)) * channels + z;
                        sum += h_original[flat_nbr_idx];
                    }
                }

                int flat_idx = (y * width  + x) * channels + z;
                h_blurred[flat_idx] = (unsigned char)(sum / (kernel_size * kernel_size));
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Elapsed Times: %f ms\n", elapsed_time);

    cv::Mat blurred_image(height, width, CV_8UC3, h_blurred);
    cv::imwrite("h_mh_blurred.jpg", blurred_image);

    free(h_blurred);

    return 0; 
}

int main(int argc, char *argv[]){

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
        }
    }

    return 0;
}