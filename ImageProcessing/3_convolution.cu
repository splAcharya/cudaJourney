#include <stdio.h>
#include <cstring>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

/*** Mask Configuration */
const int MASK_RADIUS = 1;
const int MASK_DIM = (2 * MASK_RADIUS + 1); // 3

/* Block Configuration */
const int BLOCK_DIM = 16;

const int GX_MASK[MASK_DIM][MASK_DIM] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}    
};

const int GY_MASK[MASK_DIM][MASK_DIM] = {
    {-1, -2, -1},
    {0,   0,  0},
    {1,   2,  1}    
};


#define cuda_check_errors(msg) \
    do \
    { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) \
        { \
            printf("Fatal Error:%s, (%s at %s:%d)\n", \
                    msg, \
                    cudaGetErrorString(__err), \
                    __FILE__, \
                    __LINE__);\
        } \
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

int read_image(unsigned char **gray_image_serial, int *width, int *height)
{
    // Read image directly as grayscale
    //Mat gray = imread("manhattan_traffic.jpg", IMREAD_GRAYSCALE);
    //Mat gray = imread("monarch_in_may.jpg", IMREAD_GRAYSCALE);
    Mat gray = imread("edgeflower.jpg", IMREAD_GRAYSCALE);
    
    if (gray.empty())
    {
        printf("Error: Could not load image\n");
        return -1;
    }

    *width  = gray.cols;
    *height = gray.rows;
    int size = gray.cols * gray.rows;

    if (size != gray.total())
    {
        printf("Error In Computed Dimension\n");
        return -1;
    }

    *gray_image_serial = (unsigned char *)malloc(size * sizeof(unsigned char));
    memcpy(*gray_image_serial, gray.data, size);
    return 0;
}



void hostk_basic(unsigned char *gm_gray, unsigned char *gm_edge, int width, int height) 
{
    /* FOr each output pixel apple mask */
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int gx_rsum = 0, gy_rsum = 0;
            for (int my = -1; my <= 1; my ++)
            {
                for (int mx = -1; mx <= 1; mx ++)
                {
                    if((y + my >= 0) && (y + my < height)
                       && (x + mx >= 0) &&  (x + mx < width))
                    {
                        int row_idx_gray = (y + my) * width; //Num elements per row
                        int col_idx_gray = (x + mx);
                        int flat_idx_gray = row_idx_gray + col_idx_gray;
                    
                        gx_rsum += (gm_gray[flat_idx_gray] * GX_MASK[my + 1][mx + 1]);
                        gy_rsum += (gm_gray[flat_idx_gray] * GY_MASK[my + 1][mx + 1]);
                    }
                }
            }

            int magnitude = (int)(sqrt((gx_rsum * gx_rsum) + (gy_rsum * gy_rsum)));
            magnitude = min(255, magnitude);
            
            int row_idx_edge = y * width;
            int col_idx_edge = x;
            int flat_idx_edge = row_idx_edge + col_idx_edge;
            gm_edge[flat_idx_edge] = (unsigned char)magnitude;
        }
    }
}

int host_main_basic(int thres_lo, int thres_hi, bool write_image)
{
    printf("Executing host_main_basic \n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned char *h_gray, *h_edge;
    int width, height;

    read_image(&h_gray, &width, &height);
    printf("Input Dim: (Height:%d, Width:%d)\n", height, width);

    h_edge = (unsigned char *)malloc(width * height* sizeof(unsigned char));

    cudaEventRecord(start, 0);

    hostk_basic(h_gray, h_edge, width, height);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Basic Elaped Time: %f ms \n", elapsed_time);

    if (write_image)
    {
        Mat gray_image(height, width, CV_8UC1, h_edge);
        Mat bw_image, inverted;
        threshold(gray_image, bw_image, thres_lo, thres_hi, THRESH_BINARY);  // adjust 100 if needed
        imwrite("h_edge_bw.jpg", bw_image);
        //bitwise_not(bw_image, inverted);
        //imwrite("h_edge_bwi.jpg", inverted);
    }

    free(h_edge);
    free(h_gray);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    return 0;
}


__global__ void devicek_basic(
    const unsigned char *gm_gray,
    unsigned char *gm_edge,
    const int *gm_mask_gx,
    const int *gm_mask_gy,
    int width,
    int height,
    int mask_dim)
{
    int tid_gl_x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int tid_gl_y = (blockDim.y * blockIdx.y) + threadIdx.y;

    /* Each thread is responsible for 1 output element */
    if ((tid_gl_y < height) && (tid_gl_x < width))
    {
        int gx_rsum = 0, gy_rsum = 0;        

        for (int my = -1; my <= 1; my ++) 
        {
            for (int mx = -1; mx <= 1; mx ++)
            {
                if ((tid_gl_y + my >= 0) && (tid_gl_y + my < height)
                   && (tid_gl_x + mx >= 0) && (tid_gl_x + mx < width))      
                {
                    int row_idx_gray = (tid_gl_y + my) * width;
                    int col_idx_gray = (tid_gl_x + mx) ;
                    int flat_idx_gray = row_idx_gray + col_idx_gray;

                    int row_idx_mask = (my + 1) * mask_dim;
                    int col_idx_mask = mx + 1;
                    int flat_idx_mask = row_idx_mask + col_idx_mask;

                    gx_rsum += (gm_gray[flat_idx_gray] * gm_mask_gx[flat_idx_mask]);
                    gy_rsum += (gm_gray[flat_idx_gray] * gm_mask_gy[flat_idx_mask]);                    
                } 
            }
        }
        
        int magnitude = (int)(sqrtf( ( (float)(gx_rsum * gx_rsum) + (gy_rsum * gy_rsum) ) ) );
        magnitude = min(255, magnitude);

        int row_idx_edge = tid_gl_y * width;
        int col_idx_edge = tid_gl_x;
        int flat_idx_edge = row_idx_edge + col_idx_edge;
        gm_edge[flat_idx_edge] = (unsigned char)magnitude; 
    }
}


int device_main_basic(int thres_lo, int thres_hi, bool write_image)
{
    printf("Executing device_main_basic \n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned char *h_gray, *h_edge;
    unsigned char *d_gray, *d_edge;
    int *d_mask_gx, *d_mask_gy;
    int width, height;

    read_image(&h_gray, &width, &height);
    printf("Input Dim: (Height:%d, Width:%d)\n", height, width);

    h_edge = (unsigned char *)malloc(width * height* sizeof(unsigned char));
    
    cudaEventRecord(start, 0);

    size_t alloc_size_image = width * height * sizeof(unsigned char);
    size_t alloc_size_mask =  (MASK_DIM * MASK_DIM * sizeof(int));
    
    cudaMalloc(&d_gray, alloc_size_image);
    cuda_check_errors("D_GRAY ALLOC");
    
    cudaMalloc(&d_edge, alloc_size_image);
    cuda_check_errors("D_EDGE_ALLOC");
    
    cudaMalloc(&d_mask_gx, alloc_size_mask);
    cuda_check_errors("D_MASK_ALLOC_GX");
    
    cudaMalloc(&d_mask_gy, alloc_size_mask);
    cuda_check_errors("D_MASK_ALLOC_GY");

    cudaMemcpy(d_gray, h_gray, alloc_size_image, cudaMemcpyHostToDevice);
    cuda_check_errors("Gray H2D MCP");

    cudaMemcpy(d_mask_gx, GX_MASK, alloc_size_mask, cudaMemcpyHostToDevice);
    cuda_check_errors("Gray H2D MCP");
    
    cudaMemcpy(d_mask_gy, GY_MASK, alloc_size_mask, cudaMemcpyHostToDevice);
    cuda_check_errors("Gray H2D MCP");

    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    int num_block_x = (width + block_size.x - 1) / block_size.x;
    int num_block_y = (height + block_size.y - 1) / block_size.y;
    dim3 grid_size(num_block_x, num_block_y);
    //printf("NUM BLOCK(x:%d, y:%d), BLock_size:%d\n", num_block_x, num_block_y, BLOCK_DIM);
    devicek_basic<<<grid_size, block_size>>>(d_gray, d_edge, d_mask_gx,  d_mask_gy, width, height, MASK_DIM);
    cudaDeviceSynchronize();
    cuda_check_errors("Klaunch an synch");
    
    cudaMemcpy(h_edge, d_edge, alloc_size_image, cudaMemcpyDeviceToHost);
    cuda_check_errors("Edge D2H MCP");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Host Basic Elapsed Time: %f ms \n", elapsed_time);

    if (write_image)
    {
        Mat gray_image(height, width, CV_8UC1, h_edge);
        Mat bw_image, inverted;
        threshold(gray_image, bw_image, thres_lo, thres_hi, THRESH_BINARY);  // adjust 100 if needed
        imwrite("d_basic_edge_bw.jpg", bw_image);
        //bitwise_not(bw_image, inverted);
        //imwrite("d_basic_edge_bwi.jpg", inverted);
    }

    free(h_edge);
    free(h_gray);

    cudaFree(d_mask_gx);
    cudaFree(d_mask_gy);
    cudaFree(d_gray);
    cudaFree(d_edge);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    
    return 0;
}


/* Possibly limited ot 64k, here
   3 * 3 * sizeof(int) => 9 * 4 bytes = 36 bytes
*/
__constant__ int D_GX_MASK[MASK_DIM][MASK_DIM];
__constant__ int D_GY_MASK[MASK_DIM][MASK_DIM];

__global__ void devicek_const(
    const unsigned char *gm_gray,
    unsigned char *gm_edge,
    int width, 
    int height)
{
    int tid_lc_x = threadIdx.x;
    int tid_lc_y = threadIdx.y;
    int tid_gl_x = (blockDim.x * blockIdx.x) + tid_lc_x;
    int tid_gl_y = (blockDim.y * blockIdx.y) + tid_lc_y;

    if ( ( tid_gl_x < width ) && ( tid_gl_y < height ) )
    {
        int gx_rsum = 0, gy_rsum = 0; 
        
        for (int my = -1; my <= 1; my++)
        {
            for (int mx = -1; mx <= 1; mx++)
            {
                if ((tid_gl_y + my >= 0) && (tid_gl_y + my < height) && 
                    (tid_gl_x + mx >= 0) && (tid_gl_x + mx < width))      
                {
                    int row_idx_gray = (tid_gl_y + my) * width;
                    int col_idx_gray = (tid_gl_x + mx);
                    int flat_idx_gray = row_idx_gray + col_idx_gray;

                    gy_rsum += (gm_gray[flat_idx_gray] * D_GY_MASK[my + 1][mx + 1]);
                    gx_rsum += (gm_gray[flat_idx_gray] * D_GX_MASK[my + 1][mx + 1]);
                } 
            }
        }

        int magnitude = (int)(sqrtf( ( (float)(gx_rsum * gx_rsum) + (gy_rsum * gy_rsum) ) ) );
        magnitude = min(255, magnitude);

        int row_idx_edge = tid_gl_y * width;
        int col_idx_edge = tid_gl_x;
        int flat_idx_edge = row_idx_edge + col_idx_edge;
        gm_edge[flat_idx_edge] = (unsigned char)magnitude;
    }
}


int device_main_const(int thres_lo, int thres_hi, bool write_image)
{
    printf("Executing device_main_const \n");

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned char *h_gray, *h_edge;
    unsigned char *d_gray, *d_edge;
    int width, height;

    read_image(&h_gray, &width, &height);
    printf("Input Dim: (Height:%d, Width:%d)\n", height, width);

    h_edge = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    cudaEventRecord(begin, 0);

    size_t alloc_sz_image = width * height * sizeof(unsigned char);
    size_t alloc_sz_mask = MASK_DIM * MASK_DIM * sizeof(int);

    cudaMalloc(&d_gray, alloc_sz_image);
    cuda_check_errors("D_ALC_GRAY");

    cudaMalloc(&d_edge, alloc_sz_image);
    cuda_check_errors("D_ALC_EDGE");

    cudaMemcpy(d_gray, h_gray, alloc_sz_image, cudaMemcpyHostToDevice);
    cuda_check_errors("GRAY_H2D");

    cudaMemcpyToSymbol(D_GX_MASK, GX_MASK, alloc_sz_mask);
    cuda_check_errors("GX_H2DSYM");

    cudaMemcpyToSymbol(D_GY_MASK, GY_MASK, alloc_sz_mask);
    cuda_check_errors("GY_H2DSYM");

    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    int num_blocks_x = (width + block_size.x - 1) / block_size.x;
    int num_blocks_y = (height + block_size.y - 1) / block_size.y;
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_const<<<grid_size, block_size>>>(d_gray, d_edge, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_edge, d_edge, alloc_sz_image, cudaMemcpyDeviceToHost);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device Const Elapsed Time:%f\n", elapsed_time);

    if (write_image)
    {
        Mat gray_image(height, width, CV_8UC1, h_edge);
        Mat bw_image, inverted;
        threshold(gray_image, bw_image, thres_lo, thres_hi, THRESH_BINARY);  // adjust 100 if needed
        imwrite("d_const_edge_bw.jpg", bw_image);
        //bitwise_not(bw_image, inverted);
        //imwrite("d_const_edge_bwi.jpg", inverted);
    }
    
    cudaFree(d_edge);
    cudaFree(d_gray);

    free(h_edge);
    free(h_gray);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


/* NOTE: 
No need to making DUCKING COMPLICATED, soom threads can simply load 1, 2 or 3 elements, 
elements next to their actual element. For larger mask radius, each block can load elements
in a block stride manner */
__global__ void devicek_tiled(
    const unsigned char *gm_gray,
    unsigned char *gm_edge,
    int width, 
    int height)
{
    int lx = threadIdx.x, ly = threadIdx.y;
    int gx = (blockDim.x * blockIdx.x) + lx;
    int gy = (blockDim.y * blockIdx.y) + ly;

    /* define shared memory to load input tile, including the halo portion */
    __shared__ unsigned char sm_gray[BLOCK_DIM + 2][BLOCK_DIM + 2];

    /* Top left corner halo */
    if (ly == 0 && lx == 0)
    {
        if (gy > 0 && gx > 0) 
            sm_gray[0][0] = gm_gray[((gy - 1 ) * width + gx - 1)];
        else
            sm_gray[0][0] = 0;
    }
   
    /* Halo between top left and top right corner halos*/
    if (ly == 0)
    {
        if (gy > 0 && gx < width)
            sm_gray[ly][lx + 1] = gm_gray[((gy - 1) * width + gx)];
        else
            sm_gray[ly][lx + 1] = 0;
    }

    /* Top right corner halo */
    if (ly == 0 && lx == blockDim.x - 1)
    {
        if (gy > 0 && gx < width - 1)
            sm_gray[ly][lx + 2] = gm_gray[ ( (gy - 1) * width + gx + 1) ];
        else
            sm_gray[ly][lx + 2] = 0;
    }
      
    /* bottom left corner */
    if (ly == blockDim.y - 1 && lx == 0)
    {
        if (gy < height - 1 && gx > 0)
            sm_gray[ly + 2][lx] = gm_gray[ ((gy + 1) * width + gx - 1) ];
        else
            sm_gray[ly + 2][lx] = 0;
    }

    /* middle rows, between bottom left and rigth corner */
    if (ly == blockDim.y - 1)
    {
        if (gy < height - 1 && gx > 0 && gx < width )
            sm_gray[ly + 2][lx + 1] = gm_gray[( gy  + 1) * width + gx];
        else
            sm_gray[ly + 2][lx + 1] = 0;
    }

    /* bottom right corner */
    if ((ly == blockDim.y - 1) && (lx == blockDim.x - 1))
    {
        if (gy < height - 1 && gx < width - 1)    
            sm_gray[ly + 2][lx + 2] = gm_gray[((gy + 1) * width + (gx + 1))];
        else
            sm_gray[ly + 2][lx + 2] = 0;
    }

    /* between top corner and bottom corner on the left side */
    if (lx == 0)
    {
        if (gy > 0 && gy < height - 1 && gx > 0 && gx < width - 1)
            sm_gray[ly + 1][lx] = gm_gray[( (gy * width) + (gx - 1))];
        else
            sm_gray[ly + 1][lx] = 0;
    }

    /* between top and bottom corner on the right side */
    if (lx == blockDim.x - 1)
    {
        if (gy > 0 && gy < height - 1 && gx > 0 && gx < width - 1)
            sm_gray[ly + 1][lx + 2] = gm_gray[((gy * width) + (gx + 1))];
        else
            sm_gray[ly + 1][lx + 2] = 0;
    }

    /* actual image */
    if (gy < height && gx < width)
    {
        sm_gray[ly + 1][lx + 1] = gm_gray[( (gy * width) + gx )];
    }
    else
    {
        sm_gray[ly + 1][lx + 1] = 0;
    }

    __syncthreads();

    /* convolution */
    if (gx < width && gy < height)
    {
        int gx_rsum = 0, gy_rsum = 0;

        for (int my = -1; my <= 1; my++)
        {
            for (int mx = -1; mx <= 1; mx++)
            {
                gx_rsum += (sm_gray[ly + 1 + my][lx + 1 + mx] * D_GX_MASK[my + 1][mx + 1]);
                gy_rsum += (sm_gray[ly + 1 + my][lx + 1 + mx] * D_GY_MASK[my + 1][mx + 1]);
            }
        }

        int magnitude = (int)(sqrtf( ( (float)(gx_rsum * gx_rsum) + (gy_rsum * gy_rsum) ) ) );
        magnitude = min(255, magnitude);

        gm_edge[(gy * width) + gx] = (unsigned char)magnitude;
    }
}


int device_main_tiled(int thres_lo, int thres_hi, bool write_image)
{
    printf("Executing device_main_tiled \n");

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    unsigned char *h_gray, *h_edge;
    unsigned char *d_gray, *d_edge;
    int width, height;

    read_image(&h_gray, &width, &height);
    printf("Input Dim: (Height:%d, Width:%d)\n", height, width);

    h_edge = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    cudaEventRecord(begin, 0);

    size_t alloc_sz_image = width * height * sizeof(unsigned char);
    size_t alloc_sz_mask = MASK_DIM * MASK_DIM * sizeof(int);

    cudaMalloc(&d_gray, alloc_sz_image);
    cuda_check_errors("D_ALC_GRAY");

    cudaMalloc(&d_edge, alloc_sz_image);
    cuda_check_errors("D_ALC_EDGE");

    cudaMemcpy(d_gray, h_gray, alloc_sz_image, cudaMemcpyHostToDevice);
    cuda_check_errors("GRAY_H2D");

    cudaMemcpyToSymbol(D_GX_MASK, GX_MASK, alloc_sz_mask);
    cuda_check_errors("GX_H2DSYM");

    cudaMemcpyToSymbol(D_GY_MASK, GY_MASK, alloc_sz_mask);
    cuda_check_errors("GY_H2DSYM");

    dim3 block_size(BLOCK_DIM, BLOCK_DIM);
    int num_blocks_x = (width + block_size.x - 1) / block_size.x;
    int num_blocks_y = (height + block_size.y - 1) / block_size.y;
    dim3 grid_size(num_blocks_x, num_blocks_y);
    devicek_tiled<<<grid_size, block_size>>>(d_gray, d_edge, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_edge, d_edge, alloc_sz_image, cudaMemcpyDeviceToHost);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    printf("Device Tiled Elapsed Time:%f\n", elapsed_time);

    if (write_image)
    {
        Mat gray_image(height, width, CV_8UC1, h_edge);
        Mat bw_image;
        threshold(gray_image, bw_image, thres_lo, thres_hi, THRESH_BINARY);  // adjust 100 if needed
        imwrite("d_tiled_edge_bw.jpg", bw_image);
    }
    
    cudaFree(d_edge);
    cudaFree(d_gray);

    free(h_edge);
    free(h_gray);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return 0;
}


int main(int argc, char *argv[]) 
{
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <mode> [submode] [ThresLo] [ThresHi] [WriteImage]\n", argv[0]);
        return 1;
    }

    int thres_lo = atoi(argv[3]);
    int thres_hi = atoi(argv[4]);
    bool write_image = (strcmp(argv[5], "true") == 0) ? true : false;

    if (strcmp(argv[1], "HOST") == 0) 
    {
        if (strcmp(argv[2], "BASIC") == 0) 
        {
            host_main_basic(thres_lo, thres_hi, write_image);
        } 
        else 
        {
            //host_main_tiled(verif, M, N, P);
        }
    } 
    else if (strcmp(argv[1], "DEVICE") == 0) 
    {
        print_device_prop();

        if (strcmp(argv[2], "BASIC") == 0) 
        {
            device_main_basic(thres_lo, thres_hi, write_image);
        } 
        else if (strcmp(argv[2], "CONST") == 0)
        {
            device_main_const(thres_lo, thres_hi, write_image);
        }
        else if (strcmp(argv[2], "TILED") == 0)
        {
            device_main_tiled(thres_lo, thres_hi, write_image);
        }
        else
        {

        }
    } 
    else 
    {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}