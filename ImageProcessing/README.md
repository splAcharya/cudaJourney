**# RGB TO Grayscale**
------------------------------------------------------------------------
### Basic Kernel:

Input Image (Read Using OpenCV), is in row-major format, Each thread get a pixel i.e the pixels R,G,B channel converts it writes it output matrix. 
The HOST comparison is using a nested loop to iterate through pixel 1 at a time. The elapsed times:
HOST: 4.782784 ms VS DEVICE: 2.177376 ms .. sort of 2x faster, the device time does include the memcpy copy cost from H2D and D2H.
NOTE: Assumes a BLOCK_SIZE of 256, despite 1024 threads could be used. Device saturation is not stressed here.


Global memory Efficiency:
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 20.98
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct 36.15
This seems to suggested low coalesced access.

Top Level Profile From Nsight Compute:
![alt text](grayscale_conversion_basic.png)

** Optimization Possibilities **:
--------------------------------------------------------
This kernel could benefits from cudastreams and chunking
as long as all the chunks contains all the 3 channels of a pixel.
Another optimization attempt was done by using shared memory,
threads local individual element (as opposed to a pixel)
from global memory into shared memory. The 2 out of every 3
threads adds up 3 position within shared memory to get the grayscale conversion. Then 85/BLOCK_SIZE threads then
write data into global memory/. BLOCK 0 would write from 0 --> 85 (Assuming block size of 256), block 1 would write from 85 --> 169 and so on. I still have not figured out that yet.


**# Image Blur (3 x 3) kernel **
------------------------------------------------------------------------
### Basic Kernel: Exactly like the Grasycale conversion. Host Processes each element in a triple nested loop.
Elapsed Times: 72.84 ms (HOST) Vs 3.83 ms (Device)
Compared to grayscale, even basic GPU kernel is 94% faster.
Again a BLOCK_SIZE of 256 is used, so device staturation is not stressed here.

GLobal memory Efficiency:
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 20.68
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct 20.79

Top Level Profile From Nsight Compute:
![alt text](blur_basic.png)



**### DEVICE PROPERTIES**
Device Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Max Blocks per Multiprocessor: 16Threads per Multiprocessor: 1536
Max Block: 1024
Max Threads per 