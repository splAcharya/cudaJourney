#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cuda_runtime.h>


#define cuda_check_errors(msg) \
    do{\
        cudaError_t __err = cudaGetLastError();\
        if (__err != cudaSuccess){\
            printf("Fatal Error:%s=>()"
            )
        }\
    }while(0)