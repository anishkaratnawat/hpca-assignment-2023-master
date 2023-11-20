// #include <cuda_runtime.h>

// // CUDA kernel for the computation
// __global__ void gpuThreadKernel(int input_row, int input_col,
//                                 int *input, int kernel_row, int kernel_col,
//                                 int *kernel, int output_row, int output_col,
//                                 long long unsigned int *output) {
//     // Calculate thread indices
//     int output_i = blockIdx.y * blockDim.y + threadIdx.y;
//     int output_j = blockIdx.x * blockDim.x + threadIdx.x;
//     int dilation = 2;
//     int input_i, input_j;
//     long long unsigned int partial_sum;
//     // Check if thread is within the output dimensions
//     if (output_i < output_row && output_j < output_col) {
 
//         int output_row_skip = output_i * output_col;
//         partial_sum = 0;
//         input_i = output_i;
        
//         for (int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {
//             int kernel_row_skip = kernel_i * kernel_col;
//             int input_row_skip = input_i * input_col;
//             input_j = output_j;

//             for (int kernel_j = 0; kernel_j < kernel_col; ++kernel_j) {

//                 partial_sum = partial_sum + input[input_row_skip + input_j] * kernel[kernel_row_skip + kernel_j];

//                 input_j = input_j + dilation;
//                 if(input_j >= input_col)
//                     input_j = input_j % input_col;
//             }

//             input_i = input_i + dilation;
//             if(input_i >= input_row)
//                     input_i = input_i % input_row;
//         }

//         output[output_row_skip + output_j] = partial_sum;
//     }
    
// }

// // __global__ void gpuThreadKernel(int input_row, int input_col,
// //                                 int *input, int kernel_row, int kernel_col,
// //                                 int *kernel, int output_row, int output_col,
// //                                 long long unsigned int *output) {
// //     // Calculate thread indices
// //     int output_i = blockIdx.y * blockDim.y + threadIdx.y;
// //     int output_j = blockIdx.x * blockDim.x + threadIdx.x;

// //     // Check if thread is within the output dimensions
// //     if (output_i < output_row && output_j < output_col) {
// //         for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++) {
// //             for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++) {
// //                 int input_i = (output_i + 2 * kernel_i) % input_row;
// //                 int input_j = (output_j + 2 * kernel_j) % input_col;
// //                 int kernel_index = kernel_i * kernel_col + kernel_j;

// //                 atomicAdd(&output[output_i * output_col + output_j],
// //                           input[input_i * input_col + input_j] * kernel[kernel_index]);
// //             }
// //         }
// //     }
// // }

// // Wrapper function to call the CUDA kernel
// void gpuThread(int input_row, int input_col, int *input,
//                int kernel_row, int kernel_col, int *kernel,
//                int output_row, int output_col, long long unsigned int *output) {

//     // Define block and grid dimensions
//     dim3 blockSize(16,16);
//     dim3 gridSize((output_col + blockSize.x - 1) / blockSize.x,
//                   (output_row + blockSize.y - 1) / blockSize.y);

//     // Allocate device memory
//     int *device_input, *device_kernel;
//     long long unsigned int *device_output;
    
//     cudaMalloc((void **)&device_input, input_row * input_col * sizeof(int));
//     cudaMalloc((void **)&device_kernel, kernel_row * kernel_col * sizeof(int));
//     cudaMalloc((void **)&device_output, output_row * output_col * sizeof(long long unsigned int));

//     // Copy input and kernel data to device
//     cudaMemcpy(device_input, input, input_row * input_col * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(device_kernel, kernel, kernel_row * kernel_col * sizeof(int), cudaMemcpyHostToDevice);

//     // Launch the kernel
//     gpuThreadKernel<<<gridSize, blockSize>>>(input_row, input_col, device_input,
//                                              kernel_row, kernel_col, device_kernel,
//                                              output_row, output_col, device_output);

//     // Copy the result back to the host
//     cudaMemcpy(output, device_output, output_row * output_col * sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_input);
//     cudaFree(device_kernel);
//     cudaFree(device_output);
// }

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <string.h>
using namespace std;
 
#define THREADS_PER_BLOCK 16
 
__global__
void dilatedConvolutionKernel(int input_row,
                                         int input_col,
                                         int *input,
                                         int kernel_row,
                                         int kernel_col,
                                         int *kernel,
                                         int output_row,
                                         int output_col,
                                         long long unsigned int *output)
{
    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;
    int input_i , input_j;
    int value=0;
   
    if (output_i < output_row && output_j < output_col)
    {
        int output_offset = output_i * output_col;
        input_i  = output_i-2;
    for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                int kernel_offset=kernel_i*kernel_col ;
 
                if(input_i +2 >= input_row)
                {
                    if(input_i+2==input_col)
                        input_i=0;
                    else
                    input_i=1;
                }
                else
                    input_i=(input_i+2);
 
 
                int input_offset = input_i*input_col;
 
                 input_j= output_j-2;
 
                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    if(input_j +2 >= input_col)
                    {
                        if(input_j+2==input_col)
                            input_j=0;
                        else
                            input_j=1;
                    }
 
                    else
                        input_j=(input_j+2);
 
                       
                    value+= input[ input_offset+input_j] * kernel[kernel_offset+kernel_j];
                }
            }
 
            output[ output_offset+ output_j] = value;
        }
 
       
           
 
 
    }
 
 
void gpuThread(int input_row,
               int input_col,
               int *input,
               int kernel_row,
               int kernel_col,
               int *kernel,
               int output_row,
               int output_col,
               long long unsigned int *output)
{
    int* d_input, *d_kernel;
    long long unsigned int *d_output;
    cudaMalloc((void**)&d_input, sizeof(int) * input_row * input_col);
    cudaMalloc((void**)&d_kernel, sizeof(int) * kernel_row * kernel_col);
    cudaMalloc((void**)&d_output, sizeof(int) * output_row * output_col);
 
    // Copy input and kernel to device
    cudaMemcpy(d_input, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice);
 
    // Launch kernel
    dim3 blockDim(2, 2);
    dim3 gridDim((output_row + blockDim.x - 1) / blockDim.x, (output_col + blockDim.y - 1) / blockDim.y);
 
    dilatedConvolutionKernel<<<gridDim, blockDim>>>(input_row, input_col, d_input, kernel_row, kernel_col,d_kernel, output_row, output_col,d_output);
 
    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(int) * output_row * output_col, cudaMemcpyDeviceToHost);
 
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
 
}