#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel to perform dilated convolution on GPU
__global__ void dilatedConvolutionKernel(int input_row, int input_col, const int* input,
                                         int kernel_row, int kernel_col, const int* kernel,
                                         int output_row, int output_col, unsigned long long int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < output_row && j < output_col) {
        unsigned long long int sum = 0;

        for (int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {
            for (int kernel_j = 0; kernel_j < kernel_col; ++kernel_j) {
                int kernel_index = kernel_i * kernel_col + kernel_j;
                
                int input_i = (i + 2 * kernel_i) % input_row;
                int input_j = (j + 2 * kernel_j) % input_col;
                
                sum += static_cast<unsigned long long int>(input[input_i * input_col + input_j]) * 
                       static_cast<unsigned long long int>(kernel[kernel_index]);
            }
        }

        int outputIndex = i * output_col + j;
        output[outputIndex] = sum;
    }
}

// Function to perform dilated convolution on GPU
void gpuThread(int input_row, int input_col, int* input, 
               int kernel_row, int kernel_col, int* kernel,
               int output_row, int output_col, 
               unsigned long long int* output) {

    // Allocate device memory
    int* d_input, *d_kernel;
    unsigned long long int* d_output;
    cudaMalloc((void**)&d_input, input_row * input_col * sizeof(int));
    cudaMalloc((void**)&d_kernel, kernel_row * kernel_col * sizeof(int));
    cudaMalloc((void**)&d_output, output_row * output_col * sizeof(unsigned long long int));

    // Copy input and kernel matrices from host to device
    cudaMemcpy(d_input, input, input_row * input_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_row * kernel_col * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((output_row + blockSize.x - 1) / blockSize.x, (output_col + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    dilatedConvolutionKernel<<<gridSize, blockSize>>>(input_row, input_col, d_input,
                                                       kernel_row, kernel_col, d_kernel,
                                                       output_row, output_col, d_output);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, output_row * output_col * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}