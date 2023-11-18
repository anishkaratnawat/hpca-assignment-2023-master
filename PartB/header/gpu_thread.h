#include <cuda_runtime.h>

// CUDA kernel for the computation
__global__ void gpuThreadKernel(int input_row, int input_col,
                                int *input, int kernel_row, int kernel_col,
                                int *kernel, int output_row, int output_col,
                                long long unsigned int *output) {
    // Calculate thread indices
    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within the output dimensions
    if (output_i < output_row && output_j < output_col) {
        for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++) {
            for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++) {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_j = (output_j + 2 * kernel_j) % input_col;
                int kernel_index = kernel_i * kernel_col + kernel_j;

                atomicAdd(&output[output_i * output_col + output_j],
                          input[input_i * input_col + input_j] * kernel[kernel_index]);
            }
        }
    }
}

// Wrapper function to call the CUDA kernel
void gpuThread(int input_row, int input_col, int *input,
               int kernel_row, int kernel_col, int *kernel,
               int output_row, int output_col, long long unsigned int *output) {
    // Define block and grid dimensions
    dim3 blockSize(64,64); // Adjust block size as needed
    dim3 gridSize((output_col + blockSize.x - 1) / blockSize.x,
                  (output_row + blockSize.y - 1) / blockSize.y);

    // Allocate device memory
    int *d_input, *d_kernel;
    long long unsigned int *d_output;
    cudaMalloc((void **)&d_input, input_row * input_col * sizeof(int));
    cudaMalloc((void **)&d_kernel, kernel_row * kernel_col * sizeof(int));
    cudaMalloc((void **)&d_output, output_row * output_col * sizeof(long long unsigned int));

    // Copy input and kernel data to device
    cudaMemcpy(d_input, input, input_row * input_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_row * kernel_col * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    gpuThreadKernel<<<gridSize, blockSize>>>(input_row, input_col, d_input,
                                             kernel_row, kernel_col, d_kernel,
                                             output_row, output_col, d_output);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, output_row * output_col * sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
