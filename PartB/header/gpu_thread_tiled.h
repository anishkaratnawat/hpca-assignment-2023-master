#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Define the tile width

// CUDA kernel for the tiled 2D convolution with dilation
__global__ void gpuTiledKernelDilation(int input_row, int input_col,
                                       int *input, int kernel_row, int kernel_col,
                                       int *kernel, int output_row, int output_col,
                                       long long unsigned int *output, int dilation) {
    // Shared memory for the input tile (adjusted for dilation and kernel size)
    __shared__ int shared_input[TILE_WIDTH + 3][TILE_WIDTH + 3]; 
    __shared__ int shared_kernel[3][3]; // Shared memory for kernel (assuming kernel is 3x3)

    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int output_i = blockIdx.y * TILE_WIDTH + ty;
    int output_j = blockIdx.x * TILE_WIDTH + tx;

    // Load the kernel into shared memory
    if (ty < kernel_row && tx < kernel_col) {
        shared_kernel[ty][tx] = kernel[ty * kernel_col + tx];
    }

    // Load input into shared memory, accounting for dilation
    int input_i = output_i * dilation;
    int input_j = output_j * dilation;
    
    if (input_i < input_row && input_j < input_col) {
        shared_input[ty][tx] = input[input_i * input_col + input_j];
    } else {
        shared_input[ty][tx] = 0;  // Pad with zeros if out of bounds
    }

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    long long unsigned int partial_sum = 0;

    // Perform the convolution only if the thread corresponds to a valid output element
    if (output_i < output_row && output_j < output_col) {
        for (int i = 0; i < kernel_row; ++i) {
            for (int j = 0; j < kernel_col; ++j) {
                int input_y = ty + i * dilation;
                int input_x = tx + j * dilation;

                // Ensure we don't access out-of-bounds shared memory
                if (input_y < TILE_WIDTH + 2 && input_x < TILE_WIDTH + 2) {
                    partial_sum += shared_input[input_y][input_x] * shared_kernel[i][j];
                }
            }
        }
        // Store the result in the output array
        output[output_i * output_col + output_j] = partial_sum;
    }
}

// Wrapper function to call the CUDA kernel
void gpuTiledWithDilation(int input_row, int input_col, int *input,
                          int kernel_row, int kernel_col, int *kernel,
                          int output_row, int output_col, long long unsigned int *output, int dilation) {

    // Define block and grid dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((output_col + TILE_WIDTH - 1) / TILE_WIDTH,
                  (output_row + TILE_WIDTH - 1) / TILE_WIDTH);

    // Allocate device memory
    int *device_input, *device_kernel;
    long long unsigned int *device_output;
    
    cudaMalloc((void **)&device_input, input_row * input_col * sizeof(int));
    cudaMalloc((void **)&device_kernel, kernel_row * kernel_col * sizeof(int));
    cudaMalloc((void **)&device_output, output_row * output_col * sizeof(long long unsigned int));

    // Copy input and kernel data to device
    cudaMemcpy(device_input, input, input_row * input_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel, kernel, kernel_row * kernel_col * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with dilation
    gpuTiledKernelDilation<<<gridSize, blockSize>>>(input_row, input_col, device_input,
                                                    kernel_row, kernel_col, device_kernel,
                                                    output_row, output_col, device_output, dilation);

    // Copy the result back to the host
    cudaMemcpy(output, device_output, output_row * output_col * sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_kernel);
    cudaFree(device_output);
}
