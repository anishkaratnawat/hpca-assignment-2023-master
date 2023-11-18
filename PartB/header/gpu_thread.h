// Create other necessary functions here

// Fill in this function
void gpuThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_row && col < output_row) {
        int sum = 0;

        // Loop over the kernel and input matrix to perform convolution
        for (int ki = 0; ki < kernel_col; ++ki) {
            for (int kj = 0; kj < kernel_col; ++kj) {
                int inputIndex = (row + ki) * input_col + (col + kj);
                int kernelIndex = ki * kernel_col + kj;
                sum += input[inputIndex] * kernel[kernelIndex];
            }
        }

        // Assign the result to the output matrix
        int outputIndex = row * output_col + col;
        output[outputIndex] = sum;
    }
}

// Host function to perform dilated convolution on GPU
void dilatedConvolutionGPU(const int* h_input, const int* h_kernel, int* h_output,
                            int inputSize, int kernelSize, int outputSize) {
    int* d_input, * d_kernel, * d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, inputSize * inputSize * sizeof(int));
    cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(int));
    cudaMalloc((void**)&d_output, outputSize * outputSize * sizeof(int));

    // Copy input and kernel matrices from host to device
    cudaMemcpy(d_input, h_input, inputSize * inputSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x, (outputSize + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    gpuThread<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, inputSize, kernelSize, outputSize);

    // Copy the result back from device to host
    cudaMemcpy(h_output, d_output, outputSize * outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory on GPU
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

}
