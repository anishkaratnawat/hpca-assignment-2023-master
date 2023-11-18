#include <cstdint>
#include <smmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

// Optimize this function

void singleThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
{


    const int vectorSize = 8;

    // Iterate over the output matrix in vectorized blocks
    for (int output_i = 0; output_i < output_row; ++output_i) {
        for (int output_j = 0; output_j < output_col; ++output_j) {

            int output_index = output_i * output_col + output_j;

            // Initialize the result vector with zeros
            __m256 resultVector = _mm256_setzero_ps();

            // Iterate over the kernel matrix
            for (int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {
                int input_i_offset = 2 * kernel_i;
                int input_i = (output_i + input_i_offset) % input_row;

                for (int kernel_j = 0; kernel_j < kernel_col; ++kernel_j) {
                    int input_j_offset = 2 * kernel_j;
                    int input_j = (output_j + input_j_offset) % input_col;

                    int input_index = input_i * input_col + input_j;
                    int kernel_index = kernel_i * kernel_col + kernel_j;

                    // Load vectors
                    __m256 inputVector = _mm256_loadu_ps(&input[input_index]);
                    __m256 kernelVector = _mm256_loadu_ps(&kernel[kernel_index]);

                    // Multiply and accumulate
                    resultVector = _mm256_fmadd_ps(inputVector, kernelVector, resultVector);
                }
            }

            // Store the result vector in the output matrix
            _mm256_storeu_ps(&output[output_index], resultVector);
        }
    }
  
}