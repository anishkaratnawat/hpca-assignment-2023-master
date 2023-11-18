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


    for (int i = 0; i < output_row; ++i) {
        for (int j = 0; j < output_col; ++j) {
            __m256i sum = _mm256_setzero_si256();

            for (int ki = 0; ki < kernel_row; ++ki) {
                for (int kj = 0; kj < kernel_col; ++kj) {
                    __m256i inputVal = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[(i + 2*ki) * input_col + (j + 2*kj)]));
                    __m256i kernelVal = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&kernel[ki * kernel_col + kj]));
                    sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(inputVal, kernelVal));
                }
                  _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[i * output_col + j]), sum);
            }
        }
    }
  
}