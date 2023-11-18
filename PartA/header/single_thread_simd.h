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
            __m128i sum = _mm_setzero_si128();

            for (int ki = 0; ki < kernel_row; ++ki) {
                for (int kj = 0; kj < kernel_col; ++kj) {
                    __m128i inputVal = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(i + ki) * input_col + (j + kj)]));
                    __m128i kernelVal = _mm_set1_epi32(kernel[ki * kernel_col + kj]);
                    sum = _mm_add_epi32(sum, _mm_mullo_epi32(inputVal, kernelVal));
                }
                  _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[i * output_col + j]), sum);
            }
        }
    }
  
}
// void singleThread(int input_row, int input_col, int *input, int kernel_row,
//                   int kernel_col, int *kernel, int output_row, int output_col,
//                   long long unsigned int *output) {

//     // Iterate through the input
//     for (int i = 0; i < input_row; ++i) {
//         for (int j = 0; j < input_col; ++j) {
//             int rowIndex = i * 2;
//             int colIndex = j * 2;

//             // Iterate through the kernel
//             for (int m = 0; m < kernel_row; ++m) {
//                 for (int n = 0; n < kernel_col; ++n) {
//                     int outputRowIndex = rowIndex + m;
//                     int outputColIndex = colIndex + n;

//                     // Compute the indices
//                     int inputIndex = i * input_col + j;
//                     int kernelIndex = m * kernel_col + n;
//                     int outputIndex = outputRowIndex * output_col + outputColIndex;

//                     __m256i inputVector = _mm256_loadu_si256((const __m256i_u *)(&input[inputIndex]));
//                     __m256i kernelVector = _mm256_loadu_si256((const __m256i_u *)(&kernel[kernelIndex]));

//                     // Multiply input and kernel vectors element-wise, then add to the result vector
//                     __m256i resultVector = _mm256_loadu_si256((const __m256i_u *)(&output[outputIndex]));
//                     resultVector = _mm256_add_epi32(resultVector, _mm256_mullo_epi32(inputVector, kernelVector));

//                     // Store the result vector
//                     _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[outputIndex]), resultVector);
//                 }
//             }
//         }
//     }
// }
// void singleThread(int input_row, int input_col, int *input, int kernel_row,
//                   int kernel_col, int *kernel, int output_row, int output_col,
//                   long long unsigned int *output) {

//   for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++) {  
//       for (int kernel_j = 0; kernel_j < kernel_col; kernel_j+=4) { 
//         for (int output_i = 0; output_i < output_row; output_i++) {
//             for (int output_j = 0; output_j < output_col; output_j+=2) {
//           int input_i = (output_i + 2 * kernel_i) % input_row;
//           int input_j = (output_j + 2 * kernel_j) % input_col;
//           __m256i vinput, vkernel;

//           bool lastColCheck = (input_j + 7) < input_col;
          
//           if(lastColCheck)
//             vinput = _mm256_loadu_si256(
//                 (const __m256i_u *)&input[input_i * input_col + input_j]);
//           else {
//             vinput = _mm256_set_epi32(
//                 input[input_i * input_col +
//                       ((output_j + 1) + 2 * (kernel_j + 3)) % input_col],
//                 input[input_i * input_col +
//                       (output_j + 2 * (kernel_j + 3)) % input_col],
//                 input[input_i * input_col +
//                       ((output_j + 1) + 2 * (kernel_j + 2)) % input_col],
//                 input[input_i * input_col +
//                       (output_j + 2 * (kernel_j + 2)) % input_col],
//                 input[input_i * input_col +
//                       ((output_j + 1) + 2 * (kernel_j + 1)) % input_col],
//                 input[input_i * input_col +
//                       (output_j + 2 * (kernel_j + 1)) % input_col],
//                 input[input_i * input_col +
//                       ((output_j + 1) + 2 * kernel_j) % input_col],
//                 input[input_i * input_col +
//                       (output_j + 2 * kernel_j) % input_col]);

//           } 
//             vkernel = _mm256_set_epi32(
//                 (kernel_j + 3 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 3]
//                        : 0,
//                 (kernel_j + 3 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 3]
//                        : 0,
//                 (kernel_j + 2 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 2]
//                        : 0,
//                 (kernel_j + 2 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 2]
//                        : 0,
//                 (kernel_j + 1 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 1]
//                        : 0,
//                 (kernel_j + 1 < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j + 1]
//                        : 0,
//                 (kernel_j < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j]
//                        : 0,
//                 (kernel_j < kernel_col)
//                        ? kernel[kernel_i * kernel_col + kernel_j]
//                        : 0);

           

//           __m256i output_vec = _mm256_mullo_epi32(vinput, vkernel);
//           output_vec =
//               _mm256_shuffle_epi32(output_vec, _MM_SHUFFLE(3, 1, 2, 0));
//           __m128i sum128 =
//               _mm_add_epi32(_mm256_castsi256_si128(output_vec),
//                             _mm256_extracti128_si256(output_vec, 1));
//           output[output_i * output_col + output_j] += _mm_extract_epi32(sum128, 0);
//           output[output_i * output_col + output_j + 1] += _mm_extract_epi32(sum128, 2);
//           output[output_i * output_col + output_j] += _mm_extract_epi32(sum128, 1);
//           output[output_i * output_col + output_j + 1] += _mm_extract_epi32(sum128, 3);
//         }
//       }
//     }
//   }
// }
