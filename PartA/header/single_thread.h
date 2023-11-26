#include <vector>

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
    //dilation factor
    int dilation = 2;
    int input_i, input_j;
    long long unsigned int partial_sum;
    long long unsigned int partial_sum_1;
    long long unsigned int partial_sum_2;
    long long unsigned int partial_sum_3;
    long long unsigned int partial_sum_4;
    long long unsigned int partial_sum_5;
    long long unsigned int partial_sum_6;
    long long unsigned int partial_sum_7;

    for (int output_i = 0; output_i < output_row; ++output_i) {
        int output_row_skip = output_i * output_col;

        for (int output_j = 0; output_j < output_col; output_j+=8) {
            
            //loop unrolling
            partial_sum = 0;
            partial_sum_1 = 0;
            partial_sum_2 = 0;
            partial_sum_3 = 0;
            partial_sum_4 = 0;
            partial_sum_5 = 0;
            partial_sum_6 = 0;
            partial_sum_7 = 0;

            input_i = output_i;

            for (int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {

                int kernel_row_skip = kernel_i * kernel_col;

                //reducing ALU operations
                int input_row_skip = input_i * input_col;
                input_j = output_j;

                for (int kernel_j = 0; kernel_j < kernel_col; ++kernel_j) {

                    //loop unrolling
                    partial_sum = partial_sum + input[input_row_skip + input_j] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_1 = partial_sum_1 + input[input_row_skip + (input_j+1)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_2 = partial_sum_2 + input[input_row_skip + (input_j+2)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_3 = partial_sum_3 + input[input_row_skip + (input_j+3)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_4 = partial_sum_4 + input[input_row_skip + (input_j+4)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_5 = partial_sum_5 + input[input_row_skip + (input_j+5)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_6 = partial_sum_6 + input[input_row_skip + (input_j+6)%input_col] * kernel[kernel_row_skip + kernel_j];
                    partial_sum_7 = partial_sum_7 + input[input_row_skip + (input_j+7)%input_col] * kernel[kernel_row_skip + kernel_j];
                    input_j = input_j + dilation;

                    if(input_j >= input_col)
                        input_j = input_j % input_col;
                }

                input_i = input_i + dilation;
                if(input_i >= input_row)
                        input_i = input_i % input_row;
            }
            
            //loop unrolling
            output[output_row_skip + output_j] = partial_sum;
            if(output_j + 1 < output_col)
            output[output_row_skip + output_j + 1] = partial_sum_1;
            if(output_j + 2 < output_col)
            output[output_row_skip + output_j + 2] = partial_sum_2;
            if(output_j + 3 < output_col)
            output[output_row_skip + output_j + 3] = partial_sum_3;
            if(output_j + 4 < output_col)
            output[output_row_skip + output_j + 4] = partial_sum_4;
            if(output_j + 5 < output_col)
            output[output_row_skip + output_j + 5] = partial_sum_5;
            if(output_j + 6 < output_col)
            output[output_row_skip + output_j + 6] = partial_sum_6;
            if(output_j + 7 < output_col)
            output[output_row_skip + output_j + 7] = partial_sum_7;
        }
    }

}
