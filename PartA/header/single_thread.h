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
    
    for(int i = 0; i < output_row * output_col; ++i)
        output[i] = 0;

    for(int output_i = 0; output_i< output_row; output_i++)
    {
        for(int output_j = 0; output_j< output_col; output_j++)
        {
            for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2*kernel_i) % input_row;
                    int input_j = (output_j + 2*kernel_j) % input_col;
                    output[output_i * output_col + output_j] += input[input_i*input_col +input_j] 
                                                                * kernel[kernel_i*kernel_col +kernel_j];
                }
            }
        }
    }
}
