int * interleave(int *array, int row, int col)
{
    int *interleaved_array = new int[row * col];
    int f_i=0, s_i=0;
    for(int i = 0; i < row/2; ++i, f_i+=2) //even-even 
    {   
        s_i=0;
        for(int j = 0; j < col/2; ++j, s_i+=2)
            interleaved_array[(i*row+j)] = array[f_i*col+s_i];
    }
    f_i=0,s_i=1;
    for(int i = 0; i < row/2; ++i, f_i+=2) //even-odd
    {    
        s_i=1;
        for(int j = col/2; j < col; ++j, s_i+=2)
            interleaved_array[i*row+j] = array[f_i*col+s_i];
    }

    f_i=1,s_i=0;
    for(int i = row/2; i < row; ++i, f_i+=2) //odd-even
    {    
        s_i=0;
        for(int j = 0; j < col/2; ++j, s_i+=2)
            interleaved_array[i*row+j] = array[f_i*col+s_i];
    }

    f_i=1,s_i=1;
    for(int i = row/2; i < row; ++i, f_i+=2) //odd-odd
    {   
        s_i=1;
        for(int j = col/2; j < col; ++j, s_i+=2)
            interleaved_array[i*row+j] = array[f_i*col+s_i];
    }
    return interleaved_array;
}

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
    int * input_new = interleave(input, input_row, input_col);
    //Approach 1
    //even-even
    int kernel_index;
    for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
    {
        for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
        {
            kernel_index = kernel_i*kernel_col +kernel_j;
            for(int output_i = 0; output_i< output_row; output_i+=2)
            {
                int input_i = ((output_i/2 + kernel_i)%(input_row/2));
                for(int output_j = 0; output_j< output_col; output_j+=2)
                {
                    int input_j = (output_j/2 + kernel_j)%(input_col/2);
                    output[output_i* output_col+output_j]+= input_new[input_i*input_col + input_j]* kernel[kernel_index];
                }
            }
        }
    }
    //even-odd
    for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
    {
        for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
        {
            kernel_index = kernel_i*kernel_col +kernel_j;
            for(int output_i = 0; output_i< output_row; output_i+=2)
            {
                int input_i = ((output_i/2 + kernel_i)%(input_row/2));
                for(int output_j = 1; output_j< output_col; output_j+=2)
                {
                    int input_j = (input_col/2)+(output_j/2 + kernel_j)%(input_col/2);
                    output[output_i* output_col+output_j]+= input_new[input_i*input_col + input_j]* kernel[kernel_index];
                }
            }
        }
    }
    //odd-even
    for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
    {
        for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
        {
            kernel_index = kernel_i*kernel_col +kernel_j;
            for(int output_i = 1; output_i< output_row; output_i+=2)
            {
                int input_i = (input_row/2)+((output_i/2 + kernel_i)%(input_row/2));
                for(int output_j = 0; output_j< output_col; output_j+=2)
                {
                    int input_j = (output_j/2 + kernel_j)%(input_col/2);
                    output[output_i* output_col+output_j]+= input_new[input_i*input_col + input_j]* kernel[kernel_index];
                }
            }
        }
    }
    //odd-odd
    for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
    {
        for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
        {
            kernel_index = kernel_i*kernel_col +kernel_j;
            for(int output_i = 1; output_i< output_row; output_i+=2)
            {
                int input_i = (input_row/2)+((output_i/2 + kernel_i)%(input_row/2));
                for(int output_j = 1; output_j< output_col; output_j+=2)
                {
                    int input_j = (input_col/2)+(output_j/2 + kernel_j)%(input_col/2);
                    output[output_i* output_col+output_j]+= input_new[input_i*input_col + input_j]* kernel[kernel_index];
                }
            }
        }
    }
}
