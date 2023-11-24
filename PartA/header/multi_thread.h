#include <pthread.h>
#include <thread>

// Create other necessary functions here

int num_threads = 8;
const auto cores = std::thread::hardware_concurrency();

struct ThreadData {
    int *input;
    int *kernel;
    long long unsigned int *output;
    int input_row, input_col, output_row, output_col, kernel_row, kernel_col;
    int start_i, end_i; // Range of rows for this thread
    int dilation;
};


void* threadDilatedConvolution(void* arg) {
    ThreadData* data = (ThreadData*)(arg);
    long long unsigned int partial_sum;
    long long unsigned int partial_sum_1;
    long long unsigned int partial_sum_2;
    long long unsigned int partial_sum_3;
    long long unsigned int partial_sum_4;
    long long unsigned int partial_sum_5;
    long long unsigned int partial_sum_6;
    long long unsigned int partial_sum_7;
    int input_i, input_j;

        for (int output_i = data->start_i; output_i < data->end_i; ++output_i) {
        int output_row_skip = output_i * data->output_col;

        for (int output_j = 0; output_j < data->output_col; output_j+=8) {
            partial_sum = 0;
            partial_sum_1 = 0;
            partial_sum_2 = 0;
            partial_sum_3 = 0;
            partial_sum_4 = 0;
            partial_sum_5 = 0;
            partial_sum_6 = 0;
            partial_sum_7 = 0;
            
            input_i = output_i;
            for (int kernel_i = 0; kernel_i < data->kernel_row; ++kernel_i) {
                int kernel_row_skip = kernel_i * data->kernel_col;

                int input_row_skip = input_i * data->input_col;
                input_j = (output_j) % data->input_col;

                for (int kernel_j = 0; kernel_j < data->kernel_col; ++kernel_j) {
                    partial_sum = partial_sum + data->input[input_row_skip + input_j] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_1 = partial_sum_1 + data->input[input_row_skip + (input_j+1)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_2 = partial_sum_2 + data->input[input_row_skip + (input_j+2)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_3 = partial_sum_3 + data->input[input_row_skip + (input_j+3)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_4 = partial_sum_4 + data->input[input_row_skip + (input_j+4)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_5 = partial_sum_5 + data->input[input_row_skip + (input_j+5)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_6 = partial_sum_6 + data->input[input_row_skip + (input_j+6)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    partial_sum_7 = partial_sum_7 + data->input[input_row_skip + (input_j+7)%data->input_col] * data->kernel[kernel_row_skip + kernel_j];
                    

                    input_j = input_j + data->dilation;
                    if(input_j >= data->input_row)
                        input_j = input_j % data->input_row;

                }
                 input_i = input_i + data->dilation;
                if(input_i >= data->input_col)
                    input_i = input_i % data->input_col;

            }

            data->output[output_row_skip + output_j] = (partial_sum);
            if(output_j + 1 < data->output_col)
            data->output[output_row_skip + output_j + 1] = partial_sum_1;
            if(output_j + 2 < data->output_col)
            data->output[output_row_skip + output_j + 2] = partial_sum_2;
            if(output_j + 3 < data->output_col)
            data->output[output_row_skip + output_j + 3] = partial_sum_3;
            if(output_j + 4 < data->output_col)
            data->output[output_row_skip + output_j + 4] = partial_sum_4;
            if(output_j + 5 < data->output_col)
            data->output[output_row_skip + output_j + 5] = partial_sum_5;
            if(output_j + 6 < data->output_col)
            data->output[output_row_skip + output_j + 6] = partial_sum_6;
            if(output_j + 7 < data->output_col)
            data->output[output_row_skip + output_j + 7] = partial_sum_7;
            
        }
    }

    pthread_exit(nullptr);
}

// Fill in this function
void multiThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output )  {

    if(cores !=0)
        num_threads = cores;
    pthread_t threads[num_threads];
    struct ThreadData threadData[num_threads];

    // Calculate the number of rows per thread
    int rows_per_thread = output_row / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threadData[i].input = input;
        threadData[i].kernel = kernel;
        threadData[i].output = output;
        threadData[i].input_row = input_row;
        threadData[i].input_col = input_col;
        threadData[i].output_row = output_row;
        threadData[i].output_col = output_col;
        threadData[i].kernel_row = kernel_row;
        threadData[i].kernel_col = kernel_col;
        threadData[i].start_i = i * rows_per_thread;
        if ( i == num_threads - 1 )
            threadData[i].end_i = output_row;
        else
            threadData[i].end_i = ( i + 1 ) * rows_per_thread;
        threadData[i].dilation = 2;
        int partial_sum = pthread_create(&threads[i], nullptr, threadDilatedConvolution, &threadData[i]);
        if (partial_sum != 0) {
            std::cerr << "Error creating thread " << i << ": " << partial_sum << std::endl;
            exit(EXIT_FAILURE);
        }
        
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}