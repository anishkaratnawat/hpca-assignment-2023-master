#include <pthread.h>
#define num_threads 8
// Create other necessary functions here


// Fill in this function

struct ThreadData {
    int *input;
    int *kernel;
    long long unsigned int *output;
    int input_row, input_col, output_row, output_col, kernel_row, kernel_col;
    int start_i, end_i; // Range of rows for this thread
    int dilation;
};

void* threadDilatedConvolution(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    long long unsigned int partial_sum;
    int input_i, input_j;

        for (int output_i = data->start_i; output_i < data->end_i; ++output_i) {
        int output_row_skip = output_i * data->output_col;

        for (int output_j = 0; output_j < data->output_col; ++output_j) {
            partial_sum = 0;

            input_i = output_i;
            for (int kernel_i = 0; kernel_i < data->kernel_row; ++kernel_i) {
                int kernel_row_skip = kernel_i * data->kernel_col;

                int input_row_skip = input_i * data->input_col;
                input_j = (output_j) % data->input_col;

                for (int kernel_j = 0; kernel_j < data->kernel_col; ++kernel_j) {
                    partial_sum += data->input[input_row_skip + input_j] * data->kernel[kernel_row_skip + kernel_j];
                    input_j = input_j + data->dilation;
                    if(input_j >= data->input_row)
                        input_j = input_j % data->input_row;

                }
                 input_i = input_i + data->dilation;
                if(input_i >= data->input_col)
                    input_i = input_i % data->input_col;

            }

            data->output[output_row_skip + output_j] = static_cast<int>(partial_sum);
        }
    }

    pthread_exit(nullptr);
}

void multiThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output )  {

    pthread_t threads[num_threads];
    struct ThreadData threadData[num_threads];

    // Calculate the number of rows each thread will handle
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
        threadData[i].end_i = ((i == num_threads - 1) ? output_row : (i + 1) * rows_per_thread);
        threadData[i].dilation = 2;
        int partial_sumult = pthread_create(&threads[i], nullptr, threadDilatedConvolution, &threadData[i]);
        if (partial_sumult != 0) {
            std::cerr << "Error creating thread " << i << ": " << partial_sumult << std::endl;
            exit(EXIT_FAILURE);
        }
        


        
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}