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
};

void* threadDilatedConvolution(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

        for (int output_i = data->start_i; output_i < data->end_i; ++output_i) {
        int output_offset = output_i * data->output_col;

        for (int output_j = 0; output_j < data->output_col; ++output_j) {
            long long unsigned int res = 0;

            // Adjust input_i to handle boundary conditions
            int input_i = (output_i - 2 + data->input_row) % data->input_row;

            for (int kernel_i = 0; kernel_i < data->kernel_row; ++kernel_i) {
                int kernel_offset = kernel_i * data->kernel_col;

                // Adjust input_i to handle boundary conditions
                if (input_i + 2 >= data->input_row) {
                    input_i = (input_i + 2) % data->input_row;
                } else {
                    input_i = (input_i + 2);
                }

                int input_offset = input_i * data->input_col;
                int input_j = (output_j - 2 + data->input_col) % data->input_col;

                for (int kernel_j = 0; kernel_j < data->kernel_col; ++kernel_j) {
                    // Adjust input_j to handle boundary conditions
                    if (input_j + 2 >= data->input_col) {
                        input_j = (input_j + 2) % data->input_col;
                    } else {
                        input_j = (input_j + 2);
                    }

                    res += data->input[input_offset + input_j] * data->kernel[kernel_offset + kernel_j];
                }
            }

            data->output[output_offset + output_j] = static_cast<int>(res);
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
        int result = pthread_create(&threads[i], nullptr, threadDilatedConvolution, &threadData[i]);
        if (result != 0) {
            std::cerr << "Error creating thread " << i << ": " << result << std::endl;
            exit(EXIT_FAILURE);
        }
        


        
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}