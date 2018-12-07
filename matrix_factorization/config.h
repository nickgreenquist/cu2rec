#ifndef CU2REC_CONFIG
#define CU2REC_CONFIG

#include <iostream>

namespace config {
    // CUDA variables
    __constant__ int cur_iterations = 0;
    __constant__ int total_iterations = 1000;
    __constant__ int n_factors = 10;
    __constant__ float learning_rate = 1e-4;
    __constant__ int seed = 42;
    __constant__ float P_reg = 1e-1;
    __constant__ float Q_reg = 1e-1;
    __constant__ float user_bias_reg = 1e-1;
    __constant__ float item_bias_reg = 1e-1;
    __constant__ bool is_train = true;

    class Config {
        public:
            // CUDA variables
            int cur_iterations = 0;
            int total_iterations = 1000;
            int n_factors = 50;
            float learning_rate = 0.01;
            int seed = 42;
            float P_reg = 0.08;
            float Q_reg = 0.1;
            float user_bias_reg = 0.003;
            float item_bias_reg = 0.003;
            bool is_train = true;

            // Host variables
            int n_threads = 32; // Must be 2^0 to 2^9
            int check_error = 100;
            float patience = 3;
            float learning_rate_decay = 0.2;

            bool read_config(std::string file_path);
            bool write_config(std::string file_path);
            bool set_cuda_variables();
            bool get_cuda_variables();
            void print_config();
    };    
}

#endif