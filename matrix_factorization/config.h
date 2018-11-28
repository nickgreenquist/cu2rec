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
            int cur_iterations = 0;
            int total_iterations = 1000;
            int n_factors = 10;
            float learning_rate = 1e-5;
            int seed = 42;
            float P_reg = 1e-2;
            float Q_reg = 1e-1;
            float user_bias_reg = 1e-2;
            float item_bias_reg = 1e-1;
            bool is_train = true;

            bool read_config(std::string file_path);
            bool write_config(std::string file_path);
            bool set_cuda_variables();
            bool get_cuda_variables();
            void print_config();
    };    
}

#endif