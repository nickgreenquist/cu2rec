#ifndef CU2REC_CONFIG
#define CU2REC_CONFIG

#include <iostream>

namespace config {
    // CUDA constant variables
    // These get cached heavily, and don't eat up registers
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
            // Current iteration count
            int cur_iterations = 0;
            // Total iteration count
            int total_iterations = 5000;
            // Number of latent factors to use
            int n_factors = 50;
            // The learning rate for SGD
            float learning_rate = 0.01;
            // The seed for the random number generator
            int seed = 42;
            // The regularization parameter for the user matrix
            float P_reg = 0.02;
            // The regularization parameter for the item matrix
            float Q_reg = 0.02;
            // The regularization parameter for user biases
            float user_bias_reg = 0.02;
            // The regularization parameter for item biases
            float item_bias_reg = 0.02;
            // Whether we're doing full training or partial fit
            bool is_train = true;
            // The number of threads in a block
            int n_threads = 32; // Must be 2^0 to 2^9
            // The number of iterations before calculating loss 
            int check_error = 500;
            // The number of times loss can stay constant or increase
            // before triggering a learning rate decay
            float patience = 2;
            // The amount of decay for learning rate. When patience
            // reaches zero, learning rate gets multiplied by this amount.
            float learning_rate_decay = 0.2;

            bool read_config(std::string file_path);
            bool write_config(std::string file_path);
            bool set_cuda_variables();
            bool get_cuda_variables();
            void print_config();
    };    
}

#endif