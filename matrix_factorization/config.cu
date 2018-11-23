#include <fstream>

#include "cuda.h"

#include "config.h"

namespace config {
    bool Config::read_config(std::string file_path) {
        std::ifstream config_file(file_path);
        config_file >> cur_iterations >> total_iterations >> n_factors >> learning_rate >>
            seed >> P_reg >> Q_reg >> user_bias_reg >> item_bias_reg;
        config_file.close();
        return true;
    }

    bool Config::write_config(std::string file_path) {
        std::ofstream config_file(file_path);
        config_file << cur_iterations << " " << total_iterations << " " << n_factors << " " <<
        learning_rate << " " << seed << " " << P_reg << " " << Q_reg << " " << user_bias_reg <<
        " " << item_bias_reg << "\n";
        config_file.close();
        return true;
    }

    bool Config::set_cuda_variables() {
        cudaMemcpyToSymbol(config::cur_iterations, &cur_iterations, sizeof(int));
        cudaMemcpyToSymbol(config::total_iterations, &total_iterations, sizeof(int));
        cudaMemcpyToSymbol(config::n_factors, &n_factors, sizeof(int));
        cudaMemcpyToSymbol(config::learning_rate, &learning_rate, sizeof(float));
        cudaMemcpyToSymbol(config::seed, &seed, sizeof(int));
        cudaMemcpyToSymbol(config::P_reg, &P_reg, sizeof(float));
        cudaMemcpyToSymbol(config::Q_reg, &Q_reg, sizeof(float));
        cudaMemcpyToSymbol(config::user_bias_reg, &user_bias_reg, sizeof(float));
        cudaMemcpyToSymbol(config::item_bias_reg, &item_bias_reg, sizeof(float));

        cudaError_t lastError;
        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
            return false;
        }
        return true;
    }

    bool Config::get_cuda_variables() {
        cudaMemcpyFromSymbol(&cur_iterations, config::cur_iterations, sizeof(int));
        cudaMemcpyFromSymbol(&total_iterations, config::total_iterations, sizeof(int));
        cudaMemcpyFromSymbol(&n_factors, config::n_factors, sizeof(int));
        cudaMemcpyFromSymbol(&learning_rate, config::learning_rate, sizeof(float));
        cudaMemcpyFromSymbol(&seed, config::seed, sizeof(int));
        cudaMemcpyFromSymbol(&P_reg, config::P_reg, sizeof(float));
        cudaMemcpyFromSymbol(&Q_reg, config::Q_reg, sizeof(float));
        cudaMemcpyFromSymbol(&user_bias_reg, config::user_bias_reg, sizeof(float));
        cudaMemcpyFromSymbol(&item_bias_reg, config::item_bias_reg, sizeof(float));

        cudaError_t lastError;
        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
            return false;
        }
        return true;
    }
}
