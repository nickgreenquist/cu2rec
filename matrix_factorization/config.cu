#include <fstream>
#include "cuda.h"
#include "config.h"
#include "util.h"

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
        CHECK_CUDA(cudaMemcpyToSymbol(config::cur_iterations, &cur_iterations, sizeof(int)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::total_iterations, &total_iterations, sizeof(int)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::n_factors, &n_factors, sizeof(int)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::learning_rate, &learning_rate, sizeof(float)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::seed, &seed, sizeof(int)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::P_reg, &P_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::Q_reg, &Q_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::user_bias_reg, &user_bias_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyToSymbol(config::item_bias_reg, &item_bias_reg, sizeof(float)));
        return true;
    }

    bool Config::get_cuda_variables() {
        CHECK_CUDA(cudaMemcpyFromSymbol(&cur_iterations, config::cur_iterations, sizeof(int)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&total_iterations, config::total_iterations, sizeof(int)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&n_factors, config::n_factors, sizeof(int)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&learning_rate, config::learning_rate, sizeof(float)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&seed, config::seed, sizeof(int)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&P_reg, config::P_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&Q_reg, config::Q_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&user_bias_reg, config::user_bias_reg, sizeof(float)));
        CHECK_CUDA(cudaMemcpyFromSymbol(&item_bias_reg, config::item_bias_reg, sizeof(float)));
        return true;
    }

    void Config::print_config() {
        printf("Hyperparameters:\n");
        printf("total_iterations: %d\n", total_iterations);
        printf("n_factors: %d\n", n_factors);
        printf("learning_rate: %f\n", learning_rate);
        printf("P_reg: %f\n", P_reg);
        printf("Q_reg: %f\n", Q_reg);
        printf("user_bias_reg: %f\n", user_bias_reg);
        printf("item_bias_reg: %f\n", item_bias_reg);
        printf("is_train: %s\n", is_train?"true":"false");
    }
}
