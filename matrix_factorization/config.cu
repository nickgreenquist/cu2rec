#include <fstream>

#include "config.h"

bool Config::read_config(string file_path) {
    std::ifstream config_file(file_path);
    config_file >> cur_iterations >> total_iterations >> n_factors >> learning_rate >>
        seed >> P_reg >> Q_reg >> user_bias_reg >> item_bias_reg;
    config_file.close();
    return true;
}

bool Config::write_config(string file_path) {
    std::ofstream config_file(file_path);
    config_file << cur_iterations << " " << total_iterations << " " << n_factors << " " <<
    learning_rate << " " << seed << " " << P_reg << " " << Q_reg << " " << user_bias_reg <<
    " " << item_bias_reg;
    config_file.close();
    return true;
}

bool Config::set_cuda_variables() {
    return false;
}

bool Config::get_cuda_variables() {
    return false;
}
