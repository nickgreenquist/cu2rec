#include <assert.h>

#include "../config.h"

string file_path = "../../data/test/test_config.cfg";
string gen_path = "../../data/test/gen/test_config.cfg";

void test_config_load(string file_path) {
    config::Config *cfg = new config::Config();
    cfg->read_config(file_path);
    assert(cfg->total_iterations == 100);
    assert(fabs(cfg->P_reg - 2e-1) < 1e-4);
    delete cfg;
}

void test_config_save() {
    config::Config *cfg = new config::Config();
    cfg->total_iterations = 100;
    cfg->P_reg = 2e-1;
    cfg->write_config(gen_path);
    test_config_load(gen_path);
    delete cfg;
}

void test_config_set_cuda_variables() {
    config::Config *cfg = new config::Config();
    cfg->total_iterations = 100;
    cfg->P_reg = 2e-1;
    assert(cfg->set_cuda_variables());
    delete cfg;
}

void test_config_get_cuda_variables() {
    config::Config *cfg = new config::Config();
    cfg->total_iterations = 100;
    cfg->P_reg = 2e-1;
    assert(cfg->get_cuda_variables());
    delete cfg;
}

int main() {
    cout << "Testing config load...\n";
    test_config_load(file_path);
    cout << "PASSED\n";
    cout << "Testing config save...\n";
    test_config_save();
    cout << "PASSED\n";
    cout << "Testing config cuda set variables...\n";
    test_config_set_cuda_variables();
    cout << "PASSED\n";
    cout << "Testing config cuda get variables...\n";
    test_config_get_cuda_variables();
    cout << "PASSED\n";

    return 0;
}
