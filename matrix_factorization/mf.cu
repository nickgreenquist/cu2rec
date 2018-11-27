#include <getopt.h>

#include "config.h"
#include "matrix.h"
#include "util.h"
#include "training.h"
#include "loss.h"

int main(int argc, char **argv){
    if(argc < 2) {
        return -1;
    }
    std::string filename_config;
    int o;
    while((o = getopt(argc, argv, "c:")) != -1) {
        switch(o) {
            case 'c':
                filename_config = optarg;
                break;
            default:
                cout << "Unknown option.\n";
                return 1;
        }
    }

    // Load in data
    string file_path = argv[optind++];
    int rows, cols;
    float global_bias;
    std::vector<Rating> ratings = readCSV(file_path, &rows, &cols, &global_bias);
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    config::Config *cfg = new config::Config();
    if(!filename_config.empty())
        cfg->read_config(filename_config);
    cfg->print_config();

    // Create components and train on ratings
    float *P, *Q, *losses, *user_bias, *item_bias;
    train(matrix, cfg, &P, &Q, &losses, &user_bias, &item_bias, global_bias);

    // Write output to files
    // TODO: make this work on Windows, because it will probably fail
    size_t dir_index = file_path.find_last_of("/"); 
    string parent_dir, filename;
    if(dir_index != string::npos) {
        parent_dir = file_path.substr(0, dir_index);
        filename = file_path.substr(dir_index + 1);
    } else {
        // doesn't have directory, therefore working on current directory
        parent_dir = ".";
        filename = file_path;
    }
    size_t dot_index = filename.find_last_of("."); 
    string basename = filename.substr(0, dot_index);

    // Put global_bias into array in order to use generalized writeToFile
    float *global_bias_array = new float[1];
    global_bias_array[0] = global_bias;

    // Write components to file
    writeToFile(parent_dir, basename, "csv", "p", P, rows, cfg->n_factors, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "q", Q, cols, cfg->n_factors, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "user_bias", user_bias, rows, 1, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "item_bias", item_bias, cols, 1, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "global_bias", global_bias_array, 1, 1, cfg->n_factors);

    // Free memory
    delete cfg;
    delete matrix;
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_array;
}