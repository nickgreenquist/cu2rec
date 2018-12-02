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

    // Check free memory
    size_t free_bytes, total_bytes;
    free_bytes = getFreeBytes(0, &total_bytes);
    long free_bytes_before = (long)free_bytes;
    printf("Free memory: %ld\n\n", free_bytes_before);

    // Load in train data
    string file_path_train = argv[optind++];
    int rows, cols;
    float global_bias;
    std::vector<Rating> train_ratings = readCSV(file_path_train, &rows, &cols, &global_bias);
    cu2rec::CudaCSRMatrix* train_matrix = createSparseMatrix(&train_ratings, rows, cols);

    // Load in test data
    string file_path_test = argv[optind++];
    int r, c;
    float gb;
    std::vector<Rating> test_ratings = readCSV(file_path_test, &r, &c, &gb);
    cu2rec::CudaCSRMatrix* test_matrix = createSparseMatrix(&test_ratings, r, c);

    // Hyperparams
    config::Config *cfg = new config::Config();
    if(!filename_config.empty())
        cfg->read_config(filename_config);
    cfg->print_config();

    // Create components and train on ratings
    float *P, *Q, *losses, *user_bias, *item_bias;
    train(train_matrix, test_matrix, cfg, &P, &Q, &losses, &user_bias, &item_bias, global_bias);

    // Write output to files
    // TODO: make this work on Windows, because it will probably fail
    size_t dir_index = file_path_train.find_last_of("/"); 
    string parent_dir, filename;
    if(dir_index != string::npos) {
        parent_dir = file_path_train.substr(0, dir_index);
        filename = file_path_train.substr(dir_index + 1);
    } else {
        // doesn't have directory, therefore working on current directory
        parent_dir = ".";
        filename = file_path_train;
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
    delete train_matrix;
    delete test_matrix;
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_array;
}