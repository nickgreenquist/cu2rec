#include "matrix.h"
#include "util.h"
#include "training.h"
#include "loss.h"

int main(int argc, char **argv){
    if(argc < 2) {
        return -1;
    }

    // Load in data
    string file_path = argv[1];
    int rows, cols;
    float global_bias;
    std::vector<Rating> ratings = readCSV(file_path, &rows, &cols, &global_bias);
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 2;
    int n_iterations = 1000;
    int seed = 42;
    float learning_rate = 1e-3;
    float P_reg = 1e-1;
    float Q_reg = 1e-1;
    float user_bias_reg = 1e-1;
    float item_bias_reg = 1e-1;

    // Create components and train on ratings
    float *P, *Q, *losses, *user_bias, *item_bias;
    train(matrix, n_iterations, n_factors, learning_rate, seed, &P, &Q, &losses, &user_bias, &item_bias, global_bias,
          P_reg, Q_reg, user_bias_reg, item_bias_reg);

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
    writeToFile(parent_dir, basename, "csv", "p", P, rows, n_factors, n_factors);
    writeToFile(parent_dir, basename, "csv", "q", Q, cols, n_factors, n_factors);
    writeToFile(parent_dir, basename, "csv", "user_bias", user_bias, rows, 1, n_factors);
    writeToFile(parent_dir, basename, "csv", "item_bias", item_bias, cols, 1, n_factors);
    writeToFile(parent_dir, basename, "csv", "global_bias", global_bias_array, 1, 1, n_factors);

    // Free memory
    delete matrix;
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_array;
}