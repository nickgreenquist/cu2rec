#include "matrix.h"
#include "util.h"
#include "training.h"

int main(int argc, char **argv){
    if(argc < 2) {
        return -1;
    }

    // Load in data
    string filename = argv[1];
    int rows, cols;
    float global_bias;
    std::vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 2;
    int n_iterations = 200;
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

    // Output loss every 10th iteration
    cout << "Loss:\n";
    for(int i = 0; i < n_iterations; ++i) {
        if((i + 1) % 10 == 0) {
            cout << "Iteration " << i + 1 << ": " << losses[i] << "\n";
        }
    }

    // Write output to files
    size_t lastindex = filename.find_last_of("."); 
    string filepath = filename.substr(0, lastindex);

    // Put global_bias into array in order to use generalized writeToFile
    float *global_bias_array = new float[1];
    global_bias_array[0] = global_bias;

    // Write components to file
    writeToFile(filepath, "csv", "p", P, rows, n_factors, n_factors);
    writeToFile(filepath, "csv", "q", Q, cols, n_factors, n_factors);
    writeToFile(filepath, "csv", "user_bias", user_bias, rows, 1, n_factors);
    writeToFile(filepath, "csv", "item_bias", item_bias, cols, 1, n_factors);
    writeToFile(filepath, "csv", "global_bias", global_bias_array, 1, 1, n_factors);

    // Free memory
    delete matrix;
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_array;
}