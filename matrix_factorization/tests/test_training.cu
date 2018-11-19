#include <assert.h>
#include <vector>

#include "../util.h"
#include "../matrix.h"
#include "../loss.h"
#include "../training.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

using namespace std;

string filename = "../../data/test_ratings.csv";

void test_training_loop() {
    // Initalize the input matrix
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);
    CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 2;
    int n_iterations = 10;
    int seed = 42;
    float learning_rate = 1e-3;

    float *P, *Q, *losses, *user_bias, *item_bias;

    train(matrix, n_iterations, n_factors, learning_rate, seed, &P, &Q, &losses, &user_bias, &item_bias);

    cout << "Losses: ";
    for(int i = 0; i < n_iterations; ++i) {
        cout << losses[i] << " ";
    }
    cout << endl;

    assert(losses[0] >= losses[n_iterations - 1]);

    // Write updated P and Q to file
    char filename_char[filename.length()+1];  
    strcpy(filename_char, filename.c_str());
    char p_filename [255];
    string test_filename = "test_output.csv";
    sprintf(p_filename, "%s_%d_p.txt", test_filename.c_str(), n_factors);

    FILE *fp;
    fp = fopen(p_filename, "w");
    for(int u = 0; u < rows; u++) {
        for(int f = 0; f < n_factors - 1; f++) {
            fprintf(fp, "%f,", P[index(u, f, n_factors)]);
        }
        fprintf(fp, "%f", P[index(u, n_factors - 1, n_factors)]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Free memory
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete matrix;
}

int main() {
    cout << "Testing training loop...\n";
    test_training_loop();
    cout << "PASSED\n";
    return 0;
}
