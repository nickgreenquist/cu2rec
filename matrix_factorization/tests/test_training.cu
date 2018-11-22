#include <assert.h>
#include <vector>

#include "../util.h"
#include "../matrix.h"
#include "../loss.h"
#include "../training.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

using namespace std;

string filename = "../../data/test/test_ratings.csv";

void test_training_loop() {
    // Initalize the input matrix
    int rows, cols;
    float global_bias;
    vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);
    CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 2;
    int n_iterations = 10;
    int seed = 42;
    float learning_rate = 1e-3;
    float P_reg = 1e-1;
    float Q_reg = 1e-1;
    float user_bias_reg = 1e-1;
    float item_bias_reg = 1e-1;

    float *P, *Q, *losses, *user_bias, *item_bias;

    train(matrix, n_iterations, n_factors, learning_rate, seed, &P, &Q, &losses, &user_bias, &item_bias, global_bias,
          P_reg, Q_reg, user_bias_reg, item_bias_reg);

    cout << "Losses: ";
    for(int i = 0; i < n_iterations; ++i) {
        cout << losses[i] << " ";
    }
    cout << endl;

    assert(losses[0] >= losses[n_iterations - 1]);

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
