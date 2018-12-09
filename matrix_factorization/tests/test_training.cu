/** Tests the whole training loop
 */

#include <assert.h>
#include <vector>

#include "../config.h"
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
    config::Config *cfg = new config::Config();
    cfg->total_iterations = 10;
    cfg->seed = 42;
    cfg->n_factors = n_factors;
    cfg->learning_rate = 1e-3;
    cfg->P_reg = 1e-1;
    cfg->Q_reg = 1e-1;
    cfg->user_bias_reg = 1e-1;
    cfg->item_bias_reg = 1e-1;

    float *P, *Q, *losses, *user_bias, *item_bias;

    // user matrix as both train and test for sake of testing
    train(matrix, matrix, cfg, &P, &Q, &losses, &user_bias, &item_bias, global_bias);

    // we only calculate loss every 10th iteration, so compare first and last loss
    assert(losses[0] >= losses[9]);

    // Free memory
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete matrix;
    delete cfg;
}

int main() {
    cout << "Testing training loop...\n";
    test_training_loop();
    cout << "PASSED\n";
    return 0;
}
