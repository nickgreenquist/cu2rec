#include <assert.h>
#include <vector>

#include "../read_csv.h"
#include "../matrix.h"
#include "../loss.h"
#include "../training.h"

using namespace std;

string filename = "../../data/test_ratings.csv";

void test_training_loop() {
    // Initalize the input matrix
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);
    CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 1;
    int n_iterations = 10;
    int seed = 42;
    float learning_rate = 1e-3;

    float *P, *Q, *losses;

    train(matrix, n_iterations, n_factors, learning_rate, seed, &P, &Q, &losses);

    cout << "Losses: ";
    for(int i = 0; i < n_iterations; ++i) {
        cout << losses[i] << " ";
    }
    cout << endl;

    delete [] P;
    delete [] Q;
    delete [] losses;
    delete matrix;
}

int main() {
    cout << "Testing training loop...\n";
    test_training_loop();
    cout << "PASSED\n";
    return 0;
}
