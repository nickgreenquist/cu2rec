#include "../read_csv.h"
#include "../matrix.h"
#include "../loss.h"

#include <assert.h>

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace std;

string filename = "../../data/test_ratings.csv";

void test_loss_sequential() {
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // copy matrix from Device to Host
    int nonzeros =  (int)(ratings.size());
    int * indptr, * indices;
    float * data;
    indptr = new int[(rows + 1)];
    indices = new int[nonzeros];
    data = new float[nonzeros];

    cudaMemcpy(indptr, matrix->indptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, matrix->indices,  nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data, matrix->data, nonzeros * sizeof(float), cudaMemcpyDeviceToHost);

    // create temp P and Q
    int factors = 2;
    int user_count = 6;
    int item_count = 5;
    float *P = new float[user_count * factors];
    float *Q = new float[item_count * factors];
    for(int u = 0; u < user_count; u++) {
        for(int f = 0; f < factors; f++) {
            P[index(u, f, factors)] = 1.0;
        }
    }
    for(int i = 0; i < item_count; i++) {
        for(int f = 0; f < factors; f++) {
            Q[index(i, f, factors)] = 1.0;
        }
    }

    float loss = calculate_loss_sequential(factors, user_count, item_count, P, Q, indptr, indices, data);

    cout << "\nLoss: " << loss << "\n";

    assert(loss == 80.0);

    //free memory
    delete indptr;
    delete indices;
    delete data;
    delete matrix;
    delete [] P;
    delete [] Q;
}

void test_loss() {
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // create temp P and Q
    int factors = 2;
    int user_count = 6;
    int item_count = 5;
    float *P = new float[user_count * factors];
    float *Q = new float[item_count * factors];
    for(int u = 0; u < user_count; u++) {
        for(int f = 0; f < factors; f++) {
            P[index(u, f, factors)] = 1.0;
        }
    }
    for(int i = 0; i < item_count; i++) {
        for(int f = 0; f < factors; f++) {
            Q[index(i, f, factors)] = 1.0;
        }
    }

    float loss = calculate_loss_gpu(factors, user_count, item_count, P, Q, matrix);
    cout << "\nLoss: " << loss << "\n";

    assert(loss == 80.0);

    //free memory
    delete matrix;
    delete [] P;
    delete [] Q;
}

int main() {
    cout << "Testing Sequential Loss Function on test ratings...";
    test_loss_sequential();
    cout << "PASSED\n";

    cout << "Testing Parallel Loss Function on test ratings...";
    test_loss();
    cout << "PASSED\n";

    return 0;
}