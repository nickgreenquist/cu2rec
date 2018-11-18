#include <assert.h>
#include <math.h>       /* pow */
#include <time.h>

#include "../read_csv.h"
#include "../matrix.h"
#include "../loss.h"

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace std;

string filename = "../../data/test_ratings.csv";
int factors = 2;

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
    int user_count = rows;
    int item_count = cols;
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
    int user_count = rows;
    int item_count = cols;
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

    float * error = new float[ratings.size()];
    calculate_loss_gpu(factors, user_count, item_count, ratings.size(), P, Q, matrix, error);
    float loss = 0.0;
    for(int i = 0; i < ratings.size(); i++) {
        loss += pow(error[i], 2);
    }

    cout << "\nLoss: " << loss << "\n";
    assert(loss == 80.0);

    //free memory
    delete matrix;
    delete [] P;
    delete [] Q;
    delete [] error;
}

int main() {
    double time_taken;
    clock_t start, end;

    cout << "Testing Sequential Loss Function on test ratings...";
    start = clock();
    test_loss_sequential();
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    cout << "FACTORS: " << factors << "\n";
    cout << "CPU TIME TAKEN: " << time_taken << "\n";
    cout << "PASSED\n";

    cout << "Testing Parallel Loss Function on test ratings...";
    start = clock();
    test_loss();
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    cout << "FACTORS: " << factors << "\n";
    cout << "GPU TIME TAKEN: " << time_taken << "\n";
    cout << "PASSED\n";

    return 0;
}