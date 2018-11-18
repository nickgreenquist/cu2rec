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

    // make copy of error array
    float* error = new float[ratings.size()];
    float* error_d;
    cudaMalloc((void **) &error_d, ratings.size() * sizeof(float));

    // create user and item bias arrays
    float *user_bias = new float[user_count];
    for(int u = 0; u < user_count; u++) {
        user_bias[u] = 1.0;
    }
    float *user_bias_device;
    cudaMalloc((void **) &user_bias_device, user_count * sizeof(float));
    cudaMemcpy(user_bias_device, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);

    float *item_bias = new float[item_count];
    for(int i = 0; i < item_count; i++) {
        item_bias[i] = 1.0;
    }
    float *item_bias_device;
    cudaMalloc((void **) &item_bias_device, item_count * sizeof(float));
    cudaMemcpy(item_bias_device, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Turn P and Q into CudaDenseMatrices on GPU and calculate the loss using GPU
    CudaDenseMatrix* P_d = new CudaDenseMatrix(user_count, factors, P);
    CudaDenseMatrix* Q_d = new CudaDenseMatrix(item_count, factors, Q);
    calculate_loss_gpu(P_d, Q_d, factors, user_count, item_count, ratings.size(), matrix, error_d, user_bias_device, item_bias_device);

    // move array of errors back to host
    cudaMemcpy(error, error_d, ratings.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float loss = 0.0;
    for(int i = 0; i < ratings.size(); i++) {
        loss += pow(error[i], 2);
    }

    cout << "\nLoss: " << loss << "\n";
    assert(loss == 40.0);

    //free memory
    delete matrix;
    delete [] P;
    delete [] Q;
    delete [] error;
    delete [] user_bias;
    cudaFree(error_d);
    cudaFree(user_bias_device);
    cudaFree(item_bias_device);
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
