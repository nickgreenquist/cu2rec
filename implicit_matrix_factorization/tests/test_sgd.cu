#include <assert.h>

#include "../matrix.h"
#include "../read_csv.h"
#include "../sgd.cu"

void test_sgd() {
    // Initalize the input matrix
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 1;
    float learning_rate = 1e-3;

    // Initialize P and Q
    float *P = new float[rows * n_factors];
    for(int i = 0; i < rows; ++i) {
        P[i] = 1;
    }
    float *Q = new float[cols * n_factors];
    for(int i = 0; i < cols; ++i) {
        Q[i] = 1;
    }

    // Copy P and Q to device memory
    float *P_device, *P_device_target;
    int P_size = rows * n_factors * sizeof(float);
    cudaMalloc(&P_device, P_size);
    cudaMemcpy(P_device, P, P_size, cudaMemcpyHostToDevice);
    cudaMalloc(&P_device_target, P_size);
    cudaMemcpy(P_device_target, P, P_size, cudaMemcpyHostToDevice);
    float *Q_device, *Q_device_target;
    int Q_size = cols * n_factors * sizeof(float);
    cudaMalloc(&Q_device, Q_size);
    cudaMemcpy(Q_device, Q, Q_size, cudaMemcpyHostToDevice);
    cudaMalloc(&Q_device_target, Q_size);
    cudaMemcpy(Q_device_target, Q, Q_size, cudaMemcpyHostToDevice);

    // Create the errors - we would get this through the loss function
    float *errors = new float[matrix->nonzero];
    for(int i = 0; i < matrix->nonzero; ++i) {
        errors[i] = 1;
    }
    float *errors_device;
    cudaMalloc(&errors_device, matrix->nonzero * sizeof(float));
    cudaMemcpy(errors_device, errors, matrix->nonzero * sizeof(float), cudaMemcpyHostToDevice);

    // Dimensions
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(rows / n_threads + 1);
    sgd_update<<<dimGrid, dimBlock>>>(matrix, P_device, Q_device, P_device_target, Q_device_target, n_factors, errors_device, rows, cols, learning_rate);
    std::swap(P_device, P_device_target);
    std::swap(Q_device, Q_device_target);

    // Copy updated P and Q back
    float *P_updated = new float[rows * n_factors];
    float *Q_updated = new float[cols * n_factors];
    cudaMemcpy(P_updated, P_device, P_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q_updated, Q_device, Q_size, cudaMemcpyDeviceToHost);

    // For now, print the matrices
    cout << "Updated P: ";
    for(int i = 0; i < rows; ++i) {
        cout << P[i] << " ";
    }
    cout << endl << "Updated Q: ";
    for(int i = 0; i < cols; ++i) {
        cout << Q[i] << " ";
    }
    cout << endl;

    // Clean up
    cudaFree(P_device);
    cudaFree(P_device_target);
    cudaFree(Q_device);
    cudaFree(Q_device_target);
    cudaFree(errors_device);
    delete matrix;
    delete P;
    delete P_updated;
    delete Q;
    delete Q_updated;
    delete errors;
}

int main() {
    test_sgd();
    return 0;
}