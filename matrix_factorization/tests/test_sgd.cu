#include <assert.h>
#include <vector>
#include <math.h>

#include "../config.h"
#include "../matrix.h"
#include "../util.h"
#include "../sgd.h"

using namespace cu2rec;
using namespace std;

string filename = "../../data/test/test_ratings.csv";

void test_sgd() {
    // Initalize the input matrix
    int rows, cols;
    float global_bias;
    vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);
    CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    int n_factors = 1;
    config::Config *cfg = new config::Config();
    cfg->n_factors = n_factors;
    cfg->learning_rate = 0.07;
    cfg->P_reg = 1e-1;
    cfg->Q_reg = 1e-1;
    cfg->user_bias_reg = 1e-1;
    cfg->item_bias_reg = 1e-1;
    assert(cfg->set_cuda_variables());

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
    float *errors = new float[matrix->nonzeros];
    for(int i = 0; i < matrix->nonzeros; ++i) {
        errors[i] = 1;
    }
    float *errors_device;
    cudaMalloc(&errors_device, matrix->nonzeros * sizeof(float));
    cudaMemcpy(errors_device, errors, matrix->nonzeros * sizeof(float), cudaMemcpyHostToDevice);

    // Create the bias arrays
    float *user_bias = new float[rows];
    for(int u = 0; u < rows; u++) {
        user_bias[u] = 1.0;
    }
    float *user_bias_device;
    cudaMalloc((void **) &user_bias_device, rows * sizeof(float));
    cudaMemcpy(user_bias_device, user_bias, rows * sizeof(float), cudaMemcpyHostToDevice);

    float *item_bias = new float[cols];
    for(int i = 0; i < cols; i++) {
        item_bias[i] = 1.0;
    }
    float *item_bias_device;
    cudaMalloc((void **) &item_bias_device, cols * sizeof(float));
    cudaMemcpy(item_bias_device, item_bias, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Create bias targets
    float *user_bias_target, *item_bias_target;
    cudaMalloc(&user_bias_target, rows * sizeof(float));
    cudaMemcpy(user_bias_target, user_bias, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&item_bias_target, cols * sizeof(float));
    cudaMemcpy(item_bias_target, item_bias, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Create curand state
    curandState *d_state;
    cudaMalloc(&d_state, rows * sizeof(curandState));

    // Dimensions
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(rows / n_threads + 1);

    // Set up random state using iteration as seed
    initCurand<<<dimGrid, dimBlock>>>(d_state, 1, matrix->rows);

    // Call SGD kernel
    float shared_mem_size = rows * sizeof(float);
    sgd_update<<<dimGrid, dimBlock, shared_mem_size>>>(matrix->indptr, matrix->indices, matrix->data, P_device, Q_device, Q_device_target,
                                      errors_device, rows, cols, user_bias_device, item_bias_device,
                                      item_bias_target, d_state,
                                      global_bias);
    std::swap(Q_device, Q_device_target);
    std::swap(item_bias_device, item_bias_target);

    // Copy updated P and Q back
    float *P_updated = new float[rows * n_factors];
    float *Q_updated = new float[cols * n_factors];
    cudaMemcpy(P_updated, P_device, P_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q_updated, Q_device, Q_size, cudaMemcpyDeviceToHost);

    // Copy updated biases back
    float *user_bias_updated = new float[rows];
    float *item_bias_updated = new float[cols];
    cudaMemcpy(user_bias_updated, user_bias_device, rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(item_bias_updated, item_bias_device, cols * sizeof(float), cudaMemcpyDeviceToHost);

    // check for correct components
    for(int i = 0; i < rows; ++i) {
        assert(!std::isnan(P_updated[i]));
    }
    for(int i = 0; i < cols; ++i) {
        assert(!std::isnan(Q_updated[i]));
    }
    for(int i = 0; i < rows; i++) {
        assert(!std::isnan(user_bias_updated[i]));
    }
    for(int i = 0; i < cols; i++) {
        assert(!std::isnan(item_bias_updated[i]));
    }

    // Clean up
    cudaFree(P_device);
    cudaFree(P_device_target);
    cudaFree(Q_device);
    cudaFree(Q_device_target);
    cudaFree(errors_device);
    cudaFree(user_bias_device);
    cudaFree(user_bias_target);
    cudaFree(item_bias_device);
    cudaFree(item_bias_target);
    delete cfg;
    delete matrix;
    delete P;
    delete P_updated;
    delete Q;
    delete Q_updated;
    delete errors;
    delete [] user_bias;
    delete [] item_bias;
    delete [] user_bias_updated;
    delete [] item_bias_updated;
}

int main() {
    cout << "Testing a single gradient update...\n";
    test_sgd();
    cout << "PASSED\n";
    return 0;
}