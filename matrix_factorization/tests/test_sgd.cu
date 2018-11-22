#include <assert.h>
#include <vector>

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
    float learning_rate = 1e-3;
    float P_reg = 1e-1;
    float Q_reg = 1e-1;
    float user_bias_reg = 1e-1;
    float item_bias_reg = 1e-1;

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


    // Dimensions
    int n_threads = 32;
    dim3 dimBlock(n_threads);
    dim3 dimGrid(rows / n_threads + 1);
    sgd_update<<<dimGrid, dimBlock>>>(matrix->indptr, matrix->indices, P_device, Q_device, P_device_target, Q_device_target, n_factors,
                                      errors_device, rows, cols, learning_rate, user_bias_device, item_bias_device, user_bias_target,
                                      item_bias_target, P_reg, Q_reg, user_bias_reg, item_bias_reg);
    std::swap(P_device, P_device_target);
    std::swap(Q_device, Q_device_target);
    std::swap(user_bias_device, user_bias_target);
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

    // Assert all updated components are correct
    vector<float> P_expected = {1.0036, 1.0027, 1.0027, 1.0027, 1.0027, 1.0018};
    vector<float> Q_expected = {1.0036, 1.0045, 1.0036, 1.0018, 1.0027};
    for(int i = 0; i < rows; ++i) {
        assert(fabs(P_expected.at(i) - P_updated[i]) < 1e-4);
    }
    for(int i = 0; i < cols; ++i) {
        assert(fabs(Q_expected.at(i) - Q_updated[i]) < 1e-4);
    }

    // Bias for user will be (0.0009) * num of items they have rated
    vector<float> user_bias_expected = {1.0036, 1.0027, 1.0027, 1.0027, 1.0027, 1.0018};
    // Bias for item will be (0.0009) * num of users have rated it
    vector<float> item_bias_expected = {1.0036, 1.0045, 1.0036, 1.0018, 1.0027};
    for(int i = 0; i < rows; i++) {
        assert(fabs(user_bias_expected.at(i) - user_bias_updated[i]) < 1e-4);
    }
    for(int i = 0; i < cols; i++) {
        assert(fabs(item_bias_expected.at(i) - item_bias_updated[i]) < 1e-4);
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