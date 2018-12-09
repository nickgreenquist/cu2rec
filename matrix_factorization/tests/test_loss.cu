/** Tests calculating the errors and the total loss
 */

#include <assert.h>
#include <math.h>       /* pow */
#include <time.h>
#include <tuple>
#include <vector>

#include "../config.h"
#include "../util.h"
#include "../matrix.h"
#include "../loss.h"

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace std;

string filename = "../../data/test/test_ratings.csv";

/** Tests the calculation of errors
 */
void test_loss() {
    int rows, cols;
    float global_bias;
    vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);

    // set global_bias to 1.0 for easier testing
    global_bias = 1.0;

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Hyperparams
    config::Config *cfg = new config::Config();
    cfg->n_factors = 2;

    // create temp P and Q
    int user_count = rows;
    int item_count = cols;
    float *P = new float[user_count * cfg->n_factors];
    float *Q = new float[item_count * cfg->n_factors];
    for(int u = 0; u < user_count; u++) {
        for(int f = 0; f < cfg->n_factors; f++) {
            P[index(u, f, cfg->n_factors)] = 1.0;
        }
    }
    for(int i = 0; i < item_count; i++) {
        for(int f = 0; f < cfg->n_factors; f++) {
            Q[index(i, f, cfg->n_factors)] = 1.0;
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
    CudaDenseMatrix* P_d = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_d = new CudaDenseMatrix(item_count, cfg->n_factors, Q);
    calculate_loss_gpu(P_d, Q_d, cfg, user_count, item_count, ratings.size(), matrix, error_d, user_bias_device, item_bias_device, global_bias);

    // move array of errors back to host
    cudaMemcpy(error, error_d, ratings.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float loss = 0.0;
    for(int i = 0; i < ratings.size(); i++) {
        loss += pow(error[i], 2);
    }

    cout << "\nLoss: " << loss << "\n";
    assert(loss == 74.0);

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

/** Tests the calculation of total loss with
 * multiple sizes.
 */
void test_total_loss() {
    vector<int> problem_sizes = { 1, 33, 1<<10, 1<<16 };
    vector<int> grid_sizes = { 1, 20, 1000 };
    vector<int> block_sizes = { 1, 16, 64 };
    for (std::vector<int>::iterator i = problem_sizes.begin(); i != problem_sizes.end(); ++i) {
        for (std::vector<int>::iterator j = grid_sizes.begin(); j != grid_sizes.end(); ++j) {
            for (std::vector<int>::iterator k = block_sizes.begin(); k != block_sizes.end(); ++k) {
                // Set up experiment sizes
                int problem_size = (*i);
                int grid_size = (*j);
                int block_size = (*k);

                // Input and output arrays
                float *in_errors = new float[problem_size];
                for (int i = 0; i < problem_size; ++i) {
                    in_errors[i] = 1.0;
                }
                float *in_errors_device;
                cudaMalloc(&in_errors_device, problem_size * sizeof(float));
                cudaMemcpy(in_errors_device, in_errors, problem_size * sizeof(float), cudaMemcpyHostToDevice);

                double *out_errors = new double[grid_size];
                double *out_errors_device;
                cudaMalloc(&out_errors_device, grid_size * sizeof(double));

                // Call the kernel
                float mae, rmse;
                std::tie(mae, rmse) = get_error_metrics_gpu(in_errors_device, out_errors_device, out_errors, problem_size, grid_size, block_size);
                
                // Since all errors are 1.0, we expect the RMSE and MAE to be 1
                // This makes sure the kernel covers all problem_size elements
                assert(mae == 1);
                assert(rmse == 1);

                cudaFree(in_errors_device);
                cudaFree(out_errors_device);
                delete [] in_errors;
                delete [] out_errors;
            }
        }
    }
}

int main() {
    cout << "Testing Parallel Loss Function on test ratings...";
    test_loss();
    cout << "PASSED\n";

    cout << "Testing calculation of total loss...";
    test_total_loss();
    cout << "PASSED\n";

    return 0;
}
