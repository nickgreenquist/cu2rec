#include <random>
#include <cmath>
#include <time.h>
#include <tuple>
#include <limits>

#include "config.h"
#include "loss.h"
#include "matrix.h"
#include "sgd.h"
#include "util.h"

using namespace cu2rec;

/** Main loop for training the Matrix Factorization model. It takes all its configurations from the cfg
 * object, uses the train_matrix as the training data and test_matrix as the test data.
 * When it is done, it puts the trained P, Q, user_bias, and item_bias files into its respective pointers
 * to return them. The training loop also implements learning rate scheduling, and outputs the time
 * training takes (doesn't include the memcpy calls).
 */
void train(CudaCSRMatrix* train_matrix, CudaCSRMatrix* test_matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float *Q, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float *item_bias, float global_bias) {
    int user_count = train_matrix->rows;
    int item_count = train_matrix->cols;
    cfg->set_cuda_variables();

    // Initialize P, Q has already been initialized
    float *P = initialize_normal_array(user_count * cfg->n_factors, cfg->n_factors);
    float *losses = new float[cfg->total_iterations];
    *P_ptr = P;
    *losses_ptr = losses;

    // Copy P and Q to device memory
    CudaDenseMatrix* P_device = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device = new CudaDenseMatrix(item_count, cfg->n_factors, Q);
    CudaDenseMatrix* P_device_target = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device_target = new CudaDenseMatrix(item_count, cfg->n_factors, Q);

    // Create the errors
    float *errors = new float[train_matrix->nonzeros];
    float *errors_device;
    CHECK_CUDA(cudaMalloc(&errors_device, train_matrix->nonzeros * sizeof(float)));

    float *errors_test = new float[test_matrix->nonzeros];
    float *errors_test_device;
    CHECK_CUDA(cudaMalloc(&errors_test_device, test_matrix->nonzeros * sizeof(float)));

    // Create the bias array
    float *user_bias = initialize_normal_array(user_count, cfg->n_factors);
    *user_bias_ptr = user_bias;
    
    float *user_bias_device;
    CHECK_CUDA(cudaMalloc(&user_bias_device, user_count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(user_bias_device, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice));

    float *item_bias_device;
    CHECK_CUDA(cudaMalloc(&item_bias_device, item_count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(item_bias_device, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice));

    // Create bias targets
    float *user_bias_target, *item_bias_target;
    CHECK_CUDA(cudaMalloc(&user_bias_target, user_count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(user_bias_target, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&item_bias_target, item_count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(item_bias_target, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice));

    // Dimensions
    dim3 dim_block(cfg->n_threads);
    dim3 dim_grid_sgd(user_count / cfg->n_threads + 1);
    dim3 dim_grid_loss(256); // TODO: figure out a way to use a config value for this
    dim3 dim_block_loss(2 * cfg->n_threads);
    dim3 dim_grid_P_reg_loss(P_device->rows * P_device->cols / cfg->n_threads + 1);
    dim3 dim_grid_Q_reg_loss(Q_device->rows * Q_device->cols / cfg->n_threads + 1);
    dim3 dim_grid_user_bias_reg_loss(user_count / cfg->n_threads + 1);
    dim3 dim_grid_item_bias_reg_loss(item_count / cfg->n_threads + 1);

    // Create loss per block
    double *block_errors_host = new double[dim_grid_loss.x];
    double *block_errors_device;
    CHECK_CUDA(cudaMalloc(&block_errors_device, dim_grid_loss.x * sizeof(double)));
    CHECK_CUDA(cudaMemset(block_errors_device, 0, dim_grid_loss.x * sizeof(double)));

    // Create curand state
    curandState *d_state;
    CHECK_CUDA(cudaMalloc(&d_state, user_count * sizeof(curandState)));
    // Set up random state using iteration as seed
    initCurand<<<dim_grid_sgd, dim_block>>>(d_state, cfg->seed, user_count);
    CHECK_CUDA(cudaGetLastError());

    // to measure time taken by a specific part of the code 
    double time_taken;
    clock_t start, end;

    // Adaptive learning rate setup
    float train_rmse, train_mae, validation_rmse, validation_mae, last_validation_rmse;
    validation_rmse = validation_mae = std::numeric_limits<float>::max();
    int current_patience = cfg->patience;

    // Training loop
    start = clock();
    for (int i = 0; i < cfg->total_iterations; ++i) {

        // Run single iteration of SGD
        sgd_update<<<dim_grid_sgd, dim_block>>>(train_matrix->indptr, train_matrix->indices, train_matrix->data, P_device->data, Q_device->data, 
                                                Q_device_target->data, user_count, user_bias_device, item_bias_device, item_bias_target, d_state,
                                                global_bias);
        CHECK_CUDA(cudaGetLastError());

        // Calculate total loss first, last, and every check_error iterations
        if((i + 1) % cfg->check_error == 0 || i == 0 || (i + 1) % cfg->total_iterations == 0) {

            // Calculate error on train ratings
            calculate_loss_gpu(P_device, Q_device, cfg, user_count, item_count, train_matrix->nonzeros, train_matrix,
                               errors_device, user_bias_device, item_bias_device, global_bias);

            // Calculate error on test ratings
            calculate_loss_gpu(P_device, Q_device, cfg, test_matrix->rows, test_matrix->cols, test_matrix->nonzeros, test_matrix,
                               errors_test_device, user_bias_device, item_bias_device, global_bias);

            // save previous metrics
            last_validation_rmse = validation_rmse;

            // TODO add this as a param to train function
            bool use_gpu = true;
            if(use_gpu) {
                std::tie(train_mae, train_rmse) = get_error_metrics_gpu(errors_device, block_errors_device, block_errors_host, train_matrix->nonzeros, dim_grid_loss.x, dim_block_loss.x);
                printf("TRAIN: Iteration %d GPU MAE: %f RMSE: %f\n", i + 1, train_mae, train_rmse);
                std::tie(validation_mae, validation_rmse) = get_error_metrics_gpu(errors_test_device, block_errors_device, block_errors_host, test_matrix->nonzeros, dim_grid_loss.x, dim_block_loss.x);
                printf("TEST: Iteration %d GPU MAE: %f RMSE: %f\n", i + 1, validation_mae, validation_rmse);
            } else {
                std::tie(train_mae, train_rmse) = get_error_metrics_cpu(errors, errors_device, train_matrix->nonzeros);
                printf("TRAIN: Iteration %d MAE: %f RMSE: %f\n", i + 1, train_mae, train_rmse);
                std::tie(validation_mae, validation_rmse) = get_error_metrics_cpu(errors_test, errors_test_device, test_matrix->nonzeros);
                printf("TEST: Iteration %d MAE: %f RMSE: %f\n", i + 1, validation_mae, validation_rmse);
            }

            // Update learning rate if needed
            if(last_validation_rmse < validation_rmse) {
                current_patience--;
            }
            if(current_patience <= 0) {
                current_patience = cfg->patience;
                cfg->learning_rate *= cfg->learning_rate_decay;
                cfg->set_cuda_variables();

                printf("New Learning Rate: %f\n: ", cfg->learning_rate);
            }

            // TODO: Do we still need to store this?
            losses[i] = validation_rmse;
        }

        CHECK_CUDA(cudaGetLastError());

        // Swap item related components
        swap(Q_device, Q_device_target);
        swap(item_bias_device, item_bias_target);

        cfg->cur_iterations += 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    end = clock();

    // Output time taken
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;   
    printf("Time taken for %d of iterations is %lf\n", cfg->total_iterations, time_taken);

    // Copy updated P and Q back
    P_device->to_host(P);
    Q_device->to_host(Q);

    // Copy updated bias arrays back
    CHECK_CUDA(cudaMemcpy(user_bias, user_bias_device, user_count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(item_bias, item_bias_device, item_count * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CHECK_CUDA(cudaFree(errors_device));
    CHECK_CUDA(cudaFree(errors_test_device));
    CHECK_CUDA(cudaFree(block_errors_device));
    CHECK_CUDA(cudaFree(user_bias_device));
    CHECK_CUDA(cudaFree(item_bias_device));
    CHECK_CUDA(cudaFree(user_bias_target));
    CHECK_CUDA(cudaFree(item_bias_target));
    CHECK_CUDA(cudaFree(d_state));
    delete P_device;
    delete P_device_target;
    delete Q_device;
    delete Q_device_target;
    delete [] errors;
    delete [] errors_test;
    delete [] block_errors_host;
}

/** Initializes Q and item bias randomly and passes them into the training method.
 */
void train(CudaCSRMatrix* train_matrix, CudaCSRMatrix* test_matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float global_bias) {
    int item_count = train_matrix->cols;
    // Initialize for regular training
    float *Q = initialize_normal_array(item_count * cfg->n_factors, cfg->n_factors);
    float *item_bias = initialize_normal_array(item_count, cfg->n_factors);
    *Q_ptr = Q;
    *item_bias_ptr = item_bias;
    train(train_matrix, test_matrix, cfg, P_ptr, Q_ptr, Q, losses_ptr, user_bias_ptr, item_bias_ptr, item_bias, global_bias);
}