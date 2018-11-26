#include <random>

#include "config.h"
#include "loss.h"
#include "matrix.h"
#include "sgd.h"
#include "util.h"

#define index(i, j, N)  ((i)*(N)) + (j)

using namespace cu2rec;
using namespace std;

void train(CudaCSRMatrix* matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float *Q, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float *item_bias, float global_bias) {
    int user_count = matrix->rows;
    int item_count = matrix->cols;
    cfg->set_cuda_variables();

    // Initialize P, Q has already been initialized
    float *P = initialize_normal_array(user_count * cfg->n_factors);
    float *losses = new float[cfg->total_iterations];
    *P_ptr = P;
    *losses_ptr = losses;

    // Copy P and Q to device memory
    CudaDenseMatrix* P_device = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device = new CudaDenseMatrix(item_count, cfg->n_factors, Q);
    CudaDenseMatrix* P_device_target = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device_target = new CudaDenseMatrix(item_count, cfg->n_factors, Q);

    // Create the errors
    float *errors_device;
    cudaMalloc(&errors_device, matrix->nonzeros * sizeof(float));

    // Create the total losses
    float *losses_device;
    cudaMalloc(&losses_device, cfg->total_iterations * sizeof(float));
    cudaMemset(losses_device, 0, cfg->total_iterations * sizeof(float));

    // Create the bias array
    float *user_bias = initialize_normal_array(user_count);
    *user_bias_ptr = user_bias;
    
    float *user_bias_device;
    cudaMalloc(&user_bias_device, user_count * sizeof(float));
    cudaMemcpy(user_bias_device, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);

    float *item_bias_device;
    cudaMalloc(&item_bias_device, item_count * sizeof(float));
    cudaMemcpy(item_bias_device, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Create bias targets
    float *user_bias_target, *item_bias_target;
    cudaMalloc(&user_bias_target, user_count * sizeof(float));
    cudaMemcpy(user_bias_target, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&item_bias_target, item_count * sizeof(float));
    cudaMemcpy(item_bias_target, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Dimensions
    int n_threads = 32;
    dim3 dim_block(n_threads);
    dim3 dim_grid_sgd(user_count / n_threads + 1);
    dim3 dim_grid_loss(matrix->nonzeros / n_threads + 1);
    dim3 dim_grid_P_reg_loss(P_device->rows * P_device->cols / n_threads + 1);
    dim3 dim_grid_Q_reg_loss(Q_device->rows * Q_device->cols / n_threads + 1);
    dim3 dim_grid_user_bias_reg_loss(user_count / n_threads + 1);
    dim3 dim_grid_item_bias_reg_loss(item_count / n_threads + 1);

    // Training loop
    cudaError_t lastError;
    for (int i = 0; i < cfg->total_iterations; ++i) {
        // Calculate initial error per each rating
        calculate_loss_gpu(P_device, Q_device, cfg->n_factors, user_count, item_count, matrix->nonzeros, matrix,
                           errors_device, user_bias_device, item_bias_device, global_bias);

        // Run single iteration of SGD
        sgd_update<<<dim_grid_sgd, dim_block>>>(matrix->indptr, matrix->indices, P_device->data, Q_device->data,
                                                P_device_target->data, Q_device_target->data, errors_device,
                                                user_count, item_count, user_bias_device, item_bias_device,
                                                user_bias_target, item_bias_target);
        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
        }

        // Calculate total loss to check for improving loss
        total_loss_kernel<<<dim_grid_loss, dim_block>>>(errors_device, losses_device, matrix->nonzeros, i, 1);
        if(cfg->P_reg > 0)
            total_loss_kernel<<<dim_grid_P_reg_loss, dim_block>>>(P_device->data, losses_device, P_device->rows * P_device->cols, i, cfg->P_reg);
        if(cfg->Q_reg > 0)
            total_loss_kernel<<<dim_grid_Q_reg_loss, dim_block>>>(Q_device->data, losses_device, Q_device->rows * Q_device->cols, i, cfg->Q_reg);
        if(cfg->user_bias_reg > 0)
            total_loss_kernel<<<dim_grid_user_bias_reg_loss, dim_block>>>(user_bias_device, losses_device, user_count, i, cfg->user_bias_reg);
        if(cfg->item_bias_reg > 0)
            total_loss_kernel<<<dim_grid_item_bias_reg_loss, dim_block>>>(item_bias_device, losses_device, item_count, i, cfg->item_bias_reg);

        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
        }

        // The loss kernels modify P, Q, user_bias, and item_bias
        // Copy them back
        // TODO: avoid this entirely
        if(cfg->P_reg > 0)
            cudaMemcpy(P_device->data, P_device_target->data, user_count * cfg->n_factors * sizeof(float), cudaMemcpyDeviceToDevice);
        if(cfg->Q_reg > 0)
            cudaMemcpy(Q_device->data, Q_device_target->data, item_count * cfg->n_factors * sizeof(float), cudaMemcpyDeviceToDevice);
        if(cfg->user_bias_reg > 0)
            cudaMemcpy(user_bias_device, user_bias_target, user_count * sizeof(float), cudaMemcpyDeviceToDevice);
        if(cfg->item_bias_reg > 0)
            cudaMemcpy(item_bias_device, item_bias_target, item_count * sizeof(float), cudaMemcpyDeviceToDevice);

        lastError = cudaGetLastError();
        if(cudaSuccess != lastError) {
            printf("ERROR: %s\n", cudaGetErrorName(lastError));
        }

        // Swap old and new P and Q
        swap(P_device, P_device_target);
        swap(Q_device, Q_device_target);

        // Swap old and new bias arrays
        swap(user_bias_device, user_bias_target);
        swap(item_bias_device, item_bias_target);

        cfg->cur_iterations += 1;

        // Output current loss for this iteration
        // WARNING - SLOW: Remove after debugging due to cudaMemcpy just for printing loss
        if((i + 1) % 10 == 0) {
            cudaMemcpy(losses, losses_device, cfg->total_iterations * sizeof(float), cudaMemcpyDeviceToHost);
            cout << "Loss for Iteration " << i + 1 << ": " << losses[i] << "\n";
        }
    }
    
    // Copy array of losses back to host
    cudaMemcpy(losses, losses_device, cfg->total_iterations * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy updated P and Q back
    P_device->to_host(P);
    Q_device->to_host(Q);

    // Copy updated bias arrays back
    cudaMemcpy(user_bias, user_bias_device, user_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(item_bias, item_bias_device, item_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(errors_device);
    cudaFree(losses_device);
    cudaFree(user_bias_device);
    cudaFree(item_bias_device);
    cudaFree(user_bias_target);
    cudaFree(item_bias_target);
    delete P_device;
    delete P_device_target;
    delete Q_device;
    delete Q_device_target;
}

void train(CudaCSRMatrix* matrix, config::Config* cfg, float **P_ptr, float **Q_ptr, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float global_bias) {
    int item_count = matrix->cols;
    // Initialize for regular training
    float *Q = initialize_normal_array(item_count * cfg->n_factors);
    float *item_bias = initialize_normal_array(item_count);
    *Q_ptr = Q;
    *item_bias_ptr = item_bias;
    train(matrix, cfg, P_ptr, Q_ptr, Q, losses_ptr, user_bias_ptr, item_bias_ptr, item_bias, global_bias);
}
