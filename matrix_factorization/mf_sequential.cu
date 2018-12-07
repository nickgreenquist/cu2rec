#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "config.h"
#include "matrix.h"
#include "util.h"
#include "training.h"
#include "loss.h"

#define index(i, j, N)  ((i)*(N)) + (j)

int main(int argc, char **argv){
    if(argc < 2) {
        return -1;
    }
    std::string filename_config;
    int o;
    while((o = getopt(argc, argv, "c:")) != -1) {
        switch(o) {
            case 'c':
                filename_config = optarg;
                break;
            default:
                cout << "Unknown option.\n";
                return 1;
        }
    }

    // Load in train data
    string file_path_train = argv[optind++];
    int rows, cols;
    float global_bias;
    std::vector<Rating> train_ratings = readCSV(file_path_train, &rows, &cols, &global_bias);

    // Create sparse matrix for train data
    std::vector<int> train_indptr_vec;
    int *train_indices = new int[train_ratings.size()];
    float *train_data = new float[train_ratings.size()];
    int lastUser = -1;
    for(int i = 0; i < train_ratings.size(); ++i) {
        Rating r = train_ratings.at(i);
        if(r.userID != lastUser) {
            while(lastUser != r.userID) {
                train_indptr_vec.push_back(i);
                lastUser++;
            }
        }
        train_indices[i] = r.itemID;
        train_data[i] = r.rating;
    }
    train_indptr_vec.push_back(train_ratings.size());
    int *train_indptr = train_indptr_vec.data();

    // Load in test data
    string file_path_test = argv[optind++];
    int r, c;
    float gb;
    std::vector<Rating> test_ratings = readCSV(file_path_test, &r, &c, &gb);

    std::vector<int> test_indptr_vec;
    int *test_indices = new int[test_ratings.size()];
    float *test_data = new float[test_ratings.size()];
    lastUser = -1;
    for(int i = 0; i < test_ratings.size(); ++i) {
        Rating r = test_ratings.at(i);
        if(r.userID != lastUser) {
            while(lastUser != r.userID) {
                test_indptr_vec.push_back(i);
                lastUser++;
            }
        }
        test_indices[i] = r.itemID;
        test_data[i] = r.rating;
    }
    test_indptr_vec.push_back(test_ratings.size());
    int *test_indptr = test_indptr_vec.data();

    // Hyperparams
    config::Config *cfg = new config::Config();
    if(!filename_config.empty())
        cfg->read_config(filename_config);
    cfg->print_config();

    // Initialize P, Q has already been initialized
    float *P = initialize_normal_array(rows * cfg->n_factors, cfg->n_factors);
    float *Q = initialize_normal_array(cols * cfg->n_factors, cfg->n_factors);
    float *item_bias = initialize_normal_array(cols, cfg->n_factors);
    float *user_bias = initialize_normal_array(rows, cfg->n_factors);

    // to measure time taken by a specific part of the code 
    double time_taken;
    clock_t start, end;

    // Training loop
    start = clock();
    for (int i = 0; i < cfg->total_iterations; ++i) {
        // Run one cycle of SGD per user
        for(int x = 0; x < rows; x++) {
            // Pick a random item for this user
            int low = train_indptr[x];
            int high = train_indptr[x+1];
            if(low != high) {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(low, high); // define the range
                int y_i = distr(eng);

                // get needed components into variables
                int y = train_indices[y_i];
                const float * p = &P[x * cfg->n_factors];
                const float * q = &Q[y * cfg->n_factors];
                float ub = user_bias[x];
                float ib = item_bias[y];

                // get the error random item y_i
                float pred = global_bias + ub + ib;
                for (int f = 0; f < cfg->n_factors; f++)
                    pred += q[f]*p[f];
                float error_y_i = train_data[y_i] - pred;

                // Update all components
                for(int f = 0; f < cfg->n_factors; ++f) {
                    int p_index = index(x, f, cfg->n_factors);
                    int q_index = index(y, f, cfg->n_factors);

                    // update components
                    float p_old = P[p_index];
                    float q_old = Q[q_index];
                    P[p_index] += cfg->learning_rate * (error_y_i * q_old - cfg->P_reg * P[p_index]);
                    Q[q_index] += cfg->learning_rate * (error_y_i * p_old - cfg->Q_reg * Q[q_index]);
                }

                // update biases
                user_bias[x] += cfg->learning_rate * (error_y_i - cfg->user_bias_reg * ub);
                item_bias[y] += cfg->learning_rate * (error_y_i - cfg->item_bias_reg * ib);
            }
        }

        // Calculate total loss first, last, and every check_error iterations
        if((i + 1) % cfg->check_error == 0 || i == 0 || (i + 1) % cfg->total_iterations == 0) {
            float train_rmse, train_mae, validation_rmse, validation_mae; 
            train_rmse = train_mae = validation_rmse = validation_mae = 0.0;
                       
            // Calculate loss on train ratings
            for(int x = 0; x < rows; x++) {
                const float * p = &P[x * cfg->n_factors];
                float ub = user_bias[x];

                for (int k = train_indptr[x]; k < train_indptr[x + 1]; ++k) {
                    int item_id = train_indices[k];
                    const float * q = &Q[item_id * cfg->n_factors];
                    float ib = item_bias[item_id];

                    // get the error random item y_i
                    float pred = global_bias + ub + ib;
                    for (int f = 0; f < cfg->n_factors; f++)
                        pred += q[f]*p[f];
                    float error_y_i = train_data[k] - pred;

                    // update total loss metrics
                    train_mae += abs(error_y_i);
                    train_rmse += error_y_i * error_y_i;
                }
            }
            train_mae /= train_ratings.size();
            train_rmse = sqrt(train_rmse / train_ratings.size());

            // Calculate loss on test ratings
            for(int x = 0; x < r; x++) {
                const float * p = &P[x * cfg->n_factors];
                float ub = user_bias[x];

                for (int k = test_indptr[x]; k < test_indptr[x + 1]; ++k) {
                    int item_id = test_indices[k];
                    const float * q = &Q[item_id * cfg->n_factors];
                    float ib = item_bias[item_id];

                    // get the error random item y_i
                    float pred = global_bias + ub + ib;
                    for (int f = 0; f < cfg->n_factors; f++)
                        pred += q[f]*p[f];
                    float error_y_i = test_data[k] - pred;

                    // update total loss metrics
                    validation_mae += abs(error_y_i);
                    validation_rmse += error_y_i * error_y_i;
                }
            }
            validation_mae /= test_ratings.size();
            validation_rmse = sqrt(validation_rmse / test_ratings.size());

            // Print error metrics
            printf("TRAIN: Iteration %d MAE: %f RMSE: %f\n", i + 1, train_mae, train_rmse);
            printf("TEST: Iteration %d MAE: %f RMSE: %f\n", i + 1, validation_mae, validation_rmse);
        }
    }
    end = clock();

    // Output time taken
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;   
    printf("Time taken for %d of iterations is %lf\n", cfg->total_iterations, time_taken);

    // Free memory
    delete cfg;
    // delete [] train_indices;
    // delete [] train_indptr;
    // delete [] train_data;
    // delete [] test_indices;
    // delete [] test_indptr;
    // delete [] test_data;
    delete [] P;
    delete [] Q;
    delete [] user_bias;
    delete [] item_bias;
}