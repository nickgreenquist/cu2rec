#include <getopt.h>
#include <vector>

#include "config.h"
#include "matrix.h"
#include "util.h"

float* predict_ratings(float *P_u, float *Q, float user_bias, float *item_bias, float global_bias,
                       int n_items, int n_factors) {
    float *predictions = new float[n_items];
    for(int i = 0; i < n_items; i++) {
        const float * Q_i = &Q[i * n_factors];
        float pred = global_bias + user_bias + item_bias[i];
        for(int f = 0; f < n_factors; f++) {
            pred += Q_i[f]*P_u[f];
        }
        predictions[i] = pred;
    }
    return predictions;
}

void print_predictions(float *predictions, int n_items) {
    cout << "Predictions: " <<  "\n";
    cout << "[";
    for(int i = 0; i < n_items; i++) {
        cout << predictions[i] << ", ";
    }
    cout << "]\n";
}

int main(int argc, char **argv) {
    if(argc < 2) {
        return 2;
    }
    bool use_partial_fit = false;
    std::string filename_config;
    std::string filename_item_bias;
    std::string filename_global_bias;
    std::string filename_Q;
    int o;
    while((o = getopt(argc, argv, "pc:i:g:q:")) != -1) {
        switch(o) {
            case 'p':
                use_partial_fit = true;
                break;
            case 'c':
                filename_config = optarg;
                break;
            case 'i':
                filename_item_bias = optarg;
                break;
            case 'g':
                filename_global_bias = optarg;
                break;
            case 'q':
                filename_Q = optarg;
                break;
            default:
                cout << "Unknown option.\n";
                return 1;
        }
    }
    config::Config *cfg = new config::Config();
    cfg->read_config(filename_config);
    cfg->total_iterations = 10;
    cfg->learning_rate = 1e-1;
    cfg->is_train = false;
    cfg->n_factors = 2;
    cfg->set_cuda_variables();
    int n_items, n_factors;
    float *item_bias = read_array(filename_item_bias.c_str(), &n_factors, &n_items);
    float *global_bias_arr = read_array(filename_global_bias.c_str());
    float global_bias = global_bias_arr[0];
    float *Q = read_array(filename_Q.c_str(), &n_items, &n_factors);

    std::string filename_user_ratings = argv[optind++];
    int rows, cols;
    float user_mean;
    vector<Rating> ratings = readCSV(filename_user_ratings, &rows, &cols, &user_mean);
    for(std::vector<Rating>::iterator i = ratings.begin(); i != ratings.end(); ++i) {
        (*i).userID = 0;
    }
    CudaCSRMatrix* matrix = createSparseMatrix(&ratings, 1, n_items);

    float *P, *losses, *user_bias;
    train(matrix, cfg, &P, &Q, Q, &losses, &user_bias, &item_bias, item_bias, global_bias);

    float *predictions = predict_ratings(P, Q, user_bias[0], item_bias, global_bias, n_items, cfg->n_factors);
    print_predictions(predictions, n_items);

    delete cfg;
    delete matrix;
    delete [] losses;
    delete [] P;
    delete [] Q;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_arr;
    delete [] predictions;

    return 0;
}
