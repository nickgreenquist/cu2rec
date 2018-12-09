#ifndef CU2REC_UTIL
#define CU2REC_UTIL

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <cuda.h>

#include "matrix.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/** Convenience struct for holding the userid-itemid-rating tuple.
 */
struct Rating
{
    int userID;
    int itemID;
    float rating;
};

// Error checking
#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::stringstream err;
        err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}

// File read and write utils
float* read_array(const char *file_path, int *n_rows_ptr, int *n_cols_ptr);
float* read_array(const char *file_path);
std::vector<Rating> readCSV(std::string filename, int *rows, int *cols, float *global_bias);
void writeCSV(char *file_path, float *data, int rows, int cols);
void writeToFile(std::string parent_dir, std::string base_filename, std::string extension, std::string component, float *data, int rows, int cols, int factors);

// Print utils
void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);

// Array and matrix utils
float* initialize_normal_array(int size, int n_factors, float mean, float stddev, int seed);
float* initialize_normal_array(int size, int n_factors, float mean, float stddev);
float* initialize_normal_array(int size, int n_factors, int seed);
float *initialize_normal_array(int size, int n_factors);
cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols);

// GPU information helper functions
size_t getFreeBytes(const int where, size_t *total_bytes);

// device functions kernels can use
__device__ float get_prediction(int factors, const float *p, const float *q, float user_bias, float item_bias, float global_bias);

#endif
