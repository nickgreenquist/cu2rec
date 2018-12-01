#ifndef CU2REC_UTIL
#define CU2REC_UTIL

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

#include <cuda.h>

#include "matrix.h"

using namespace std;

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

struct Rating
{
    int userID;
    int itemID;
    float rating;
};

// File read and write utils
float* read_array(const char *file_path, int *n_rows_ptr, int *n_cols_ptr);
float* read_array(const char *file_path);
std::vector<Rating> readCSV(std::string filename, int *rows, int *cols, float *global_bias);
void writeCSV(char *file_path, float *data, int rows, int cols);
void writeToFile(string parent_dir, string base_filename, string extension, string component, float *data, int rows, int cols, int factors);

// Print utils
void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);

// Array and matrix utils
float* initialize_normal_array(int size, int n_factors, float mean, float stddev, int seed);
float* initialize_normal_array(int size, int n_factors, float mean, float stddev);
float* initialize_normal_array(int size, int n_factors, int seed);
float *initialize_normal_array(int size, int n_factors);
cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols);

// device functions kernels can use
__device__ float get_prediction(int factors, const float *p, const float *q, const float *data, int y_i, float user_bias, float item_bias, float global_bias);

#endif
