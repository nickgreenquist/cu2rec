#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

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

std::vector<Rating> readCSV(std::string filename, int *rows, int *cols, float *global_bias);
void writeToFile(string parent_dir, string base_filename, string extension, string component, float *data, int rows, int cols, int factors);

// Print utils

void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);

// Array and matrix utils

float* initialize_normal_array(int size, float mean, float stddev, int seed);
float* initialize_normal_array(int size, float mean, float stddev);
float* initialize_normal_array(int size, int seed);
float *initialize_normal_array(int size);
cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols);
