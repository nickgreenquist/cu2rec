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

void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);
std::vector<Rating> readCSV(std::string filename, int *rows, int *cols);
cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols);
void writeToFile(string filepath, string extension, string component, float *data, int rows, int cols, int factors);