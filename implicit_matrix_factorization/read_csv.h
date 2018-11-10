#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>

#include "matrix.h"

struct Rating
{
    int userID;
    int itemID;
    float rating;
};

//function headers
void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);
std::vector<Rating> readCSV(std::string filename, int *rows, int *cols);
cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols);