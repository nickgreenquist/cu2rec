#include <algorithm>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

#include "matrix.h"
#include "read_csv.h"

cu2rec::CudaCSRMatrix* readSparseMatrix(std::vector<Rating> *ratings, int rows, int cols) {
    //int *indptr = new int[ratings->size()];
    std::vector<int> indptr_vec;
    int *indices = new int[ratings->size()];
    float *data = new float[ratings->size()];
    int lastUser = -1;
    for(int i = 0; i < ratings->size(); ++i) {
        Rating r = ratings->at(i);
        if(r.userID != lastUser) {
            indptr_vec.push_back(r.userID);
            lastUser = r.userID;
        }
        indices[i] = r.itemID;
        data[i] = r.rating;
    }
    indptr_vec.push_back(ratings->size());
    int *indptr = indptr_vec.data();

    // Create the Sparse Matrix
    const int *indptr_c = const_cast<const int*>(indptr);
    const int *indices_c = const_cast<const int*>(indices);
    const float *data_c = const_cast<const float*>(data);
    cu2rec::CudaCSRMatrix* matrix = new cu2rec::CudaCSRMatrix(rows, cols, (int)(ratings->size()), indptr_c, indices_c, data_c);
    cudaDeviceSynchronize();

    return matrix;
}

int main(int argc, char **argv){
    int rows, cols;
    std::vector<Rating> ratings = readCSV(argv[1], &rows, &cols);
    printCSV(&ratings);
    std::cout << "Rows: " << rows << ", Cols: " << cols << "\n";
    cu2rec::CudaCSRMatrix* matrix = readSparseMatrix(&ratings, rows, cols);

    //free memory
    delete matrix;
}