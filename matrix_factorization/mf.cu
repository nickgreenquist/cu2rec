#include "matrix.h"
#include "util.h"

int main(int argc, char **argv){
    int rows, cols;
    float global_bias;
    std::vector<Rating> ratings = readCSV(argv[1], &rows, &cols, &global_bias);
    printCSV(&ratings);

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Kernels for factorization called here

    // Free memory
    delete matrix;
}