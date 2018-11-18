#include "matrix.h"
#include "read_csv.h"

int main(int argc, char **argv){
    int rows, cols;
    std::vector<Rating> ratings = readCSV(argv[1], &rows, &cols);
    printCSV(&ratings);

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // Kernels for factorization called here

    // Free memory
    delete matrix;
}