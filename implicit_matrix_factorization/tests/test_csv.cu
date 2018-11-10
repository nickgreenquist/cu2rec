#include "../read_csv.h"
#include "../matrix.h"

#include <assert.h>

using namespace std;

string filename = "../../data/test_ratings.csv";

vector<Rating> test_read_csv() {
    int rows, cols;

    vector<Rating> ratings = readCSV(filename, &rows, &cols);
    assert(rows == 6);
    assert(cols == 5);
    assert(ratings.size() == 18);

    return ratings;
}

void test_sparse_matrix() {
    // Reread CSV so this test is self contained
    int rows, cols;
    vector<Rating> ratings = readCSV(filename, &rows, &cols);

    // Create Sparse Matrix in Device memory
    cu2rec::CudaCSRMatrix* matrix = createSparseMatrix(&ratings, rows, cols);

    // copy matrix from Device to Host
    int nonzeros =  (int)(ratings.size());
    int * indptr, * indices;
    float * data;
    indptr = new int[(rows + 1)];
    indices = new int[nonzeros];
    data = new float[nonzeros];

    cudaMemcpy(indptr, matrix->indptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, matrix->indices,  nonzeros * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data, matrix->data, nonzeros * sizeof(float), cudaMemcpyDeviceToHost);

    vector<int> expected_indptr = {0,4,7,10,13,16,18};
    vector<int> expected_indices = {0,1,2,4,0,1,2,0,1,2,0,1,2,1,3,4,3,4};
    vector<float> expected_data = {1,1,1,5,3,3,3,4,4,4,5,5,5,2,4,4,5,5};

    for(int i = 0; i < expected_indptr.size(); i++) {
        assert(expected_indptr.at(i) == indptr[i]);
    }
    for(int i = 0; i < expected_indices.size(); i++) {
        assert(expected_indices.at(i) == indices[i]);
    }
    for(int i = 0; i < expected_data.size(); i++) {
        assert(expected_data.at(i) == data[i]);
    }

    //free memory
    delete indptr;
    delete indices;
    delete data;
    delete matrix;
}

int main() {
    cout << "Testing CSV is read in correctly...";
    test_read_csv();
    cout << "PASSED\n";

    cout << "Testing Sparse Matrix is Created correctly...";
    test_sparse_matrix();
    cout << "PASSED\n";

    return 0;
}