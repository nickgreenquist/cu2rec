/** Tests the utility functions
 */

#include <iostream>     // std::cout
#include <assert.h>
#include <vector>

#include "../util.h"
#include "../matrix.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

std::string test_dir = "../../data/test";
std::string test_gen_dir = "../../data/test/gen";
std::string test_filename = "test_ratings.csv";
std::string test_missing_user_ratings = "test_missing_user_ratings.csv";
std::string test_array_filename = "test_Q.csv";

std::vector<Rating> test_read_csv() {
    int rows, cols;
    float global_bias;
    std::string filename;
    filename.append(test_dir);
    filename.append("/");
    filename.append(test_filename);
    std::vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);
    assert(rows == 6);
    assert(cols == 5);
    assert(ratings.size() == 18);
    assert(fabs(global_bias - 3.5555555555555) < 1e-3);

    return ratings;
}

void test_read_array() {
    std::string filename;
    filename.append(test_dir);
    filename.append("/");
    filename.append(test_array_filename);
    float *arr = read_array(filename.c_str());
    for(int i = 0; i < 10; ++i) {
        assert(fabs(arr[i] - i) < 1e-3);
    }
    delete [] arr;
}

/** Tests writing out the factors to a file
 */
void test_output() {
    int user_count = 6;
    int item_count = 5;
    int factors = 2;

    float *P = new float[user_count * factors];
    for(int i = 0; i < user_count * factors; i++) {
        P[i] = 1.0;
    }
    
    float *Q = new float[item_count * factors];
    for(int i = 0; i < item_count * factors; i++) {
        Q[i] = 1.0;
    }

    float *user_bias = new float[user_count];
    for(int i = 0; i < user_count; i++) {
        user_bias[i] = 1.0;
    }

    float *item_bias = new float[item_count];
    for(int i = 0; i < item_count; i++) {
        item_bias[i] = 1.0;
    }

    float *global_bias = new float[1];
    global_bias[0] = 1.0;

    // Get filepath without extension
    size_t lastindex = test_filename.find_last_of("."); 
    std::string filename = test_filename.substr(0, lastindex); 

    // Write components to file
    writeToFile(test_gen_dir, filename, "csv", "p", P, user_count, factors, factors);
    writeToFile(test_gen_dir, filename, "csv", "q", Q, item_count, factors, factors);
    writeToFile(test_gen_dir, filename, "csv", "user_bias", user_bias, user_count, 1, factors);
    writeToFile(test_gen_dir, filename, "csv", "item_bias", item_bias, item_count, 1, factors);
    writeToFile(test_gen_dir, filename, "csv", "global_bias", global_bias, 1, 1, factors);

    // Free memory
    delete [] P;
    delete [] Q;
    delete [] user_bias;
    delete [] item_bias;
}

/** Tests reading in a sparse matrix
 */
void test_sparse_matrix() {
    // Reread CSV so this test is self contained
    int rows, cols;
    float global_bias;
    std::string filename;
    filename.append(test_dir);
    filename.append("/");
    filename.append(test_filename);
    std::vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);

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

    std::vector<int> expected_indptr = {0,4,7,10,13,16,18};
    std::vector<int> expected_indices = {0,1,2,4,0,1,2,0,1,2,0,1,2,1,3,4,3,4};
    std::vector<float> expected_data = {1,1,1,5,3,3,3,4,4,4,5,5,5,2,4,4,5,5};

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

/** Tests reading in a CSV with missing users.
 */
void test_missing_users() {
    int rows, cols;
    float global_bias;
    std::string filename;
    filename.append(test_dir);
    filename.append("/");
    filename.append(test_missing_user_ratings);
    std::vector<Rating> ratings = readCSV(filename, &rows, &cols, &global_bias);

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

    std::vector<int> expected_indptr = {0,4,4,7,10,13,15};
    std::vector<int> expected_indices = {0,1,2,4,0,1,2,0,1,2,1,3,4,3,4};
    std::vector<float> expected_data =  {1,1,1,5,4,4,4,5,5,5,2,4,4,5,5};

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
    std::cout << "Testing CSV is read in correctly...";
    test_read_csv();
    std::cout << "PASSED\n";

    std::cout << "Testing reading components from files...";
    test_read_array();
    std::cout << "PASSED\n";

    std::cout << "Testing Sparse Matrix is Created correctly...";
    test_sparse_matrix();
    test_missing_users();
    std::cout << "PASSED\n";

    std::cout << "Testing writing components to files...";
    test_output();
    std::cout << "PASSED\n";

    return 0;
}
