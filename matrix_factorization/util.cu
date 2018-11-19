#include "util.h"

using namespace std;

void writeToFile(string filepath, string extension, string component, float *data, int rows, int cols, int factors) {
    char filename [255];
    sprintf(filename, "%s_%d_%s.%s", filepath.c_str(), factors, component.c_str(), extension.c_str());

    FILE *fp;
    fp = fopen(filename, "w");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols - 1; j++) {
            fprintf(fp, "%f,", data[index(i, j, cols)]);
        }
        fprintf(fp, "%f", data[index(i, cols - 1, cols)]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

//function to print info
void printRating(Rating r){
    std::cout << r.userID << "  "<< r.itemID <<"  "<< r.rating << "\n";
}

void printCSV(std::vector<Rating> *ratings) {
    // Print the vector
    std::cout  << "UserID" << "   ItemID" << "   Rating\n";
    for (int x(0); x < ratings->size(); ++x){
        printRating(ratings->at(x));
    }
}

std::vector<Rating> readCSV(std::string filename, int *rows, int *cols, float *global_bias) {
    int max_row = 0;
    int max_col = 0;
    float sum_ratings = 0;
    std::ifstream ratingsFile(filename);
    std::vector<Rating> ratings;

    if (ratingsFile.is_open()){
        int userID, itemID;
        float rating;
        int timestamp;
        char delimiter;

        // Read the file line by line and skip the header
        ratingsFile.ignore(1000, '\n');
        while(ratingsFile >> userID >> delimiter >> itemID >> delimiter >> rating >> delimiter >> timestamp) {
            ratings.push_back({userID - 1, itemID - 1, rating});
            max_row = std::max(userID, max_row);
            max_col = std::max(itemID, max_col);
            sum_ratings += rating;
        }
        *rows = max_row;
        *cols = max_col;
        *global_bias = sum_ratings / (1.0 * ratings.size());
        return ratings;
    }
    else{
        std::cerr<<"ERROR: The file isnt open.\n";
        return ratings;
    }
}

cu2rec::CudaCSRMatrix* createSparseMatrix(std::vector<Rating> *ratings, int rows, int cols) {
    //int *indptr = new int[ratings->size()];
    std::vector<int> indptr_vec;
    int *indices = new int[ratings->size()];
    float *data = new float[ratings->size()];
    int lastUser = -1;
    for(int i = 0; i < ratings->size(); ++i) {
        Rating r = ratings->at(i);
        if(r.userID != lastUser) {
            indptr_vec.push_back(i);
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

