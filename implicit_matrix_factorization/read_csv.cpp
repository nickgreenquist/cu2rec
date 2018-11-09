
#include "read_csv.h"

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

std::vector<Rating> readCSV(char * filename, int *rows, int *cols) {
    int max_row = 0;
    int max_col = 0;
    std::ifstream ratingsFile(filename);
    std::vector<Rating> ratings;

    if (ratingsFile.is_open()){
        // Read the file;
        int userID, itemID;
        float rating;
        int timestamp;
        char delimiter;

        // Read the file.
        ratingsFile.ignore(1000, '\n');
        while(ratingsFile >> userID >> delimiter >> itemID >> delimiter >> rating >> delimiter >> timestamp) {
            ratings.push_back({userID, itemID, rating});
            max_row = std::max(userID, max_row);
            max_col = std::max(itemID, max_col);
        }
        *rows = max_row;
        *cols = max_col;
        return ratings;
    }
    else{
        std::cerr<<"ERROR: The file isnt open.\n";
        return ratings;
    }
}

// cu2rec::CudaCSRMatrix* readSparseMatrix(std::vector<Rating> *ratings, int rows, int cols) {
//     //int *indptr = new int[ratings->size()];
//     std::vector<int> indptr_vec;
//     int *indices = new int[ratings->size()];
//     float *data = new float[ratings->size()];
//     int lastUser = -1;
//     for(int i = 0; i < ratings->size(); ++i) {
//         Rating r = ratings->at(i);
//         if(r.userID != lastUser) {
//             indptr_vec.push_back(r.userID);
//             lastUser = r.userID;
//         }
//         indices[i] = r.itemID;
//         data[i] = r.rating;

//     }
//     indptr_vec.push_back(ratings->size());
//     int *indptr = indptr_vec.data();
//     const int *indptr_c = const_cast<const int*>(indptr);
//     const int *indices_c = const_cast<const int*>(indices);
//     const float *data_c = const_cast<const float*>(data);
//     cu2rec::CudaCSRMatrix* matrix = new cu2rec::CudaCSRMatrix(rows, cols, (int)(ratings->size()), indptr_c, indices_c, data_c);
    
//     // Delete host arrays
//     delete[] indptr;
//     delete[] indices;
//     delete[] data;
//     return matrix;
// }

// int main(int argc, char **argv){
//     int rows, cols;
//     std::vector<Rating> ratings = readCSV(argv[1], &rows, &cols);
//     // cu2rec::CudaCSRMatrix* matrix = readSparseMatrix(&ratings, rows, cols);
//     printCSV(&ratings);
// }