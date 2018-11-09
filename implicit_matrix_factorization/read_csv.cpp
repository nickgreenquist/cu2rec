
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
            ratings.push_back({userID - 1, itemID - 1, rating});
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