#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

struct Rating
{
    int userID;
    int itemID;
    float rating;
};

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

std::vector<Rating> readCSV(char * filename) {
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
        while(ratingsFile >> userID >> delimiter >> itemID >> delimiter >> rating >> delimiter >> timestamp)
        {
            ratings.push_back({userID, itemID, rating});
        }

        return ratings;
    }
    else{
        std::cerr<<"ERROR: The file isnt open.\n";
        return ratings;
    }
}

int main(int argc, char **argv){
    std::vector<Rating> ratings = readCSV(argv[1]);
    printCSV(&ratings);
}