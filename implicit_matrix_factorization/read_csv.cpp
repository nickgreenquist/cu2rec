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

int main(){
    std::ifstream ratingsFile("../data/ratings.csv");

    if (ratingsFile.is_open() ){
        std::vector<Rating> ratings;

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

        // Print the vector
        std::cout  << "UserID" << "   ItemID" << "   Rating\n";
        for (int x(0); x < ratings.size(); ++x){
            printRating(ratings.at(x) );
        }
    }
    else{
        std::cerr<<"ERROR: The file isnt open.\n";
    }
}