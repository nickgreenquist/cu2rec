#include <algorithm>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

struct Rating
{
    int userID;
    int itemID;
    float rating;
};

//function headers
void printRating(Rating r);
void printCSV(std::vector<Rating> *ratings);
std::vector<Rating> readCSV(char * filename, int *rows, int *cols);