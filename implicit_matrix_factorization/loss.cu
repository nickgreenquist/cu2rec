#include <stdexcept>
#include <sstream>
#include <iostream>     // std::cout
#include <math.h>       /* pow */

#define index(i, j, N)  ((i)*(N)) + (j)

float dot_product_sequential(const float *Qi, const float *p, int n)
{
    float result = 0.0;
    for (int i = 0; i < n; i++)
        result += Qi[i]*p[i];
    return result;
}
float calculate_loss_sequential(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, const int * indices, const float * data) {
    float total_loss = 0;
    for(int u = 0; u < user_count; u++) {
        // get this user's factors
        float *p = new float[factors];
        for(int f = 0; f < factors; f++) {
            p[f] = P[index(u, f, factors)];
        }

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            // get this item's factors
            float *Qi = new float[factors];
            int item_id = indices[i];
            for(int f = 0; f < factors; f++) {
                Qi[f] = Q[index(item_id, f, factors)];
            }

            // update loass with this rating and prediction
            float rating = data[i];
            float pred = dot_product_sequential(Qi, p, factors);

            float loss = pow(rating - pred, 2);
            total_loss += loss;

            delete [] Qi;
        }
        delete [] p;
    }
    return total_loss;
}