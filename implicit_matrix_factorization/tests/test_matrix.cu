#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>
#include <vector>
#include <string>
#include <assert.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "../matrix.h"
#include "../utils.cuh"

using namespace std;

__global__ void dot_kernel(float* a, float* b, float* c) {
    // for some reason, this kernel has to be called len(a) times for c to get the full dot product????
    *c = cu2rec::dot(a,b);
}

void test_dot() {
    vector<float> va = {1,2,3,4,5};
    vector<float> vb = {5,4,3,2,1};
    int size = va.size() * sizeof(float);

    float* a, *b, *c;
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(sizeof(float));

    // Convert vector to float array
    for(int i = 0; i < va.size(); i++) {
        a[i] = va.at(i);
        b[i] = vb.at(i);
    }

    // Allocate and copy memory to device arrays
    float* a_d, *b_d, *c_d;

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, sizeof(float));

    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, sizeof(float), cudaMemcpyHostToDevice);

    dot_kernel<<<1, va.size()>>>(a_d,b_d,c_d);

    cudaMemcpy(c, c_d, sizeof(float), cudaMemcpyDeviceToHost);
    
    assert(*c == 35);

    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
}

int main() {
    cout << "Testing dot() function returns correct result...";
    test_dot();
    cout << "PASSED\n";

    return 0;
}