// Credit: https://github.com/benfred/implicit

#ifndef CU2REC_CUDA_MATRIX_H_
#define CU2REC_CUDA_MATRIX_H_

namespace cu2rec {
    /// Thin wrappers of CUDA memory: copies to from host, frees in destructor

    // This stores the data in a manner that makes it easily accessible by row (ie user)
    // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    struct CudaCSRMatrix {
        CudaCSRMatrix(int rows, int cols, int nonzeros,
                    const int * indptr, const int * indices, const float * data);
        ~CudaCSRMatrix();
        int * indptr, * indices;
        float * data;
        int rows, cols, nonzeros;
    };

    struct CudaDenseMatrix {
        CudaDenseMatrix(int rows, int cols, const float * data);
        ~CudaDenseMatrix();

        void to_host(float * output) const;

        int rows, cols;
        float * data;
    };
}  // namespace cu2rec
#endif  // CU2REC_CUDA_MATRIX_H_