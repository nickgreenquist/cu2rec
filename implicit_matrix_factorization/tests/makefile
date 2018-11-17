read_csv:
	nvcc -std=c++11 -o test_csv -arch=sm_52 ../read_csv.cu ../matrix.cu test_csv.cu
	nvcc -std=c++11 -o test_matrix -arch=sm_52 ../read_csv.cu ../matrix.cu test_matrix.cu
	nvcc -std=c++11 -o test_sgd -arch=sm_52 ../read_csv ../matrix.cu test_sgd.cu

clean:
	rm -f test_csv
	rm -f test_matrix
	rm -f test_sgd
