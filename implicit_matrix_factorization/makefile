read_csv:
	nvcc -std=c++11 -o mf -arch=sm_52 read_csv.cu matrix.cu mf.cu
	nvcc -std=c++11 -o test_csv -arch=sm_52 read_csv.cu matrix.cu tests/test_csv.cu

clean:
	rm -f read_csv