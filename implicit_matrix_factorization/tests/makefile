all: test_csv test_matrix test_loss test_sgd

test_csv: create_bin
	nvcc -std=c++11 -o bin/test_csv -arch=sm_52 ../read_csv.cu ../matrix.cu test_csv.cu

test_matrix: create_bin
	nvcc -std=c++11 -o bin/test_matrix -arch=sm_52 ../read_csv.cu ../matrix.cu test_matrix.cu

test_loss: create_bin
	nvcc -std=c++11 -o bin/test_loss -arch=sm_52 ../read_csv.cu ../matrix.cu ../loss.cu test_loss.cu

test_sgd: create_bin
	nvcc -std=c++11 -o bin/test_sgd -arch=sm_52 -G ../read_csv.cu ../matrix.cu ../sgd.cu test_sgd.cu

create_bin:
	mkdir -p bin

clean:
	rm -rf bin/
