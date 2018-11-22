mf: create_bin
	nvcc -std=c++11 -o bin/mf -arch=sm_52 util.cu matrix.cu mf.cu loss.cu sgd.cu training.cu

create_bin:
	mkdir -p bin

clean:
	rm -rf bin/
