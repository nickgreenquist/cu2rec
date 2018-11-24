mf: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h config.cu utils.cuh matrix.h matrix.cu loss.h loss.cu util.h util.cu sgd.h sgd.cu training.h training.cu mf.cu > bin/mf_all.cu
	nvcc -std=c++11 -o bin/mf -arch=sm_52 -G bin/mf_all.cu

create_bin:
	mkdir -p bin

clean:
	rm -rf bin/
