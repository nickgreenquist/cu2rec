all: mf mf_cpu predict

mf: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h matrix.h util.h loss.h sgd.h training.h config.cu matrix.cu util.cu loss.cu sgd.cu training.cu mf.cu > bin/mf_all.cu
	nvcc -std=c++11 -o bin/mf -arch=sm_35 bin/mf_all.cu

mf_cpu: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h matrix.h util.h config.cu matrix.cu util.cu mf_sequential.cu > bin/mf_cpu_all.cu
	nvcc -std=c++11 -o bin/mf_cpu -arch=sm_35 bin/mf_cpu_all.cu

predict: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h matrix.h util.h loss.h sgd.h training.h config.cu matrix.cu util.cu loss.cu sgd.cu training.cu predict.cu > bin/predict_all.cu
	nvcc -std=c++11 -o bin/predict -arch=sm_35 -G bin/predict_all.cu

create_bin:
	mkdir -p bin

clean:
	rm -rf bin/
