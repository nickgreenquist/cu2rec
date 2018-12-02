all: mf predict

mf: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h matrix.h util.h loss.h sgd.h training.h config.cu matrix.cu util.cu loss.cu sgd.cu training.cu mf.cu > bin/mf_all.cu
	nvcc -std=c++11 -o bin/mf -arch=sm_52 bin/mf_all.cu

predict: create_bin
	sed '/#include "[a-zA-Z./]*"/d' config.h matrix.h util.h loss.h sgd.h training.h config.cu matrix.cu util.cu loss.cu sgd.cu training.cu predict.cu > bin/predict_all.cu
	nvcc -std=c++11 -o bin/predict -arch=sm_52 -G bin/predict_all.cu

create_bin:
	mkdir -p bin

clean:
	rm -rf bin/
