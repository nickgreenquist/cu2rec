all: compile_all run_all

compile_all: test_config test_loss test_sgd test_training test_util

run_all:
	@bin/test_config 1>/dev/null || echo "test_config failed!"
	@bin/test_loss 1>/dev/null || echo "test_loss failed!"
	@bin/test_sgd 1>/dev/null || echo "test_sgd failed!"
	@bin/test_training 1>/dev/null || echo "test_training failed!"
	@bin/test_util 1>/dev/null || echo "test_util failed!"

test_config: create_bin create_test_gen
	nvcc -std=c++11 -o bin/test_config -arch=sm_52 ../config.cu test_config.cu

test_loss: create_bin
	sed '/#include "[a-zA-Z./]*"/d' ../config.h ../config.cu ../matrix.h ../matrix.cu ../util.h ../util.cu ../loss.h ../loss.cu test_loss.cu > bin/test_loss_all.cu
	nvcc -std=c++11 -o bin/test_loss -arch=sm_52 -G bin/test_loss_all.cu

test_sgd: create_bin
	sed '/#include "[a-zA-Z./]*"/d' ../config.h ../config.cu ../matrix.h ../matrix.cu ../util.h ../util.cu ../sgd.h ../sgd.cu test_sgd.cu > bin/test_sgd_all.cu
	nvcc -std=c++11 -o bin/test_sgd -arch=sm_52 -G bin/test_sgd_all.cu

test_training: create_bin
	sed '/#include "[a-zA-Z./]*"/d' ../config.h ../config.cu ../matrix.h ../matrix.cu ../util.h ../util.cu ../loss.h ../loss.cu ../sgd.h ../sgd.cu ../training.h ../training.cu test_training.cu > bin/test_training_all.cu
	nvcc -std=c++11 -o bin/test_training -arch=sm_52 -G bin/test_training_all.cu

test_util: create_bin create_test_gen
	nvcc -std=c++11 -o bin/test_util -arch=sm_52 -G ../util.cu ../matrix.cu test_util.cu

create_bin:
	mkdir -p bin

create_test_gen:
	mkdir -p ../../data/test/gen

clean:
	rm -rf bin/
	rm -rf ../../data/test/gen/
