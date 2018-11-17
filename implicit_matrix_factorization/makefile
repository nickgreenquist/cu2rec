read_csv:
	nvcc -std=c++11 -o mf -arch=sm_52 read_csv.cu matrix.cu mf.cu loss.cu

clean:
	rm -f read_csv