INC_PATH = "/home/theo/intel/oneapi/dnnl/latest/cpu_gomp/include"
LIB_PATH = "/home/theo/intel/oneapi/dnnl/latest/cpu_gomp/lib"


all:
	g++ -o cnn_inference_int8 -std=c++11 -I$(INC_PATH) -L$(LIB_PATH) convolution_with_quantization.cpp -ldnnl -O3
