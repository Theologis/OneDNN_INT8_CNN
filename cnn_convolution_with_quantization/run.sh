#!/bin/bash

LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/latest/cpu_gomp/lib

export LD_LIBRARY_PATH

echo "/home/theo/Desktop/PhD/projects/Fast_CNN_c++/oneDNN_code/input_test_files/300_300.txt" | DNNL_VERBOSE=1 ./cnn_inference_int8



