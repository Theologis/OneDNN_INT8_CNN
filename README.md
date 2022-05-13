# OneDNN_INT8_CNN

This C++ API example demonstrates how to run a 3D-convolution and relu3 layer with int8 data type using Intel's [OneDNN](https://github.com/oneapi-src/oneDNN) library. 
To run example do the following steps:
- Open terminal
- Cd cnn_convolution_with_quantization
- Open the Makefile, update INC_PATH and LIB_PATH variables and save it
- Type "make" in terminal
- Run the created output as follow:
  DNNL_VERBOSE=1 ./cnn_inference_int8
- type /.../input_test_files/300_300.txt or /.../input_test_files/1000_1000_7_7.txt

Code will create the output under the ../input_test_files/ref_output_{input_name}.txt
		
		
*** Please take a look at section 1.1 of link below and check if I am right about supported platforms of int8 convolution***
*** https://docs.oneapi.io/versions/latest/onednn/dev_guide_int8_computations.html ***
