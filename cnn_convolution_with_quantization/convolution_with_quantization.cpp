/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example convolution_compare.cpp
/// @copybrief convolution_compare_cpp

/// This C++ API example compares two kernels of convolution
///
/// The example implements JUST a convolution layer
/// with two different convolution kernel and compare them.

#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>
#include <string.h>
#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"


using namespace dnnl;


void load_dims(char *filepath, int	*input_h, int *input_w, int *filter_h, int *filter_w,
               int *stride_h, int *stride_w, int *bias_size)
{
    FILE	*fp;
//	char	filepath[100];// = "/home/zahra/Documents/test_files/";
//	strcat(filepath, filename);
//	printf("Enter complete filepath(path+name):%s\n", filepath);
    fp = fopen(filepath, "r");
    if(!fp){
        printf("Error in opening file...\n");
        perror("fopen:");
        exit(0);
    }
    fscanf(fp, "%d %d %d %d %d %d %d\n", input_h, input_w, filter_h, filter_w, stride_h, stride_w, bias_size);
//	printf("Read Values:%d %d %d %d %d %d %d \n", *input_h, *input_w, *filter_h, *filter_w, *stride_h, *stride_w, *bias_size);
//	fflush(stdout);
    fclose(fp);
    return;
}

void load_src_weights(std::vector<float> *user_src, std::vector<float> *conv_weights,
                      std::vector<float> *conv_bias, char *filepath)
{
    FILE	*fp;
//	char	filepath[100] = "/home/zahra/Documents/test_files/";
    float	ftemp;
    int		i, j;
    int		input_h, input_w, filter_h, filter_w, bias_size ,stride_h, stride_w;


//	strcat(filepath, filename);
//	printf("filepath:%s\n", filepath);
    fp = fopen(filepath, "r");
    if(!fp){
        printf("Error in opening file...\n");
        perror("fopen:");
        exit(0);
    }
    fscanf(fp, "%d %d %d %d %d %d %d\n", &input_h, &input_w, &filter_h, &filter_w, &stride_h, &stride_w, &bias_size);
//	printf("Read Values:%d %d %d %d %d %d %d\n", input_h, input_w, filter_h, filter_w, stride_h, stride_w, bias_size);
//	fflush(stdout);
    for(i = 0 ; i < input_h ; i++){
        for(j = 0 ; j < input_w ; j++){
            fscanf(fp, "%f ", &ftemp);
            user_src->insert(user_src->begin() + (i*input_h) + j, ftemp);
        }
        fscanf(fp, "\n");
    }

    for(i = 0 ; i < filter_h ; i++){
        for(j = 0 ; j < filter_w ; j++){
            fscanf(fp, "%f ", &ftemp);
            conv_weights->insert(conv_weights->begin() + (i*filter_w) + j, ftemp);
        }
        fscanf(fp, "\n");
    }
    for(i = 0 ; i < bias_size ; i++){
        fscanf(fp, "%f ", &ftemp);
        conv_bias->insert(conv_bias->begin() + i, ftemp);
//		printf("conv_bias:%f\n", *(conv_bias->data()+i));

    }
    fclose(fp);
    return;
}
void simple_net_ref(engine::kind engine_kind, char *filepath, int times = 100) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    char	output_filename[200];// = "/home/zahra/Documents/test_files/onDNN_ref_out_10_10_3_3.txt";
    FILE	*output_fp;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet convolution_compare.cpp Initialize engine and stream
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    //[create output filepath]
    strcpy(output_filename, filepath);
    *(strrchr(output_filename, '/')+1) = 0;
    strcat(strrchr(output_filename, '/')+1, "ref_output_");
    strcat(output_filename, strrchr(filepath, '/')+1);
    //[create output filepath]


    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet convolution_compare.cpp Create network
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    const memory::dim batch = 1;
    int		input_h, input_w, filter_h, filter_w, c_bias;
    int		stride_h, stride_w;
    load_dims(filepath, &input_h, &input_w, &filter_h, &filter_w, &stride_h, &stride_w, &c_bias);
    printf("%d\n", input_h);
    //  conv
    // {batch, 1, 10, 10} (x) {1, 1, 3, 3} -> {batch, 1, 7, 7}
    // strides: {1, 1}
    int dest_h = (input_h-(filter_h-stride_h))/stride_h;
    int dest_w = (input_w-(filter_w-stride_w))/stride_w;
    memory::dims conv_src_tz = {batch, 1, input_h, input_w};
    memory::dims conv_weights_tz = {1, 1, filter_h, filter_w};
    memory::dims conv_bias_tz = {c_bias};
    memory::dims conv_dst_tz = {batch, 1, dest_h, dest_w};
    memory::dims conv_strides = {stride_h, stride_w};
    memory::dims conv_padding = {0, 0};

    /// Next, the example configures the scales used to quantize f32 data
    /// into int8. For this example, the scaling value is chosen as an
    /// arbitrary number, although in a realistic scenario, it should be
    /// calculated from a set of precomputed values as previously mentioned.
    /// @snippet cnn_inference_int8.cpp Choose scaling factors


    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet convolution_compare.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * 1 * input_h * input_w);
    std::vector<float> user_dst(batch * dest_h* dest_w);
    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));
    //[Allocate buffers]

    load_src_weights(&user_src, &conv_weights, &conv_bias, filepath);
    /*    printf("%f\n", user_src[0]);
    fflush(0);
    exit(0);
*/

    /// Create memory that describes data layout in the buffers. This example uses
    /// tag::nchw (batch-channels-height-width) for input data and tag::oihw
    /// for weights.
    /// @snippet cnn_inference_f32.cpp Create user memory
    //[Allocate buffers]
    auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);

    auto user_weights_memory
            = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv_weights.data(), user_weights_memory);

    auto user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv_bias.data(), user_bias_memory);
    //[Allocate buffers]

    //[Choose scaling factors]
    // Choose scaling factors for input, weight, output and bias quantization
    const std::vector<float> src_scales = {1.8f};
    const std::vector<float> weight_scales = {2.0f};
    const std::vector<float> bias_scales = {1.0f};
    const std::vector<float> dst_scales = {0.55f};

    // Choose channel-wise scaling factors for convolution
    std::vector<float> conv_scales(c_bias);
    const int scales_half = c_bias / 2;
    std::fill(conv_scales.begin(), conv_scales.begin() + scales_half, 0.3f);
    std::fill(conv_scales.begin() + scales_half + 1, conv_scales.end(), 0.8f);
    //[Choose scaling factors]

    /// The *source, weights, bias* and *destination* datasets use the single-scale
    /// format with mask set to '0', while the *output* from the convolution
    /// (conv_scales) will use the array format where mask = 2 corresponding
    /// to the output dimension.
    /// @snippet cnn_inference_int8.cpp Set scaling mask
    //[Set scaling mask]
    const int dst_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
//     const int dst_mask = 0;
    const int conv_mask = 2; // 1 << output_channel_dim
    //[Set scaling mask]

    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    /// @snippet cnn_inference_f32.cpp Create convolution memory descriptors

    //[Create convolution memory descriptors]
    auto conv_src_md = memory::desc({conv_src_tz}, dt::u8, tag::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::s8, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::u8, tag::any);
    //[Create convolution memory descriptors]




    /// Create a convolution descriptor by specifying propagation kind,
    /// [convolution algorithm](@ref dev_guide_convolution), shapes of input,
    /// weights, bias, output, convolution strides, padding, and kind of padding.
    /// Propagation kind is set to prop_kind::forward_inference to optimize for
    /// inference execution and omit computations that are necessary only for
    /// backward propagation.
    /// @snippet cnn_inference_f32.cpp Create convolution descriptor

    //[Create convolution descriptor]    memory::dims conv_src_tz = {batch, 1, 10, 10};
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
                                               algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                               conv_bias_md, conv_dst_md, conv_strides, conv_padding,
                                               conv_padding);
    //[Create convolution descriptor]



    /// Configuring int8-specific parameters in an int8 primitive is done
    /// via the Attributes Primitive. Create an attributes object for the
    /// convolution and configure it accordingly.
    /// @snippet cnn_inference_int8.cpp Configure scaling
    //[Configure scaling]
    primitive_attr conv_attr;
    conv_attr.set_output_scales(conv_mask, conv_scales);
    //[Configure scaling]


    /// The ReLU layer from Alexnet is executed through the PostOps feature. Create
    /// a PostOps object and configure it to execute an _eltwise relu_ operation.
    /// @snippet cnn_inference_int8.cpp Configure post-ops


    //[Configure post-ops]
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);
    //[Configure post-ops]

    // check if int8 convolution is supported
    try {
        convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented)
            throw example_allows_unimplemented {
                    "No int8 convolution implementation is available for this "
                    "platform.\n"
                    "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    //[Create convolution primitive descriptor]
    auto conv_prim_desc
            = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
    //[Create convolution primitive descriptor]

    //[Quantize Source data,weights and Bias ]
    //  Source Quantiaztion
    auto conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
    primitive_attr src_attr;
    src_attr.set_output_scales(dst_mask, src_scales);
    auto src_reorder_pd
            = reorder::primitive_desc(eng, user_src_memory.get_desc(), eng,
                                      conv_src_memory.get_desc(), src_attr);
    auto src_reorder = reorder(src_reorder_pd);
    src_reorder.execute(s, user_src_memory, conv_src_memory);

    //  Weigts Quantiaztion
    auto conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
    primitive_attr weight_attr;
    weight_attr.set_output_scales(weight_mask , weight_scales); //weight_mask = 0 & weight_scales = 1.0
    auto weight_reorder_pd
            = reorder::primitive_desc(eng, user_weights_memory.get_desc(), eng,
                                      conv_weights_memory.get_desc(), weight_attr);
    auto weight_reorder = reorder(weight_reorder_pd);
    weight_reorder.execute(s, user_weights_memory, conv_weights_memory);

    //  Weigts Debugging
    float *handler1 = (float *)user_weights_memory.get_data_handle();
    printf("First two CNN weigts: %f, %f\n", *handler1,handler1[1]);
    std::vector<int8_t> quan_conv_weights(product(conv_weights_tz));
    read_from_dnnl_memory(quan_conv_weights.data(), conv_weights_memory);
    printf("A = %" PRId8 ", B = %" PRIi8 "\n", quan_conv_weights[0],quan_conv_weights[1]);

        //Bias Quantiaztion
    auto conv_bias_memory = memory(conv_prim_desc.bias_desc(), eng);
    primitive_attr bias_attr;
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd
            = reorder::primitive_desc(eng, user_bias_memory.get_desc(), eng,
                                      conv_bias_memory.get_desc(), bias_attr);
    auto bias_reorder = reorder(bias_reorder_pd);
    bias_reorder.execute(s, user_bias_memory, conv_bias_memory);
    //[Quantize Source data,weights and Bias ]

    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    //[Create convolution primitive]
    auto conv = convolution_forward(conv_prim_desc);
    conv.execute(s,
                 {{DNNL_ARG_SRC, conv_src_memory},
                  {DNNL_ARG_WEIGHTS, conv_weights_memory},
                  {DNNL_ARG_BIAS, conv_bias_memory},
                  {DNNL_ARG_DST, conv_dst_memory}});
    //[Create convolution primitive]

    auto user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_dst.data(), user_dst_memory);
    primitive_attr dst_attr;
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = reorder::primitive_desc(eng, conv_dst_memory.get_desc(), eng,
                                      user_dst_memory.get_desc(), dst_attr);
    auto dst_reorder = reorder(dst_reorder_pd);
    dst_reorder.execute(s, conv_dst_memory, user_dst_memory);
    //[Dequantize the result]

    setvbuf(stdout, NULL, _IONBF, 0);
    read_from_dnnl_memory(user_dst.data(), user_dst_memory);
    output_fp = fopen(output_filename, "w");
    if(!output_fp)
    {
        perror("fopen:can not open file:");
        fflush(0);
        exit(0);
    }

    for(int i = 0; i < dest_h; i++){
        for(int j = 0 ; j < dest_w ; j++){
            fprintf(output_fp, "%.1f ", user_dst[(i*dest_h)+j]);
        }
        fprintf(output_fp, "\n");
    }

    s.wait();

}


void cnn_inference_int8(engine::kind engine_kind) {
    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count(); //VASILIS: THE TIMER SHOULD START JUST BEFORE THE MAIN ROUTINE. NOT HERE. SCANF IS NOT INCLUDED INSIDE THE TIMER WHICH IS WRONG

    int times = 1;
    char	filename[200];
    printf("\nEnter Input/Parameters filename:");
    scanf("%s", filename);


    simple_net_ref(engine_kind, filename, times);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    std::cout << "Use time(int8): " << (end - begin) / (times + 0.0)
              << " ms per iteration." << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            cnn_inference_int8, parse_engine_kind(argc, argv));
}
