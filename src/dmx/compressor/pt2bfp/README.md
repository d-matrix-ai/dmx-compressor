# pt2-bfp
Implementation of Block Floating Point supporting the Pytorch 2.0 export quantization flow

## fx
- Contains files for creating preparing and creating the graphs for Quantization of PyTorch modules

## pt2e
- Contains methods for preparing a given GraphModulewith the observers and Quantizers

## quant/triton
- [bfp.py](quant/triton/bfp.py): Contains Triton kernels supporting Stochastic, Nearest, Up and Down quantization of a given FP32 tensor
- [bfp_ops.py](quant/triton/bfp_ops.py): Contains helper functions and the `_quantize_bfp` function which decide the quantization is to be done in CUDA or Triton
- [common_lib.py](quant/triton/common_lib.py): Contains functions for sanity checks (`check_diff`) and other helper functions
- [custom_extensions.py](quant/triton/custom_extensions.py): Python interface for CUDA implementations of the quantization kernels
- [elemwise_ops.py](quant/triton/elemwise_ops.py): Contains helper functions for bitwise manipulations for teh Python implementation of the MX kernels
- [formats.py](quant/triton/formats.py): Contains helper functions to get the `max_norm`, `min_norm`, `exponent bits` and `mantissa` bits of a given format
- [funcs.py](quant/triton/funcs.py): Contains wrapper functions to call the main functions for MX (in [here](quant/triton/mx.py)) and BFP (in [here](quant/triton/bfp.py)) quantization
- [mx.py](quant/triton/mx.py): Contains the Triton kernel suppporting MX quantization
- [mx_ops.py](quant/triton/mx_ops.py): Contains helper functions and the `_quantize_mx` function which decide the quantization is to be done in Python, CUDA or Triton
- [quantize.py](quant/triton/quantize.py): Contains helper functions for bitshifting and performing the actual quantization on the given blocksize of the triton kernel

# torchao
Contains files used for initial experimentations

# src
* [fake_quantize.py](fake_quantize.py): Contains `FakeQuantize` constructors which supports the `BlockFloatingPoint` dtype
* [observer.py](observer.py): Implemented more observers inheriting from `DMXBaseObserver`
* [qconfig_mapping.py](qconfig_mapping.py): Contains classes and functions for setting the Quantization Config Mapping of given model
* [qconfig.py](qconfig.py): Describes how to quantize a layer or a part of the network by providing settings (pt2bfp observer classes) for activations and weights respectively
* [quantization_mapping.py](quantization_mapping.py): Contains the mapping for static and dynamic quantized modules
* [quantize_fx.py](quantize_fx.py): Contains the `prepare` and `convert` modules which takes a graph module, adds observers and calibrates it according to the mapping
* [quantize_pt2e.py](quantize_pt2e.py): Contains initial wrapper functions to `prepare` (prepare_pt2e) and `convert` (convert_pt2e) the model for configuration and quantization
* [quantize.py](quantize.py): Prepare and convert submodules of  the input module based on the config and calibrations
* [quantizer.py](quantizer.py): Contains classe definitions for the BFP annotation specs and BFP quantizer
* [utils.py](utils.py): Constains helper functions for annotation, fucntions to add qspecs for the input, weight and bias.
* [x86_inductor_quantizer.py](x86_inductor_quantizer.py): Quantizer which can lower model to inductor level, not used yet, need to be validated

## tests
- [test_benchmark.py](../../../../tests/test_benchmark.py): Test suite for for benchmarking the triton kernels across various BLOCKSIZEs and warps
- [test_bfp_ptq.py](../../../../tests/test_bfp_ptq.py): Testing the bfp quantization with the triton kernels inttegrated (work in progress)
- [test_fake_bfp.py](../../../../tests/test_fake_bfp.py): Testing the fake bfp quantization flow for various models - supports all linear layers. Need to add support for gpt models (conv1d layers)
- [test_fake_ptq.py](../../../../tests/test_fake_ptq.py): Initial test for fake ptq with unit8
- [test_quantize_bfp.py](../../../../tests/test_quantize_bfp.py): Tests for logical similarity of quantized representation of the Triton generated tensor with the CUDA generated BFP tensor
- [test_quantize_mx.py](../../../../tests/test_quantize_mx.py): Tests for logical similarity of quantized representation of the Triton generated tensor with the CUDA and Python generated MX tensor
- [test_toy_ptq.py](../../../../tests/test_toy_ptq.py): Another test for ptq using the PyTorch 2.0 flow

## Notes
- Made changes in local graph.py of (torch.fx), which are reflected in [graph.py](fx/graph.py)
