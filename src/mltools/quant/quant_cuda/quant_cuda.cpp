#include <torch/torch.h>
#include "quant_cuda.h"
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// fixed point quantizations
Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl,
                                                                bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_mask_cuda(a, wl, fl, symmetric);
}

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_cuda(a, wl, fl, use_clamp, symmetric);
}

// For research
Tensor fixed_point_quantize_up(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_up_cuda(a, wl, fl, use_clamp, symmetric);
}

// For research
Tensor fixed_point_quantize_down(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_down_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl,
                                                             bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_mask_cuda(a, wl, fl, symmetric);
}

// block floating point quantizations
Tensor block_quantize_stochastic(Tensor a, int wl, int dim, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_stochastic_cuda(a, wl, dim, symmetric);
}

Tensor block_quantize_sim_stochastic(Tensor a, int wl, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_sim_stochastic_cuda(a, wl, symmetric);
}

Tensor block_quantize_down(Tensor a, int wl, int dim, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_down_cuda(a, wl, dim, symmetric);
}

Tensor block_quantize_up(Tensor a, int wl, int dim, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_up_cuda(a, wl, dim, symmetric);
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_nearest_cuda(a, wl, dim, symmetric);
}

Tensor block_quantize_sim_nearest(Tensor a, int wl, bool symmetric)
{
  CHECK_INPUT(a);
  return block_quantize_sim_nearest_cuda(a, wl, symmetric);
}

// Low-bitwidth floating point quantizations
Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal)
{
  CHECK_INPUT(a);
  return float_quantize_stochastic_cuda(a, man_bits, exp_bits, exp_bias, flush_subnormal);
}

Tensor float_quantize_down(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal)
{
  CHECK_INPUT(a);
  return float_quantize_down_cuda(a, man_bits, exp_bits, exp_bias, flush_subnormal);
}

Tensor float_quantize_up(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal)
{
  CHECK_INPUT(a);
  return float_quantize_up_cuda(a, man_bits, exp_bits, exp_bias, flush_subnormal);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal)
{
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a, man_bits, exp_bits, exp_bias, flush_subnormal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // fixed point quantizations
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_stochastic_mask", &fixed_point_quantize_stochastic_mask, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("fixed_point_quantize_up", &fixed_point_quantize_up, "Fixed Point Number Rounding Up Quantization (CUDA)");
  m.def("fixed_point_quantize_down", &fixed_point_quantize_down, "Fixed Point Number Rounding Down Quantization (CUDA)");

  // block floating point quantizations
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_sim_stochastic", &block_quantize_sim_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_down", &block_quantize_down, "Block Floating Point Number Rounding Down Quantization (CUDA)");
  m.def("block_quantize_up", &block_quantize_up, "Block Floating Point Number Rounding Up Quantization (CUDA)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_sim_nearest", &block_quantize_sim_nearest, "Block Floating Point Number Stochastic Quantization (CUDA)");

  // Low-bitwidth floating point quantizations
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_down", &float_quantize_down, "Low-Bitwidth Floating Point Number Rounding Down Quantization (CUDA)");
  m.def("float_quantize_up", &float_quantize_up, "Low-Bitwidth Floating Point Number Rounding Up Quantization (CUDA)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
}
