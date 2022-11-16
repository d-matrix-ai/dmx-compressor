#include <ATen/ATen.h>
#include <tuple>

using namespace at;

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * and option of having a symmetric number range.
 * Stochastic Rounding.
 **/
Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl, bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * and option of having a symmetric number range.
 * Nearest Rounding.
 **/
Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl, bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the clamping into a mask,
 * with the option of having a symmetric number range
 * Stochastic Rounding.
 **/
std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask_cuda(Tensor a, int wl, int fl, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the clamping into a mask,
 * with the option of having a symmetric number range
 * Nearest Rounding.
 **/
std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask_cuda(Tensor a, int wl, int fl, bool symmetric);


/**
 * quantize a FloatTensor into block floating point number with word length [wl]
 * and block floating point exponent dimension [dim]
 * Stochastic Rounding.
 **/
Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into block floating point with word length [wl]
 * Stochastic Rounding.
 **/
Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl);

/**
 * quantize a FloatTensor into block floating point number with word length [wl]
 * and block floating point exponent dimension [dim]
 * Nearest Rounding.
 **/
Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into block floating point with word length [wl]
 * Nearest Rounding.
 **/
Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl);

/**
 * quantize a FloatTensor into block floating point number with word length [wl]
 * and block floating point exponent dimension [dim]
 * Down Rounding.
 **/
Tensor block_quantize_down_cuda(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into block floating point number with word length [wl]
 * and block floating point exponent dimension [dim]
 * Up Rounding.
 **/
Tensor block_quantize_up_cuda(Tensor a, int wl, int dim);


/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * with the option of supporting denormals or flushing them to zero.
 * Does not handle NaN, and Inf.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * with the option of supporting denormals or flushing them to zero.
 * Does not handle NaN, and Inf.
 * Down Rounding.
 **/
Tensor float_quantize_down_cuda(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * with the option of supporting denormals or flushing them to zero.
 * Does not handle NaN, and Inf.
 * Up Rounding.
 **/
Tensor float_quantize_up_cuda(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * with the option of supporting denormals or flushing them to zero.
 * Does not handle NaN, and Inf.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits, int exp_bias, bool flush_subnormal);
