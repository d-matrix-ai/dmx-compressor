#include "quant_kernel.h"
#include "bit_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa, stochastic rounding
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits, 
                                        int exp_bias, 
                                        bool flush_subnormal) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127; 
    // int min_exp = -((1 << (exp_bits - 1)) - 2);
    int min_exp = -(exp_bias - 1);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      if (!flush_subnormal) {
        float shift_float,val;
        int shift_bits = ((127 + min_exp)<<23) | (target >> 31 <<31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val=a[index]+shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
      }
      else {// flush subnormal numbers to zero
        quantized = 0;
      }
    }
    else {
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa, down rounding
__global__ void float_kernel_down(float* __restrict__ a,
                                  float* o, int size,
                                  int man_bits,
                                  int exp_bits, 
                                  int exp_bias, 
                                  bool flush_subnormal) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127; 
    // int min_exp = -((1 << (exp_bits - 1)) - 2);
    int min_exp = -(exp_bias - 1);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      if (!flush_subnormal) {
        float shift_float,val;
        int shift_bits = ((127 + min_exp)<<23) | (target >> 31 <<31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val=a[index]+shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_down(target, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
      }
      else {// flush subnormal numbers to zero
        quantized = 0;
      }
    }
    else {
      quantize_bits = round_bitwise_down(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa, up rounding
__global__ void float_kernel_up(float* __restrict__ a,
                                float* o, int size,
                                int man_bits,
                                int exp_bits, 
                                int exp_bias, 
                                bool flush_subnormal) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127; 
    // int min_exp = -((1 << (exp_bits - 1)) - 2);
    int min_exp = -(exp_bias - 1);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      if (!flush_subnormal) {
        float shift_float,val;
        int shift_bits = ((127 + min_exp)<<23) | (target >> 31 <<31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val=a[index]+shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_up(target, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
      }
      else {// flush subnormal numbers to zero
        quantized = 0;
      }
    }
    else {
      quantize_bits = round_bitwise_up(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa, nearest rounding
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits, 
                                     int exp_bias, 
                                     bool flush_subnormal) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127; 
    // int min_exp = -((1 << (exp_bits - 1)) - 2);
    int min_exp = -(exp_bias - 1);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      if (!flush_subnormal) {
        float shift_float,val;
        int shift_bits = ((127 + min_exp)<<23) | (target >> 31 <<31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val=a[index]+shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_nearest(target, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
      }
      else {// flush subnormal numbers to zero
        quantized = 0;
      }
    }
    else {
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}
