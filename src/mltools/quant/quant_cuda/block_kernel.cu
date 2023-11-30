#include "quant_kernel.h"
#include "sim_helper.cu"
#include "bit_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r,
                                        float *o, int size,
                                        float *__restrict__ max_entry,
                                        int man_bits,
                                        bool symmetric)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    if (!symmetric)
    {
      if (a[index] == -max_entry[index] && (max_entry_bits >> 16 << 25) == (unsigned int)0xFE000000)
      {
        max_entry_bits = (max_entry_bits >> 23) + 1 << 23;
      }
    }
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6 * BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index] + base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int rand_prob = (unsigned int)r[index];
    unsigned int quantized = round_bitwise_stochastic(target_bits, rand_prob, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

    unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float);
    unsigned int clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits);
    quantize_float = BITS_TO_FLOAT(&clip_quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_nearest(float *__restrict__ a,
                                     float *o, int size,
                                     float *__restrict__ max_entry,
                                     int man_bits,
                                     bool symmetric)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    if (!symmetric)
    {
      if (a[index] == -max_entry[index] && (max_entry_bits >> 16 << 25) == (unsigned int)0xFE000000)
      {
        max_entry_bits = (max_entry_bits >> 23) + 1 << 23;
      }
    }
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6 * BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index] + base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int quantized = round_bitwise_nearest(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

    unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float);
    unsigned int clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits); // sign bit, virtual bit
    quantize_float = BITS_TO_FLOAT(&clip_quantize);

    o[index] = quantize_float;
  }
}
__global__ void block_kernel_down(float *__restrict__ a,
                                  float *o, int size,
                                  float *__restrict__ max_entry,
                                  int man_bits,
                                  bool symmetric)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    if (!symmetric)
    {
      if (a[index] == -max_entry[index] && (max_entry_bits >> 16 << 25) == (unsigned int)0xFE000000)
      {
        max_entry_bits = (max_entry_bits >> 23) + 1 << 23;
      }
    }
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6 * BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index] + base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int quantized = round_bitwise_down(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

    unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float);
    unsigned int clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits); // sign bit, virtual bit
    quantize_float = BITS_TO_FLOAT(&clip_quantize);

    o[index] = quantize_float;
  }
}

__global__ void block_kernel_up(float *__restrict__ a,
                                float *o, int size,
                                float *__restrict__ max_entry,
                                int man_bits,
                                bool symmetric)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    if (!symmetric)
    {
      if (a[index] == -max_entry[index] && (max_entry_bits >> 16 << 25) == (unsigned int)0xFE000000)
      {
        max_entry_bits = (max_entry_bits >> 23) + 1 << 23;
      }
    }
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6 * BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index] + base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int quantized = round_bitwise_up(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

    unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float);
    unsigned int clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits); // sign bit, virtual bit
    quantize_float = BITS_TO_FLOAT(&clip_quantize);

    o[index] = quantize_float;
  }
}
// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_sim_stochastic(float *__restrict__ a,
                                            float *__restrict__ r,
                                            float *o, int size,
                                            float *max_entry,
                                            int wl,
                                            bool symmetric)
{
  // TODO: implement symmetric/asymmetric encoding below
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    int exponent = ((int)extract_exponent(max_entry));
    int sigma = exponent - (wl - 1);
    o[index] = round(a[index], r[index], sigma);
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_sim_nearest(float *__restrict__ a,
                                         float *o, int size,
                                         float *max_entry,
                                         int wl,
                                         bool symmetric)
{
  // TODO: implement symmetric/asymmetric encoding below
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    int exponent = ((int)extract_exponent(max_entry));
    int sigma = exponent - (wl - 1);
    o[index] = nearest_round(a[index], sigma);
  }
}
