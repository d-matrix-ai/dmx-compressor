#include <stdio.h>                                                                                                                                                 
#include <math.h>

typedef struct {
  int     index;
  double  mu;
} igen_op;

// float gelu(float value);
void  gen_GELULUT(float *LUT);
void  idx_gen(double *xi, long mnum_precal, int shift, igen_op *idx_mu);
void  intrp_lut(igen_op *idx_mu, int lutout_unsigned, float LUT[128], double *xop);

