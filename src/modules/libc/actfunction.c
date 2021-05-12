#include "actfunction.h"                                                                                                  
#define MSL 384                                               

#if 0
int main(void) { 

// Input, intermediate index/mu and output vectors
double  xi[128];
igen_op idx_mu[128];
float   xop[128];

//LUT: Can be populated for GELU, Tanh, and/or Sigmoid
float LUT[128]; 

//Standard Parameters will be set when HW is implemented
long mnum_precal; 
int  l2span_nfrac;
int  lutout_unsigned;

int i; 
float value; 
float outzeroPoint;
  
  mnum_precal = 0; l2span_nfrac = 0; lutout_unsigned = 1;
  outzeroPoint = ceil(0.17*256/8.17);

  // 128-vector values input to the GELU testing
  value = -4.0;
  for(i=0; i<128; i++) { 
    xi[i] = value; 
    value += 0.125;
  }

  // Populate LUT, Generate Index/Mu values and interpolate
  gen_GELULUT(LUT);
  idx_gen(xi, mnum_precal, l2span_nfrac, (igen_op *) idx_mu);
  intrp_lut((igen_op *) idx_mu, lutout_unsigned, LUT, xop);

  // Export values into Excel to draw xi-xop response
  printf(" \n");
  for(i=0; i<128; i++) {
    printf("Xi=%f, \tXop(flp)=%f\n",xi[i], xop[i]-outzeroPoint);
    //printf("%f,%f\n",xi[i], xop[i]-outzeroPoint);
  }

return 0;
}
#endif

// extern "C" {

double* gelu(double *xi) { 

  static igen_op idx_mu[MSL*4098];
  static double  xop[MSL*4098];

  static float LUT[128]; 

  //Standard Parameters will be set when HW is implemented
  long mnum_precal; 
  int  l2span_nfrac;
  int  lutout_unsigned;

  int i; 
  float value; 
  float outzeroPoint;
  
  mnum_precal = 0; l2span_nfrac = 0; lutout_unsigned = 1;
  outzeroPoint = ceil(0.17*256/8.17);

  // Populate LUT, Generate Index/Mu values and interpolate
  gen_GELULUT(LUT);
  idx_gen(xi, mnum_precal, l2span_nfrac, (igen_op *) idx_mu);
  intrp_lut((igen_op *) idx_mu, lutout_unsigned, LUT, xop);

  for(i=0; i<MSL*4096; i++) {
    xop[i] -= outzeroPoint;
    xop[i] /= 32.0;
  }

  return xop;
}

  
// } //end extern "C"


// Interpolate function - Input: index and mu value
// Generates interpolated output based on LUT values
void  intrp_lut(igen_op *idx_mu, int lutout_unsigned, float LUT[128], double *xop){
int    vec;
double lutdiff_xmu;
double lut_outcode1, lut_outcode2;
double intrp_val, intrp_rnd;

  for(vec = 0; vec<MSL*4096; vec++) {
    //printf("   index = %d and Mu = %f and LUT content=%f \n", idx_mu[vec].index, idx_mu[vec].mu, LUT[idx_mu[vec].index]);
    lut_outcode1 = (double) (1.0*LUT[idx_mu[vec].index]);
    lut_outcode2 = (double) (1.0*LUT[(int) fmin((idx_mu[vec].index+1), 127)]);

    lutdiff_xmu  = (double) floor(idx_mu[vec].mu*(lut_outcode2-lut_outcode1));
    intrp_val    = lut_outcode1 + lutdiff_xmu;
    //intrp_rnd    = (double) floor(intrp_val+0.5);
    //xop[vec]     = intrp_rnd;

    //printf("\n lut_outcode1=%f, lut_outcode2=%f, lutdiff_xmu=%f, intrp_val=%f, and intrp_rnd=%f", lut_outcode1, lut_outcode2, lutdiff_xmu, intrp_val, intrp_rnd);
    if(lutout_unsigned == 1) {
      //xop[vec] = fmin(intrp_rnd, 255);
      xop[vec] = fmin(intrp_val, 255);
      xop[vec] = fmax(xop[vec], 0);
    } else {
      //xop[vec] = fmin(intrp_rnd, 127);
      xop[vec] = fmin(intrp_val, 127);
      xop[vec] = fmax(xop[vec], -128);
    }
    //printf("\n XOP = %f ", xop[vec]);
  }

return;
}

// Index and MU value generation function
// Assumption - Input value, xi is scale/shifted value
// Fixed parameters: xmin, log2span and #of addr bits
// mnum_precal will be added later, currently set to 0
void idx_gen(double *xi, long mnum_precal, int shift, igen_op *idx_mu) {

int     vec;
double  igen_sum;
double  igen_dbl;
double  igen_dbl0;
double  xmin  = -4;
int     log2span = 4; 
int     addr_nbits = 7;

  for(vec = 0; vec<MSL*4096; vec++) {
    igen_sum = xi[vec] + (double) (1.0*mnum_precal);  // mnum_precal range: -8 to 8
    igen_dbl0 = (double) floor(igen_sum*4096)/4096.0;  // Signed 34.13 representation

    igen_dbl = (igen_dbl0 -(xmin)) / (double) pow(2.0, (log2span-addr_nbits));
    //igen_dbl = fmax(igen_dbl, 0);
    //igen_dbl = fmin(igen_dbl, 127/128.0);

    //idx_mu[vec].index = (int) floor(igen_dbl*128);
    //idx_mu[vec].mu    = (double) (igen_dbl*128 - idx_mu[vec].index);
    idx_mu[vec].index = (int) floor(igen_dbl);
    idx_mu[vec].mu    = (double) (igen_dbl - idx_mu[vec].index);
    //if(vec < 4) printf(" The index = %d and Mu = %f \n", idx_mu[vec].index, idx_mu[vec].mu);
  }
return;
}


void gen_GELULUT(float *LUT) { 
  LUT[0] = 6; LUT[1] = 6; LUT[2] = 6; LUT[3] = 6; LUT[4] = 6; LUT[5] = 6; LUT[6] = 6; LUT[7] = 6;
  LUT[8] = 5.75; LUT[9] = 5.75; LUT[10] = 5.75; LUT[11] = 5.75; LUT[12] = 5.5; LUT[13] = 5.25; LUT[14] = 5.25; LUT[15] = 5;
  LUT[16] = 4.5; LUT[17] = 4.25; LUT[18] = 3.75; LUT[19] = 3.25; LUT[20] = 2.75; LUT[21] = 2.25; LUT[22] = 1.75; LUT[23] = 1.5;
  LUT[24] = 1; LUT[25] = 0.75; LUT[26] = 0.75; LUT[27] = 0.75; LUT[28] = 1.25; LUT[29] = 1.75; LUT[30] = 2.75; LUT[31] = 4.25;
  LUT[32] = 6; LUT[33] = 8.25; LUT[34] = 10.75; LUT[35] = 13.5; LUT[36] = 16.75; LUT[37] = 20.25; LUT[38] = 24.25; LUT[39] = 28.25;
  LUT[40] = 32.25; LUT[41] = 36.75; LUT[42] = 41; LUT[43] = 45.5; LUT[44] = 49.75; LUT[45] = 54.25; LUT[46] = 58.75; LUT[47] = 63;
  LUT[48] = 67.25; LUT[49] = 71.5; LUT[50] = 75.75; LUT[51] = 79.75; LUT[52] = 83.75; LUT[53] = 88; LUT[54] = 92; LUT[55] = 96;
  LUT[56] = 100; LUT[57] = 103.75; LUT[58] = 107.75; LUT[59] = 111.75; LUT[60] = 115.75; LUT[61] = 119.5; LUT[62] = 123.5; LUT[63] = 127.5;
  LUT[64] = 131.25; LUT[65] = 135.25; LUT[66] = 139.25; LUT[67] = 143; LUT[68] = 147; LUT[69] = 151; LUT[70] = 154.75; LUT[71] = 158.75;
  LUT[72] = 162.75; LUT[73] = 166.5; LUT[74] = 170.5; LUT[75] = 174.5; LUT[76] = 178.25; LUT[77] = 182.25; LUT[78] = 186.25; LUT[79] = 190;
  LUT[80] = 194; LUT[81] = 198; LUT[82] = 201.75; LUT[83] = 205.75; LUT[84] = 209.75; LUT[85] = 213.5; LUT[86] = 217.5; LUT[87] = 221.5;
  LUT[88] = 225.25; LUT[89] = 229.25; LUT[90] = 233.25; LUT[91] = 237; LUT[92] = 241; LUT[93] = 245; LUT[94] = 248.75; LUT[95] = 252.75;
  LUT[96] = 255; LUT[97] = 255; LUT[98] = 255; LUT[99] = 255; LUT[100] = 255; LUT[101] = 255; LUT[102] = 255; LUT[103] = 255;
  LUT[104] = 255; LUT[105] = 255; LUT[106] = 255; LUT[107] = 255; LUT[108] = 255; LUT[109] = 255; LUT[110] = 255; LUT[111] = 255;
  LUT[112] = 255; LUT[113] = 255; LUT[114] = 255; LUT[115] = 255; LUT[116] = 255; LUT[117] = 255; LUT[118] = 255; LUT[119] = 255;
  LUT[120] = 255; LUT[121] = 255; LUT[122] = 255; LUT[123] = 255; LUT[124] = 255; LUT[125] = 255; LUT[126] = 255; LUT[127] = 255;       
return; 
}
