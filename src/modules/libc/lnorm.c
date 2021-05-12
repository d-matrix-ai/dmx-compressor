#include "lnorm.h"                                                                                                                                                 

int main(void) { 

double *lrn_result;
int    *dmlrn_result;
int    idealnorm[vecblk_len];

int    i, j;
int    input[vecblk_len];
float  NormLOutScale;
float  NormOutZpoint;
float  flpinput[vecblk_len];

float  meanSqError;
float  IntrSqError[vecblk_row];
float  normerror[vecblk_row][vecblk_len];

double epsilon_prime = 0.0;

input_cfg inp_cfg;
lrn_cfg   lcfg; 

  set_lrn_cfg (&lcfg, beta, vgamma, epsilon, vecblk_len);
  specific_cfg ();
  set_input_cfg (&inp_cfg, MaxData, imcout_nrbits); 

  epsilon_prime = epsilon/pow(inp_cfg.InputScaleFP, 2);
  NormLOutScale = (ymax-ymin)/255; 
  NormOutZpoint = round((-1*ymin)/NormLOutScale); 

  printf(" \n");
  for(i=0; i<vecblk_row; i++) {   // 64 rows
    for(j=0; j<vecblk_len; j++) {
      input[j] = (int) round(inpvector[i][j]/inp_cfg.InputScaleFP);
      input[j] = fmin(input[j], pow(2, 31)-1);
      input[j] = fmax(input[j], -1*pow(2,31));
      flpinput[j] = input[j]*inp_cfg.InputScaleFP;
    } 
    // Floating point Layer normalization
    lrn_result   = lrespnorm(flpinput, lcfg);   
    for(j=0; j<vecblk_len; j++) 
      idealnorm[j] = quantize8b(lrn_result[j], NormLOutScale, NormOutZpoint, 0); 
    
    // Fixed point Layer Normalization
    dmlrn_result = dmlrespnorm(input, lcfg);
    
    // Normalization Error
    for(j=0; j<vecblk_len; j++) normerror[i][j] = pow((dmlrn_result[j] - idealnorm[j]), 2);
    IntrSqError[i] = get_statmean(normerror[i], vecblk_len);
  }

  meanSqError = get_statmean(IntrSqError, vecblk_row);
  printf(" \n FINAL RESULT Mean Square Error of Ideal and Quantized DM norm o/p = %f \n\n", meanSqError);
  //for(i=0; i<vecblk_len; i++) if (i<16) printf("Yi[%2d] = %f, \tIdealQuantYi[%2d] = %d, \tDMatQuantYi[%2d] = %d \n", i, lrn_result[i], i, idealnorm[i], i, dmlrn_result[i]);
 
return 0;
}

// Fixed point Local Response Normalization
int * dmlrespnorm(int *ipval, lrn_cfg cfg){
static int dmlrn_res[vecblk_len];

int     i;
int     ai, ni, intrp_out;
float   tmp, deltak;
float   vmu, mbvar;
double  sigma2;
float   xi_ar_prod[vecblk_len];
igen_op idx_mu;

  // Convert 8-bit input to Float
  for(i=0; i<vecblk_len; i++) xi_ar_prod[i] = (float) ipval[i] * 1.0;
  
  vmu    = floor(get_statmean(xi_ar_prod, vecblk_len));  
  mbvar  = floor(get_mbvariance(xi_ar_prod, vecblk_len, vmu)); 
  sigma2 = mbvar;  // should be -epsilon*Xmin  .. Since epsilon is set to 0 hence ...
  idx_gen(&sigma2, 1, vcfg, isqrtlut_cfg, &idx_mu);
  intrp_lut(&idx_mu, 1, 1, ISQRTLUT, &intrp_out);
  
  for(i=0; i<vecblk_len; i++) {
    // NormOutScale = 0.4928 for this particular input dataset
    tmp = vgamma[i]*isqrtlut_cfg.OutDelta/0.4928;  
    mapfloattoHWformat(tmp, 0, 0.001*tmp, &(ai), &(ni));
    deltak = round(beta[i]*(pow(2, ni))/0.4928 + 28*(pow(2, ni)));
    
    dmlrn_res[i] = (int) round((ai*(intrp_out-isqrtlut_cfg.OutZeroPoint)*(((float) ipval[i])-vmu)+deltak)/pow(2,ni)); 
  }
  return dmlrn_res;
}

// Floating point Local Response Normalization
double * lrespnorm(float *ipval, lrn_cfg cfg) { 
static double lrn_res[vecblk_len];

int    i;
float  normx[vecblk_len];
float  vmean; 
float  mbvar; 
float  rmsval; 

  vmean  = get_statmean(ipval, vecblk_len);
  mbvar  = get_mbvariance(ipval, vecblk_len, vmean); 
  rmsval = sqrt((mbvar)+cfg.epsilon);

  //printf(" \n The mean value = %f, mbvar = %f, and rmsval = %f \n", vmean, mbvar, rmsval);
  for(i=0; i<vecblk_len; i++) { 
    normx[i]   = (ipval[i]-vmean)/(rmsval);
    lrn_res[i] = normx[i]*cfg.vgamma[i] + cfg.beta[i];
  }
  return lrn_res;
}

void specific_cfg() {

  ymax          = 111.9911; 
  ymin          = -13.6620; 
  MaxData       = 5;
  MinData       = -5;
  imcout_nrbits = 16;

  isqrtlut_cfg.m        = 8870689;
  isqrtlut_cfg.step     = 262144;
  isqrtlut_cfg.OutDelta = 0.00000067271;
  isqrtlut_cfg.log2span = 25;
  
  isqrtlut_cfg.m_num          = 8870689;
  isqrtlut_cfg.addr_nbit      = 7;
  isqrtlut_cfg.m_log2denum    = 0;
  isqrtlut_cfg.OutIsSigned    = 0;
  isqrtlut_cfg.OutZeroPoint   = -244;
  isqrtlut_cfg.nbrbitperentry = 10;
  
  vcfg.InputScaleFP            = 0;             
  vcfg.imcout_nrbits           = 0;
  vcfg.InputScale32bHW         = 0;
  vcfg.nmbredecimalsbelowcomma = 0;

return; 
}

#define INSTRUMENT 0

// extern "C" {

double* layernorm(double *ipval, double *gamma, double *beta) { 

  int    *dmlrn_result;

  static int    input[vecblk_len];
  static double output[vecblk_len];

  input_cfg inp_cfg;
  lrn_cfg   lcfg; 

  set_lrn_cfg (&lcfg, beta, gamma, 0.0f, vecblk_len);
  specific_cfg ();
  set_input_cfg (&inp_cfg, MaxData, imcout_nrbits); 

  float NormLOutScale = (ymax-ymin)/255; 
  float NormOutZpoint = round((-1*ymin)/NormLOutScale); 

  for(int j=0; j<vecblk_len; j++) {
    input[j] = (int) round(ipval[j]/inp_cfg.InputScaleFP);
    input[j] = fmin(input[j], pow(2, 31)-1);
    input[j] = fmax(input[j], -1*pow(2,31));
  } 

  // Fixed point Layer Normalization
  dmlrn_result = dmlrespnorm(input, lcfg);

  for(int j=0; j<vecblk_len; j++) {
    output[j] = ((double)dmlrn_result[j]-NormOutZpoint)*NormLOutScale;
  } 

#if INSTRUMENT
  printf("\n\ninput\n\n");
  for(int j=0; j<10; j++)
    printf("%f, ", ipval[j]);

  printf("\n\ngamma\n\n");
  for(int j=0; j<10; j++)
    printf("%f, ", gamma[j]);

  printf("\n\nbeta\n\n");
  for(int j=0; j<10; j++)
    printf("%f, ", beta[j]);

  printf("\n\nquantized input\n\n");
  for(int j=0; j<10; j++)
    printf("%d, ", input[j]);

  printf("\n\nquantized output\n\n");
  for(int j=0; j<10; j++)
    printf("%d, ", dmlrn_result[j]);

  printf("\n\noutput\n\n");
  for(int j=0; j<10; j++)
    printf("%f, ", output[j]);
#endif

  return output;
}


// } //end extern "C"

