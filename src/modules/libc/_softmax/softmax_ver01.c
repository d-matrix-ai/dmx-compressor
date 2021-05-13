#include "softmax.h"                                                                                                                                                 
int * softmax(double *x32b, input_cfg cfg);
int * dmsoftmax(double *x32b, input_cfg cfg, linlut_cfg ecfg, linlut_cfg icfg);

int main(void) { 

int    MaxData = 8;
int    MinData = -0;

int    MinSum = 255;
int    MaxSum = 4096;

int    i;
int    *dmtrx8bsmax;
int    *ideal8bsmax;

input_cfg  inp_cfg; 
linlut_cfg exp_linlut_cfg;
linlut_cfg inv_linlut_cfg;

  set_input_cfg (&inp_cfg, MaxData);
  set_linlut_cfg(&exp_linlut_cfg, MinData, MaxData, 0); // inv=0 means exp linlut  
  set_linlut_cfg(&inv_linlut_cfg, MinSum, MaxSum, 1);   // inv=1 means inv linlut  

  GenerateLinLUT(exp_linlut_cfg, 0, 0, ExpLUT);
  GenerateLinLUT(inv_linlut_cfg, 0, 1, InvLUT);

  ideal8bsmax = softmax(x32b, inp_cfg);   
  dmtrx8bsmax = dmsoftmax(x32b, inp_cfg, exp_linlut_cfg, inv_linlut_cfg);

  printf(" \n FINAL COMPARISON \n");
  for(i=0; i<128; i++) if (i<16) printf(" ideal8bsmax[%0d] = %d,  dmtrx8bsmax[%0d] = %d \n", i, ideal8bsmax[i], i, dmtrx8bsmax[i]);
 
return;
}

//Ideal Softmax function
int * softmax(double *x32b, input_cfg cfg) { 

static int ideal8bsmax[vecblk_len];

double input[vecblk_len]; 
double inputFP[vecblk_len];
double idsftmax[vecblk_len];
double sum_idsftmax; 

int    i;

  sum_idsftmax = 0;
  for(i=0; i<vecblk_len; i++) {
    // Read out the stim file .. Currently hardcoded input data
    input[i] = x32b[i]*cfg.InputScale32bHW; 
    inputFP[i] = input[i]*(pow(2, (-1*cfg.nmbredecimalsbelowcomma))); 
    idsftmax[i] = exp(inputFP[i]);
    sum_idsftmax += idsftmax[i]; 
  }
  
  for(i=0; i<vecblk_len; i++) { 
    idsftmax[i] = idsftmax[i]/sum_idsftmax; 
    ideal8bsmax[i] = FPQuantize(idsftmax[i]*255, 8, 0, "rnd");
  }
  return ideal8bsmax;
}

int * dmsoftmax(double *x32b, input_cfg cfg, linlut_cfg ecfg, linlut_cfg icfg){ 
int    i;
double mx; 
double input[vecblk_len]; 
igen_op idx_mu[vecblk_len];
igen_op inv_idx_mu;
int    exp_vec[vecblk_len];
long    act_vec[vecblk_len];
int    inv_val;
double sumexp = 0;
static int    dmtrx8bsmax[vecblk_len];

  // Read out the stim file .. Currently hardcoded input data
  for(i=0; i<vecblk_len; i++)
    input[i] = x32b[i]*cfg.InputScale32bHW;

  mx = maxarray(input, vecblk_len);
  for(i=0; i<vecblk_len; i++) 
    input[i] -= mx;

  idx_gen(input, 128, cfg, ecfg, idx_mu); 
  intrp_lut(idx_mu, 128, 1, ExpLUT, exp_vec);

  for(i=0; i<128; i++) sumexp += exp_vec[i];
  printf(" The sum is %ld ", sumexp);

  cfg.nmbredecimalsbelowcomma = 0;
  idx_gen(&sumexp, 1, cfg, icfg, &inv_idx_mu);
  intrp_lut(&inv_idx_mu, 1, 1, InvLUT, &inv_val);

  //LS_Scale=(mean(mean((FinalNHSoftMaxOut(I).*Ideal8bSoftMax(I))))/mean(mean(Ideal8bSoftMax(I).^2))); optimal scale
  //Equated to 230.0743  ... Should be evaluated in s/w domain 8b*8b and quantize back to 8 bit
  for(i=0; i<128; i++) {
  act_vec[i] = exp_vec[i]*(inv_val - icfg.OutZeroPoint);
  dmtrx8bsmax[i] = FPQuantize(act_vec[i]/230.0743, 8, 0, "tr ");
  //if(i<16) printf("  value[%0d] = %ld i.e. dmtrx8bsmax = %d \n ", i, act_vec[i], dmtrx8bsmax[i]);
  }
  
return dmtrx8bsmax; 
}

