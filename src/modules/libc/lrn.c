#include "lrn.h"                                                                                                                                                 
// double * lrespnorm(double *x32b, int *gamma, int *beta, int epsilon, float *vmean, float *mbvar, float *rmsval, float *normx);
// void set_lrn_cfg(lrn_cfg *cfg, int beta, int gamma, int epsilon);

#if 0
int main(void) { 
                                                                                                                                                                        
double   *lrn_result;

int      skip[1024]; 
int      bias; 
int      beta; 
int      gamma; 
int      epsilon; 

lrn_cfg  lcfg; 
int      i;
double   inpvalue[1024];

// Debug values
float    mean; 
float    mbvar; 
float    rmsval; 
float    normx[vecblk_len]; 

  // Required parameters beta, gamma, and epsilon
  // Populate x32b vector (of size 1024), with meaningful input values 
  // Populate skip vector (of size 1024) and bias with meaningful values.

  beta    = rand()%256; gamma = rand()%256; 
  epsilon = rand()%256; bias = 0; //bias  = rand()%256;

  for(i=0; i<vecblk_len; i++) x32b[i] = rand()%256;
  for(i=0; i<vecblk_len; i++) skip[i] = 0;  // skip[i] = rand()%256;

  // Set the LRN configution specifically beta, gamma, and epsilon
  set_lrn_cfg (&lcfg, beta, gamma, epsilon);   

  for(i=0; i<vecblk_len; i++) inpvalue[i] = x32b[i] + skip[i];
  for(i=0; i<vecblk_len; i++) inpvalue[i] += bias; 

  lrn_result = lrespnorm(inpvalue, lcfg, &mean, &mbvar, &rmsval, normx);   

  printf(" \n FINAL RESULT \n\n");
  printf(" Assumption - Beta=%d, Gamma=%d, and Epsilon=%d \n", lcfg.beta, lcfg.gamma, lcfg.epsilon);
  printf(" Batch specific results - Mean=%f, MBVariance=%f, RMS=%f \n\n", mean, mbvar, rmsval);
  for(i=0; i<vecblk_len; i++) if (i<16) printf(" Yi[%0d] = %f,  Normalized-Xi[%0d]=%f, Xi[%0d] = %f \n", i, lrn_result[i], i, normx[i], i, x32b[i]);
 
return;
}
#endif

// Local Response Normalization
double * lrespnorm(float *ipval, int *gamma, int *beta, int epsilon, float *vmean, float *mbvar, float *rmsval, float *normx) { 

static double lrn_res[vecblk_len];

int    i;
double value[vecblk_len];

  *vmean  = get_statmean(ipval, vecblk_len);
  //printf("MEAN %f\n", *vmean);
  *mbvar  = get_mbvariance(ipval, vecblk_len, *vmean); 
  //printf("VAR %f\n", *mbvar);
  *rmsval = sqrt((*mbvar)+epsilon);
  //printf("RMS %f\n", *rmsval);

  for(i=0; i<vecblk_len; i++) { 
    normx[i]   = (ipval[i]-*vmean)/(*rmsval);
    lrn_res[i] = normx[i]*gamma[i] + beta[i];
  }
  //printf("[%f %f %d %d]", normx[0], lrn_res[0], gamma[0], beta[0]);
  //printf("\n\n\n");
  return lrn_res;
}

int      first = 1;
lrn_cfg  lcfg; 

// Debug values
float    mean; 
float    mbvar; 
float    rmsval; 
float    normx[vecblk_len]; 

// extern "C" {

double * layernorm(double *ipval, int *gamma, int *beta, int epsilon) { 

  // Set the LRN configution specifically beta, gamma, and epsilon
  // set_lrn_cfg (&lcfg, gamma, beta, epsilon);

  return lrespnorm(ipval, gamma, beta, epsilon, &mean, &mbvar, &rmsval, normx);

}


// } //end extern "C"
