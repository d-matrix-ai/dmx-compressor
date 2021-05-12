#include "softmax.h"
#include <time.h>
#include <pthread.h>

double   natlog_unit(double f_sum);
double * exp_unit(int *inp, double lnF, int inpShift, int pass_select, int FractionalBits, double* ey);


typedef struct {
  double *in;
  int *out;
  input_cfg *inp_cfg;
} smdata_struct;


//Ideal Softmax function
void idsoftmax(double *inputFP, int *ideal8bsmax) { 

  double idsftmax[vecblk_len];
  double sum_idsftmax; 

  int    i;

  sum_idsftmax = 0;
  for(i=0; i<vecblk_len; i++) {
    idsftmax[i] = exp(inputFP[i]);
    sum_idsftmax += idsftmax[i]; 
  }
  
  for(i=0; i<vecblk_len; i++) { 
    idsftmax[i] = idsftmax[i]/sum_idsftmax; 
    ideal8bsmax[i] = FPQuantize(idsftmax[i]*255, 8, 0, "rnd");
    //if(i<8) printf(" Ideal values are %d \n", ideal8bsmax[i]); 
  }
}

//D-Matrix softmax function2
//int * dmsoftmax(int *input){
void* dmsoftmax_impl(smdata_struct* smdata){

  double   ey1[vecblk_len];
  double   ey2[vecblk_len];
  double   sum = 0; 
  double   inputFP[vecblk_len];
  int      tmp[vecblk_len];

  for(int j=0; j<vecblk_len; j++) {
    tmp[j] = (int) round(smdata->in[j]/smdata->inp_cfg->InputScaleFP);
    tmp[j] = fmin(tmp[j], pow(2, 31)-1);
    tmp[j] = fmax(tmp[j], -1*pow(2,31));
    tmp[j] = tmp[j]*smdata->inp_cfg->InputScale32bHW; 
    inputFP[j] = tmp[j]/pow(2, smdata->inp_cfg->nmbredecimalsbelowcomma); 
//    printf("%f %f\n", smdata->in[j], inputFP[j]);
  } 

  int input[vecblk_len];

  //This is the quantization scale for the final softmax output. Can be optimized based on data.
  int      QuantizationScaler=243;
  int      i;
  for(i=0; i<vecblk_len; i++) input[i] = (int) (inputFP[i]*pow(2, 7));
  exp_unit(input, 0, 7, 1, 4, ey1); 
  for(i=0; i<vecblk_len; i++) sum += ey1[i];
  exp_unit(input, natlog_unit(sum), 7, 2, 4, ey2);
  for(i=0; i<vecblk_len; i++){ 
    smdata->out[i] = FPQuantize(ey2[i]*QuantizationScaler, 8, 0, "rnd");
    smdata->out[i] = FPQuantize(smdata->out[i], 9, 0, "sat");
  }

  //int sqerr = 0; 
  //int ideal8bsmax[vecblk_len];

  //idsoftmax(inputFP, ideal8bsmax);
  //for(int j=0; j<vecblk_len; j++) {
  //  sqerr += pow((smdata->out[j] - ideal8bsmax[j]), 2); };

  //double mse = (sqerr*1.0)/vecblk_len;
  //printf(" The MSE is %f with SqErr=%d\n", mse, sqerr);


return 0;
}

// Natural Logaritmic unit 
double natlog_unit(double f_sum) { 
double LnF; 
double ln2, wexp, kval; 
  
  ln2  = 0.6875;   // Approximation of binary(loge 2)
  wexp = floor(log2(f_sum));
  kval = f_sum/pow(2, wexp);
  LnF  = ln2*(wexp + kval -1);

return LnF; 
}

// Exponential Unit implementation 
// inpShift is input scaling right shift
// pass_select differentiates 1st and 2nd call
// FractionalBits is no of bits below comma (used for truncation) 
double * exp_unit(int *inp, double lnF, int inpShift, int pass_select, int FractionalBits, double* ey){ 

  double log2e, d1_d2; 
  double inpflt[vecblk_len]; 
  double y2[vecblk_len]; 
  double z[vecblk_len]; 
  double u_int[vecblk_len]; 
  double v_frac[vecblk_len]; 
  double two_pow_v[vecblk_len]; 
  double two_pow_u[vecblk_len]; 
  double mx; 
  int i;

  // Currently maxarray function supports only float number
  for (i=0; i<vecblk_len; i++) inpflt[i] = (double) inp[i];
  mx = maxarray(inpflt, vecblk_len);

  // Log2e is binary 1.0111 i.e. 1+0.25+0.125+0.0625
  log2e = 1.4375;    

  // Based on pass_select, 0.94 is optimal solution for linearizing =2^x for 0<x<1 
  d1_d2 = 0.94;        // Should be 0.95?

  for(i=0; i<vecblk_len; i++){ 
    //y2[i] = inpint[i]/pow(2, inpShift);

    y2[i] = (inp[i] - (int) mx)/pow(2, inpShift);
    y2[i] = floor(y2[i]*pow(2, FractionalBits))/pow(2, FractionalBits);
    z[i]  = (y2[i]-lnF)*log2e; 

    u_int[i]  = floor(z[i]); 
    v_frac[i] = z[i] - u_int[i];

    two_pow_v[i] = v_frac[i] + d1_d2; 
    two_pow_u[i] = pow(2, u_int[i]);

    ey[i]        = two_pow_u[i]*two_pow_v[i];
  }

return ey; 
}

// extern "C" {

int * dmsoftmax(double *x32b) {

  static int dmtrx8bsmax[vecblk_len*vecblk_len];

  pthread_t the_threads[vecblk_len];
  smdata_struct inputs[vecblk_len];

  int    MaxData       = 16;
  int    imcout_nrbits = 12;

  input_cfg  inp_cfg;

  set_input_cfg (&inp_cfg, MaxData, imcout_nrbits);

  for(int batch=0 ; batch<vecblk_len ; ++batch) {

    smdata_struct *inp = &inputs[batch];
    inp->in = &x32b[batch*vecblk_len];
    inp->out = &dmtrx8bsmax[batch*vecblk_len];
    inp->inp_cfg = &inp_cfg;

    pthread_create(&the_threads[batch], NULL, dmsoftmax_impl, inp);
  }

  for(int batch=0 ; batch<vecblk_len ; ++batch) {
      pthread_join(the_threads[batch], NULL);
  }
  return dmtrx8bsmax;
}

// } //end extern "C"
