#include "mtlib.h"

// Max value is returned from array of "size"=size
double maxarray(double *inp, int size) { 
int    i; 
double max = pow(2.0, -1*pow(2,31));

  for(i=0; i<size; i++) 
    if(inp[i]>max) max = inp[i];

  return max; 
}
// Set basic parameters related to input as well as IMC OUT
// VGP - Accept precision as parameter
void set_input_cfg (input_cfg *cfg, int maxX, int imcoutbits) {

  cfg->imcout_nrbits = imcoutbits;
  cfg->InputScaleFP = maxX/pow(2.0, (cfg->imcout_nrbits-1));
  mapfloattoHWformat(cfg->InputScaleFP, 1, 0.00001, &(cfg->InputScale32bHW), &(cfg->nmbredecimalsbelowcomma));
  //printf(" \n SET INPUT CFG - \n");
  //printf( "imcout_nrbits = %d, InputScale32bHW =%d, nmbredecimalsbelowcomma=%d , InputScaleFP=%lf \n", cfg->imcout_nrbits, cfg->InputScale32bHW, cfg->nmbredecimalsbelowcomma, cfg->InputScaleFP);

return;
}

// Set LinLUT cfg based on min and max values
// VGP - Accept precision as parameter
void set_linlut_cfg(linlut_cfg *cfg, int minX, int maxX, int inv) {

  if(inv == 1) { 
    cfg->m = minX;
    cfg->OutDelta = (1.0/minX)/256;
  } else {   // only for exp
    cfg->m = minX - maxX;
    cfg->OutDelta = 1.0/256;
  }

  cfg->addr_nbit      = 7;
  cfg->OutIsSigned    = 0; 
  cfg->OutZeroPoint   = 0;
  cfg->nbrbitperentry = 10; 

  cfg->log2span = ceil(log2(maxX - minX));
  cfg->step     = pow(2, (cfg->log2span - cfg->addr_nbit));

  mapfloattoHWformat(cfg->m, 1, 0.0001, &(cfg->m_num), &(cfg->m_log2denum));
 
  //printf(" \n SET LINLUT CFG - \n");
  //printf(" m=%f, outdelta=%f, log2span=%f, step=%f m_num=%d \n", cfg->m, cfg->OutDelta, cfg->log2span, cfg->step, cfg->m_num); 
  //printf(" addr_nbit=%d, m_log2denum=%d, OutIsSignedOutIsSigned=%d, OutZeroPoint=%d, nbrbitperentry=%d \n", cfg->addr_nbit, cfg->m_log2denum, cfg->OutIsSigned, cfg->OutZeroPoint, cfg->nbrbitperentry);
return; 
}

// Translate Float data ito H/W i.e. fixed format
void mapfloattoHWformat(float m, int issigned, float precision, int *num, int *m_log2denum) {
double d;
int    n = 0 ;
int    goodenough = 0;

  while((goodenough==0)&&(n<32)) {
    d = round(m*pow(2.0, n));
    if (issigned == 0) {
      d = fmin(d, pow(2, 32)-1);
      d = fmax(d, 0);
    } else {
      d = fmin(d, pow(2, 31)-1);
      d = fmax(d, -1*pow(2,31));
    }
    if(fabs(m-d/(pow(2, n))) < precision)
        goodenough = 1; 
    else
        n = n+1;
  }
  *num = d;
  *m_log2denum = n;

return;
}

// Floating point to Fixed point - Quantize function
// VGP - Implement % (mod) function for wrap
double FPQuantize(double fpval, int a, int b, char flag[3]){ 
double qval; 

  switch (flag[0]) {
    case 't':
        qval = floor(fpval*pow(2,b))/pow(2,b);
        break;
    case 'r':
        if (fpval>=0)
          qval=round(fpval*pow(2,b))/pow(2,b);
        else
          qval=floor(fpval*pow(2,b)+0.5)/pow(2,b);
        break;
     
    case 's':
        qval = fmin(fpval, (pow(2,(a-1-b))-(pow(2,-1*b))));
        qval = fmax(qval, -1*pow(2,(a-1-b)));
        break;
    case 'w': 
        printf("wrap Flag");
        break;
        //qval = (((fpval*pow(2,b) + pow(2,(a-1)))%pow(2,a))-pow(2,(a-1)))/pow(2,b);
    default:
        printf("Invalid Flag");
        break;
  }
  return qval;
} 

// Calculate mnum_precal based on linlut cfg
// Used in Index and Mu generation function
double get_mnum_precal(linlut_cfg cfg) { 
double mnum_precal; 

  // prec=-m_num/2^(p+m_log2denum);
  mnum_precal = -1*(cfg.m_num/(pow(2.0, cfg.log2span+cfg.m_log2denum))); 
  // prec=FPQuantize(prec,45,13,'tr');
  mnum_precal = FPQuantize(mnum_precal, 45, 13, "tr ");

return mnum_precal;
}

// Index generation for LUT path of non-linear function
void idx_gen(double *scaled_vec, int size, input_cfg icfg, linlut_cfg lcfg, igen_op *idx_mu) {

int     vec;
int     shift;

double  shresult;
double  igen_sum;

double  mnum_precal = get_mnum_precal(lcfg);

  shift = lcfg.log2span + icfg.nmbredecimalsbelowcomma;
  for(vec = 0; vec<size; vec++) {
    shresult = scaled_vec[vec]/(double) pow(2.0,shift);
    shresult = FPQuantize(shresult, 46, 13, "tr ");
    shresult = FPQuantize(shresult, 46, 13, "sat ");

    igen_sum = shresult + mnum_precal;
    igen_sum = FPQuantize(igen_sum, 13, 12, "tr ");
    igen_sum = fmax(igen_sum, 0);
    igen_sum = fmin(igen_sum, 127/128.0);

    idx_mu[vec].index = (int) floor(igen_sum*128);
    idx_mu[vec].mu    = (double) (igen_sum*128 - idx_mu[vec].index);
    //if(vec < 16)
    //  printf(" shresult=%f The mnum_precal was  =%f  index = %d and Mu = %f \n", shresult, mnum_precal, idx_mu[vec].index, idx_mu[vec].mu);
  }
return;
}

// Interpolate the LUT values of non-linear function 
void  intrp_lut(igen_op *idx_mu, int size, int lutout_unsigned, float LUT[128], int *op_vec){
int i; 
double lut_outcode1;
double lut_outcode2;
double lutdiff_xmu;
double intrp_val;                                                                                                                               
double intrp_rnd;

  for(i=0; i<size; i++) {
    lut_outcode1 = LUT[idx_mu[i].index];
    lut_outcode2 = LUT[(int) fmin((idx_mu[i].index+1), 127)];
    lutdiff_xmu   = FPQuantize(idx_mu[i].mu*(lut_outcode2-lut_outcode1), 12, 2, "tr ");
    intrp_val    = lut_outcode1 + lutdiff_xmu;
    intrp_rnd    = FPQuantize(intrp_val, 10, 0, "rnd");

    if(lutout_unsigned == 1) {
      op_vec[i] = fmin(intrp_rnd, 255);
      op_vec[i] = fmax(op_vec[i], 0);
    } else {
      op_vec[i] = fmin(intrp_rnd, 127);
      op_vec[i] = fmax(op_vec[i], -128);
    }
    //if(i<4) printf(" The index = %d and mu=%lf op_vec= %d\n", idx_mu[i].index, idx_mu[i].mu, op_vec[i]); 
  }
return;
}

// Generate Lin LUT. It also supports only exp(x-lambda) for softmax
// VGP - Add remaining functions such as GELU etc
void GenerateLinLUT(linlut_cfg lin_lut_config, float lambda, int func_id, float *LUT){

int    i; 
double func_value; 
double value; 
double code; 

  //printf(" index value final = %lf and %lf ", pow(2, lin_lut_config.addr_nbit), pow(2, lin_lut_config.addr_nbit)-1);
  for (i=0; i<(int) pow(2, lin_lut_config.addr_nbit); i++) { 
    LUT[i] = 0;
    value = (lin_lut_config.m_num/pow(2, lin_lut_config.m_log2denum)) + (i*lin_lut_config.step) + (2*lin_lut_config.step)/2;
    switch (func_id) {
      case 0: 
        func_value = exp(value - lambda); 
        break;
      case 1: 
        func_value = 1/value; 
        break;
      case 2: 
        func_value = exp(-1*value); 
        break;
      case 3: 
        func_value = 0.5*value*(1+erf(value/sqrt(2)));
        break;
    }

    // Gelu uses following code which looks exactly same hence it should worK 
    // code=round((func_value/delta + Zp)*2^(lin_lut_config.nbrbitperentry-8))*2^-(lin_lut_config.nbrbitperentry-8);
    // 10 and 2 in FPQuantize call should be replaced by 
    // lin_lut_config.nbrbitperentry in place of 10 and 2 = lin_lut_config.nbrbitperentry-8
    code = FPQuantize((func_value/lin_lut_config.OutDelta + lin_lut_config.OutZeroPoint), 10, 2, "rnd");

    //if(i==127) printf(" \n Code = %lf because func_value=%lf and value=%lf \n ", code, func_value, value);
    if(lin_lut_config.OutIsSigned == 0){
        code = fmin(code,255);
        code = fmax(code,0);
    } else {
        code = fmin(code,127);
        code = fmax(code,-128);
    } 
    LUT[i] = (float) code;
  }

return;
}

int quantize8b(float value, float delta, float zerop, int OutIsSigned){
int   qvalue; 
float fvalue; 

  fvalue = round(value/delta + zerop); 
  if(OutIsSigned == 0){
    qvalue = fmin(fvalue,255);
    qvalue = fmax(fvalue,0);
  } else {
    qvalue = fmin(fvalue,127);
    qvalue = fmax(fvalue,-128);
  }

return qvalue; 
}

// Statistical mean is calculated from array of "size"=size
float get_statmean(float *ipval, int size){ 
float mean; 

double sum; 
int i; 

  sum = 0;
  for(i=0; i<size; i++) sum += ipval[i];
  mean = sum/size;
return mean;
}

//Mini Batch variance is calculated based on mean and array of "size"=size
double get_mbvariance(float *ipval, int size, float mean) { 
double mbvar;
 
int    i; 

  mbvar = 0;
  for(i=0; i<size; i++) mbvar += ipval[i]*ipval[i];
  mbvar /= size; 
  mbvar -= (mean*mean);

return mbvar; 
} 

// Expect the beta, gamma and  epsilon to be set/provided as parameters
void set_lrn_cfg(lrn_cfg *cfg, double *beta, double *vgamma, double epsilon, int size) { 
int i; 

  for(i=0; i<size; i++){
    cfg->beta[i]    = beta[i]; 
    cfg->vgamma[i]  = vgamma[i]; 
    cfg->epsilon    = epsilon;
  } 

return; 
}

