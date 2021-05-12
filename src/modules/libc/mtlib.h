#include <stdio.h>
#include <math.h>

typedef struct {
  int     index;
  double  mu;
} igen_op;

typedef struct {
  float m;
  float OutDelta;
  float log2span;
  float step;

  int   m_num;
  int   addr_nbit;
  int   m_log2denum;
  int   OutIsSigned;
  int   OutZeroPoint;
  int   nbrbitperentry;
} linlut_cfg;

typedef struct {
  int    imcout_nrbits;
  int    InputScale32bHW;
  int    nmbredecimalsbelowcomma;
  double InputScaleFP; 
} input_cfg;

typedef struct {
  double    beta[1024];
  double    vgamma[1024];
  double    epsilon;
} lrn_cfg;

double maxarray(double *inp, int size);

void   set_input_cfg (input_cfg *cfg, int maxX, int imcoutbits);
void   set_linlut_cfg(linlut_cfg *cfg, int minX, int maxX, int inv);
void   mapfloattoHWformat(float m, int issigned, float precision, int *num, int *m_log2denum);
double FPQuantize(double fpval, int a, int b, char flag[3]);

double get_mnum_precal(linlut_cfg cfg);                                                                                                         
void   idx_gen(double *scaled_vec, int size, input_cfg icfg, linlut_cfg lcfg, igen_op *idx_mu);
void   intrp_lut(igen_op *idx_mu, int size, int lutout_unsigned, float LUT[128], int *op_vec);

void   GenerateLinLUT(linlut_cfg lin_lut_config, float lambda, int func_id, float *LUT);
int    quantize8b(float value, float delta, float zerop, int OutIsSigned);

double get_mbvariance(float *ipval, int size, float mean);
float  get_statmean(float *ipval, int size);
void   set_lrn_cfg(lrn_cfg *cfg, double *beta, double *vgamma, double epsilon, int size);
