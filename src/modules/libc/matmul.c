// Experiments with different precisions for quantized matmul

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
typedef unsigned char uchar;
typedef unsigned short ushort;
// x: M x K, W: K x N
#define M 384 // input
#define N 384 // output
#define K 64//1024
#define ARANGE 50.0
#define BRANGE 3.0
#define M2 384
#define N2 64
#define K2 1024

#define M3 384
#define N3 64
#define K3 384

// keep range small to avoid breaking scale factor logic hack

void fp32matmul(float *A, float *B, float *C, int m, int n, int k);

//void dmmatmul(float *A, float *B, float *C);

// void bfpmatmul(char *A, char *B, int *C, int scaleA, int scaleB, int zeroA,
//                int *scaleC, int *zeroC,
//                int m, int n, int k, bool returnBFP);

void quantmat(float *A, char *B, int *scaleA, int *zeroA, int m, int n);

float matmse(float *A, int *B, int scale, int zero, int m, int n);

#if 0
int main()
{
    float A[M][K]; char qA[M][K];
    float B[K][N]; char qB[K][N];
    float C[M][N]; int qC[M][N];
    int i, j;
    int scaleA, scaleB;
    int zeroA, zeroB;
    int scaleC, zeroC;
    float sqerr;
    
    srand(time(NULL));
    
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
        {
            A[i][j] = ARANGE * (float) rand() / (float) RAND_MAX;
        }
    }
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
        {
            B[i][j] = BRANGE * (float) rand() / (float) RAND_MAX;
        }
    }
    
    fp32matmul(A[0], B[0], C[0], M, N, K);
    
    printf("\nA matrix:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
            printf("%f\t", A[i][j]);
        printf("\n");
    }

    printf("\nB matrix:\n");
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
            printf("%f\t", B[i][j]);
        printf("\n");
    }

    printf("\nC matrix:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%f\t", C[i][j]);
        printf("\n");
    }
    
    
    dmmatmul(A[0], B[0], C[0]);
    printf("\nC matrix after DMMM:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%f\t", C[i][j]);
        printf("\n");
    }


    /*
    quantmat(A[0], qA[0], &scaleA, &zeroA, M, K);
    quantmat(B[0], qB[0], &scaleB, &zeroB, K, N);
    
    printf("\nScale factors %d, %d\n", scaleA, scaleB);
    printf("\nqA matrix:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
            printf("%d\t", qA[i][j]);
        printf("\n");
    }
    printf("\nqB matrix:\n");
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d\t", qB[i][j]);
        printf("\n");
    }
    
    // get result in full INT32
    bfpmatmul(qA[0], qB[0], qC[0], scaleA, scaleB, 0, &scaleC, &zeroC, M, N, K, false);
    
    printf("\nqC matrix at scale %d:\n", scaleC);
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d\t", qC[i][j]);
        printf("\n");
    }
    
    printf("\nqC deq matrix:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%f\t", (float)qC[i][j] / (float)scaleC);
        printf("\n");
    }
    sqerr = matmse(C[0], qC[0], scaleC, zeroC, M, N);
    printf("\nFP32 vs INT32: %f\n", sqerr);
    
    // get result in BFP 8.8
    bfpmatmul(qA[0], qB[0], qC[0], scaleA, scaleB, 0, &scaleC, &zeroC, M, N, K, true);
    
    printf("\nqC BFP matrix at scale %d:\n", scaleC);
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d\t", qC[i][j]);
        printf("\n");
    }
    
    printf("\nqC BFP deq matrix:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%f\t", (float)qC[i][j]);
        printf("\n");
    }
    
    sqerr = matmse(C[0], qC[0], 1, zeroC, M, N);
    printf("\nFP32 vs BFP16: %f\n", sqerr);
    */
    
    printf("All done\n");
    return 0;
}
#endif

void fp32matmul(float *A, float *B, float *C, int m, int n, int k)
{
    int x, y, z;
    for (x = 0; x < m; x++)
        for (y = 0; y < n; y++)
            *(C + x*m + y) = 0;
    
    for (x = 0; x < m; x++)
    {
        for (y = 0; y < k; y++)
        {
            for (z = 0; z < n; z++)
            {
                *(C + x*m + z) += *(A + x*m + y) * *(B + y*k + z);
            }
        }
    }
}


void quantmat(float *A, char *B, int *scaleA, int *zeroA, int m, int n)
{
    float min, max, fscale, fzero;
    int i, j;
    min = *(A+0);
    max = *(A+0);
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (min > *(A + i*m + j))
                min = *(A + i*m + j);
            if (max < *(A + i*m + j))
                max = *(A + i*m + j);
        }
    }
    
    fscale = 256.0 / (max - 0);
    fzero = 0.0; //Ignore zero point since values are in [0, RANGE] and unsigned
    *zeroA = 0;
    *scaleA = (int) fscale;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(B + i*m + j) = (char) ((float) *scaleA * *(A + i*m + j));
        }
    }
    
}

void doublequantmat(double *A, char *B, int *scaleA, int *zeroA, int m, int n)
{
    float min, max, fscale, fzero;
    int i, j;
    min = *(A+0);
    max = *(A+0);
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (min > *(A + i*n + j))
                min = *(A + i*n + j);
            if (max < *(A + i*n + j))
                max = *(A + i*n + j);
        }
    }
    
    if (min == max) max = min + 1;
    
    fscale = 255.0 / (max - min);
    fzero = -1.0 * fscale * min;
    *zeroA = (int)(fzero + 0.5) - 128;
    *scaleA = (int) fscale;
    if (*scaleA == 0) *scaleA = 1;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(B + i*n + j) = (char) (((float) *scaleA * *(A + i*n + j)) + *zeroA);
        }
    }
    
}


void bfpmatmul(char *A, char *B, int *C, int scaleA, int scaleB, int zeroA, int zeroB,
               int *scaleC, int *zeroC, int m, int n, int k, bool returnBFP)
{
    int x, y, z;
    int cmax = 0;
    int bfpscale = 0;
    for (x = 0; x < m; x++)
        for (y = 0; y < n; y++)
            *(C + x*n + y) = 0;
    //printf("BFPMM - 1\n");
    
    for (x = 0; x < m; x++)
    {
        for (y = 0; y < k; y++)
        {
            for (z = 0; z < n; z++)
            {
                *(C + x*n + z) += (*(A + x*k + y) - zeroA) * (*(B + y*n + z) - zeroB);
                //printf("[x y z]: [%d, %d, %d]\n", x, y, z);
            }
        }
    }
    
    //printf("BFPMM - 2\n");
    *scaleC = scaleA * scaleB;

    if (returnBFP)
    {
        for (x = 0; x < m; x++)
        {
            for (y = 0; y < n; y++)
            {
                if (cmax < *(C + x*m + y))
                    cmax = *(C + x*m + y);
            }
        }
                
        while (cmax > 256)
        {
            bfpscale++;
            cmax >>= 1;
        }
        *scaleC <<= bfpscale;
        
        for (x = 0; x < m; x++)
        {
            for (y = 0; y < n; y++)
            {
                *(C + x*m + y) = *(C + x*m + y) >> bfpscale;
            }
        }
        
    }
    
    *zeroC = 0;
}

float matmse(float *A, int *B, int scale, int zero, int m, int n)
{
    int i, j;
    float sum = 0.0;
    float diff;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            diff = (float) *(B + i*m + j) / (float) scale;
            diff = *(A + i*m + j) - diff;
            sum += diff * diff;
        }
    }
    return (sum);
}

// extern "C" {
double * dmmatmul(double *A, double *B)
{
    static double C[M*N];
    char qA[M][K];
    char qB[K][N];
    int qC[M][N];
    int i, j;
    int scaleA, scaleB;
    int zeroA, zeroB;
    int scaleC, zeroC;
    
    doublequantmat(A, qA[0], &scaleA, &zeroA, M, K);
    doublequantmat(B, qB[0], &scaleB, &zeroB, K, N);
        
    // get result in full INT32
    bfpmatmul(qA[0], qB[0], qC[0], scaleA, scaleB, zeroA, zeroB, &scaleC, &zeroC, M, N, K, false);
        
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            *(C + i*N + j) = (float)qC[i][j] / (float)scaleC;
    }
        
    #if 0
    FILE* f3 = fopen("/homes/asrivastava/matrixC.dat", "a");
    fprintf(f3, "A: %d, B: %d, C: %d\n", scaleA, scaleB, scaleC);
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            fprintf(f3, "%f", *(C + i*4 + j));
        }
    }
    fprintf(f3, "\n");
    fclose(f3);
    
    FILE* f1 = fopen("/homes/asrivastava/matrixA.dat", "w");
    FILE* f2 = fopen("/homes/asrivastava/matrixB.dat", "w");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
        {
            fprintf(f1, "%f", *(A + i*K + j));
        }
    }
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(f2, "%f", *(B + i*N + j));
        }
    }
    fclose(f1);
    fclose(f2);    
    printf("DM-MAT-MUL M=%d, N=%d, K=%d\n", M, N, K);
    printf("Honk Honk: A=%f, scaleA=%d, zeroA=%d, qA=%d", *A, scaleA, zeroA, qA[0][0]);
    printf("Honk Honk: B=%f, scaleA=%d, zeroA=%d, qA=%d", *B, scaleB, zeroB, qB[0][0]);
    printf("Honk Honk: C=%f, scaleA=%d, zeroA=%d, qA=%d", *C, scaleC, zeroC, qC[0][0]);
    #endif

    return C;
}
    
double * dmmatmul2(double *A, double *B)
{
    static double C2[M2*N2];
    char qA[M2][K2];
    char qB[K2][N2];
    int qC[M2][N2];
    int i, j;
    int scaleA, scaleB;
    int zeroA, zeroB;
    int scaleC, zeroC;
    
    doublequantmat(A, qA[0], &scaleA, &zeroA, M2, K2);
    doublequantmat(B, qB[0], &scaleB, &zeroB, K2, N2);
    // get result in full INT32
    bfpmatmul(qA[0], qB[0], qC[0], scaleA, scaleB, zeroA, zeroB, &scaleC, &zeroC, M2, N2, K2, false);
        
    for (i = 0; i < M2; i++)
    {
        for (j = 0; j < N2; j++)
        {
            *(C2 + i*N2 + j) = (float)qC[i][j] / (float)scaleC;
        }
    }
    return C2;
}

double * dmmatmul3(double *A, double *B)
{
    static double C3[M3*N3];
    char qA[M3][K3];
    char qB[K3][N3];
    int qC[M3][N3];
    int i, j;
    int scaleA, scaleB;
    int zeroA, zeroB;
    int scaleC, zeroC;
    
    //printf("Cluck 1\n");
    doublequantmat(A, qA[0], &scaleA, &zeroA, M3, K3);
    doublequantmat(B, qB[0], &scaleB, &zeroB, K3, N3);
    // get result in full INT32
    
    //printf("Cluck 2\n");
    bfpmatmul(qA[0], qB[0], qC[0], scaleA, scaleB, zeroA, zeroB, &scaleC, &zeroC, M3, N3, K3, false);
        
    //printf("Cluck 4\n");
    for (i = 0; i < M3; i++)
    {
        for (j = 0; j < N3; j++)
        {
            *(C3 + i*N3 + j) = (float)qC[i][j] / (float)scaleC;
        }
    }
    //printf("Cluck 5\n");
    return C3;
}
    
// }

