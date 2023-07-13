/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) fcost_gradient_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[59] = {55, 1, 0, 55, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
static const casadi_int casadi_s1[113] = {1, 55, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* fcost_gradient:(i0[55])->(o0[1x55]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6;
    a0=1.0000000000000000e-02;
    a1=arg[0] ? arg[0][0] : 0;
    a2=(a1+a1);
    a2=(a0*a2);
    a1=(a1+a1);
    a2=(a2+a1);
    if (res[0]!=0) res[0][0]=a2;
    a2=arg[0] ? arg[0][1] : 0;
    a1=(a2+a2);
    a1=(a0*a1);
    a2=(a2+a2);
    a1=(a1+a2);
    if (res[0]!=0) res[0][1]=a1;
    a1=arg[0] ? arg[0][2] : 0;
    a2=(a1+a1);
    a2=(a0*a2);
    a1=(a1+a1);
    a2=(a2+a1);
    if (res[0]!=0) res[0][2]=a2;
    a2=9.0185760300002804e-02;
    a1=arg[0] ? arg[0][3] : 0;
    a1=(a1+a1);
    a1=(a2*a1);
    if (res[0]!=0) res[0][3]=a1;
    a1=arg[0] ? arg[0][4] : 0;
    a1=(a1+a1);
    a1=(a2*a1);
    if (res[0]!=0) res[0][4]=a1;
    a1=arg[0] ? arg[0][5] : 0;
    a1=(a1+a1);
    a1=(a2*a1);
    if (res[0]!=0) res[0][5]=a1;
    a1=1.4981423969999721e-01;
    a3=arg[0] ? arg[0][6] : 0;
    a3=(a3+a3);
    a3=(a1*a3);
    if (res[0]!=0) res[0][6]=a3;
    a3=arg[0] ? arg[0][7] : 0;
    a3=(a3+a3);
    a3=(a1*a3);
    if (res[0]!=0) res[0][7]=a3;
    a3=arg[0] ? arg[0][8] : 0;
    a3=(a3+a3);
    a3=(a1*a3);
    if (res[0]!=0) res[0][8]=a3;
    a3=1.4981423969999719e-01;
    a4=arg[0] ? arg[0][9] : 0;
    a4=(a4+a4);
    a4=(a3*a4);
    if (res[0]!=0) res[0][9]=a4;
    a4=arg[0] ? arg[0][10] : 0;
    a4=(a4+a4);
    a4=(a3*a4);
    if (res[0]!=0) res[0][10]=a4;
    a4=arg[0] ? arg[0][11] : 0;
    a4=(a4+a4);
    a4=(a3*a4);
    if (res[0]!=0) res[0][11]=a4;
    a4=9.0185760300002832e-02;
    a5=arg[0] ? arg[0][12] : 0;
    a5=(a5+a5);
    a5=(a4*a5);
    if (res[0]!=0) res[0][12]=a5;
    a5=arg[0] ? arg[0][13] : 0;
    a5=(a5+a5);
    a5=(a4*a5);
    if (res[0]!=0) res[0][13]=a5;
    a5=arg[0] ? arg[0][14] : 0;
    a5=(a5+a5);
    a5=(a4*a5);
    if (res[0]!=0) res[0][14]=a5;
    a5=arg[0] ? arg[0][15] : 0;
    a6=(a5+a5);
    a6=(a0*a6);
    a5=(a5+a5);
    a5=(a0*a5);
    a6=(a6+a5);
    if (res[0]!=0) res[0][15]=a6;
    a6=arg[0] ? arg[0][16] : 0;
    a5=(a6+a6);
    a5=(a0*a5);
    a6=(a6+a6);
    a6=(a0*a6);
    a5=(a5+a6);
    if (res[0]!=0) res[0][16]=a5;
    a5=arg[0] ? arg[0][17] : 0;
    a6=(a5+a5);
    a6=(a0*a6);
    a5=(a5+a5);
    a5=(a0*a5);
    a6=(a6+a5);
    if (res[0]!=0) res[0][17]=a6;
    a6=arg[0] ? arg[0][18] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][18]=a6;
    a6=arg[0] ? arg[0][19] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][19]=a6;
    a6=arg[0] ? arg[0][20] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][20]=a6;
    a6=arg[0] ? arg[0][21] : 0;
    a6=(a6+a6);
    a6=(a1*a6);
    if (res[0]!=0) res[0][21]=a6;
    a6=arg[0] ? arg[0][22] : 0;
    a6=(a6+a6);
    a6=(a1*a6);
    if (res[0]!=0) res[0][22]=a6;
    a6=arg[0] ? arg[0][23] : 0;
    a6=(a6+a6);
    a6=(a1*a6);
    if (res[0]!=0) res[0][23]=a6;
    a6=arg[0] ? arg[0][24] : 0;
    a6=(a6+a6);
    a6=(a3*a6);
    if (res[0]!=0) res[0][24]=a6;
    a6=arg[0] ? arg[0][25] : 0;
    a6=(a6+a6);
    a6=(a3*a6);
    if (res[0]!=0) res[0][25]=a6;
    a6=arg[0] ? arg[0][26] : 0;
    a6=(a6+a6);
    a6=(a3*a6);
    if (res[0]!=0) res[0][26]=a6;
    a6=arg[0] ? arg[0][27] : 0;
    a6=(a6+a6);
    a6=(a4*a6);
    if (res[0]!=0) res[0][27]=a6;
    a6=arg[0] ? arg[0][28] : 0;
    a6=(a6+a6);
    a6=(a4*a6);
    if (res[0]!=0) res[0][28]=a6;
    a6=arg[0] ? arg[0][29] : 0;
    a6=(a6+a6);
    a6=(a4*a6);
    if (res[0]!=0) res[0][29]=a6;
    a6=arg[0] ? arg[0][30] : 0;
    a6=(a6+a6);
    a6=(a0*a6);
    if (res[0]!=0) res[0][30]=a6;
    a6=arg[0] ? arg[0][31] : 0;
    a6=(a6+a6);
    a6=(a0*a6);
    if (res[0]!=0) res[0][31]=a6;
    a6=arg[0] ? arg[0][32] : 0;
    a6=(a6+a6);
    a6=(a0*a6);
    if (res[0]!=0) res[0][32]=a6;
    a6=arg[0] ? arg[0][33] : 0;
    a6=(a6+a6);
    a6=(a0*a6);
    if (res[0]!=0) res[0][33]=a6;
    a6=arg[0] ? arg[0][34] : 0;
    a6=(a6+a6);
    a6=(a0*a6);
    if (res[0]!=0) res[0][34]=a6;
    a6=arg[0] ? arg[0][35] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][35]=a6;
    a6=arg[0] ? arg[0][36] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][36]=a6;
    a6=arg[0] ? arg[0][37] : 0;
    a6=(a6+a6);
    a6=(a1*a6);
    if (res[0]!=0) res[0][37]=a6;
    a6=arg[0] ? arg[0][38] : 0;
    a6=(a6+a6);
    a6=(a1*a6);
    if (res[0]!=0) res[0][38]=a6;
    a6=arg[0] ? arg[0][39] : 0;
    a6=(a6+a6);
    a6=(a3*a6);
    if (res[0]!=0) res[0][39]=a6;
    a6=arg[0] ? arg[0][40] : 0;
    a6=(a6+a6);
    a6=(a3*a6);
    if (res[0]!=0) res[0][40]=a6;
    a6=arg[0] ? arg[0][41] : 0;
    a6=(a6+a6);
    a6=(a4*a6);
    if (res[0]!=0) res[0][41]=a6;
    a6=arg[0] ? arg[0][42] : 0;
    a6=(a6+a6);
    a6=(a4*a6);
    if (res[0]!=0) res[0][42]=a6;
    a6=arg[0] ? arg[0][43] : 0;
    a5=(a6+a6);
    a5=(a0*a5);
    a6=(a6+a6);
    a6=(a0*a6);
    a5=(a5+a6);
    if (res[0]!=0) res[0][43]=a5;
    a5=arg[0] ? arg[0][44] : 0;
    a6=(a5+a5);
    a6=(a0*a6);
    a5=(a5+a5);
    a5=(a0*a5);
    a6=(a6+a5);
    if (res[0]!=0) res[0][44]=a6;
    a6=arg[0] ? arg[0][45] : 0;
    a6=(a6+a6);
    a6=(a2*a6);
    if (res[0]!=0) res[0][45]=a6;
    a6=arg[0] ? arg[0][46] : 0;
    a6=(a6+a6);
    a2=(a2*a6);
    if (res[0]!=0) res[0][46]=a2;
    a2=arg[0] ? arg[0][47] : 0;
    a2=(a2+a2);
    a2=(a1*a2);
    if (res[0]!=0) res[0][47]=a2;
    a2=arg[0] ? arg[0][48] : 0;
    a2=(a2+a2);
    a1=(a1*a2);
    if (res[0]!=0) res[0][48]=a1;
    a1=arg[0] ? arg[0][49] : 0;
    a1=(a1+a1);
    a1=(a3*a1);
    if (res[0]!=0) res[0][49]=a1;
    a1=arg[0] ? arg[0][50] : 0;
    a1=(a1+a1);
    a3=(a3*a1);
    if (res[0]!=0) res[0][50]=a3;
    a3=arg[0] ? arg[0][51] : 0;
    a3=(a3+a3);
    a3=(a4*a3);
    if (res[0]!=0) res[0][51]=a3;
    a3=arg[0] ? arg[0][52] : 0;
    a3=(a3+a3);
    a4=(a4*a3);
    if (res[0]!=0) res[0][52]=a4;
    a4=arg[0] ? arg[0][53] : 0;
    a4=(a4+a4);
    a4=(a0*a4);
    if (res[0]!=0) res[0][53]=a4;
    a4=arg[0] ? arg[0][54] : 0;
    a4=(a4+a4);
    a0=(a0*a4);
    if (res[0]!=0) res[0][54]=a0;
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int fcost_gradient(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
    return casadi_f0(arg, res, iw, w, mem);
}

extern "C" CASADI_SYMBOL_EXPORT void fcost_gradient_incref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT void fcost_gradient_decref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT casadi_int fcost_gradient_n_in(void) { return 1;}

extern "C" CASADI_SYMBOL_EXPORT casadi_int fcost_gradient_n_out(void) { return 1;}

extern "C" CASADI_SYMBOL_EXPORT const char* fcost_gradient_name_in(casadi_int i){
    switch (i) {
        case 0: return "i0";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* fcost_gradient_name_out(casadi_int i){
    switch (i) {
        case 0: return "o0";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* fcost_gradient_sparsity_in(casadi_int i) {
    switch (i) {
        case 0: return casadi_s0;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* fcost_gradient_sparsity_out(casadi_int i) {
    switch (i) {
        case 0: return casadi_s1;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int fcost_gradient_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 1;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

