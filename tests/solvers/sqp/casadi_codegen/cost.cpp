/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) fcost_ ## ID
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
#define casadi_sq CASADI_PREFIX(sq)

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
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* fcost:(i0[55])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a2, a3, a4, a5, a6, a7, a8, a9;
    a0=arg[0] ? arg[0][0] : 0;
    a1=casadi_sq(a0);
    a2=arg[0] ? arg[0][1] : 0;
    a3=casadi_sq(a2);
    a1=(a1+a3);
    a3=arg[0] ? arg[0][2] : 0;
    a4=casadi_sq(a3);
    a1=(a1+a4);
    a4=2.5000000000000000e-01;
    a5=4.0000000000000001e-02;
    a0=casadi_sq(a0);
    a2=casadi_sq(a2);
    a0=(a0+a2);
    a3=casadi_sq(a3);
    a0=(a0+a3);
    a3=arg[0] ? arg[0][33] : 0;
    a3=casadi_sq(a3);
    a2=arg[0] ? arg[0][34] : 0;
    a2=casadi_sq(a2);
    a3=(a3+a2);
    a0=(a0+a3);
    a0=(a5*a0);
    a3=3.6074304120001122e-01;
    a2=arg[0] ? arg[0][3] : 0;
    a2=casadi_sq(a2);
    a6=arg[0] ? arg[0][4] : 0;
    a6=casadi_sq(a6);
    a2=(a2+a6);
    a6=arg[0] ? arg[0][5] : 0;
    a6=casadi_sq(a6);
    a2=(a2+a6);
    a6=arg[0] ? arg[0][35] : 0;
    a6=casadi_sq(a6);
    a7=arg[0] ? arg[0][36] : 0;
    a7=casadi_sq(a7);
    a6=(a6+a7);
    a2=(a2+a6);
    a2=(a3*a2);
    a0=(a0+a2);
    a2=5.9925695879998886e-01;
    a6=arg[0] ? arg[0][6] : 0;
    a6=casadi_sq(a6);
    a7=arg[0] ? arg[0][7] : 0;
    a7=casadi_sq(a7);
    a6=(a6+a7);
    a7=arg[0] ? arg[0][8] : 0;
    a7=casadi_sq(a7);
    a6=(a6+a7);
    a7=arg[0] ? arg[0][37] : 0;
    a7=casadi_sq(a7);
    a8=arg[0] ? arg[0][38] : 0;
    a8=casadi_sq(a8);
    a7=(a7+a8);
    a6=(a6+a7);
    a6=(a2*a6);
    a0=(a0+a6);
    a6=5.9925695879998875e-01;
    a7=arg[0] ? arg[0][9] : 0;
    a7=casadi_sq(a7);
    a8=arg[0] ? arg[0][10] : 0;
    a8=casadi_sq(a8);
    a7=(a7+a8);
    a8=arg[0] ? arg[0][11] : 0;
    a8=casadi_sq(a8);
    a7=(a7+a8);
    a8=arg[0] ? arg[0][39] : 0;
    a8=casadi_sq(a8);
    a9=arg[0] ? arg[0][40] : 0;
    a9=casadi_sq(a9);
    a8=(a8+a9);
    a7=(a7+a8);
    a7=(a6*a7);
    a0=(a0+a7);
    a7=3.6074304120001133e-01;
    a8=arg[0] ? arg[0][12] : 0;
    a8=casadi_sq(a8);
    a9=arg[0] ? arg[0][13] : 0;
    a9=casadi_sq(a9);
    a8=(a8+a9);
    a9=arg[0] ? arg[0][14] : 0;
    a9=casadi_sq(a9);
    a8=(a8+a9);
    a9=arg[0] ? arg[0][41] : 0;
    a9=casadi_sq(a9);
    a10=arg[0] ? arg[0][42] : 0;
    a10=casadi_sq(a10);
    a9=(a9+a10);
    a8=(a8+a9);
    a8=(a7*a8);
    a0=(a0+a8);
    a8=arg[0] ? arg[0][15] : 0;
    a9=casadi_sq(a8);
    a10=arg[0] ? arg[0][16] : 0;
    a11=casadi_sq(a10);
    a9=(a9+a11);
    a11=arg[0] ? arg[0][17] : 0;
    a12=casadi_sq(a11);
    a9=(a9+a12);
    a12=arg[0] ? arg[0][43] : 0;
    a13=casadi_sq(a12);
    a14=arg[0] ? arg[0][44] : 0;
    a15=casadi_sq(a14);
    a13=(a13+a15);
    a9=(a9+a13);
    a9=(a5*a9);
    a0=(a0+a9);
    a0=(a4*a0);
    a8=casadi_sq(a8);
    a10=casadi_sq(a10);
    a8=(a8+a10);
    a11=casadi_sq(a11);
    a8=(a8+a11);
    a12=casadi_sq(a12);
    a14=casadi_sq(a14);
    a12=(a12+a14);
    a8=(a8+a12);
    a8=(a5*a8);
    a12=arg[0] ? arg[0][18] : 0;
    a12=casadi_sq(a12);
    a14=arg[0] ? arg[0][19] : 0;
    a14=casadi_sq(a14);
    a12=(a12+a14);
    a14=arg[0] ? arg[0][20] : 0;
    a14=casadi_sq(a14);
    a12=(a12+a14);
    a14=arg[0] ? arg[0][45] : 0;
    a14=casadi_sq(a14);
    a11=arg[0] ? arg[0][46] : 0;
    a11=casadi_sq(a11);
    a14=(a14+a11);
    a12=(a12+a14);
    a3=(a3*a12);
    a8=(a8+a3);
    a3=arg[0] ? arg[0][21] : 0;
    a3=casadi_sq(a3);
    a12=arg[0] ? arg[0][22] : 0;
    a12=casadi_sq(a12);
    a3=(a3+a12);
    a12=arg[0] ? arg[0][23] : 0;
    a12=casadi_sq(a12);
    a3=(a3+a12);
    a12=arg[0] ? arg[0][47] : 0;
    a12=casadi_sq(a12);
    a14=arg[0] ? arg[0][48] : 0;
    a14=casadi_sq(a14);
    a12=(a12+a14);
    a3=(a3+a12);
    a2=(a2*a3);
    a8=(a8+a2);
    a2=arg[0] ? arg[0][24] : 0;
    a2=casadi_sq(a2);
    a3=arg[0] ? arg[0][25] : 0;
    a3=casadi_sq(a3);
    a2=(a2+a3);
    a3=arg[0] ? arg[0][26] : 0;
    a3=casadi_sq(a3);
    a2=(a2+a3);
    a3=arg[0] ? arg[0][49] : 0;
    a3=casadi_sq(a3);
    a12=arg[0] ? arg[0][50] : 0;
    a12=casadi_sq(a12);
    a3=(a3+a12);
    a2=(a2+a3);
    a6=(a6*a2);
    a8=(a8+a6);
    a6=arg[0] ? arg[0][27] : 0;
    a6=casadi_sq(a6);
    a2=arg[0] ? arg[0][28] : 0;
    a2=casadi_sq(a2);
    a6=(a6+a2);
    a2=arg[0] ? arg[0][29] : 0;
    a2=casadi_sq(a2);
    a6=(a6+a2);
    a2=arg[0] ? arg[0][51] : 0;
    a2=casadi_sq(a2);
    a3=arg[0] ? arg[0][52] : 0;
    a3=casadi_sq(a3);
    a2=(a2+a3);
    a6=(a6+a2);
    a7=(a7*a6);
    a8=(a8+a7);
    a7=arg[0] ? arg[0][30] : 0;
    a7=casadi_sq(a7);
    a6=arg[0] ? arg[0][31] : 0;
    a6=casadi_sq(a6);
    a7=(a7+a6);
    a6=arg[0] ? arg[0][32] : 0;
    a6=casadi_sq(a6);
    a7=(a7+a6);
    a6=arg[0] ? arg[0][53] : 0;
    a6=casadi_sq(a6);
    a2=arg[0] ? arg[0][54] : 0;
    a2=casadi_sq(a2);
    a6=(a6+a2);
    a7=(a7+a6);
    a5=(a5*a7);
    a8=(a8+a5);
    a4=(a4*a8);
    a0=(a0+a4);
    a1=(a1+a0);
    if (res[0]!=0) res[0][0]=a1;
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int fcost(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
    return casadi_f0(arg, res, iw, w, mem);
}

extern "C" CASADI_SYMBOL_EXPORT void fcost_incref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT void fcost_decref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT casadi_int fcost_n_in(void) { return 1;}

extern "C" CASADI_SYMBOL_EXPORT casadi_int fcost_n_out(void) { return 1;}

extern "C" CASADI_SYMBOL_EXPORT const char* fcost_name_in(casadi_int i){
    switch (i) {
        case 0: return "i0";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* fcost_name_out(casadi_int i){
    switch (i) {
        case 0: return "o0";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* fcost_sparsity_in(casadi_int i) {
    switch (i) {
        case 0: return casadi_s0;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* fcost_sparsity_out(casadi_int i) {
    switch (i) {
        case 0: return casadi_s1;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int fcost_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 1;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

