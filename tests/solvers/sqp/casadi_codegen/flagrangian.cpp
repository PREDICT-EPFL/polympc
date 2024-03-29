/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) flagrangian_ ## ID
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
#define casadi_s2 CASADI_PREFIX(s2)
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
static const casadi_int casadi_s1[37] = {33, 1, 0, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* flagrangian:(i0[55],i1[33])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a8, a9;
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
    a6=casadi_sq(a0);
    a7=casadi_sq(a2);
    a6=(a6+a7);
    a7=casadi_sq(a3);
    a6=(a6+a7);
    a7=arg[0] ? arg[0][33] : 0;
    a8=casadi_sq(a7);
    a9=arg[0] ? arg[0][34] : 0;
    a10=casadi_sq(a9);
    a8=(a8+a10);
    a6=(a6+a8);
    a6=(a5*a6);
    a8=3.6074304120001122e-01;
    a10=arg[0] ? arg[0][3] : 0;
    a11=casadi_sq(a10);
    a12=arg[0] ? arg[0][4] : 0;
    a13=casadi_sq(a12);
    a11=(a11+a13);
    a13=arg[0] ? arg[0][5] : 0;
    a14=casadi_sq(a13);
    a11=(a11+a14);
    a14=arg[0] ? arg[0][35] : 0;
    a15=casadi_sq(a14);
    a16=arg[0] ? arg[0][36] : 0;
    a17=casadi_sq(a16);
    a15=(a15+a17);
    a11=(a11+a15);
    a11=(a8*a11);
    a6=(a6+a11);
    a11=5.9925695879998886e-01;
    a15=arg[0] ? arg[0][6] : 0;
    a17=casadi_sq(a15);
    a18=arg[0] ? arg[0][7] : 0;
    a19=casadi_sq(a18);
    a17=(a17+a19);
    a19=arg[0] ? arg[0][8] : 0;
    a20=casadi_sq(a19);
    a17=(a17+a20);
    a20=arg[0] ? arg[0][37] : 0;
    a21=casadi_sq(a20);
    a22=arg[0] ? arg[0][38] : 0;
    a23=casadi_sq(a22);
    a21=(a21+a23);
    a17=(a17+a21);
    a17=(a11*a17);
    a6=(a6+a17);
    a17=5.9925695879998875e-01;
    a21=arg[0] ? arg[0][9] : 0;
    a23=casadi_sq(a21);
    a24=arg[0] ? arg[0][10] : 0;
    a25=casadi_sq(a24);
    a23=(a23+a25);
    a25=arg[0] ? arg[0][11] : 0;
    a26=casadi_sq(a25);
    a23=(a23+a26);
    a26=arg[0] ? arg[0][39] : 0;
    a27=casadi_sq(a26);
    a28=arg[0] ? arg[0][40] : 0;
    a29=casadi_sq(a28);
    a27=(a27+a29);
    a23=(a23+a27);
    a23=(a17*a23);
    a6=(a6+a23);
    a23=3.6074304120001133e-01;
    a27=arg[0] ? arg[0][12] : 0;
    a29=casadi_sq(a27);
    a30=arg[0] ? arg[0][13] : 0;
    a31=casadi_sq(a30);
    a29=(a29+a31);
    a31=arg[0] ? arg[0][14] : 0;
    a32=casadi_sq(a31);
    a29=(a29+a32);
    a32=arg[0] ? arg[0][41] : 0;
    a33=casadi_sq(a32);
    a34=arg[0] ? arg[0][42] : 0;
    a35=casadi_sq(a34);
    a33=(a33+a35);
    a29=(a29+a33);
    a29=(a23*a29);
    a6=(a6+a29);
    a29=arg[0] ? arg[0][15] : 0;
    a33=casadi_sq(a29);
    a35=arg[0] ? arg[0][16] : 0;
    a36=casadi_sq(a35);
    a33=(a33+a36);
    a36=arg[0] ? arg[0][17] : 0;
    a37=casadi_sq(a36);
    a33=(a33+a37);
    a37=arg[0] ? arg[0][43] : 0;
    a38=casadi_sq(a37);
    a39=arg[0] ? arg[0][44] : 0;
    a40=casadi_sq(a39);
    a38=(a38+a40);
    a33=(a33+a38);
    a33=(a5*a33);
    a6=(a6+a33);
    a6=(a4*a6);
    a33=casadi_sq(a29);
    a38=casadi_sq(a35);
    a33=(a33+a38);
    a38=casadi_sq(a36);
    a33=(a33+a38);
    a38=casadi_sq(a37);
    a40=casadi_sq(a39);
    a38=(a38+a40);
    a33=(a33+a38);
    a33=(a5*a33);
    a38=arg[0] ? arg[0][18] : 0;
    a40=casadi_sq(a38);
    a41=arg[0] ? arg[0][19] : 0;
    a42=casadi_sq(a41);
    a40=(a40+a42);
    a42=arg[0] ? arg[0][20] : 0;
    a43=casadi_sq(a42);
    a40=(a40+a43);
    a43=arg[0] ? arg[0][45] : 0;
    a44=casadi_sq(a43);
    a45=arg[0] ? arg[0][46] : 0;
    a46=casadi_sq(a45);
    a44=(a44+a46);
    a40=(a40+a44);
    a8=(a8*a40);
    a33=(a33+a8);
    a8=arg[0] ? arg[0][21] : 0;
    a40=casadi_sq(a8);
    a44=arg[0] ? arg[0][22] : 0;
    a46=casadi_sq(a44);
    a40=(a40+a46);
    a46=arg[0] ? arg[0][23] : 0;
    a47=casadi_sq(a46);
    a40=(a40+a47);
    a47=arg[0] ? arg[0][47] : 0;
    a48=casadi_sq(a47);
    a49=arg[0] ? arg[0][48] : 0;
    a50=casadi_sq(a49);
    a48=(a48+a50);
    a40=(a40+a48);
    a11=(a11*a40);
    a33=(a33+a11);
    a11=arg[0] ? arg[0][24] : 0;
    a40=casadi_sq(a11);
    a48=arg[0] ? arg[0][25] : 0;
    a50=casadi_sq(a48);
    a40=(a40+a50);
    a50=arg[0] ? arg[0][26] : 0;
    a51=casadi_sq(a50);
    a40=(a40+a51);
    a51=arg[0] ? arg[0][49] : 0;
    a52=casadi_sq(a51);
    a53=arg[0] ? arg[0][50] : 0;
    a54=casadi_sq(a53);
    a52=(a52+a54);
    a40=(a40+a52);
    a17=(a17*a40);
    a33=(a33+a17);
    a17=arg[0] ? arg[0][27] : 0;
    a40=casadi_sq(a17);
    a52=arg[0] ? arg[0][28] : 0;
    a54=casadi_sq(a52);
    a40=(a40+a54);
    a54=arg[0] ? arg[0][29] : 0;
    a55=casadi_sq(a54);
    a40=(a40+a55);
    a55=arg[0] ? arg[0][51] : 0;
    a56=casadi_sq(a55);
    a57=arg[0] ? arg[0][52] : 0;
    a58=casadi_sq(a57);
    a56=(a56+a58);
    a40=(a40+a56);
    a23=(a23*a40);
    a33=(a33+a23);
    a23=arg[0] ? arg[0][30] : 0;
    a40=casadi_sq(a23);
    a56=arg[0] ? arg[0][31] : 0;
    a58=casadi_sq(a56);
    a40=(a40+a58);
    a58=arg[0] ? arg[0][32] : 0;
    a59=casadi_sq(a58);
    a40=(a40+a59);
    a59=arg[0] ? arg[0][53] : 0;
    a60=casadi_sq(a59);
    a61=arg[0] ? arg[0][54] : 0;
    a62=casadi_sq(a61);
    a60=(a60+a62);
    a40=(a40+a60);
    a5=(a5*a40);
    a33=(a33+a5);
    a33=(a4*a33);
    a6=(a6+a33);
    a1=(a1+a6);
    a6=arg[1] ? arg[1][0] : 0;
    a33=8.5000000000000000e+00;
    a5=(a33*a0);
    a40=-1.0472135954999581e+01;
    a60=(a40*a10);
    a5=(a5+a60);
    a60=2.8944271909999162e+00;
    a62=(a60*a15);
    a5=(a5+a62);
    a62=-1.5278640450004206e+00;
    a63=(a62*a21);
    a5=(a5+a63);
    a63=1.1055728090000840e+00;
    a64=(a63*a27);
    a5=(a5+a64);
    a64=-5.0000000000000000e-01;
    a65=(a64*a29);
    a5=(a5+a65);
    a65=cos(a3);
    a65=(a7*a65);
    a66=cos(a9);
    a65=(a65*a66);
    a65=(a4*a65);
    a5=(a5-a65);
    a6=(a6*a5);
    a5=arg[1] ? arg[1][1] : 0;
    a65=(a33*a2);
    a66=(a40*a12);
    a65=(a65+a66);
    a66=(a60*a18);
    a65=(a65+a66);
    a66=(a62*a24);
    a65=(a65+a66);
    a66=(a63*a30);
    a65=(a65+a66);
    a66=(a64*a35);
    a65=(a65+a66);
    a66=sin(a3);
    a66=(a7*a66);
    a67=cos(a9);
    a66=(a66*a67);
    a66=(a4*a66);
    a65=(a65-a66);
    a5=(a5*a65);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][2] : 0;
    a65=(a33*a3);
    a66=(a40*a13);
    a65=(a65+a66);
    a66=(a60*a19);
    a65=(a65+a66);
    a66=(a62*a25);
    a65=(a65+a66);
    a66=(a63*a31);
    a65=(a65+a66);
    a66=(a64*a36);
    a65=(a65+a66);
    a9=sin(a9);
    a7=(a7*a9);
    a7=(a4*a7);
    a65=(a65-a7);
    a5=(a5*a65);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][3] : 0;
    a65=2.6180339887498953e+00;
    a7=(a65*a0);
    a9=-1.1708203932499370e+00;
    a66=(a9*a10);
    a7=(a7+a66);
    a66=-2.;
    a67=(a66*a15);
    a7=(a7+a67);
    a67=8.9442719099991586e-01;
    a68=(a67*a21);
    a7=(a7+a68);
    a68=-6.1803398874989479e-01;
    a69=(a68*a27);
    a7=(a7+a69);
    a69=2.7639320225002101e-01;
    a70=(a69*a29);
    a7=(a7+a70);
    a70=cos(a13);
    a70=(a14*a70);
    a71=cos(a16);
    a70=(a70*a71);
    a70=(a4*a70);
    a7=(a7-a70);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][4] : 0;
    a7=(a65*a2);
    a70=(a9*a12);
    a7=(a7+a70);
    a70=(a66*a18);
    a7=(a7+a70);
    a70=(a67*a24);
    a7=(a7+a70);
    a70=(a68*a30);
    a7=(a7+a70);
    a70=(a69*a35);
    a7=(a7+a70);
    a70=sin(a13);
    a70=(a14*a70);
    a71=cos(a16);
    a70=(a70*a71);
    a70=(a4*a70);
    a7=(a7-a70);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][5] : 0;
    a7=(a65*a3);
    a70=(a9*a13);
    a7=(a7+a70);
    a70=(a66*a19);
    a7=(a7+a70);
    a70=(a67*a25);
    a7=(a7+a70);
    a70=(a68*a31);
    a7=(a7+a70);
    a70=(a69*a36);
    a7=(a7+a70);
    a16=sin(a16);
    a14=(a14*a16);
    a14=(a4*a14);
    a7=(a7-a14);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][6] : 0;
    a7=-7.2360679774997905e-01;
    a14=(a7*a0);
    a16=2.;
    a70=(a16*a10);
    a14=(a14+a70);
    a70=-1.7082039324993659e-01;
    a71=(a70*a15);
    a14=(a14+a71);
    a71=-1.6180339887498949e+00;
    a72=(a71*a21);
    a14=(a14+a72);
    a72=(a67*a27);
    a14=(a14+a72);
    a72=-3.8196601125010515e-01;
    a73=(a72*a29);
    a14=(a14+a73);
    a73=cos(a19);
    a73=(a20*a73);
    a74=cos(a22);
    a73=(a73*a74);
    a73=(a4*a73);
    a14=(a14-a73);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][7] : 0;
    a14=(a7*a2);
    a73=(a16*a12);
    a14=(a14+a73);
    a73=(a70*a18);
    a14=(a14+a73);
    a73=(a71*a24);
    a14=(a14+a73);
    a73=(a67*a30);
    a14=(a14+a73);
    a73=(a72*a35);
    a14=(a14+a73);
    a73=sin(a19);
    a73=(a20*a73);
    a74=cos(a22);
    a73=(a73*a74);
    a73=(a4*a73);
    a14=(a14-a73);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][8] : 0;
    a14=(a7*a3);
    a73=(a16*a13);
    a14=(a14+a73);
    a73=(a70*a19);
    a14=(a14+a73);
    a73=(a71*a25);
    a14=(a14+a73);
    a73=(a67*a31);
    a14=(a14+a73);
    a73=(a72*a36);
    a14=(a14+a73);
    a22=sin(a22);
    a20=(a20*a22);
    a20=(a4*a20);
    a14=(a14-a20);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][9] : 0;
    a14=3.8196601125010515e-01;
    a20=(a14*a0);
    a22=-8.9442719099991586e-01;
    a73=(a22*a10);
    a20=(a20+a73);
    a73=1.6180339887498949e+00;
    a74=(a73*a15);
    a20=(a20+a74);
    a74=1.7082039324993681e-01;
    a75=(a74*a21);
    a20=(a20+a75);
    a75=(a66*a27);
    a20=(a20+a75);
    a75=7.2360679774997894e-01;
    a76=(a75*a29);
    a20=(a20+a76);
    a76=cos(a25);
    a76=(a26*a76);
    a77=cos(a28);
    a76=(a76*a77);
    a76=(a4*a76);
    a20=(a20-a76);
    a5=(a5*a20);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][10] : 0;
    a20=(a14*a2);
    a76=(a22*a12);
    a20=(a20+a76);
    a76=(a73*a18);
    a20=(a20+a76);
    a76=(a74*a24);
    a20=(a20+a76);
    a76=(a66*a30);
    a20=(a20+a76);
    a76=(a75*a35);
    a20=(a20+a76);
    a76=sin(a25);
    a76=(a26*a76);
    a77=cos(a28);
    a76=(a76*a77);
    a76=(a4*a76);
    a20=(a20-a76);
    a5=(a5*a20);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][11] : 0;
    a20=(a14*a3);
    a76=(a22*a13);
    a20=(a20+a76);
    a76=(a73*a19);
    a20=(a20+a76);
    a76=(a74*a25);
    a20=(a20+a76);
    a76=(a66*a31);
    a20=(a20+a76);
    a76=(a75*a36);
    a20=(a20+a76);
    a28=sin(a28);
    a26=(a26*a28);
    a26=(a4*a26);
    a20=(a20-a26);
    a5=(a5*a20);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][12] : 0;
    a20=-2.7639320225002101e-01;
    a0=(a20*a0);
    a26=6.1803398874989479e-01;
    a10=(a26*a10);
    a0=(a0+a10);
    a15=(a22*a15);
    a0=(a0+a15);
    a21=(a16*a21);
    a0=(a0+a21);
    a21=1.1708203932499357e+00;
    a27=(a21*a27);
    a0=(a0+a27);
    a27=-2.6180339887498936e+00;
    a15=(a27*a29);
    a0=(a0+a15);
    a15=cos(a31);
    a15=(a32*a15);
    a10=cos(a34);
    a15=(a15*a10);
    a15=(a4*a15);
    a0=(a0-a15);
    a5=(a5*a0);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][13] : 0;
    a2=(a20*a2);
    a12=(a26*a12);
    a2=(a2+a12);
    a18=(a22*a18);
    a2=(a2+a18);
    a24=(a16*a24);
    a2=(a2+a24);
    a30=(a21*a30);
    a2=(a2+a30);
    a30=(a27*a35);
    a2=(a2+a30);
    a30=sin(a31);
    a30=(a32*a30);
    a24=cos(a34);
    a30=(a30*a24);
    a30=(a4*a30);
    a2=(a2-a30);
    a5=(a5*a2);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][14] : 0;
    a3=(a20*a3);
    a13=(a26*a13);
    a3=(a3+a13);
    a19=(a22*a19);
    a3=(a3+a19);
    a25=(a16*a25);
    a3=(a3+a25);
    a31=(a21*a31);
    a3=(a3+a31);
    a31=(a27*a36);
    a3=(a3+a31);
    a34=sin(a34);
    a32=(a32*a34);
    a32=(a4*a32);
    a3=(a3-a32);
    a5=(a5*a3);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][15] : 0;
    a3=(a33*a29);
    a32=(a40*a38);
    a3=(a3+a32);
    a32=(a60*a8);
    a3=(a3+a32);
    a32=(a62*a11);
    a3=(a3+a32);
    a32=(a63*a17);
    a3=(a3+a32);
    a32=(a64*a23);
    a3=(a3+a32);
    a32=cos(a36);
    a32=(a37*a32);
    a34=cos(a39);
    a32=(a32*a34);
    a32=(a4*a32);
    a3=(a3-a32);
    a5=(a5*a3);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][16] : 0;
    a3=(a33*a35);
    a32=(a40*a41);
    a3=(a3+a32);
    a32=(a60*a44);
    a3=(a3+a32);
    a32=(a62*a48);
    a3=(a3+a32);
    a32=(a63*a52);
    a3=(a3+a32);
    a32=(a64*a56);
    a3=(a3+a32);
    a32=sin(a36);
    a32=(a37*a32);
    a34=cos(a39);
    a32=(a32*a34);
    a32=(a4*a32);
    a3=(a3-a32);
    a5=(a5*a3);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][17] : 0;
    a33=(a33*a36);
    a40=(a40*a42);
    a33=(a33+a40);
    a60=(a60*a46);
    a33=(a33+a60);
    a62=(a62*a50);
    a33=(a33+a62);
    a63=(a63*a54);
    a33=(a33+a63);
    a64=(a64*a58);
    a33=(a33+a64);
    a39=sin(a39);
    a37=(a37*a39);
    a37=(a4*a37);
    a33=(a33-a37);
    a5=(a5*a33);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][18] : 0;
    a33=(a65*a29);
    a37=(a9*a38);
    a33=(a33+a37);
    a37=(a66*a8);
    a33=(a33+a37);
    a37=(a67*a11);
    a33=(a33+a37);
    a37=(a68*a17);
    a33=(a33+a37);
    a37=(a69*a23);
    a33=(a33+a37);
    a37=cos(a42);
    a37=(a43*a37);
    a39=cos(a45);
    a37=(a37*a39);
    a37=(a4*a37);
    a33=(a33-a37);
    a5=(a5*a33);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][19] : 0;
    a33=(a65*a35);
    a37=(a9*a41);
    a33=(a33+a37);
    a37=(a66*a44);
    a33=(a33+a37);
    a37=(a67*a48);
    a33=(a33+a37);
    a37=(a68*a52);
    a33=(a33+a37);
    a37=(a69*a56);
    a33=(a33+a37);
    a37=sin(a42);
    a37=(a43*a37);
    a39=cos(a45);
    a37=(a37*a39);
    a37=(a4*a37);
    a33=(a33-a37);
    a5=(a5*a33);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][20] : 0;
    a65=(a65*a36);
    a9=(a9*a42);
    a65=(a65+a9);
    a9=(a66*a46);
    a65=(a65+a9);
    a9=(a67*a50);
    a65=(a65+a9);
    a68=(a68*a54);
    a65=(a65+a68);
    a69=(a69*a58);
    a65=(a65+a69);
    a45=sin(a45);
    a43=(a43*a45);
    a43=(a4*a43);
    a65=(a65-a43);
    a5=(a5*a65);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][21] : 0;
    a65=(a7*a29);
    a43=(a16*a38);
    a65=(a65+a43);
    a43=(a70*a8);
    a65=(a65+a43);
    a43=(a71*a11);
    a65=(a65+a43);
    a43=(a67*a17);
    a65=(a65+a43);
    a43=(a72*a23);
    a65=(a65+a43);
    a43=cos(a46);
    a43=(a47*a43);
    a45=cos(a49);
    a43=(a43*a45);
    a43=(a4*a43);
    a65=(a65-a43);
    a5=(a5*a65);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][22] : 0;
    a65=(a7*a35);
    a43=(a16*a41);
    a65=(a65+a43);
    a43=(a70*a44);
    a65=(a65+a43);
    a43=(a71*a48);
    a65=(a65+a43);
    a43=(a67*a52);
    a65=(a65+a43);
    a43=(a72*a56);
    a65=(a65+a43);
    a43=sin(a46);
    a43=(a47*a43);
    a45=cos(a49);
    a43=(a43*a45);
    a43=(a4*a43);
    a65=(a65-a43);
    a5=(a5*a65);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][23] : 0;
    a7=(a7*a36);
    a65=(a16*a42);
    a7=(a7+a65);
    a70=(a70*a46);
    a7=(a7+a70);
    a71=(a71*a50);
    a7=(a7+a71);
    a67=(a67*a54);
    a7=(a7+a67);
    a72=(a72*a58);
    a7=(a7+a72);
    a49=sin(a49);
    a47=(a47*a49);
    a47=(a4*a47);
    a7=(a7-a47);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][24] : 0;
    a7=(a14*a29);
    a47=(a22*a38);
    a7=(a7+a47);
    a47=(a73*a8);
    a7=(a7+a47);
    a47=(a74*a11);
    a7=(a7+a47);
    a47=(a66*a17);
    a7=(a7+a47);
    a47=(a75*a23);
    a7=(a7+a47);
    a47=cos(a50);
    a47=(a51*a47);
    a49=cos(a53);
    a47=(a47*a49);
    a47=(a4*a47);
    a7=(a7-a47);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][25] : 0;
    a7=(a14*a35);
    a47=(a22*a41);
    a7=(a7+a47);
    a47=(a73*a44);
    a7=(a7+a47);
    a47=(a74*a48);
    a7=(a7+a47);
    a47=(a66*a52);
    a7=(a7+a47);
    a47=(a75*a56);
    a7=(a7+a47);
    a47=sin(a50);
    a47=(a51*a47);
    a49=cos(a53);
    a47=(a47*a49);
    a47=(a4*a47);
    a7=(a7-a47);
    a5=(a5*a7);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][26] : 0;
    a14=(a14*a36);
    a7=(a22*a42);
    a14=(a14+a7);
    a73=(a73*a46);
    a14=(a14+a73);
    a74=(a74*a50);
    a14=(a14+a74);
    a66=(a66*a54);
    a14=(a14+a66);
    a75=(a75*a58);
    a14=(a14+a75);
    a53=sin(a53);
    a51=(a51*a53);
    a51=(a4*a51);
    a14=(a14-a51);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][27] : 0;
    a14=(a20*a29);
    a51=(a26*a38);
    a14=(a14+a51);
    a51=(a22*a8);
    a14=(a14+a51);
    a51=(a16*a11);
    a14=(a14+a51);
    a51=(a21*a17);
    a14=(a14+a51);
    a51=(a27*a23);
    a14=(a14+a51);
    a51=cos(a54);
    a51=(a55*a51);
    a53=cos(a57);
    a51=(a51*a53);
    a51=(a4*a51);
    a14=(a14-a51);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][28] : 0;
    a14=(a20*a35);
    a51=(a26*a41);
    a14=(a14+a51);
    a51=(a22*a44);
    a14=(a14+a51);
    a51=(a16*a48);
    a14=(a14+a51);
    a51=(a21*a52);
    a14=(a14+a51);
    a51=(a27*a56);
    a14=(a14+a51);
    a51=sin(a54);
    a51=(a55*a51);
    a53=cos(a57);
    a51=(a51*a53);
    a51=(a4*a51);
    a14=(a14-a51);
    a5=(a5*a14);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][29] : 0;
    a20=(a20*a36);
    a26=(a26*a42);
    a20=(a20+a26);
    a22=(a22*a46);
    a20=(a20+a22);
    a16=(a16*a50);
    a20=(a20+a16);
    a21=(a21*a54);
    a20=(a20+a21);
    a27=(a27*a58);
    a20=(a20+a27);
    a57=sin(a57);
    a55=(a55*a57);
    a55=(a4*a55);
    a20=(a20-a55);
    a5=(a5*a20);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][30] : 0;
    a20=5.0000000000000000e-01;
    a29=(a20*a29);
    a55=-1.1055728090000840e+00;
    a38=(a55*a38);
    a29=(a29+a38);
    a38=1.5278640450004206e+00;
    a8=(a38*a8);
    a29=(a29+a8);
    a8=-2.8944271909999157e+00;
    a11=(a8*a11);
    a29=(a29+a11);
    a11=1.0472135954999574e+01;
    a17=(a11*a17);
    a29=(a29+a17);
    a17=-8.4999999999999947e+00;
    a23=(a17*a23);
    a29=(a29+a23);
    a23=cos(a58);
    a23=(a59*a23);
    a57=cos(a61);
    a23=(a23*a57);
    a23=(a4*a23);
    a29=(a29-a23);
    a5=(a5*a29);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][31] : 0;
    a35=(a20*a35);
    a41=(a55*a41);
    a35=(a35+a41);
    a44=(a38*a44);
    a35=(a35+a44);
    a48=(a8*a48);
    a35=(a35+a48);
    a52=(a11*a52);
    a35=(a35+a52);
    a56=(a17*a56);
    a35=(a35+a56);
    a56=sin(a58);
    a56=(a59*a56);
    a52=cos(a61);
    a56=(a56*a52);
    a56=(a4*a56);
    a35=(a35-a56);
    a5=(a5*a35);
    a6=(a6+a5);
    a5=arg[1] ? arg[1][32] : 0;
    a20=(a20*a36);
    a55=(a55*a42);
    a20=(a20+a55);
    a38=(a38*a46);
    a20=(a20+a38);
    a8=(a8*a50);
    a20=(a20+a8);
    a11=(a11*a54);
    a20=(a20+a11);
    a17=(a17*a58);
    a20=(a20+a17);
    a61=sin(a61);
    a59=(a59*a61);
    a4=(a4*a59);
    a20=(a20-a4);
    a5=(a5*a20);
    a6=(a6+a5);
    a1=(a1+a6);
    if (res[0]!=0) res[0][0]=a1;
    return 0;
}

extern "C" CASADI_SYMBOL_EXPORT int flagrangian(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
    return casadi_f0(arg, res, iw, w, mem);
}

extern "C" CASADI_SYMBOL_EXPORT void flagrangian_incref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT void flagrangian_decref(void) {
}

extern "C" CASADI_SYMBOL_EXPORT casadi_int flagrangian_n_in(void) { return 2;}

extern "C" CASADI_SYMBOL_EXPORT casadi_int flagrangian_n_out(void) { return 1;}

extern "C" CASADI_SYMBOL_EXPORT const char* flagrangian_name_in(casadi_int i){
    switch (i) {
        case 0: return "i0";
        case 1: return "i1";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const char* flagrangian_name_out(casadi_int i){
    switch (i) {
        case 0: return "o0";
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* flagrangian_sparsity_in(casadi_int i) {
    switch (i) {
        case 0: return casadi_s0;
        case 1: return casadi_s1;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT const casadi_int* flagrangian_sparsity_out(casadi_int i) {
    switch (i) {
        case 0: return casadi_s2;
        default: return 0;
    }
}

extern "C" CASADI_SYMBOL_EXPORT int flagrangian_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 2;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}


