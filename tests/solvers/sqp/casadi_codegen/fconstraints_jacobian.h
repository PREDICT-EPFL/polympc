/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

extern "C" int fconstraints_jacobian(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem);
extern "C" void fconstraints_jacobian_incref(void);
extern "C" void fconstraints_jacobian_decref(void);
extern "C" casadi_int fconstraints_jacobian_n_in(void);
extern "C" casadi_int fconstraints_jacobian_n_out(void);
extern "C" const char* fconstraints_jacobian_name_in(casadi_int i);
extern "C" const char* fconstraints_jacobian_name_out(casadi_int i);
extern "C" const casadi_int* fconstraints_jacobian_sparsity_in(casadi_int i);
extern "C" const casadi_int* fconstraints_jacobian_sparsity_out(casadi_int i);
extern "C" int fconstraints_jacobian_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
