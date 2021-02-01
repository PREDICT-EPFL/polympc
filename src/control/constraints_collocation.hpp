#ifndef CONSTRAINTS_COLLOCATION_HPP
#define CONSTRAINTS_COLLOCATION_HPP


#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/AutoDiff"
#include <type_traits>


namespace polympc {

template<int NX, int NU, int NP, typename Polynomial, int NumSegments = 1>
class constraint_collocation
{
public:
    using Scalar     = typename Polynomial::scalar_t;
    using nodes_t    = typename Polynomial::nodes_t;

    enum
    {
        POLY_ORDER = Polynomial::POLY_ORDER,
        NUM_NODES = POLY_ORDER + 1,

        VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX,
        VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU,
        VARP_SIZE = NP,
    };

    /** type to store optimization variable var = [x, u, p] */
    using var_t     = Eigen::Matrix<Scalar, VARX_SIZE + VARU_SIZE + VARP_SIZE, 1>;

    /** compute value and gradient */
    /** linearized approximation */
    using Derivatives = Eigen::Matrix<Scalar, NX + NU, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    /** AD variables */
    using ADx_t = Eigen::Matrix<ADScalar, NX, 1>;
    using ADu_t = Eigen::Matrix<ADScalar, NU, 1>;

    /** return type of generic functor */
    template<int NY>
    using value_t = Eigen::Matrix<Scalar, (NumSegments * POLY_ORDER + 1) * NY, 1>;

    ADx_t m_ADx, m_ADy;
    ADu_t m_ADu;

    void initialize_derivatives();

    /** second order derivatives */
    using local_hessian_t = Eigen::Matrix<Scalar, NX + NU, NX + NU>;
    using outer_Derivatives = Eigen::Matrix<ADScalar, NX + NU, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outer_Derivatives>;
    Eigen::Matrix<outerADScalar, NX, 1> m_outADx;
    Eigen::Matrix<outerADScalar, NU, 1> m_outADu;

    void initialize_second_derivatives();

    constraint_collocation();
    ~constraint_collocation(){}

    /** collocate a generic function */
    template<typename Function, int NY>
    void generic_function(const Function &func, const var_t &var, value_t<NY> &value);

    template<typename Function, int NY>
    auto generic_function(Function &func, const var_t &var) -> Eigen::Matrix<Scalar, (NumSegments * POLY_ORDER + 1) * NY, 1>;

    template<typename Function, int NY>
    void generic_function_jacoabian(const Function &func, var_t &var,
                                    Eigen::Matrix<Scalar, (NumSegments * POLY_ORDER + 1) * NY, VARX_SIZE + VARU_SIZE> &jac);



private:
    Polynomial m_basis_f;
};

/** constructor */
template<int NX, int NU, int NP, typename Polynomial, int NumSegments>
constraint_collocation<NX, NU, NP, Polynomial, NumSegments>::constraint_collocation()
{
    initialize_derivatives();
    initialize_second_derivatives();
}

/** void initialize derivatives */
template<int NX, int NU, int NP, typename Polynomial, int NumSegments>
void constraint_collocation<NX, NU, NP, Polynomial, NumSegments>::initialize_derivatives()
{
    int deriv_num = NX + NU;
    int deriv_idx = 0;

    for(int i = 0; i < NX; i++)
    {
        m_ADx[i].derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
    for(int i = 0; i < NU; i++)
    {
        m_ADu[i].derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
}

/** void initialize second derivatives */
template<int NX, int NU, int NP, typename Polynomial, int NumSegments>
void constraint_collocation<NX, NU, NP, Polynomial, NumSegments>::initialize_second_derivatives()
{
    /** initialize derivatives */
    int div_size = NX + NU;
    int derivative_idx = 0;
    for(int i = 0; i < NX; ++i)
    {
        m_outADx(i).value().derivatives() = Derivatives::Unit(div_size, derivative_idx);
        m_outADx(i).derivatives() =  Derivatives::Unit(div_size, derivative_idx);
        // initialize hessian matrix to zero
        for(int idx = 0; idx < div_size; idx++)
        {
            m_outADx(i).derivatives()(idx).derivatives()  = Derivatives::Zero();
        }
        derivative_idx++;
    }

    for(int i = 0; i < NU; ++i)
    {
        m_outADu(i).value().derivatives() = Derivatives::Unit(div_size, derivative_idx);
        m_outADu(i).derivatives() = Derivatives::Unit(div_size, derivative_idx);
        for(int idx = 0; idx < div_size; idx++)
        {
            m_outADu(i).derivatives()(idx).derivatives()  = Derivatives::Zero();
        }
        derivative_idx++;
    }
}


template<int NX, int NU, int NP, typename Polynomial, int NumSegments>
template<typename Function, int NY>
void constraint_collocation<NX, NU, NP, Polynomial, NumSegments>::generic_function(const Function &func, const var_t &var,
                                                                                   value_t<NY> &value)
{
    Eigen::Matrix<Scalar, NY, 1> f_res;

    int n = 0;
    int m = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        func(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
             var.template segment<NP>(VARX_SIZE + VARU_SIZE), f_res);

        value. template segment<NY>(m) = f_res;
        n += NU;
        m += NY;
    }
}


template<int NX, int NU, int NP, typename Polynomial, int NumSegments>
template<typename Function, int NY>
auto constraint_collocation<NX, NU, NP, Polynomial, NumSegments>::generic_function(Function &func, const var_t &var)
-> Eigen::Matrix<Scalar, (NumSegments * POLY_ORDER + 1) * NY, 1>
{
    Eigen::Matrix<Scalar, NY, 1> f_res;
    Eigen::Matrix<Scalar, (NumSegments * POLY_ORDER + 1) * NY, 1> value;

    int n = 0;
    int m = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        func(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
             var.template segment<NP>(VARX_SIZE + VARU_SIZE), f_res);

        value. template segment<NY>(m) = f_res;
        n += NU;
        m += NY;
    }
    return value;
}


// end of namespace
}



#endif // CONSTRAINTS_COLLOCATION_HPP
