#ifndef COST_COLLOCATION_HPP
#define COST_COLLOCATION_HPP

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/unsupported/Eigen/AutoDiff"
#include <type_traits>
#include <iostream>


/** ------------------------------- */
template <typename T, typename Dummy>
struct has_lagrange
{
    template <class, class>class checker;

    template <typename C> static std::true_type test(checker<C, decltype(&C::template operator()<Dummy, Dummy, Dummy, Dummy>)> *);
    template <typename C> static std::false_type test(...);

    using type = decltype(test<T>(nullptr));
    static constexpr bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <typename T, typename Dummy>
struct has_mayer
{
    template <class, class>class checker;

    template <typename C> static std::true_type test(checker<C, decltype(&C::template operator()<Dummy, Dummy>)> *);
    template <typename C> static std::false_type test(...);

    using type = decltype(test<T>(nullptr));
    static constexpr bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};


namespace polympc {

template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments = 1>
class cost_collocation
{
public:
    using Scalar = typename LagrangeTerm::Scalar;
    using weights_t  = typename Polynomial::q_weights_t;
    using nodes_t    = typename Polynomial::nodes_t;

    enum
    {
        NX = LagrangeTerm::State::RowsAtCompileTime,
        NU = LagrangeTerm::Control::RowsAtCompileTime,
        NP = LagrangeTerm::Parameters::RowsAtCompileTime,
        POLY_ORDER = Polynomial::POLY_ORDER,
        NUM_NODES = POLY_ORDER + 1,

        VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX,
        VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU,
        VARP_SIZE = NP,

        HAS_LAGRANGE = has_lagrange<LagrangeTerm, Scalar>::value,
        HAS_MAYER    = has_mayer<MayerTerm, Scalar>::value
    };

    /** type to store optimization variable var = [x, u, p] */
    using var_t     = Eigen::Matrix<Scalar, VARX_SIZE + VARU_SIZE + VARP_SIZE, 1>;
    void operator() (const var_t &var, Scalar &cost_value,
                     const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1) ) const;

    /** compute value and gradient */
    /** linearized approximation */
    using gradient_t = Eigen::Matrix<Scalar, var_t::RowsAtCompileTime, 1>;
    using local_gradient_t = Eigen::Matrix<Scalar, NX + NU, 1>;
    using Derivatives = Eigen::Matrix<Scalar, NX + NU, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    /** AD variables */
    using ADx_t = Eigen::Matrix<ADScalar, NX, 1>;
    using ADu_t = Eigen::Matrix<ADScalar, NU, 1>;
    ADx_t m_ADx, m_ADy;
    ADu_t m_ADu;

    void initialize_derivatives();
    void value_gradient(const var_t &var, Scalar &cost_value, var_t &cost_gradient,
                        const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1) );


    using hessian_t = Eigen::Matrix<Scalar, var_t::RowsAtCompileTime, var_t::RowsAtCompileTime>;
    using local_hessian_t = Eigen::Matrix<Scalar, NX + NU, NX + NU>;
    using outer_Derivatives = Eigen::Matrix<ADScalar, NX + NU, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outer_Derivatives>;
    Eigen::Matrix<outerADScalar, NX, 1> m_outADx;
    Eigen::Matrix<outerADScalar, NU, 1> m_outADu;

    void initialize_second_derivatives();
    void value_gradient_hessian(const var_t &var, Scalar &cost_value, var_t &cost_gradient,
                                hessian_t &cost_hessian, const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1));


    cost_collocation();
    cost_collocation(const LagrangeTerm &L, const MayerTerm &M){}
    ~cost_collocation(){}

private:
    LagrangeTerm m_Lagrange;
    MayerTerm m_Mayer;
    Polynomial m_basis_f;

    weights_t m_weights;
};

/** constructor */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::cost_collocation()
{
    std::cout << HAS_LAGRANGE << "\n";
    std::cout << HAS_MAYER << "\n";

    m_weights = m_basis_f.CCQWeights(); // change interface, probably integration weights? Both in Ebyshev and Legendre
    initialize_derivatives();
    initialize_second_derivatives();
}


template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::operator ()(const var_t &var, Scalar &cost_value,
                                                                                const Scalar &t0, const Scalar &tf) const
{
    cost_value = Scalar(0);

    if(HAS_MAYER)
    {
        Scalar Mayer = Scalar(0);
        m_Mayer(var. template head<NX>(), Mayer);
        cost_value += Mayer;
    }

    if(HAS_LAGRANGE)
    {
        Scalar t_scale = (tf - t0) / (2 * NumSegments);
        Scalar Lagrange = Scalar(0);
        int n = 0, it = 0;
        for(int k = 0; k < VARX_SIZE; k += NX)
        {
            m_Lagrange(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
                    var.template segment<NP>(VARX_SIZE + VARU_SIZE), Lagrange);

            cost_value += t_scale * m_weights[it % NUM_NODES] * Lagrange;
            if( ((it % NUM_NODES) == 0) && (it != 0))
                cost_value += t_scale * m_weights[it % NUM_NODES] * Lagrange; // add twice at the border points

            n += NU;
            it++;
        }
    }
}


/** evaluate cost and gradient */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::value_gradient(const var_t &var, Scalar &cost_value, var_t &cost_gradient,
                                                                                        const Scalar &t0, const Scalar &tf)
{
    cost_value = Scalar(0);
    cost_gradient = var_t::Zero();

    if(HAS_LAGRANGE)
    {
        Scalar t_scale = (tf - t0) / (2 * NumSegments);
        Scalar coeff;
        ADScalar ad_value;

        int n = 0, it = 0;
        for(int k = 0; k < VARX_SIZE; k += NX)
        {
            for(int i = 0; i < NX; i++)
                m_ADx(i).value() = var.template segment<NX>(k)(i);

            for(int i = 0; i < NU; i++)
                m_ADu(i).value() = var.template segment<NU>(n + VARX_SIZE)(i);

            m_Lagrange(m_ADx, m_ADu,
                    var.template segment<NP>(VARX_SIZE + VARU_SIZE), ad_value);

            coeff = t_scale * m_weights[it % POLY_ORDER];
            cost_value += coeff * ad_value.value();

            /** extract gradient */
            cost_gradient. template segment<NX>(k) = coeff * ad_value.derivatives(). template head<NX>();
            cost_gradient. template segment<NU>(n + VARX_SIZE) = coeff * ad_value.derivatives(). template tail<NU>();

            /** @note: better way to identify junction points? */
            if( ((it % POLY_ORDER) == 0) && (it != 0) && (it < NumSegments * POLY_ORDER))
            {
                cost_value += coeff * ad_value.value(); // add twice at the border points
                cost_gradient. template segment<NX>(k) *= Scalar(2);
                cost_gradient. template segment<NU>(n + VARX_SIZE) *= Scalar(2);
            }

            n += NU;
            it++;
        }

    }

    if(HAS_MAYER)
    {
        ADScalar ad_value;
        for(int i = 0; i < NX; i++)
            m_ADx(i).value() = var.template head<NX>()(i);

        m_Mayer(m_ADx, ad_value);
        cost_value += ad_value.value();
        cost_gradient.template head<NX + NU>() += ad_value.derivatives();
    }

}


/** evaluate cost, gradient and hessian */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::value_gradient_hessian(const var_t &var, Scalar &cost_value, var_t &cost_gradient,
                                                                                                hessian_t &cost_hessian, const Scalar &t0, const Scalar &tf)
{
    cost_value = Scalar(0);
    cost_gradient = var_t::Zero();
    cost_hessian  = hessian_t::Zero();

    if(HAS_LAGRANGE)
    {
        Scalar t_scale = (tf - t0) / (2 * NumSegments);
        Scalar coeff;
        outerADScalar ad_value;
        local_hessian_t hes;

        int n = 0, it = 0;
        for(int k = 0; k < VARX_SIZE; k += NX)
        {
            for(int i = 0; i < NX; i++)
                m_outADx(i).value().value() = var.template segment<NX>(k)(i);

            for(int i = 0; i < NU; i++)
                m_outADu(i).value().value() = var.template segment<NU>(n + VARX_SIZE)(i);

            m_Lagrange(m_outADx, m_outADu,
                    var.template segment<NP>(VARX_SIZE + VARU_SIZE), ad_value);

            coeff = t_scale * m_weights[it % POLY_ORDER];
            cost_value += coeff * ad_value.value().value();

            /** extract gradient */
            cost_gradient. template segment<NX>(k) = coeff * ad_value.value().derivatives(). template head<NX>();
            cost_gradient. template segment<NU>(n + VARX_SIZE) = coeff * ad_value.value().derivatives(). template tail<NU>();

            /** extract hessian */
            for(int i = 0; i < NX + NU; ++i)
            {
                hes.template middleRows(i,1) = ad_value.derivatives()(i).derivatives().transpose();
            }
            cost_hessian.template block<NX, NX>(k, k) = coeff * hes.template topLeftCorner<NX, NX>();
            cost_hessian.template block<NX, NU>(k, n + VARX_SIZE) = coeff * hes. template topRightCorner<NX, NU>();
            cost_hessian.template block<NU, NX>(n + VARX_SIZE, k) = coeff * hes.template bottomLeftCorner<NU, NX>();
            cost_hessian.template block<NU,NU>(n + VARX_SIZE, n + VARX_SIZE) = coeff * hes.template bottomRightCorner<NU, NU>();

            /** @note: better way to identify junction points? */
            if( ((it % POLY_ORDER) == 0) && (it != 0) && (it < NumSegments * POLY_ORDER))
            {
                cost_value += coeff * ad_value.value().value(); // add twice at the border points
                cost_gradient. template segment<NX>(k) *= Scalar(2);
                cost_gradient. template segment<NU>(n + VARX_SIZE) *= Scalar(2);

                cost_hessian.template block<NX, NX>(k, k) *= Scalar(2);
                cost_hessian.template block<NX, NU>(k, n + VARX_SIZE) *= Scalar(2);
                cost_hessian.template block<NU, NX>(n + VARX_SIZE, k) *= Scalar(2);
                cost_hessian.template block<NU,NU>(n + VARX_SIZE, n + VARX_SIZE) *= Scalar(2);
            }

            n += NU;
            it++;
        }
    }

    if(HAS_MAYER)
    {
        outerADScalar ad_value;
        local_hessian_t hes;
        for(int i = 0; i < NX; i++)
            m_outADx(i).value().value() = var.template head<NX>()(i);

        m_Mayer(m_outADx, ad_value);
        cost_value += ad_value.value().value();
        cost_gradient.template head<NX + NU>() += ad_value.value().derivatives();
        for(int i = 0; i < NX + NU; ++i)
        {
            hes.template middleRows(i,1) = ad_value.derivatives()(i).derivatives().transpose();
        }
        cost_hessian.template topLeftCorner<NX + NU, NX + NU>() += hes;
    }

}


/** void initialize derivatives */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::initialize_derivatives()
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

/** void initialize derivatives */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::initialize_second_derivatives()
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




// end of namespace
}

#endif // COST_COLLOCATION_HPP
