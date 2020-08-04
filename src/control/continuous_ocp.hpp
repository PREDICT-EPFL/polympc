#ifndef CONTINUOUS_OCP_HPP
#define CONTINUOUS_OCP_HPP

#include "Eigen/Core"
//#include "eigen3/unsupported/Eigen/AutoDiff"
#include "autodiff/AutoDiffScalar.h"
//#include "unsupported/Eigen/KroneckerProduct"
#include "utils/Reshaped.h"
#include "iostream"


/** define the macro for forward declarations */
#define POLYMPC_FORWARD_DECLARATION( cNAME, cNX, cNU, cNP, cND, cNG, TYPE ) \
class cNAME;                                        \
template<>                                          \
struct polympc_traits<cNAME>                        \
{                                                   \
public:                                             \
    using Scalar = TYPE;                            \
    enum { NX = cNX, NU = cNU, NP = cNP, ND = cND, NG = cNG}; \
};                                                  \

/** define derived class traits */
template<typename Derived> struct polympc_traits;
template<typename T> struct polympc_traits<const T> : polympc_traits<T> {};

/** forward declare base class */
template<typename OCP, typename Approximation> class ContinuousOCP;

template<typename OCP, typename Approximation>
class ContinuousOCP
{
public:
    ContinuousOCP()
    {
        /** compute time nodes */
        const scalar_t t_length = (t_stop - t_start) / (NUM_SEGMENTS);
        const scalar_t t_shift  = t_length / 2;
        for(Eigen::Index i = 0; i < NUM_SEGMENTS; ++i)
            time_nodes.template segment<POLY_ORDER + 1>(i * POLY_ORDER) =  (t_length/2) * m_nodes.reverse() +
                    (t_start + t_shift + i * t_length) * Approximation::nodes_t::Ones();
        time_nodes.reverseInPlace();

        /** seed derivatives */
        seed_derivatives();
    }
    virtual ~ContinuousOCP() = default;

    enum
    {
        /** OCP dimensions */
        NX = polympc_traits<OCP>::NX,
        NU = polympc_traits<OCP>::NU,
        NP = polympc_traits<OCP>::NP,
        ND = polympc_traits<OCP>::ND,
        NG = polympc_traits<OCP>::NG,

        /** NLP dimensions */
        NUM_NODES    = Approximation::NUM_NODES,
        POLY_ORDER   = Approximation::POLY_ORDER,
        NUM_SEGMENTS = Approximation::NUM_SEGMENTS,
        VARX_SIZE  = NX * NUM_NODES,
        VARU_SIZE  = NU * NUM_NODES,
        VARP_SIZE  = NP * NUM_NODES,
        VARD_SIZE  = ND * NUM_NODES,
        VAR_SIZE   = VARX_SIZE + VARU_SIZE + VARP_SIZE
    };

    /** define types*/
    /** state */
    template<typename scalar_t>
    using state_t = Eigen::Matrix<scalar_t, NX, 1>;

    /** control */
    template<typename scalar_t>
    using control_t = Eigen::Matrix<scalar_t, NU, 1>;

    /** parameters */
    template<typename scalar_t>
    using parameter_t = Eigen::Matrix<scalar_t, NP, 1>;

    /** constraints */
    template<typename scalar_t>
    using constraint_t = Eigen::Matrix<scalar_t, NG, 1>;

    /** static parameters */
    using scalar_t = typename polympc_traits<OCP>::Scalar;
    using static_parameter_t = Eigen::Matrix<scalar_t, ND, 1>;
    using time_t   = typename Eigen::Matrix<scalar_t, NUM_NODES, 1>;

    /** AD variables */
    using derivatives_t = Eigen::Matrix<scalar_t, NX + NU + NP, 1>;
    using ad_scalar_t = Eigen::AutoDiffScalar<derivatives_t>;
    using ad_state_t   = Eigen::Matrix<ad_scalar_t, NX, 1>;
    using ad_control_t = Eigen::Matrix<ad_scalar_t, NU, 1>;
    using ad_param_t   = Eigen::Matrix<ad_scalar_t, NP, 1>;
    ad_state_t m_ad_x, m_ad_y;
    ad_control_t m_ad_u;
    ad_param_t m_ad_p;

    /** do not make constant */
    const scalar_t t_start = static_cast<OCP*>(this)->t_start;
    const scalar_t t_stop  = static_cast<OCP*>(this)->t_stop;

    /** compute collocation parameters */
    const typename Approximation::diff_mat_t  m_D     = Approximation::compute_diff_matrix();
    const typename Approximation::nodes_t     m_nodes = Approximation::compute_nodes();
    const typename Approximation::q_weights_t m_quad_weights = Approximation::compute_quad_weights();
    time_t time_nodes = time_t::Zero();

    /** NLP variables */
    using nlp_variable_t    = Eigen::Matrix<scalar_t, VAR_SIZE, 1>;
    using nlp_constraints_t = Eigen::Matrix<scalar_t, VARX_SIZE, 1>;
    using nlp_eq_jacobian_t = Eigen::Matrix<scalar_t, VARX_SIZE, VAR_SIZE>;

    /** @brief
     *
     */
    template<typename T>
    inline void inequality_constraints(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                                const static_parameter_t &d, const scalar_t &t, constraint_t<T> &g) const noexcept
    {
        static_cast<OCP*>(this)->inequality_constraints_impl(x,u,p,d,t,g);
    }

    /** @brief
     *
     */
    template<typename T>
    inline void dynamics(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                         const Eigen::Ref<const parameter_t<T>> p, const static_parameter_t d,
                         const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        static_cast<const OCP*>(this)->dynamics_impl(x,u,p,d,t,xdot);
    }

    /** @brief
     *
     */
    template<typename T>
    inline void mayer_term(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                    const static_parameter_t &d, const scalar_t &t, T &mayer) const noexcept
    {
        static_cast<OCP*>(this)->mayer_term_impl(x,u,p,d,t,mayer);
    }

    /** @brief
     *
     */
    template<typename T>
    inline void lagrange_term(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                       const static_parameter_t &d, const scalar_t &t, T &lagrange) const noexcept
    {
        static_cast<OCP*>(this)->lagrange_term_impl(x,u,p,d,t,lagrange);
    }

    /** seed edrivatives */
    void seed_derivatives();

    /** @brief
     *
     */
    template<typename T>
    inline void final_inequality_constraints(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                                      const static_parameter_t &d, const scalar_t &t, constraint_t<T> &h) const noexcept
    {
        static_cast<OCP*>(this)->final_inequality_constraints(x,u,p,d,t,h);
    }

    /** equality constraint */
    void equalities(const Eigen::Ref<const nlp_variable_t> var,
                             const Eigen::Ref<const static_parameter_t> p,
                             Eigen::Ref<nlp_constraints_t> constraint) const noexcept;

    void equalities_linerised(const Eigen::Ref<const nlp_variable_t> var, const Eigen::Ref<const static_parameter_t> p,
                              Eigen::Ref<nlp_constraints_t> constraint, Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept;

};

template<typename OCP, typename Approximation>
void ContinuousOCP<OCP, Approximation>::seed_derivatives()
{
    const int deriv_num = NX + NU + NP;
    int deriv_idx = 0;

    for(int i = 0; i < NX; i++)
    {
        m_ad_x(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
    for(int i = 0; i < NU; i++)
    {
        m_ad_u(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
    for(int i = 0; i < NP; i++)
    {
        m_ad_p(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
}


template<typename OCP, typename Approximation>
void ContinuousOCP<OCP, Approximation>::equalities(const Eigen::Ref<const nlp_variable_t> var,
                                                            const Eigen::Ref<const static_parameter_t> p,
                                                            Eigen::Ref<nlp_constraints_t> constraint) const noexcept
{
    state_t<scalar_t> f_res;
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();

    //DX.transposeInPlace();

    int n = 0;
    int t = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        dynamics<scalar_t>(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
                           var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(t), f_res);
        constraint. template segment<NX>(k) = DX.transpose().col(t);
        constraint. template segment<NX>(k).noalias() -= t_scale * f_res;
        n += NU;
        ++t;
    }
}

template<typename OCP, typename Approximation>
void ContinuousOCP<OCP, Approximation>::equalities_linerised(const Eigen::Ref<const nlp_variable_t> var, const Eigen::Ref<const static_parameter_t> p,
                                                             Eigen::Ref<nlp_constraints_t> constraint,
                                                             Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
{
    jacobian = nlp_eq_jacobian_t::Zero();
    /** compute jacoabian of dynamics */
    Eigen::Matrix<scalar_t, NX, NX + NU + NP> jac;
    scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();

    /** add Diff matrix to the Jacobian */

    Eigen::DiagonalMatrix<scalar_t, NX> I; I.setIdentity();
    for(int s = 0; s < NUM_SEGMENTS; s++)
    {
        for(int i = 0; i < POLY_ORDER; i++)
        {
            for(int j = 0; j < POLY_ORDER + 1; j++)
            {
                int shift = s * POLY_ORDER * NX;
                jacobian.template block<NX, NX>(shift + i * NX, shift + j * NX) = m_D(i,j) * I;
            }
        }
    }


    /** add Diff matrix to the Jacobian */
    /**
    Eigen::DiagonalMatrix<scalar_t, NX> I; I.setIdentity();
    for(int i = 0; i < POLY_ORDER; i++)
    {
        for(int j = 0; j < POLY_ORDER + 1; j++)
            jacobian.template block<NX, NX>(i * NX, j * NX) = m_D(i,j) * I;
    }

    for(int s = 1; s < NUM_SEGMENTS; s++)
        jacobian.template block<NX * POLY_ORDER, NX * (POLY_ORDER + 1)>( s * POLY_ORDER * NX, s * POLY_ORDER * NX) =
                jacobian.template block<NX * POLY_ORDER, NX * (POLY_ORDER + 1)>(0,0);
    */

    //last segment
    jacobian.template block<NX, NX * (POLY_ORDER + 1)>(VARX_SIZE - NX, VARX_SIZE - (NX * (POLY_ORDER + 1))) =
            -jacobian.template block<NX, NX * (POLY_ORDER + 1)>(0,0).reverse();

    /** initialize AD veriables */
    int n = 0;
    int t = 0;
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        m_ad_x = var.template segment<NX>(k);
        m_ad_u = var.template segment<NU>(n + VARX_SIZE);
        double val = m_ad_x(0).value();

        dynamics<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(t), m_ad_y);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            constraint. template segment<NX>(k)(i) = -t_scale * m_ad_y(i).value();
            jac.row(i) = m_ad_y(i).derivatives();
        }
        constraint. template segment<NX>(k) += DX.transpose().col(t);
        /** finish the computation of the constraint */


        /** insert block jacobian */
        jacobian.template block<NX, NX>(k, k).noalias() -= t_scale * jac.template leftCols<NX>();
        jacobian.template block<NX, NU>(k, n + VARX_SIZE).noalias() -= t_scale * jac.template block<NX, NU>(0, NX);
        jacobian.template block<NX, NP>(k, VARX_SIZE + VARU_SIZE).noalias() -= t_scale * jac.template rightCols<NP>();

        n += NU;
        t++;
    }

}





#endif // CONTINUOUS_OCP_HPP
