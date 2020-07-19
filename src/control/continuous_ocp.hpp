#ifndef CONTINUOUS_OCP_HPP
#define CONTINUOUS_OCP_HPP

#include "Eigen/Core"
#include "eigen3/unsupported/Eigen/AutoDiff"
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
//template<typename OCP> class ContinuousOCP;

template<typename OCP>
class ContinuousOCP
{
public:
    ContinuousOCP()
    {
        //using _scalar = scalar_t;
        using _scalar = ad_scalar_t;

        state_t<_scalar> x = state_t<scalar_t>::Zero();
        control_t<_scalar> u;
        parameter_t<_scalar> p;
        static_parameter_t d;
        scalar_t time;
        state_t<_scalar> xdot;

        dynamics(x,u,p,d,time,xdot);

        constraint_t<_scalar> g;
        inequality_constraints(x,u,p,d,time,g);
    }
    virtual ~ContinuousOCP() = default;

    enum
    {
        NX = polympc_traits<OCP>::NX,
        NU = polympc_traits<OCP>::NU,
        NP = polympc_traits<OCP>::NP,
        ND = polympc_traits<OCP>::ND,
        NG = polympc_traits<OCP>::NG
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
    using ad_scalar_t = Eigen::AutoDiffScalar<state_t<scalar_t>>;

    /** @brief
     *
     */
    template<typename T>
    void inequality_constraints(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                                const static_parameter_t &d, const scalar_t &t, constraint_t<T> &g)
    {
        static_cast<OCP*>(this)->inequality_constraints_impl(x,u,p,d,t,g);
    }

    /** @brief
     *
     */
    template<typename T>
    void dynamics(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                  const static_parameter_t &d, const scalar_t &t, state_t<T> &xdot)
    {
        static_cast<OCP*>(this)->dynamics_impl(x,u,p,d,t,xdot);
    }

    /** @brief
     *
     */
    template<typename T>
    void mayer_term(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                    const static_parameter_t &d, const scalar_t &t, T &mayer)
    {
        static_cast<OCP*>(this)->mayer_term_impl(x,u,p,d,t,mayer);
    }

    /** @brief
     *
     */
    template<typename T>
    void lagrange_term(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                       const static_parameter_t &d, const scalar_t &t, T &lagrange)
    {
        static_cast<OCP*>(this)->lagrange_term_impl(x,u,p,d,t,lagrange);
    }

    /** @brief
     *
     */
    template<typename T>
    void final_inequality_constraints(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                                      const static_parameter_t &d, const scalar_t &t, constraint_t<T> &h)
    {
        static_cast<OCP*>(this)->final_inequality_constraints(x,u,p,d,t,h);
    }

};





#endif // CONTINUOUS_OCP_HPP
