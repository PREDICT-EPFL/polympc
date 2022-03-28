#ifndef POLYMPC_OCP_BASE_HPP
#define POLYMPC_OCP_BASE_HPP

#include <Eigen/Core>
#include "utils/helpers.hpp"

namespace polympc {

template<int cNX, int cNU, int cNP, int cND, int cNG, typename Scalar = double>
class OCPBase
{
public:
    enum
    {
        /** OCP dimensions */
        NX = cNX,
        NU = cNU,
        NP = cNP,
        ND = cND,
        NG = cNG,
    };

    /** static parameters */
    using scalar_t = Scalar;
    using static_parameter_t = Eigen::Matrix<scalar_t, ND, 1>;

    /** state */
    template<typename scalar_t>
    using state_t = Eigen::Matrix<scalar_t, NX, 1>;

    /** control */
    template<typename scalar_t>
    using control_t = Eigen::Matrix<scalar_t, NU, 1>;

    /** parameters */
    template<typename scalar_t>
    using parameter_t = Eigen::Matrix<scalar_t, NP, 1>;

    /** inequality constraints */
    template<typename scalar_t>
    using constraint_t = Eigen::Matrix<scalar_t, NG, 1>;

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                              const Eigen::Ref<const parameter_t<T>> &p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(xdot);

        assert(false && "You have to implement the dynamics in your OCP!");
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                                   const Eigen::Ref<const parameter_t<T>> &p, const Eigen::Ref<const static_parameter_t> &d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(lagrange);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                                const Eigen::Ref<const parameter_t<T>> &p, const Eigen::Ref<const static_parameter_t> &d,
                                const scalar_t &t, T &mayer) noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(mayer);
    }

    template<typename T>
    inline void inequality_constraints_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                                            const Eigen::Ref<const parameter_t<T>> &p,const Eigen::Ref<const static_parameter_t> &d,
                                            const scalar_t &t, Eigen::Ref<constraint_t<T>> g) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(g);
    }
};

} // polympc namespace

#endif //POLYMPC_OCP_BASE_HPP
