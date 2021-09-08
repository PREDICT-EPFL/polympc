// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "control/mpc_wrapper.hpp"
#include "polynomials/splines.hpp"
#include "solvers/box_admm.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ ParkingOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 1, /*ND*/ 1, /*NG*/ 1, /*TYPE*/ double)

using namespace Eigen;

class ParkingOCP : public ContinuousOCP<ParkingOCP, Approximation, SPARSE>
{
public:
    ~ParkingOCP() = default;

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = p(0) * u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = p(0) * u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = p(0) * u(0) * sin(u(1)) / d(0);

        polympc::ignore_unused_var(t);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        mayer = p(0);

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
    }

    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                                         const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                                         const scalar_t &t, Eigen::Ref<constraint_t<T>> g) const noexcept
    {
        g(0) = u(0) * u(0) * cos(u(1));

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(p);

    }
};

/** create solver */
template<typename Problem, typename QPSolver> class Solver;
template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ,
                                             typename Problem::scalar_t, Problem::MATRIXFMT, linear_solver_traits<ParkingOCP::MATRIXFMT>::default_solver>>
class Solver : public SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;
    using typename Base::nlp_jacobian_t;
    using typename Base::nlp_dual_t;
    using typename Base::parameter_t;
    using typename Base::nlp_constraints_t;


    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }

    EIGEN_STRONG_INLINE void hessian_regularisation_dense_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept
    {
        const int n = this->m_H.rows();
        /**Regularize by the estimation of the minimum negative eigen value--does not work with inexact Hessian update(matrix is already PSD)*/
        scalar_t aii, ri;
        for (int i = 0; i < n; i++)
        {
            aii = lag_hessian(i,i);
            ri  = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

            if (aii - ri <= 0) {lag_hessian(i,i) += (ri - aii) + scalar_t(0.01);} //All Greshgorin discs are in the positive half

        }
    }

    EIGEN_STRONG_INLINE void hessian_regularisation_sparse_impl(nlp_hessian_t& lag_hessian) noexcept
    {
        const int n = this->m_H.rows(); //132=m_H.toDense().rows()
        /**Regularize by the estimation of the minimum negative eigen value*/
        scalar_t aii, ri;
        for (int i = 0; i < n; i++)
        {
            aii = lag_hessian.coeffRef(i, i);
            ri = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

            if (aii - ri <= 0)
                lag_hessian.coeffRef(i, i) += (ri - aii) + 0.01;//All Gershgorin discs are in the positive half 0.001
        }
    }


    /** for this problem it turned out that exact linearisation not only converges faster but also with a lower computation cost per iteration *
     *
     * So we tell the solver to use the exact linearisation here to update the Hessian
     *
     */
    EIGEN_STRONG_INLINE void update_linearisation_dense_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                             const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                             Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                             Eigen::Ref<nlp_jacobian_t> A,Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        this->linearisation_dense_impl(x, p, lam, cost_grad, lag_hessian, A, b);
        polympc::ignore_unused_var(x_step);
    }

    EIGEN_STRONG_INLINE void update_linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                             const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                             Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                                                             nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        this->linearisation_sparse_impl(x, p, lam, cost_grad, lag_hessian, A, b);
        polympc::ignore_unused_var(x_step);
    }


};


int main(void)
{
    using mpc_t = MPC<ParkingOCP, Solver>;
    mpc_t mpc;
    mpc.settings().max_iter = 20;
    mpc.settings().line_search_max_iter = 10;

    // problem data
    mpc_t::static_param p; p << 1.0;          // robot wheel base
    mpc_t::state_t x0; x0 << 1.5, 0.5, 0.5;   // initial condition
    mpc_t::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
    mpc_t::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control
    mpc_t::parameter_t lbp; lbp << 0;         // lower bound on time
    mpc_t::parameter_t ubp; ubp << 10;        // upper bound on time
    mpc_t::parameter_t p0; p0 << 0.5;         // very important to set initial time estimate
    mpc_t::state_t lbx_f; lbx_f << -0.05, -0.05, -0.05;   // lower bound on final position
    mpc_t::state_t ubx_f = -lbx_f;                        // upper bound  =  lower bound
    mpc_t::constraint_t lbg, ubg; lbg << -10; ubg << 10; // lower/upper bounds on the constraints

    mpc.set_static_parameters(p);
    mpc.control_bounds(lbu, ubu);
    mpc.parameters_bounds(lbp, ubp);
    mpc.constraints_bounds(lbg, ubg);
    mpc.final_state_bounds(lbx_f, ubx_f);
    mpc.p_guess(p0);
    mpc.x_guess(x0.replicate(mpc_t::num_nodes, 1));

    mpc.initial_conditions(x0);

    polympc::time_point start = polympc::get_time();
    mpc.solve();
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "MPC status: " << mpc.info().status.value << "\n";
    std::cout << "Num iterations: " << mpc.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
    std::cout << "Solution X: \n" << mpc.solution_x_reshaped() << "\n";
    std::cout << "Solution U: \n" << mpc.solution_u_reshaped() << "\n";
    std::cout << "Solution P: \n" << mpc.solution_p() << "\n";
    std::cout << "Solution Y: \n" << mpc.solver().dual_solution().transpose() << "\n";

    return EXIT_SUCCESS;
}
