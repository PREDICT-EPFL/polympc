// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"
#include "utils/helpers.hpp"
#include <iomanip>
#include <iostream>
#include <chrono>
#include "control/simple_robot_model.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, SPARSE>
{
public:
    ~RobotOCP() = default;

    //static constexpr double t_start = 0.0;
    //static constexpr double t_stop  = 2.0;

    Eigen::DiagonalMatrix<scalar_t, 3> Q{1,1,1};
    Eigen::DiagonalMatrix<scalar_t, 2> R{1,1};
    Eigen::DiagonalMatrix<scalar_t, 3> QN{1,1,1};

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);

        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / d(0);
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);

        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = R.toDenseMatrix().template cast<T>();

        lagrange = x.dot(Qm * x) + u.dot(Rm * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(u);

        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        mayer = x.dot(Qm * x);
    }
};


int main(void)
{
    Eigen::initParallel();
    using namespace polympc;

    RobotOCP robot_nlp;
    RobotOCP::nlp_variable_t var = RobotOCP::nlp_variable_t::Ones();
    RobotOCP::nlp_eq_constraints_t eq_constr;
    RobotOCP::nlp_ineq_constraints_t ineq_constr;
    RobotOCP::nlp_constraints_t constr;
    RobotOCP::nlp_eq_jacobian_t eq_jac(static_cast<int>(RobotOCP::VARX_SIZE), static_cast<int>(RobotOCP::VAR_SIZE));
    RobotOCP::nlp_ineq_jacobian_t ineq_jac(static_cast<int>(RobotOCP::NUM_INEQ), static_cast<int>(RobotOCP::VAR_SIZE));
    RobotOCP::nlp_jacobian_t jac(static_cast<int>(RobotOCP::NUM_INEQ + RobotOCP::NUM_EQ), static_cast<int>(RobotOCP::VAR_SIZE));
    RobotOCP::nlp_cost_t cost = 0;
    RobotOCP::nlp_cost_t lagrangian = 0;                     /** suppres warn */  polympc::ignore_unused_var(lagrangian);
    RobotOCP::nlp_dual_t lam = RobotOCP::nlp_dual_t::Ones(); /** suppres warn */   polympc::ignore_unused_var(lam);
    RobotOCP::static_parameter_t p; p(0) = 2.0;
    RobotOCP::nlp_variable_t cost_gradient, lag_gradient;
    RobotOCP::nlp_hessian_t cost_hessian(static_cast<int>(RobotOCP::VAR_SIZE),  static_cast<int>(RobotOCP::VAR_SIZE));
    RobotOCP::nlp_hessian_t lag_hessian(static_cast<int>(RobotOCP::VAR_SIZE), static_cast<int>(RobotOCP::VAR_SIZE));

    robot_nlp.set_time_limits(0,2);

    //robot_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, eq_jac);
    polympc::time_point start = polympc::get_time();
    for(int i = 0; i < test_NUM_EXP; ++i)
    {
        //robot_nlp.lagrangian(var, p, lam, lagrangian);
        //robot_nlp.lagrangian_gradient(var, p, lam, lagrangian, cost_gradient, lag_gradient, constr, jac);
        //robot_nlp.lagrangian_gradient(var, p, lam, lagrangian, cost_gradient, lag_gradient, constr, jac);
        robot_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, jac);
        //robot_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, jac);
        //robot_nlp.cost_gradient_hessian(var, p, cost, cost_gradient, cost_hessian);
        //robot_nlp.cost_gradient(var, p, cost, cost_gradient);
        //robot_nlp.cost(var, p, cost);
        //robot_nlp.equalities(var, p, eq_constr);
        //robot_nlp.inequalities(var, p, ineq_constr);
        //robot_nlp.equalities_linearised(var, p, eq_constr, eq_jac);
        //robot_nlp.inequalities_linearised(var, p, ineq_constr, ineq_jac);
    }
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Collocation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) / test_NUM_EXP << " [microseconds]" << "\n";

    Eigen::IOFormat fmt(3);
    //Eigen::SparseMatrix<double> SpMat = 0.5 * cost_gradient.asDiagonal() * cost_hessian * cost_gradient.asDiagonal();
    std::cout << "Equality Constraint \n" << eq_constr.transpose() << "\n";
    std::cout << "Inequality Constraint \n" << ineq_constr.transpose() << "\n";
    //std::cout << eq_jac.template rightCols<29>().format(fmt) << "\n";
    std::cout << "Size of NLP:" << sizeof (robot_nlp) << "\n";
    //std::cout << eq_jac << "\n";//.format(fmt) << "\n";
    std::cout << "Inequality Jacoabian \n" << ineq_jac << "\n";
    //std::cout << "Hessian: \n" << cost_hessian << "\n"; // .template leftCols<30>().format(fmt) << "\n";
    std::cout << "Lagrangian: " << lagrangian << "\n";
    std::cout << "Lagrangian Gradient: \n" << lag_gradient.transpose() << "\n";
    std::cout << "Cost Gradient: \n" << cost_gradient.transpose() << "\n";
    std::cout << "Constraint: " << constr.transpose() << "\n";
    std::cout << "Constraints Jacobian \n" << jac << "\n";
    //std::cout << "Number of nonzeros: " << lag_hessian.nonZeros() << " | size: " << lag_hessian.size() << "\n";
    //std::cout << "Cost: " << cost << "\n";
    //std::cout << "Cost gradient: " << cost_gradient.transpose().format(fmt) << "\n";
    std::cout << "Hessian: \n" << lag_hessian.leftCols(18) << "\n";

    // Hessian update
    //RobotOCP::nlp_variable_t s,y;
    //s.setRandom(); y.setRandom();
    //robot_nlp.hessian_update_impl(lag_hessian, s, y);
    //std::cout << "Updated Hessian: \n" << lag_hessian << "\n";


    return EXIT_SUCCESS;
}
