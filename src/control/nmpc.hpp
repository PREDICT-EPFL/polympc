// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef NMPC_HPP
#define NMPC_HPP

#include "control/cost_collocation.hpp"
#include "control/ode_collocation.hpp"
#include "control/problem.hpp"

namespace polympc {

template<typename Problem, typename Approximation, template <typename> class Solver>
class nmpc
{
public:
    using Scalar     = typename Problem::Dynamics::Scalar;
    using State      = typename Problem::Dynamics::State;
    using Control    = typename Problem::Dynamics::Control;
    using Parameters = typename Problem::Dynamics::Parameters;

    using ode_colloc_t = ode_collocation<typename Problem::Dynamics, Approximation, 2>;
    using cost_colloc_t = cost_collocation<typename Problem::Lagrange, typename Problem::Mayer, Approximation, 2>;
    using var_t = typename cost_colloc_t::var_t;
    using SolverImpl = Solver<nmpc>;

    enum
    {
        NX = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime,
        VARX_SIZE = ode_colloc_t::VARX_SIZE,
        VARU_SIZE = ode_colloc_t::VARU_SIZE,
        VARP_SIZE = ode_colloc_t::VARP_SIZE,
        VAR_SIZE = var_t::RowsAtCompileTime,

        NUM_EQ = VARX_SIZE,
        NUM_INEQ = 0,
        NUM_CONSTR = NUM_EQ+NUM_INEQ+VAR_SIZE,
    };

    using ode_jacobian_t = typename ode_colloc_t::jacobian_t;
    using cost_gradient_t = var_t;
    using varx_t = typename Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    using varu_t = typename Eigen::Matrix<Scalar, VARU_SIZE, 1>;
    using dual_t = Eigen::Matrix<Scalar, NUM_CONSTR, 1>;

    using eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;

    cost_colloc_t m_cost_f;
    ode_colloc_t m_ps_ode;

    State m_x0;
    Parameters m_p0;
    State m_bound_xl, m_bound_xu;
    Control m_bound_ul, m_bound_uu;

    var_t m_solution;
    dual_t m_dual_solution;

    bool m_warm_start;
    SolverImpl m_solver;
    Problem m_problem;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    nmpc(){}
    ~nmpc(){}

    void enableWarmStart() {m_warm_start = true;}
    void disableWarmStart() {m_warm_start = false;}
    void computeControl(const State& x0);

    void getOptimalControl(Control& u)
    {
        varGetControl(m_solution, 0, u);
    }

    void getSolution(var_t& var)
    {
        var = m_solution;
    }

    void varGetControl(const var_t& var, int i, Control& u) const
    {
        u = var.template segment<NU>(VARX_SIZE+VARU_SIZE-NU-NU*i);
    }

    void varGetState(const var_t& var, int i, State& x) const
    {
        x = var.template segment<NX>(VARX_SIZE-NX-NX*i);
    }

    void varGetParam(const var_t& var, Parameters& p) const
    {
        p = var.template segment<NP>(VARX_SIZE+VARU_SIZE);
    }

    void setStateBounds(const State& xl, const State& xu)
    {
        m_bound_xl = xl;
        m_bound_xu = xu;
    }

    void setControlBounds(const Control& ul, const Control& uu)
    {
        m_bound_ul = ul;
        m_bound_uu = uu;
    }

    void setParameters(const Parameters& param)
    {
        m_p0 = param;
    }

    // required by solver
    void cost(const var_t& var, Scalar &cst);
    void cost_linearized(const var_t& var, cost_gradient_t &grad, Scalar &cst);
    void constraint(const var_t& var, eq_t& b_eq, ineq_t& b_ineq, var_t& lbx, var_t& ubx);
    void _set_constraints(const var_t& var, const varx_t& c_ode, eq_t& eq, ineq_t& ineq, var_t& lbx, var_t& ubx);
    void constraint_linearized(const var_t& var, A_eq_t& A_eq, eq_t& eq, A_ineq_t& A_ineq, ineq_t& ineq, var_t& lbx, var_t& ubx);
};

template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::computeControl(const State& x0)
{
    m_x0 = x0;

    var_t& var0 = m_solution;
    dual_t& dual0 = m_dual_solution;

    if (m_warm_start) {
        var0.template segment<NX>(VARX_SIZE-NX) = x0;
    } else {
        var0.setZero();
        var0.template segment<VARX_SIZE>(0) = x0.template replicate<VARX_SIZE/NX, 1>();
        var0.template segment<NP>(VARX_SIZE + VARU_SIZE) = m_p0;

        dual0.setZero();
    }

    m_solver.settings().max_iter = 100;
    m_solver.settings().line_search_max_iter = 3;
    m_solver.solve(*this, var0, dual0);

    m_solution = m_solver.primal_solution();
    m_dual_solution = m_solver.dual_solution();

    // TODO: return status?
}

template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::cost(const var_t& var, Scalar &cst)
{
    m_cost_f(var, cst);
}

template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::cost_linearized(const var_t& var, cost_gradient_t &grad, Scalar &cst)
{
    m_cost_f.value_gradient(var, cst, grad);
}

template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::constraint(const var_t& var, eq_t& b_eq, ineq_t& b_ineq, var_t& lbx, var_t& ubx)
{
    varx_t c_ode;
    m_ps_ode(var, c_ode);
    _set_constraints(var, c_ode, b_eq, b_ineq, lbx, ubx);
}

/*
 * var_t var = [xn, ..., x0, un, ..., u0, p]
 *
 * constraints:
 * xl < x < xu
 * ul < u < uu
 * x0 = x(t0)
 * ps_ode(X) = 0
 */
template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::_set_constraints(const var_t& var, const varx_t& c_ode, eq_t& eq, ineq_t& ineq, var_t& lbx, var_t& ubx)
{
    (void) ineq; // unused
    eq << c_ode;
    lbx << m_bound_xl.template replicate<VARX_SIZE/NX-1, 1>(),
           m_x0,
           m_bound_ul.template replicate<VARU_SIZE/NU, 1>(),
           m_p0;
    ubx << m_bound_xu.template replicate<VARX_SIZE/NX-1, 1>(),
           m_x0,
           m_bound_uu.template replicate<VARU_SIZE/NU, 1>(),
           m_p0;
}

template<typename Problem, typename Approximation, template <typename> class Solver>
void nmpc<Problem, Approximation, Solver>
::constraint_linearized(const var_t& var,
                           A_eq_t& A_eq,
                           eq_t& eq,
                           A_ineq_t& A_ineq,
                           ineq_t& ineq,
                           var_t& lbx,
                           var_t& ubx)
{
    varx_t c_ode;
    m_ps_ode.linearized(var, A_eq, c_ode);
    _set_constraints(var, c_ode, eq, ineq, lbx, ubx);
}

}

#endif // NMPC_HPP
