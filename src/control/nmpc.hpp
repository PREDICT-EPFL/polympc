#ifndef NMPC_HPP
#define NMPC_HPP

#include "control/cost_collocation.hpp"
#include "control/ode_collocation.hpp"
#include "control/problem.hpp"
#include "qpsolver/osqp_solver.hpp"
#include "qpsolver/sqp.hpp"

namespace polympc {

template<typename Problem, typename Approximation, typename Solver>
class nmpc
{
public:
    nmpc(){}
    ~nmpc(){}

    using Scalar     = typename Problem::Dynamics::Scalar;
    using State      = typename Problem::Dynamics::State;
    using Control    = typename Problem::Dynamics::Control;
    using Parameters = typename Problem::Dynamics::Parameters;

    using ode_colloc_t = ode_collocation<typename Problem::Dynamics, Approximation, 2>;
    using cost_colloc_t = cost_collocation<typename Problem::Lagrange, typename Problem::Mayer, Approximation, 2>;
    using var_t = typename cost_colloc_t::var_t;

    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;
    SQP<nmpc> solver;

    enum
    {
        NX = var_t::RowsAtCompileTime,
        NIEQ = 0,
        NEQ = constr_t::RowsAtCompileTime,
        // NX = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime
    };

    /*
    var_t var = [xn, ..., x0, un, ..., u0]

    x = ode(x)
    xl < x < xu
    ul < u < uu
    x0 = x(t0)

    D.X - F(X,U) = 0
    x0 - x(t0) = 0
    for x1-xn:
    -x + xl < 0
     x - xu < 0
    for u0-un:
    -u + ul < 0
     u - uu < 0
    */

    void cost(const x_t& x, Scalar &cst)
    {
        Scalar cost;
        cost_f.value(x, grad);
    }

    void cost_gradient(const x_t& x, grad_t &grad)
    {
        Scalar cost;
        cost_f.value_gradient(x, cost, grad);
    }

    void constraint(const x_t& x, constr_t &c)
    {

    }

    void constraint_linearized(const x_t& x, constr_jac_t &A, constr_t &b)
    {

    }

    var_t solve(const x_t &x0)
    {
        var_t x0;
        x0.setOnes();
        solver.solve(*this, x0);
    }
};



}



#endif // NMPC_HPP
