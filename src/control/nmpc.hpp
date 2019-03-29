#ifndef NMPC_HPP
#define NMPC_HPP

#include "control/cost_collocation.hpp"
#include "control/ode_collocation.hpp"
#include "control/problem.hpp"
#include "qpsolver/osqp_solver.hpp"


namespace polympc {

template<typename Problem, typename Approximation>
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

    /** define the QP solver */
    using qp_t = osqp_solver::QP<cost_colloc_t::hessian_t::RowsAtCompileTime,
                               ode_colloc_t::jacobian_t::RowsAtCompileTime,
                               typename Problem::Dynamics::Scalar>;
    using sovler_t = osqp_solver::OSQPSolver<qp_t>;

    using grad_t = Eigen::Matrix<Scalar, var_t::RowsAtCompileTime, 1>;
    using constr_t = typename ode_colloc_t::constr_t;
    using constr_jac_t = typename ode_colloc_t::jacobian_t;

    // sovler_t qp_solver;
    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;

    // enum
    // {
    //     NX = State::RowsAtCompileTime,
    //     NU = Control::RowsAtCompileTime,
    //     NP = Parameters::RowsAtCompileTime
    // };

    enum {
        NX = var_t::RowsAtCompileTime,
        NIEQ = 0,
        NEQ = constr_t::RowsAtCompileTime
    };

    void cost_gradient(const var_t& x, grad_t &grad)
    {
        Scalar cost;
        cost_f.value_gradient(x, cost, grad);
    }

    void constraint_linearized(const var_t& x, constr_jac_t &jac, constr_t &c)
    {
        ps_ode.linearized(x, jac, c);
    }
};



}



#endif // NMPC_HPP
