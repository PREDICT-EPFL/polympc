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

    sovler_t qp_solver;
    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;

    enum
    {
        NX = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime
    };

    void construct_subproblem(const var_t &x, qp_t &qp)
    {
        /* Construct QP from linearized cost and dynamics
         *
         * minimize     0.5 x' P x + q' x + c
         * subject to   A x + b = 0
         *
         * with:        P = cost hessian
         *              q = cost gradient
         *              c = cost value
         *              A = ode jacobian
         *              b = ode value
         */
        Scalar cost_value;
        cost_f.value_gradient_hessian(x, cost_value, qp.q, qp.P); // TODO: actually need hessian of lagrangian
        ps_ode.linearized(x, qp.A, qp.l);
        qp.l *= -1; /* OSQP equality constraint form: -b <= A x <= -b */
        qp.u = qp.l;
    }
};



}



#endif // NMPC_HPP
