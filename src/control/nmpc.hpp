#ifndef NMPC_HPP
#define NMPC_HPP

#include "control/cost_collocation.hpp"
#include "control/ode_collocation.hpp"
#include "control/problem.hpp"
#include "qpsolver/osqp_solver.hpp"


namespace polympc {

template<typename Problem, typename Approximation, typename Solver>
class nmpc
{
public:
    nmpc(){}
    ~nmpc(){}

    using State      = typename Problem::Dynamics::State;
    using Control    = typename Problem::Dynamics::Control;
    using Parameters = typename Problem::Dynamics::Parameters;

    using ode_colloc_t = ode_collocation<typename Problem::Dynamics, Approximation, 2>;
    using cost_colloc_t = cost_collocation<typename Problem::Lagrange, typename Problem::Mayer, Approximation, 2>;

    /** define the QP solver */
    using QP = osqp_solver::QP<cost_colloc_t::hessian_t::RowsAtCompileTime,
                               ode_colloc_t::jacobian_t::RowsAtCompileTime,
                               typename Problem::Dynamics::Scalar>;
    osqp_solver::OSQPSolver<QP> qp_solver;

    enum
    {
        NX = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime
    };
};



}



#endif // NMPC_HPP
