#ifndef NMPC_HPP
#define NMPC_HPP

#include "control/cost_collocation.hpp"
#include "control/ode_collocation.hpp"
#include "control/problem.hpp"
#include "solvers/sqp.hpp"

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
    };

    using ode_jacobian_t = typename ode_colloc_t::jacobian_t;
    using cost_gradient_t = var_t;
    using varx_t = typename Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    using varu_t = typename Eigen::Matrix<Scalar, VARU_SIZE, 1>;

    using sqp_t = sqp::SQP<nmpc>;
    using dual_t = typename sqp_t::dual_t;

    using eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;

    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;
    sqp_t solver;

    State _x0;
    Parameters _p0;
    State _constr_xl, _constr_xu;
    Control _constr_ul, _constr_uu;

    /*
     * var_t var = [xn, ..., x0, un, ..., u0, p]
     *
     * constraints:
     * xl < x < xu
     * ul < u < uu
     * x0 = x(t0)
     * ps_ode(X) = 0
     */

    void cost(const var_t& var, Scalar &cst)
    {
        cost_f(var, cst);
    }

    void cost_linearized(const var_t& var, cost_gradient_t &grad, Scalar &cst)
    {
        cost_f.value_gradient(var, cst, grad);
    }

    void constraint(const var_t& var, eq_t& b_eq, ineq_t& b_ineq, var_t& lbx, var_t& ubx) {
        varx_t c_ode;
        ps_ode(var, c_ode);
        set_constraints(var, c_ode, b_eq, b_ineq, lbx, ubx);
    }

    void set_constraints(const var_t& var, const varx_t& c_ode, eq_t& eq, ineq_t& ineq, var_t& lbx, var_t& ubx)
    {
        (void) ineq; // unused
        eq << c_ode;
        lbx << _constr_xl.template replicate<VARX_SIZE/NX-1, 1>(),
               _x0,
               _constr_ul.template replicate<VARU_SIZE/NU, 1>(),
               _p0;
        ubx << _constr_xu.template replicate<VARX_SIZE/NX-1, 1>(),
               _x0,
               _constr_uu.template replicate<VARU_SIZE/NU, 1>(),
               _p0;
    }

    void constraint_linearized(const var_t& var,
                               A_eq_t& A_eq,
                               eq_t& eq,
                               A_ineq_t& A_ineq,
                               ineq_t& ineq,
                               var_t& lbx,
                               var_t& ubx)
    {
        // TODO: avoid stack allocation
        varx_t c_ode;
        ode_jacobian_t ode_jac;

        ps_ode.linearized(var, ode_jac, c_ode);
        set_constraints(var, c_ode, eq, ineq, lbx, ubx);

        A_eq.setZero();
        A_eq.template topRows<VARX_SIZE>() = ode_jac;
    }

    var_t solve(const State& x0, const State& xl, const State& xu, const Control& ul, const Control& uu)
    {
        _x0 = x0;
        _p0 << 1.0;

        _constr_xl = xl;
        _constr_xu = xu;
        _constr_ul = ul;
        _constr_uu = uu;

        var_t var0;
        var0.setZero();
        var0.template segment<VARX_SIZE>(0) = x0.template replicate<VARX_SIZE/NX, 1>();
        var0.template segment<1>(VARX_SIZE+VARU_SIZE) = _p0;

        dual_t y0;
        y0.setZero();

        solver.settings().max_iter = 100;
        solver.settings().line_search_max_iter = 10;

        solver.solve(*this, var0, y0);

        return solver.primal_solution();
    }
};
}



#endif // NMPC_HPP
