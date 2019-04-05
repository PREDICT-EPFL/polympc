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

    enum
    {
        NX = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime,
        VARX_SIZE = ode_colloc_t::VARX_SIZE,
        VARU_SIZE = ode_colloc_t::VARU_SIZE,
        VARP_SIZE = ode_colloc_t::VARP_SIZE,
        VAR_SIZE = var_t::RowsAtCompileTime,

        NUM_EQ = VARX_SIZE + NX, // ode_collocation + x0
        NUM_INEQ = 0,
        NUM_BOX = VARX_SIZE - NX + VARU_SIZE, // X without x0 and U
        NUM_CONSTR  = NUM_EQ + NUM_INEQ + NUM_BOX,
    };

    using ode_jacobian_t = typename ode_colloc_t::jacobian_t;
    using cost_gradient_t = var_t;
    using varx_t = typename Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    using varu_t = typename Eigen::Matrix<Scalar, VARU_SIZE, 1>;

    using sqp_t = sqp::SQP<nmpc>;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using b_box_t = Eigen::Matrix<Scalar, NUM_BOX, 1>;
    using A_box_t = Eigen::Matrix<Scalar, NUM_BOX, VAR_SIZE>;

    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;
    sqp_t solver;

    State _x0;
    State _constr_xl, _constr_xu;
    Control _constr_ul, _constr_uu;

    /*
     * var_t var = [xn, ..., x0, un, ..., u0, p]
     *
     * constraints:
     * xl < x < xu
     * ul < u < uu
     * x0 - x(t0) = 0
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

    void constraint(const var_t& var, b_eq_t& b_eq, b_ineq_t& b_ineq, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box) {
        varx_t c_ode;
        ps_ode(var, c_ode);
        _constraint(var, c_ode, b_eq, b_ineq, b_box, l_box, u_box);
    }

    void _constraint(const var_t& var, const varx_t& c_ode, b_eq_t& b_eq, b_ineq_t& b_ineq, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box)
    {
        l_box << _constr_xl.template replicate<VARX_SIZE/NX-1, 1>(), _constr_ul.template replicate<VARU_SIZE/NU, 1>();
        u_box << _constr_xu.template replicate<VARX_SIZE/NX-1, 1>(), _constr_uu.template replicate<VARU_SIZE/NU, 1>();
        b_box << var.template head<VARX_SIZE-NX>(), var.template segment<VARU_SIZE>(VARX_SIZE);
        b_eq << c_ode, var.template segment<NX>(VARX_SIZE - NX) - _x0;
    }

    void constraint_linearized(const var_t& var,
                               A_eq_t& A_eq,
                               b_eq_t& b_eq,
                               A_ineq_t& A_ineq,
                               b_ineq_t& b_ineq,
                               A_box_t& A_box,
                               b_box_t& b_box,
                               b_box_t& l_box,
                               b_box_t& u_box)
    {
        // TODO: avoid stack allocation
        varx_t c_ode;
        ode_jacobian_t ode_jac;

        ps_ode.linearized(var, ode_jac, c_ode);
        _constraint(var, c_ode, b_eq, b_ineq, b_box, l_box, u_box);

        A_eq.setZero();
        A_eq.template topRows<VARX_SIZE>() = ode_jac;
        A_eq.template block<NX,NX>(VARX_SIZE, VARX_SIZE-NX).setIdentity();

        A_box.setZero();
        A_box.template block<VARX_SIZE-NX, VARX_SIZE-NX>(0,0).setIdentity();
        A_box.template block<VARU_SIZE, VARU_SIZE>(VARX_SIZE-NX,VARX_SIZE).setIdentity();
    }

    var_t solve(const State &x0)
    {
        _x0 = x0;
        // TODO: parametrize
        _constr_xu << 10, 10, 1e20;
        _constr_xl = -_constr_xu;
        _constr_uu << 10, 1;
        _constr_ul << -10, -1;
        var_t var0;
        var0.setZero();
        // var0.template segment<NX>(VARX_SIZE-NX) = x0;
        var0.template segment<VARX_SIZE>(0) = x0.template replicate<VARX_SIZE/NX, 1>();
        var0.template segment<1>(VARX_SIZE+VARU_SIZE).setConstant(1.0);
        solver.solve(*this, var0);
        return solver._x;
    }
};
}



#endif // NMPC_HPP
