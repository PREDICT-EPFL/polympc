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
        NX_ = State::RowsAtCompileTime,
        NU = Control::RowsAtCompileTime,
        NP = Parameters::RowsAtCompileTime,
        VARX_SIZE = ode_colloc_t::VARX_SIZE,
        VARU_SIZE = ode_colloc_t::VARU_SIZE,
        VARP_SIZE = ode_colloc_t::VARP_SIZE,

        NX = var_t::RowsAtCompileTime,
        NIEQ = 2*(VARX_SIZE - NX_ + VARU_SIZE),
        NEQ = VARX_SIZE + NX_,
        NC = NIEQ+NEQ
    };

    // using ode_constr_jac_t = Eigen::Matrix<Scalar, VARX_SIZE, NX>;
    using ode_constr_jac_t = typename ode_colloc_t::jacobian_t;
    using sqp_constr_jac_t = Eigen::Matrix<Scalar, NC, NX>;
    using cost_gradient_t = var_t;
    using varx_t = typename Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    using varu_t = typename Eigen::Matrix<Scalar, VARU_SIZE, 1>;

    using sqp_t = sqp::SQP<nmpc>;
    using sqp_constraint_t = typename sqp_t::constraint_t;

    cost_colloc_t cost_f;
    ode_colloc_t ps_ode;
    sqp_t solver;

    State _x0;
    State _constr_xl, _constr_xu;
    Control _constr_ul, _constr_uu;
    /*
    var_t var = [xn, ..., x0, un, ..., u0, p]
    NumSegments = 2
    POLY_ORDER = 3
    NUM_NODES = 4
    VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX = 21
    VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU = 14
    VARP_SIZE = NP = 1
    NX = 3
    NU = 2
    NP = 1

    xl < x < xu
    ul < u < uu
    x0 = x(t0)
    ps_ode(X) = 0

    translate to:

    for x1-xn:
    -x + xl < 0
     x - xu < 0
    -u + ul < 0
     u - uu < 0
    x0 - x(t0) = 0
    ps_ode(X) = 0

    */

    void cost(const var_t& var, Scalar &cst)
    {
        cost_f(var, cst);
    }

    void cost_gradient(const var_t& var, cost_gradient_t &grad)
    {
        Scalar cost;
        cost_f.value_gradient(var, cost, grad);
    }

    void constraint(const var_t& var, sqp_constraint_t &b) {
        varx_t c_ode;
        ps_ode(var, c_ode);
        _constraint(var, b, c_ode);
    }

    void _constraint(const var_t& var, sqp_constraint_t &b, const varx_t& c_ode)
    {
        unsigned int i = 0;
        Eigen::Matrix<Scalar, VARX_SIZE - NX_, 1> x;
        x = var.template head<VARX_SIZE - NX_>();
        for (unsigned int j = 0; j < VARX_SIZE - NX_; j++) {
            b.template segment<NX_>(i+j) = -x.template segment<NX_>(j) + _constr_xl; // lower bound
        }
        i += VARX_SIZE - NX_;
        for (unsigned int j = 0; j < VARX_SIZE - NX_; j++) {
            b.template segment<NX_>(i+j) = x.template segment<NX_>(j) - _constr_xu; // upper bound
        }
        i += VARX_SIZE - NX_;

        varu_t u = var.template segment<VARU_SIZE>(VARX_SIZE);
        for (unsigned int j = 0; j < VARU_SIZE; j++) {
            b.template segment<NU>(i+j) = -u.template segment<NU>(j) + _constr_ul; // lower bound
        }
        i += VARU_SIZE;
        for (unsigned int j = 0; j < VARU_SIZE; j++) {
            b.template segment<NU>(i+j) = u.template segment<NU>(j) - _constr_uu; // upper bound
        }
        i += VARU_SIZE;

        b.template segment<NX_>(i) = _x0 - var.template head<NX_>(VARX_SIZE - NX_);
        i += NX_;

        b.template segment<VARX_SIZE>(i) = c_ode;
    }

    void constraint_linearized(const var_t& var, sqp_constr_jac_t &A, sqp_constraint_t &b)
    {
        A.setZero();
        unsigned int row = 0;
        A.template block<VARX_SIZE-NX_, VARX_SIZE-NX_>(row,0).setIdentity();
        row += VARX_SIZE-NX_;
        A.template block<VARX_SIZE-NX_, VARX_SIZE-NX_>(row,0).setIdentity();
        row += VARX_SIZE-NX_;
        A.template block<VARU_SIZE, VARU_SIZE>(row,VARX_SIZE).setIdentity();
        row += VARU_SIZE;
        A.template block<VARU_SIZE, VARU_SIZE>(row,VARX_SIZE).setIdentity();
        row += VARU_SIZE;
        A.template block<NX_, NX_>(row,VARX_SIZE-NX_).setIdentity();
        row += NX_;

        varx_t c_ode;
        ode_constr_jac_t ode_jac;
        ps_ode.linearized(var, ode_jac, c_ode);
        A.template bottomRows<ode_constr_jac_t::RowsAtCompileTime>() = ode_jac;

        _constraint(var, b, c_ode);
    }

    var_t solve(const State &x0)
    {
        _x0 = x0;
        // TODO: parametrize
        _constr_xu << 10, 10, 1e17;
        _constr_xl = -_constr_xu;
        _constr_uu << 10, 1;
        _constr_ul << 0, -1;
        var_t var0;
        var0.setOnes();
        // var0.setZero();
        var0.template segment<NX_>(VARX_SIZE-NX_) = x0;
        solver.solve(*this, var0);
        return solver._x;
    }
};
}



#endif // NMPC_HPP
