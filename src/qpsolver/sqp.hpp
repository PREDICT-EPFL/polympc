#ifndef SQP_H
#define SQP_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include "osqp_solver.hpp"

namespace sqp {

template <typename Scalar>
struct SQPSettings {
    Scalar eta = 0.25;
    Scalar tau = 0.5;
    Scalar eps_prim = 1e-3; /**< primal step termination threshold, eps_prim > 0 */
    Scalar eps_dual = 1e-3; /**< dual step termination threshold, eps_dual > 0 */
    int max_iter = 100;
};

template <typename qp_t>
void print_qp(qp_t qp)
{
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "]", "[", "]");
    std::cout << "P = " << qp.P.format(fmt) << std::endl;
    std::cout << "q = " << qp.q.transpose().format(fmt) << std::endl;
    std::cout << "A = " << qp.A.format(fmt) << std::endl;
    std::cout << "l = " << qp.l.transpose().format(fmt) << std::endl;
    std::cout << "u = " << qp.u.transpose().format(fmt) << std::endl;
}

/*
 * minimize     f(x)
 * subject to   ci(x) <= 0
 *              ce(x)  = 0
 */
template <typename _Problem>
class SQP {
public:
    using Problem = _Problem;
    enum {
        NX = Problem::NX,
        NIEQ = Problem::NIEQ,
        NEQ = Problem::NEQ,
        NC = Problem::NEQ+Problem::NIEQ
    };

    using Scalar = typename Problem::Scalar;
    using qp_t = osqp_solver::QP<NX, NC, Scalar>;
    using qp_solver_t = osqp_solver::OSQPSolver<qp_t>;
    using Settings = SQPSettings<Scalar>;

    using x_t = Eigen::Matrix<Scalar, NX, 1>;
    using dual_t = Eigen::Matrix<Scalar, NC, 1>;
    using constraint_t = Eigen::Matrix<Scalar, NC, 1>;
    using hessian_t = Eigen::Matrix<Scalar, NX, NX>;

    // Solver state variables
    int iter;
    x_t _x;
    dual_t _lambda;

    qp_t _qp;
    qp_solver_t _qp_solver;

    Settings settings;

    // info
    Scalar _dual_step_norm;
    Scalar _primal_step_norm;
    int _qp_last_iter = 0;

    bool is_psd(hessian_t &h)
    {
        Eigen::EigenSolver<hessian_t> eigensolver(h);
        for (int i = 0; i < eigensolver.eigenvalues().RowsAtCompileTime; i++) {
            double v = eigensolver.eigenvalues()(i).real();
            if (v <= 0) {
                return false;
            }
        }
        return true;
    }

    void make_hessian_psd(hessian_t &h)
    {
        while (!is_psd(h)) {
            h += 0.1*h.Identity();
        }
    }

    // TODO: add box constraints?
    void construct_subproblem(Problem &prob)
    {
        /* Construct QP subprob
         *
         * minimize     0.5 x'.P.x + q'.x
         * subject to   l < A.x < u
         *
         * with:        P, Lagrangian hessian
         *              q, cost gradient
         *              c, cost value
         *              A, constraint gradient
         *              l,u, constraint bounds
         *                   l=u for equality constraints,
         *                   +/- INFINITY if unbounded on one side
         */
        const Scalar UNBOUNDED = 1e20;
        constraint_t b;

        _qp.P.setIdentity(); // TODO: Lagrangian hessian
        prob.cost_gradient(_x, _qp.q);

        // constraint bounds:
        // transform    l     <= A.x + b <= u
        //        to    l - b <= A.x     <= u - b
        prob.constraint_linearized(_x, _qp.A, b);

        _qp.u = -1*b;

        // inequality constraint: l = -UNBOUNDED
        _qp.l.template head<NIEQ>().setConstant(-UNBOUNDED);

        // equality constraint: l = u
        _qp.l.template tail<NEQ>() = _qp.u.template tail<NEQ>();
    }

    void solve_subproblem(x_t &p, dual_t &lambda)
    {
        _qp_solver.settings.warm_start = true;
        _qp_solver.settings.check_termination = 1;
        _qp_solver.settings.max_iter = 1000;
        _qp_solver.solve(_qp);

        _qp_last_iter = _qp_solver.iter;

        p = _qp_solver.x;
        lambda = _qp_solver.y;
    }

    void solve(Problem &prob, const x_t &x0)
    {
        x_t p; // search direction
        dual_t p_lambda; // dual search direction
        Scalar alpha; // step size
        Scalar mu;

        // initialize
        _x = x0;
        _lambda.setZero();

        // Evaluate: f, gradient f, hessian f, hessian Lagrangian, c, jacobian c
        for (iter = 1; iter <= settings.max_iter; iter++) {
            // Solve QP
            construct_subproblem(prob);
            solve_subproblem(p, p_lambda);
            p_lambda -= _lambda;

            mu = 1e5; // TODO: choose mu with sigma = 1
            alpha = 1;
            // TODO: line search using merit function
            alpha = 0.1 * alpha;

            p = alpha * p;
            p_lambda = alpha * p_lambda;

            _primal_step_norm = p.template lpNorm<Eigen::Infinity>();
            _dual_step_norm = p_lambda.template lpNorm<Eigen::Infinity>();

            // Take step
            _x = _x + p;
            _lambda = _lambda + p_lambda;

            // Evaluate: f, gradient f, hessian f, hessian Lagrangian, c, jacobian c

            if (termination_criteria()) {
                break;
            }
        }
    }

    bool termination_criteria()
    {
        if (_primal_step_norm <= settings.eps_prim &&
            _dual_step_norm <= settings.eps_dual) {
            return true;
        }
        return false;
    }
};

} // namespace sqp

#endif // SQP_H
