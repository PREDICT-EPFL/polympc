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
template <typename _Problem, typename _Scalar = double>
class SQP {
public:
    using Problem = _Problem;
    enum {
        NX = Problem::NX,
        NIEQ = Problem::NIEQ,
        NEQ = Problem::NEQ,
        NC = Problem::NEQ+Problem::NIEQ
    };

    using Scalar = _Scalar;
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

    Problem _prob;
    Settings settings;

    // info
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
    void construct_subproblem()
    {
        /* Construct QP subproblem
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
        _prob.cost_gradient(_x, _qp.q);

        // constraint bounds:
        // transform    l     <= A.x + b <= u
        //        to    l - b <= A.x     <= u - b
        _prob.constraint_jacobian(_x, _qp.A);
        _prob.constraint(_x, b);

        _qp.u = -1*b;

        // inequality constraint: l = -UNBOUNDED
        _qp.l.template head<NIEQ>().setConstant(-UNBOUNDED);

        // equality constraint: l = u
        _qp.l.template tail<NEQ>() = _qp.u.template tail<NEQ>();
    }

    void solve_subproblem(x_t &p, dual_t &lambda)
    {
        print_qp(_qp);

        _qp_solver.settings.warm_start = true;
        _qp_solver.settings.check_termination = 1;
        _qp_solver.settings.max_iter = 1000;
        _qp_solver.solve(_qp);

        if (_qp_solver.iter >= _qp_solver.settings.max_iter) {
            printf("iter %d MAX ITERATIONS\n", _qp_solver.iter);
        } else {
            printf("iter %d\n", _qp_solver.iter);
        }
        _qp_last_iter = _qp_solver.iter;

        p = _qp_solver.x;
        lambda = _qp_solver.y;
        std::cout << "QP solution: \np " <<  p.transpose() << std::endl;
        std::cout << "lambda " <<  lambda.transpose() << std::endl;
    }

    void solve(const x_t &x0)
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
            printf("\n############# SQP ITER %d #############\n", iter);

            // Solve QP
            construct_subproblem();
            solve_subproblem(p, p_lambda);
            p_lambda -= _lambda;

            mu = 1e5; // TODO: choose mu with sigma = 1
            alpha = 1;
            // TODO: line search using merit function
            alpha = 0.1 * alpha;

            // Take step
            _x = _x + alpha * p;
            _lambda = _lambda + alpha * p_lambda;

            // std::cout << "p " << p.transpose() << std::endl;
            // std::cout << "p_lambda " << p_lambda.transpose() << std::endl;
            std::cout << "STEP" << std::endl;
            std::cout << "x " << _x.transpose() << std::endl;
            std::cout << "lambda " << _lambda.transpose() << std::endl;

            // Evaluate: f, gradient f, hessian f, hessian Lagrangian, c, jacobian c

            if (termination_criteria()) {
                break;
            }
        }

        std::cout << "\n############# SQP END #############" << std::endl;
    }

    bool termination_criteria()
    {
        if (_qp_last_iter == 1) {
            return true;
        }
        return false;
    }
};

} // namespace sqp

#endif // SQP_H
