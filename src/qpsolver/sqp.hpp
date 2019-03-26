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

/*
 * minimize     f(x)
 * subject to   ci(x) >= 0
 *              ce(x)  = 0
 */
class SQP {
public:
    using Scalar = double;
    using qp_t = osqp_solver::QP<2, 4, Scalar>;
    using qp_solver_t = osqp_solver::OSQPSolver<qp_t>;
    using Settings = SQPSettings<Scalar>;

    using x_t = Eigen::Matrix<Scalar, 2, 1>;
    using dual_t = Eigen::Matrix<Scalar, 4, 1>;
    using hessian_t = Eigen::Matrix<Scalar, 2, 2>;

    // Solver state variables
    int iter;
    x_t _x;
    dual_t _lambda;

    qp_t _qp;
    qp_solver_t _qp_solver;
    qp_solver_t::Settings _qp_solver_settings;

    bool is_psd(hessian_t &h)
    {
        Eigen::EigenSolver<hessian_t> eigensolver(h);
        // std::cout << eigensolver.eigenvalues().transpose() << std::endl;
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
        // if (!is_psd(h)) {
        //     printf("not PSD\n");
        // }

        // while (!is_psd(h)) {
        //     h += 0.1*h.Identity();
        // }

        h.setIdentity();
    }

    x_t cost_gradient(const x_t& x)
    {
        x_t dx;
        // dx << -1, -1; // solution: [1 1]
        // dx << -1, 0; // solution: [1.41, 0]
        // dx << 0, -1; // solution: [0, 1.41]
        dx << 1, 1; // solution: [1, 0] or [0, 1]
        // dx << 0, 1; // solution: [0, (1, 1.41)]
        return dx;
    }

    hessian_t lagrangian_hessian(const x_t& x, const dual_t& lambda)
    {
        hessian_t L_xx;
        L_xx.setIdentity();
        L_xx += -2 * lambda(2) * hessian_t::Identity();
        L_xx += - lambda(3) * (hessian_t() << -2,0,0,0).finished();
        make_hessian_psd(L_xx);
        return L_xx;
    }

    void solve_subproblem(x_t &p, dual_t &lambda)
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
        const Scalar UNBOUNDED = 1e16;
        _qp.P = lagrangian_hessian(_x, _lambda);
        _qp.q = cost_gradient(_x);
        // constraint gradient
        _qp.A << 1, 0,
                 0, 1,
                 2*_x(0), 2*_x(1),
                 -2*_x(0), 1;
        // constraint bounds:
        // transform    l     <= A.x + b <= u
        //        to    l - b <= A.x     <= u - b
        _qp.l << -_x(0),
                 -_x(1),
                 -_x.squaredNorm() + 1,
                 -(_x(1) - _x(0)*_x(0));
        _qp.u << UNBOUNDED,
                 UNBOUNDED,
                 -_x.squaredNorm() + 2,
                 -(_x(1) - _x(0)*_x(0));

        std::cout << "P\n" << _qp.P << std::endl;
        std::cout << "q " << _qp.q.transpose() << std::endl;
        std::cout << "A\n" << _qp.A << std::endl;
        std::cout << "l " << _qp.l.transpose() << std::endl;
        std::cout << "u " << _qp.u.transpose() << std::endl;

        // TODO: warm start, which variables are reused?
        // _qp_solver.reset();

        _qp_solver.solve(_qp, _qp_solver_settings);

        printf("iter %d\n", _qp_solver.iter);
        if (_qp_solver.iter >= _qp_solver_settings.max_iter) {
            printf("QP: MAX ITERATIONS\n");
        }

        p = _qp_solver.x;
        lambda = _qp_solver.y;
        std::cout << "QP solution: " <<  p.transpose() << std::endl;
    }

    void solve(const Settings& settings)
    {
        x_t p; // search direction
        dual_t p_lambda; // dual search direction
        Scalar alpha; // step size
        Scalar mu;

        // initialize
        _x.setZero();
        _x << 1.2, 0.1; // feasible initial point
        _lambda.setZero();

        // Evaluate: f, gradient f, hessian f, hessian Lagrangian, c, jacobian c
        for (iter = 1; iter <= settings.max_iter; iter++) {
            printf("\n############# SQP ITER %d #############\n", iter);

            // Solve QP
            solve_subproblem(p, p_lambda);
            p_lambda -= _lambda;

            mu = 1e5; // TODO: choose mu with sigma = 1
            alpha = 1;
            // TODO: line search using merit function
            alpha = 0.1 * alpha;

            // Take step
            _x = _x + alpha * p;
            _lambda = _lambda + alpha * p_lambda;

            std::cout << "p " << p.transpose() << std::endl;
            std::cout << "p_lambda " << p_lambda.transpose() << std::endl;
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
        // TODO
        return false;
    }
};

} // namespace sqp

#endif // SQP_H
