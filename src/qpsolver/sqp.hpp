#ifndef SQP_H
#define SQP_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include "osqp_solver.hpp"

namespace sqp {

template <typename Scalar>
struct SQPSettings {
    Scalar tau = 0.5;       /**< line search iteration decrease, 0 < tau < 1 */
    Scalar eta = 0.25;      /**< line search parameter, 0 < eta < 1 */
    Scalar rho = 0.5;       /**< line search parameter, 0 < rho < 1 */
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
 * subject to   ce(x)  = 0
 *              ci(x) <= 0
 *         l <= cb(x) <= u
 */
template <typename _Problem>
class SQP {
public:
    using Problem = _Problem;
    enum {
        VAR_SIZE = Problem::VAR_SIZE,
        NUM_EQ = Problem::NUM_EQ,
        NUM_INEQ = Problem::NUM_INEQ,
        NUM_BOX = Problem::NUM_BOX,
        NUM_CONSTR = NUM_EQ + NUM_INEQ + NUM_BOX
    };

    using Scalar = typename Problem::Scalar;
    using qp_t = osqp_solver::QP<VAR_SIZE, NUM_CONSTR, Scalar>;
    using qp_solver_t = osqp_solver::OSQPSolver<qp_t>;
    using Settings = SQPSettings<Scalar>;

    using x_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using dual_t = Eigen::Matrix<Scalar, NUM_CONSTR, 1>;
    using cost_gradient_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;

    using constr_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using jacobian_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using constr_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using jacobian_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using constr_box_t = Eigen::Matrix<Scalar, NUM_BOX, 1>;
    using jacobian_box_t = Eigen::Matrix<Scalar, NUM_BOX, VAR_SIZE>;

    // Constants
    static constexpr Scalar DIV_BY_ZERO_REGUL = 1e-10;

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

    void construct_subproblem(Problem &prob)
    {
        /* QP from linearized NLP:
         * minimize     0.5 x'.P.x + q'.x
         * subject to   Ae.x + be  = 0
         *              Ai.x + bi <= 0
         *         l <= Ab.x + bb <= u
         *
         * with:
         *   P      Hessian of Lagrangian
         *   q      cost gradient
         *   Ae,be  linearized equality constraint
         *   Ai,bi  linearized inequality constraint
         *   Ab,bb  linearized box constraint
         *   l,u    box constraint bounds
         *
         * transform to:
         * minimize     0.5 x'.P.x + q'.x
         * subject to   l <= A.x <= u
         *
         * Where the constraint bounds l,u are l=u for equality constraints or
         * set to +-INFINITY if unbounded.
         */
        enum {
            EQ_IDX = 0,
            INEQ_IDX = NUM_EQ,
            BOX_IDX = NUM_INEQ + NUM_EQ,
        };
        const Scalar UNBOUNDED = 1e20;
        Scalar cost;

        _qp.P.setIdentity(); // TODO: Lagrangian hessian
        prob.cost_linearized(_x, _qp.q, cost);

        // TODO: avoid stack allocation (stack overflow)
        constr_eq_t b_eq;
        jacobian_eq_t A_eq;
        constr_ineq_t b_ineq;
        jacobian_ineq_t A_ineq;
        constr_box_t b_box, l_box, u_box;
        jacobian_box_t A_box;

        prob.constraint_linearized(_x, A_eq, b_eq, A_ineq, b_ineq,
                                   A_box, b_box, l_box, u_box);

        // Equality constraints
        // from        A.x + b  = 0
        // to    -b <= A.x     <= -b
        _qp.u.template segment<NUM_EQ>(EQ_IDX) = -b_eq;
        _qp.l.template segment<NUM_EQ>(EQ_IDX) = -b_eq;
        _qp.A.template block<NUM_EQ, VAR_SIZE>(EQ_IDX, 0) = A_eq;

        // Inequality constraints
        // from          A.x + b <= 0
        // to    -INF <= A.x     <= -b
        _qp.u.template segment<NUM_INEQ>(INEQ_IDX) = -b_ineq;
        _qp.l.template segment<NUM_INEQ>(INEQ_IDX).setConstant(-UNBOUNDED);
        _qp.A.template block<NUM_INEQ, VAR_SIZE>(INEQ_IDX, 0) = A_ineq;

        // Box constraints
        // from  l     <= A.x + b <= u
        // to    l - b <= A.x     <= u - b
        _qp.u.template segment<NUM_BOX>(BOX_IDX) = u_box - b_box;
        _qp.l.template segment<NUM_BOX>(BOX_IDX) = l_box - b_box;
        _qp.A.template block<NUM_BOX, VAR_SIZE>(BOX_IDX, 0) = A_box;
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
        Scalar tau; // line search step decrease
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

            /* Line search */
            mu = 1e5; // TODO: choose mu with sigma = 1
            tau = settings.tau; // 0 < tau < settings.tau
            alpha = 1;

            Scalar cost, phi_l1, Dp_phi_l1;
            cost_gradient_t cost_gradient;

            // TODO: reuse computation
            prob.cost_linearized(_x, cost_gradient, cost);
            Scalar constr_l1 = l1_constraint_violation(_x, prob);

            // TODO: get mu from merit function model using hessian of Lagrangian
            mu = cost_gradient.dot(p) / ((1 - settings.rho) * constr_l1);

            phi_l1 = cost + mu * constr_l1;
            Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

            int _line_search_iter;
            // TODO: iteration upper bound, line_search_max_iter or alpha_min
            for (_line_search_iter = 1; _line_search_iter < 100; _line_search_iter++) {
                x_t x_step = _x + alpha*p;
                prob.cost(x_step, cost);

                Scalar constr_l1 = l1_constraint_violation(x_step, prob);

                Scalar phi_l1_step = cost + mu * constr_l1;
                if (phi_l1_step <= phi_l1 + alpha * settings.eta * Dp_phi_l1) {
                    // accept step
                    break;
                } else {
                    alpha = tau * alpha;
                }
            }

            if (_line_search_iter >= 100) {
                printf("LINE SEARCH MAX ITER\n");
                break;
            }

            /* Step */
            _x = _x + alpha * p;
            _lambda = _lambda + alpha * p_lambda;

            // update step info
            _primal_step_norm = alpha * p.template lpNorm<Eigen::Infinity>();
            _dual_step_norm = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

            // Evaluate: f, gradient f, hessian f, hessian Lagrangian, c, jacobian c

            if (termination_criteria()) {
                break;
            }
        }
    }

private:
    bool termination_criteria() const
    {
        if (_primal_step_norm <= settings.eps_prim &&
            _dual_step_norm <= settings.eps_dual) {
            return true;
        }
        return false;
    }

    Scalar l1_constraint_violation(const x_t &x, Problem &prob) const
    {
        Scalar cl1 = DIV_BY_ZERO_REGUL;
        constr_eq_t c_eq;
        constr_ineq_t c_ineq;
        constr_box_t c_box, l_box, u_box;

        prob.constraint(x, c_eq, c_ineq, c_box, l_box, u_box);

        // c_eq = 0
        cl1 += c_eq.template lpNorm<1>();

        // c_ineq <= 0
        for (int i = 0; i < NUM_INEQ; i++) {
            if (c_ineq(i) > 0) {
                cl1 += c_ineq(i);
            }
        }
        // alternative: but maybe more costly in FLOP count
        // cl1 += (c_ineq.array() * (c_ineq.array() <= 0).cast<double>()).matrix().lpNorm<1>() 

        // l <= c_box <= u
        for (int i = 0; i < NUM_BOX; i++) {
            if (c_box(i) < l_box(i)) {
                cl1 += l_box(i) - c_box(i);
            } else if (u_box(i) < c_box(i)) {
                cl1 += c_box(i) - u_box(i);
            }
        }

        return cl1;
    }
};

} // namespace sqp

#endif // SQP_H
