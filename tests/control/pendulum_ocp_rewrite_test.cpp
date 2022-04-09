#include <iostream>
#include <iomanip>
#include "control/ocp_base.hpp"
#include "control/collocation_transcription.hpp"
#include "quadratures/legendre_gauss_lobatto.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/qp_preconditioners.hpp"
#include "solvers/sqp_base.hpp"
#include "solvers/line_search.hpp"
#include "solvers/ipopt_interface.hpp"
#include "solvers/qpmad_interface.hpp"
#include "control/mpc_wrapper_rewrite.hpp"

using namespace Eigen;
using namespace polympc;

class PendulumOCP : public OCPBase</*NX*/ 2, /*NU*/ 1, /*NP*/ 0, /*ND*/ 0, /*NG*/0>
{
public:
    PendulumOCP()
    {
        /** initialise weight matrices to identity */
        Q.setZero();
        Q.diagonal() << 10, 1;

        R.setZero();
        R.diagonal() << 0.001;

        QN << 3.174376532480597, 0.003777692122638,
              0.003777692122638, 0.001186581391502;
    }

    ~PendulumOCP() = default;

    Matrix<scalar_t, 2, 2> Q;
    Matrix<scalar_t, 1, 1> R;
    Matrix<scalar_t, 2, 2> QN;

    const scalar_t g = 9.81;
    const scalar_t l = 0.5;
    const scalar_t m = 0.15;
    const scalar_t b = 0.1;

    template<typename T>
    inline void dynamics_impl(const Ref<const state_t<T>> &x, const Ref<const control_t<T>> &u,
                              const Ref<const parameter_t<T>> &p, const Ref<const static_parameter_t> &d,
                              const T &t, Ref<state_t<T>> xdot) const noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);

        xdot(0) = x(1);
        xdot(1) = (m * g * l * sin(x(0)) - b * x(1) + u(0)) / (m * l * l);
    }

    template<typename T>
    inline void lagrange_term_impl(const Ref<const state_t<T>> &x, const Ref<const control_t<T>> &u,
                                   const Ref<const parameter_t<T>> &p, const Ref<const static_parameter_t> &d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);

        lagrange = x.dot(Q.template cast<T>() * x) + u.dot(R.template cast<T>() * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Ref<const state_t<T>> &x, const Ref<const control_t<T>> &u,
                                const Ref<const parameter_t<T>> &p, const Ref<const static_parameter_t> &d,
                                const scalar_t &t, T &mayer) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(u);

        mayer = x.dot(QN.template cast<T>() * x);
    }
};

#define POLY_ORDER 5
#define NUM_SEG 2

using TranscribedPendulumOCP = CollocationTranscription<PendulumOCP, NUM_SEG, POLY_ORDER, SPARSE, ClenshawCurtis<POLY_ORDER>, false>;

using QPSolver = boxADMM<TranscribedPendulumOCP::VAR_SIZE, TranscribedPendulumOCP::NUM_EQ, TranscribedPendulumOCP::scalar_t,
        TranscribedPendulumOCP::MATRIXFMT, linear_solver_traits<TranscribedPendulumOCP::MATRIXFMT>::default_solver>;
//using QPSolver = QPMAD<TranscribedPendulumOCP::VAR_SIZE, TranscribedPendulumOCP::NUM_EQ, TranscribedPendulumOCP::scalar_t, TranscribedPendulumOCP::MATRIXFMT>;

using RuizEquilibrationPreconditioner = RuizEquilibration<TranscribedPendulumOCP::scalar_t, TranscribedPendulumOCP::VAR_SIZE,
        TranscribedPendulumOCP::NUM_EQ, TranscribedPendulumOCP::MATRIXFMT>;

template<typename Problem>
class SQPSolver : public SQPBase<SQPSolver<Problem>, Problem, QPSolver>
{
public:
    using Base = SQPBase<SQPSolver<Problem>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;
    using typename Base::nlp_jacobian_t;
    using typename Base::nlp_dual_t;
    using typename Base::parameter_t;
    using typename Base::nlp_constraints_t;

    LSFilter<scalar_t> filter;

    /** change step size selection algorithm  : filter line search */
    scalar_t step_size_selection_impl(const Ref<const nlp_variable_t>& p) noexcept
    {
        const scalar_t tau = this->m_settings.tau; // line search step decrease, 0 < tau < settings.tau

        /** compute constraints at initial point */
        scalar_t constr_l1 = this->constraints_violation(this->m_x);
        scalar_t cost_1;
        this->problem.cost(this->m_x, this->m_p, cost_1);

        if(filter.is_acceptable(cost_1, constr_l1))
            filter.add(cost_1, constr_l1);

        //check for exit right here?

        //filter.print();

        scalar_t alpha = scalar_t(1.0);
        scalar_t cost_step;
        nlp_variable_t x_step;
        for (int i = 1; i < this->m_settings.line_search_max_iter; i++)
        {
            x_step.noalias() = alpha * p;
            x_step += this->m_x;

            this->problem.cost(x_step, this->m_p, cost_step);
            scalar_t constr_step = this->constraints_violation(x_step);
            this->m_cost = cost_step;

            if(filter.is_acceptable(cost_step, constr_step))
            {
                filter.add(cost_step, constr_step);
                //filter.print();

                //std::cout << "alpha: " << alpha << "\n";
                return alpha;
            }
            else
            {
                alpha *= tau;
            }

        }
        //std::cout << "alpha: " << alpha << "\n";
        return alpha;
    }

    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void
    hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t> &x_step,
                        const Eigen::Ref<const nlp_variable_t> &grad_step) noexcept {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }
};

int main()
{
//    TranscribedPendulumOCP transcribed_nlp;
//    TranscribedPendulumOCP::nlp_variable_t var = TranscribedPendulumOCP::nlp_variable_t::Ones();
//    TranscribedPendulumOCP::nlp_eq_constraints_t eq_constr;
//    TranscribedPendulumOCP::nlp_ineq_constraints_t ineq_constr;
//    TranscribedPendulumOCP::nlp_constraints_t constr;
//    TranscribedPendulumOCP::nlp_eq_jacobian_t eq_jac(static_cast<int>(TranscribedPendulumOCP::VARX_SIZE), static_cast<int>(TranscribedPendulumOCP::VAR_SIZE));
//    TranscribedPendulumOCP::nlp_ineq_jacobian_t ineq_jac(static_cast<int>(TranscribedPendulumOCP::NUM_INEQ), static_cast<int>(TranscribedPendulumOCP::VAR_SIZE));
//    TranscribedPendulumOCP::nlp_jacobian_t jac(static_cast<int>(TranscribedPendulumOCP::NUM_INEQ + TranscribedPendulumOCP::NUM_EQ), static_cast<int>(TranscribedPendulumOCP::VAR_SIZE));
//    TranscribedPendulumOCP::nlp_cost_t cost = 0;     /** suppres warn */  polympc::ignore_unused_var(cost);
//    TranscribedPendulumOCP::nlp_cost_t lagrangian = 0;                     /** suppres warn */  polympc::ignore_unused_var(lagrangian);
//    TranscribedPendulumOCP::nlp_dual_t lam = TranscribedPendulumOCP::nlp_dual_t::Ones(); /** suppres warn */   polympc::ignore_unused_var(lam);
//    TranscribedPendulumOCP::static_parameter_t p;
//    TranscribedPendulumOCP::nlp_variable_t cost_gradient, lag_gradient;
//    TranscribedPendulumOCP::nlp_hessian_t cost_hessian(static_cast<int>(TranscribedPendulumOCP::VAR_SIZE),  static_cast<int>(TranscribedPendulumOCP::VAR_SIZE));
//    TranscribedPendulumOCP::nlp_hessian_t lag_hessian(static_cast<int>(TranscribedPendulumOCP::VAR_SIZE), static_cast<int>(TranscribedPendulumOCP::VAR_SIZE));
//    transcribed_nlp.set_time_limits(0,3);
//
//    std::cout << "m_D: " << transcribed_nlp.m_D << std::endl;
//    if (transcribed_nlp.is_sparse) {
//        std::cout << "m_DiffMat: " << transcribed_nlp.m_DiffMat << std::endl;
//    }
//    std::cout << "m_nodes: " << transcribed_nlp.m_nodes << std::endl;
//    std::cout << "m_quad_weights: " << transcribed_nlp.m_quad_weights << std::endl;
//    std::cout << "time_nodes: " << transcribed_nlp.time_nodes << std::endl;
//
//    transcribed_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, jac);
//    transcribed_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, jac);
//    transcribed_nlp.cost_gradient_hessian(var, p, cost, cost_gradient, cost_hessian);
//    transcribed_nlp.cost_gradient_hessian(var, p, cost, cost_gradient, cost_hessian);
//    transcribed_nlp.equalities_linearised(var, p, eq_constr, eq_jac);
//    transcribed_nlp.equalities_linearised(var, p, eq_constr, eq_jac);
//    transcribed_nlp.inequalities_linearised(var, p, ineq_constr, ineq_jac);
//    transcribed_nlp.inequalities_linearised(var, p, ineq_constr, ineq_jac);
//
//    std::cout << "lagrangian: " << lagrangian << std::endl;
//    std::cout << "lag_gradient: " << lag_gradient << std::endl;
//    std::cout << "lag_hessian: " << lag_hessian << std::endl;
//    std::cout << "cost: " << cost << std::endl;
//    std::cout << "cost_gradient: " << cost_gradient << std::endl;
//    std::cout << "cost_hessian: " << cost_hessian << std::endl;
//    std::cout << "constr: " << constr << std::endl;
//    std::cout << "jac: " << jac << std::endl;
//    std::cout << "eq_constr: " << eq_constr << std::endl;
//    std::cout << "ineq_constr: " << ineq_constr << std::endl;
//    std::cout << "eq_jac: " << eq_jac << std::endl;
//    std::cout << "ineq_jac: " << ineq_jac << std::endl;

    using PendulumMPC = MPCRewrite<TranscribedPendulumOCP, SQPSolver>;

    PendulumMPC mpc;
    mpc.settings().max_iter = 200;
    mpc.qp_settings().max_iter = 1000;
    mpc.settings().line_search_max_iter = 10;
    mpc.set_time_limits(0, 3);

    /** problem data */
    PendulumMPC::state_t x0; x0 << M_PI, 0; // initial condition
    PendulumMPC::control_t lbu; lbu << -1;  // lower bound on control
    PendulumMPC::control_t ubu; ubu <<  1;  // upper bound on control

//    mpc.control_bounds(lbu, ubu);
    mpc.initial_conditions(x0);

    /** solve */
    time_point start = get_time();
    mpc.solve();
    time_point stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    /** retrieve solution and statistics */
    std::cout << "MPC status: " << mpc.info().status.value << "\n";
    std::cout << "Num SQP iterations: " << mpc.info().iter << "\n";
    std::cout << "Num QP iterations: " << mpc.info().qp_solver_iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) / 1000 << " ms \n";

    std::cout << "Solution X: " << "\n" << mpc.solution_x_reshaped() << "\n";
    std::cout << "Solution U: " << "\n" << mpc.solution_u_reshaped() << "\n";

    std::cout << " ------------------------------------------------ \n";

    /** sample state */
    std::cout << "x(0.0): " << mpc.solution_x_at(0.0).transpose() << "\n";
    std::cout << "x(0.5): " << mpc.solution_x_at(0.5).transpose() << "\n";
    std::cout << "x(1.0): " << mpc.solution_x_at(1.0).transpose() << "\n";
    std::cout << "x(1.5): " << mpc.solution_x_at(1.5).transpose() << "\n";
    std::cout << "x(2.0): " << mpc.solution_x_at(2.0).transpose() << "\n";
    std::cout << "x(2.5): " << mpc.solution_x_at(2.5).transpose() << "\n";
    std::cout << "x(3.0): " << mpc.solution_x_at(3.0).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    /**  sample control */
    std::cout << "u(0.0): " << mpc.solution_u_at(0.0).transpose() << "\n";
    std::cout << "u(0.5): " << mpc.solution_u_at(0.5).transpose() << "\n";
    std::cout << "u(1.0): " << mpc.solution_u_at(1.0).transpose() << "\n";
    std::cout << "u(1.5): " << mpc.solution_u_at(1.5).transpose() << "\n";
    std::cout << "u(2.0): " << mpc.solution_u_at(2.0).transpose() << "\n";
    std::cout << "u(2.5): " << mpc.solution_u_at(2.5).transpose() << "\n";
    std::cout << "u(3.0): " << mpc.solution_u_at(3.0).transpose() << "\n";

//    polympc::IpoptInterface<TranscribedPendulumOCP> solver;
//    solver.get_problem().set_time_limits(0, 3); // another way to set optimisation horizon
//    solver.settings().SetIntegerValue("print_level", 5);
//    Eigen::Matrix<TranscribedPendulumOCP::scalar_t, 2, 1> init_cond; init_cond << M_PI, 0;
//    Eigen::Matrix<TranscribedPendulumOCP::scalar_t, 1, 1> ub; ub <<  1;
//    Eigen::Matrix<TranscribedPendulumOCP::scalar_t, 1, 1> lb; lb << -1;
//
//    solver.upper_bound_x().tail(11) = ub.replicate(11, 1);
//    solver.lower_bound_x().tail(11) = lb.replicate(11, 1);
//
//    solver.upper_bound_x().segment(0, 2) = init_cond;
//    solver.lower_bound_x().segment(0, 2) = init_cond;
//
//    polympc::time_point start = polympc::get_time();
//    solver.solve();
//    polympc::time_point stop = polympc::get_time();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//
//    std::cout << "Solve status: " << solver.info().status << "\n";
//    std::cout << "Num iterations: " << solver.info().iter << "\n";
//    std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
//              << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
//    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) / 1000 << " ms\n";
//    std::cout << "Size of the solver: " << sizeof (solver) << "\n";
//    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    return EXIT_SUCCESS;
}
