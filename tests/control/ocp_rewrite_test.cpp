#include <iostream>
#include <iomanip>
#include "control/ocp_base.hpp"
#include "control/collocation_transcription.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/qp_preconditioners.hpp"
#include "solvers/sqp_base.hpp"
#include "control/mpc_wrapper_rewrite.hpp"

using namespace Eigen;
using namespace polympc;

class RobotOCP : public OCPBase</*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0>
{
public:
    RobotOCP()
    {
        /** initialise weight matrices to identity */
        Q.setIdentity();
        R.setIdentity();
        QN.setIdentity();
    }

    ~RobotOCP() = default;

    Matrix<scalar_t, NX, NX> Q;
    Matrix<scalar_t, NU, NU> R;
    Matrix<scalar_t, NX, NX> QN;

    template<typename T>
    inline void dynamics_impl(const Ref<const state_t<T>> &x, const Ref<const control_t<T>> &u,
                              const Ref<const parameter_t<T>> &p, const Ref<const static_parameter_t> &d,
                              const T &t, Ref<state_t<T>> xdot) const noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);

        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / d(0);
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

const int POLY_ORDER = 5;
const int NUM_SEG = 2;

using TranscribedRobotOCP = CollocationTranscription<RobotOCP, NUM_SEG, POLY_ORDER, SPARSE, CLENSHAW_CURTIS>;

using QPSolver = boxADMM<TranscribedRobotOCP::VAR_SIZE, TranscribedRobotOCP::NUM_EQ, TranscribedRobotOCP::scalar_t,
        TranscribedRobotOCP::MATRIXFMT, linear_solver_traits<TranscribedRobotOCP::MATRIXFMT>::default_solver>;

using Preconditioner = RuizEquilibration<TranscribedRobotOCP::scalar_t, TranscribedRobotOCP::VAR_SIZE,
        TranscribedRobotOCP::NUM_EQ, TranscribedRobotOCP::MATRIXFMT>;

template<typename Problem>
class SQPSolver : public SQPBase<SQPSolver<Problem>, Problem, QPSolver, Preconditioner>
{
public:
    using Base = SQPBase<SQPSolver<Problem>, Problem, QPSolver, Preconditioner>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;
    using typename Base::nlp_jacobian_t;
    using typename Base::nlp_dual_t;
    using typename Base::parameter_t;
    using typename Base::nlp_constraints_t;

    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void
    hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t> &x_step,
                        const Eigen::Ref<const nlp_variable_t> &grad_step) noexcept {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }
};

int main()
{
//    TranscribedRobotOCP transcribed_nlp;
//    TranscribedRobotOCP::nlp_variable_t var = TranscribedRobotOCP::nlp_variable_t::Ones();
//    TranscribedRobotOCP::nlp_eq_constraints_t eq_constr;
//    TranscribedRobotOCP::nlp_ineq_constraints_t ineq_constr;
//    TranscribedRobotOCP::nlp_constraints_t constr;
//    TranscribedRobotOCP::nlp_eq_jacobian_t eq_jac(static_cast<int>(TranscribedRobotOCP::VARX_SIZE), static_cast<int>(TranscribedRobotOCP::VAR_SIZE));
//    TranscribedRobotOCP::nlp_ineq_jacobian_t ineq_jac(static_cast<int>(TranscribedRobotOCP::NUM_INEQ), static_cast<int>(TranscribedRobotOCP::VAR_SIZE));
//    TranscribedRobotOCP::nlp_jacobian_t jac(static_cast<int>(TranscribedRobotOCP::NUM_INEQ + TranscribedRobotOCP::NUM_EQ), static_cast<int>(TranscribedRobotOCP::VAR_SIZE));
//    TranscribedRobotOCP::nlp_cost_t cost = 0;     /** suppres warn */  polympc::ignore_unused_var(cost);
//    TranscribedRobotOCP::nlp_cost_t lagrangian = 0;                     /** suppres warn */  polympc::ignore_unused_var(lagrangian);
//    TranscribedRobotOCP::nlp_dual_t lam = TranscribedRobotOCP::nlp_dual_t::Ones(); /** suppres warn */   polympc::ignore_unused_var(lam);
//    TranscribedRobotOCP::static_parameter_t p; p(0) = 2.0;
//    TranscribedRobotOCP::nlp_variable_t cost_gradient, lag_gradient;
//    TranscribedRobotOCP::nlp_hessian_t cost_hessian(static_cast<int>(TranscribedRobotOCP::VAR_SIZE),  static_cast<int>(TranscribedRobotOCP::VAR_SIZE));
//    TranscribedRobotOCP::nlp_hessian_t lag_hessian(static_cast<int>(TranscribedRobotOCP::VAR_SIZE), static_cast<int>(TranscribedRobotOCP::VAR_SIZE));
//    transcribed_nlp.set_time_limits(0,2);
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

    using RobotMPC = MPCRewrite<TranscribedRobotOCP, SQPSolver>;

    RobotMPC mpc;
    mpc.settings().max_iter = 20;
    mpc.settings().line_search_max_iter = 10;
    mpc.set_time_limits(0, 2);

    /** problem data */
    RobotMPC::static_param p; p << 2.0;          // robot wheel base
    RobotMPC::state_t x0; x0 << 0.5, 0.5, 0.5;   // initial condition
    RobotMPC::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
    RobotMPC::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control

    mpc.set_static_parameters(p);
    mpc.control_bounds(lbu, ubu);
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

    std::cout << "Solution X: " << mpc.solution_x().transpose() << "\n";
    std::cout << "Solution U: " << mpc.solution_u().transpose() << "\n";

    /** sample x solution at collocation points [0, 5, 10] */
    std::cout << "x[0]: " << mpc.solution_x_at(0).transpose() << "\n";
    std::cout << "x[5]: " << mpc.solution_x_at(5).transpose() << "\n";
    std::cout << "x[10]: " << mpc.solution_x_at(10).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    /** sample control at collocation points */
    std::cout << "u[0]: " << mpc.solution_u_at(0).transpose() << "\n";
    std::cout << "u[1]: " << mpc.solution_u_at(1).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    /** sample state at time 't = [0.0, 0.5]' */
    std::cout << "x(0.0): " << mpc.solution_x_at(0.0).transpose() << "\n";
    std::cout << "x(0.5): " << mpc.solution_x_at(0.5).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    /**  sample control at time 't = [0.0, 0.5]' */
    std::cout << "u(0.0): " << mpc.solution_u_at(0.0).transpose() << "\n";
    std::cout << "u(0.5): " << mpc.solution_u_at(0.5).transpose() << "\n";

    return EXIT_SUCCESS;
}
