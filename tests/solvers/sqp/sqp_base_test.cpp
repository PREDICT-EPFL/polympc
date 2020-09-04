#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
//#include "control/ode_collocation.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"
#include <iomanip>
#include <iostream>
#include <chrono>
#include "control/simple_robot_model.hpp"


typedef std::chrono::time_point<std::chrono::system_clock> time_point;
time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation>
{
public:
    ~RobotOCP(){}

    static constexpr double t_start = 0.0;
    static constexpr double t_stop  = 2.0;

    Eigen::DiagonalMatrix<scalar_t, 3> Q{1,1,1};
    Eigen::DiagonalMatrix<scalar_t, 2> R{1,1};
    Eigen::DiagonalMatrix<scalar_t, 3> QN{1,1,1};

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / d(0);
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        //Q.diagonal() << 1,1,1;
        //R.diagonal() << 1,1;

        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = R.toDenseMatrix().template cast<T>();

        lagrange = x.dot(Qm * x) + u.dot(Rm * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        mayer = x.dot(Qm * x);
    }
};

/** create solver */
template<typename Problem, typename QPSolver> class MySolver;

template<typename Problem, typename QPSolver = ADMM<Problem::VAR_SIZE, Problem::DUAL_SIZE, typename Problem::scalar_t>>
class MySolver : public SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;

    scalar_t step_size_selection_impl(const Ref<const nlp_variable_t>& p) noexcept
    {
        //std::cout << "taking NEW implementation \n";
        scalar_t mu, phi_l1, Dp_phi_l1;
        nlp_variable_t cost_gradient = this->m_h;
        const scalar_t tau = this->m_settings.tau; // line search step decrease, 0 < tau < settings.tau

        scalar_t constr_l1 = this->constraints_violation(this->m_x);

        // TODO: get mu from merit function model using hessian of Lagrangian
        //const scalar_t quad_term = p.dot(this->m_H * p);
        //const scalar_t qt = quad_term >= 0 ? scalar_t(0.5) * quad_term : 0;
        //mu = (abs(cost_gradient.dot(p)) ) / ((1 - this->m_settings.rho) * constr_l1);

        mu = this->m_lam_k.template lpNorm<Eigen::Infinity>();

        //std::cout << "mu: " << mu << "\n";

        scalar_t cost_1;
        this->problem.cost(this->m_x, this->m_p, cost_1);

        //std::cout << "l1: " << constr_l1 << " cost: " << cost_1 << "\n";

        phi_l1 = cost_1 + mu * constr_l1;
        Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

        scalar_t alpha = scalar_t(1.0);
        scalar_t cost_step;
        nlp_variable_t x_step;
        for (int i = 1; i < this->m_settings.line_search_max_iter; i++)
        {
            x_step.noalias() = alpha * p;
            x_step += this->m_x;
            this->problem.cost(x_step, this->m_p, cost_step);

            //std::cout << "i: " << i << " l1: " << this->constraints_violation(x_step) << " cost: " << cost_step << "\n";

            scalar_t phi_l1_step = cost_step + mu * this->constraints_violation(x_step);

            //std::cout << "phi before: " << phi_l1 << " after: " << phi_l1_step <<  " required diff: " << alpha * this->m_settings.eta * Dp_phi_l1 << "\n";

            if (phi_l1_step <= (phi_l1 + alpha * this->m_settings.eta * Dp_phi_l1))
            {
                // accept step
                return alpha;
            } else {
                alpha = tau * alpha;
            }
        }

        return alpha;
    }
};



int main(void)
{
    using admm = ADMM<RobotOCP::VAR_SIZE, RobotOCP::DUAL_SIZE, RobotOCP::scalar_t>;

    MySolver<RobotOCP> solver;
    solver.settings().max_iter = 20;
    solver.settings().line_search_max_iter = 10;
    solver.parameters()(0) = 2.0;
    Eigen::Matrix<double, 3, 1> init_cond; init_cond << 0.5, 0.5, 0.5;

    solver.upper_bound_x().segment(30, 3) = init_cond;
    solver.lower_bound_x().segment(30, 3) = init_cond;

    time_point start = get_time();
    solver.solve();
    time_point stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << solver.info().status.value << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    std::cout << "Size of the solver: " << sizeof (solver) << "\n";

    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    /** warm started iteration */
    init_cond << 0.3, 0.4, 0.45;
    solver.upper_bound_x().segment(30, 3) = init_cond;
    solver.lower_bound_x().segment(30, 3) = init_cond;

    start = get_time();
    solver.solve();
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << solver.info().status.value << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    /**
    MySolver<RobotOCP, admm>::nlp_hessian_t H;
    MySolver<RobotOCP, admm>::nlp_variable_t h;
    MySolver<RobotOCP, admm>::nlp_eq_jacobian_t A_eq;
    MySolver<RobotOCP, admm>::nlp_constraints_t b_eq;
    MySolver<RobotOCP, admm>::nlp_dual_t Alb, Aub;
    Eigen::Matrix<double, RobotOCP::DUAL_SIZE, RobotOCP::VAR_SIZE> A;
    */

    /** compute QP */
    //solver.linearisation(solver.m_x, solver.m_p, solver.m_lam, h, H, A_eq, b_eq);

    /** feel-in A matrix */
    /**
    enum {
        EQ_IDX = 0,
        INEQ_IDX = RobotOCP::NUM_EQ,
        BOX_IDX  = RobotOCP::NUM_INEQ + RobotOCP::NUM_EQ
    };

    // Equality constraints
    Aub.template segment<RobotOCP::NUM_EQ>(EQ_IDX) = -b_eq;
    Alb.template segment<RobotOCP::NUM_EQ>(EQ_IDX) = -b_eq;
    A.template block<RobotOCP::NUM_EQ, RobotOCP::VAR_SIZE>(EQ_IDX, 0) = A_eq;

    // Box constraints
    Aub.template segment<RobotOCP::VAR_SIZE>(BOX_IDX) = solver.upper_bound_x() - solver.m_x;
    Alb.template segment<RobotOCP::VAR_SIZE>(BOX_IDX) = solver.lower_bound_x() - solver.m_x;
    A.template block<RobotOCP::VAR_SIZE, RobotOCP::VAR_SIZE>(BOX_IDX, 0).setIdentity();
    */
    /** create new ADMM solver */
    /**
    admm new_admm;
    new_admm.solve(H,h,A,Alb,Aub);
    std::cout << "QP status: " << new_admm.info().status << " iterations: " << new_admm.info().iter << "\n";
    */

    /** create old ADMM solver */
    /**
    using qp_t = qp_solver::QP<RobotOCP::VAR_SIZE, RobotOCP::DUAL_SIZE, double>;
    qp_solver::QPSolver<qp_t> old_admm;
    qp_t qp;
    qp.P = H;
    qp.q = h;
    qp.A = A;
    qp.l = Alb;
    qp.u = Aub;

    old_admm.setup(qp);
    old_admm.solve(qp);

    std::cout << "Old QP status: " << old_admm.info().status << " iterations: " << old_admm.info().iter << "\n";
    std::cout << "Solvers error: " << (new_admm.primal_solution().transpose() -
                                       new_admm.primal_solution().transpose()).template lpNorm<Eigen::Infinity>() << "\n";
                                       */

    return EXIT_SUCCESS;
}

