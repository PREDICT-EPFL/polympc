#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#include "control/simple_robot_model.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/admm.hpp"
#include "solvers/qp_preconditioners.hpp"
#include "solvers/line_search.hpp"

#ifdef POLYMPC_FOUND_OSQP_EIGEN
#include "solvers/osqp_interface.hpp"
#include "solvers/qpmad_interface.hpp"
#endif

#include "boost/numeric/odeint.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;
using namespace boost::numeric;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, SPARSE>
{
public:
    ~RobotOCP() = default;
    RobotOCP() { set_time_limits(0,2); } // one way to set optimisation horizon

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

        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = R.toDenseMatrix().template cast<T>();

        lagrange = x.dot(Qm * x) + u.dot(Rm * u);

        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        mayer = x.dot(Qm * x);

        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(u);
    }

    void set_Q_coeff(const scalar_t& coeff)
    {
        Q.diagonal() << coeff, coeff, coeff;
    }
};

/** create solver */
template<typename Problem, typename QPSolver, typename Preconditioner> class MySolver;

template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ, typename Problem::scalar_t>,
         typename Preconditioner = polympc::IdentityPreconditioner>
class MySolver : public SQPBase<MySolver<Problem, QPSolver, Preconditioner>, Problem, QPSolver, Preconditioner>
{
public:
    using Base = SQPBase<MySolver<Problem, QPSolver, Preconditioner>, Problem, QPSolver, Preconditioner>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;

    //EIGEN_STRONG_INLINE const typename Base::_Problem& get_problem() const noexcept { return this->problem; }
    EIGEN_STRONG_INLINE Problem& get_problem() noexcept { return this->problem; }

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
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }
};


/** QP solvers */
using admm_solver = ADMM<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t,
                         RobotOCP::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>;

using box_admm_solver = boxADMM<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t,
                                RobotOCP::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>;

// for advanced users
using osqp_solver_t = polympc::OSQP<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t>;
//using qpmad_solver_t = polympc::QPMAD<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t>;

using preconditioner_t = polympc::RuizEquilibration<RobotOCP::scalar_t, RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::MATRIXFMT>;




int main(void)
{
    MySolver<RobotOCP, box_admm_solver, preconditioner_t> solver;
    solver.get_problem().set_Q_coeff(1.0);
    solver.get_problem().set_time_limits(0, 2); // another way to set optimisation horizon
    solver.settings().max_iter = 10;
    solver.settings().line_search_max_iter = 10;
    solver.qp_settings().max_iter = 1000;
    solver.parameters()(0) = 2.0;
    solver.filter.beta = 0.1;
    Eigen::Matrix<RobotOCP::scalar_t, 3, 1> init_cond; init_cond << 0.5, 0.5, 0.5;
    Eigen::Matrix<RobotOCP::scalar_t, 2, 1> ub; ub <<  1.5,  0.75;
    Eigen::Matrix<RobotOCP::scalar_t, 2, 1> lb; lb << -1.5, -0.75;

    solver.upper_bound_x().tail(22) = ub.replicate(11, 1);
    solver.lower_bound_x().tail(22) = lb.replicate(11, 1);

    solver.upper_bound_x().segment(30, 3) = init_cond;
    solver.lower_bound_x().segment(30, 3) = init_cond;

    polympc::time_point start = polympc::get_time();
    solver.solve();
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << solver.info().status.value << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Num of QP iter: " << solver.info().qp_solver_iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
    std::cout << "Size of the solver: " << sizeof (solver) << "\n";
    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    // warm started iteration
    init_cond << 0.3, 0.4, 0.45;
    solver.upper_bound_x().segment(30, 3) = init_cond;
    solver.lower_bound_x().segment(30, 3) = init_cond;

    start = polympc::get_time();
    solver.solve();
    stop = polympc::get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << solver.info().status.value << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";



    /**
    MySolver<RobotOCP, admm_solver>::nlp_hessian_t H((int)RobotOCP::VAR_SIZE, (int)RobotOCP::VAR_SIZE);
    MySolver<RobotOCP, admm_solver>::nlp_variable_t h, lbx, ubx;
    MySolver<RobotOCP, admm_solver>::nlp_eq_jacobian_t A_eq((int)RobotOCP::NUM_EQ, (int)RobotOCP::VAR_SIZE);
    MySolver<RobotOCP, admm_solver>::nlp_constraints_t b_eq;
    MySolver<RobotOCP, admm_solver>::nlp_constraints_t Alb, Aub;
    Eigen::Matrix<RobotOCP::scalar_t, RobotOCP::DUAL_SIZE, RobotOCP::VAR_SIZE> A;
    solver.parameters()(0) = 2.0;

    // compute QP
    solver.linearisation(solver.m_x, solver.m_p, solver.m_lam, h, H, A_eq, b_eq);
    lbx = solver.lower_bound_x() - solver.m_x;
    ubx = solver.upper_bound_x() - solver.m_x;
    //lbx = MySolver<RobotOCP, admm_solver>::nlp_variable_t::Constant(-MySolver<RobotOCP, admm_solver>::INF);
    //ubx = MySolver<RobotOCP, admm_solver>::nlp_variable_t::Constant( MySolver<RobotOCP, admm_solver>::INF);
    Aub = -b_eq;
    Alb = -b_eq;

    preconditioner_t preconditioner;
    preconditioner.compute(H, h, A_eq, Alb, Aub, lbx, ubx);

    // instantiate OSQP and QPMAD solvers
    osqp_solver_t osqp_solver;
    //osqp_solver.settings().scaling = 10;
    //osqp_solver.settings().scaled_termination = true;
    qpmad_solver_t qpmad_solver;

    // create new ADMM solver
    admm_solver new_admm;
    auto start = get_time();
    new_admm.solve(H, h, A_eq, Alb, Aub, lbx, ubx);
    auto stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "QP status: " << new_admm.info().status << " iterations: " << new_admm.info().iter
              << " size: " << sizeof (new_admm) << " time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    new_admm.solve(H, h, A_eq, Alb, Aub, lbx, ubx);

    auto admm_primal = new_admm.primal_solution();
    auto admm_dual   = new_admm.dual_solution();

    preconditioner.unscale(H, h, A_eq, Alb, Aub, lbx, ubx);

    // create box ADMM solver
    box_admm_solver box_admm;
    start = get_time();
    box_admm.solve(H, h, A_eq, Alb, Aub, lbx, ubx);
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Box QP status: " << box_admm.info().status << " iterations: " << box_admm.info().iter
              << " size: " << sizeof (box_admm) << " time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    auto b_admm_primal = box_admm.primal_solution();
    auto b_admm_dual   = box_admm.dual_solution();

    preconditioner.unscale(admm_primal, admm_dual);


    // solve with OSQP
    start = get_time();
    osqp_solver.solve(H, h, A_eq, Alb, Aub, lbx, ubx);
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "OSQP status: " << osqp_solver.info().status << " iterations: " << osqp_solver.info().iter
              << " size: " << sizeof (osqp_solver) << " time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    auto osqp_primal = osqp_solver.primal_solution();
    auto osqp_dual   = osqp_solver.dual_solution();

    std::cout << "Box ADMM (dual): \n" << b_admm_dual.transpose() << "\n";
    std::cout << "OSQP (dual): \n" << osqp_dual.transpose() << "\n";
    std::cout << "Solvers error: " <<  (admm_primal - b_admm_primal).template lpNorm<Eigen::Infinity>() << "\n";
    std::cout << "Solvers error (dual): " <<  (b_admm_dual - osqp_dual).template lpNorm<Eigen::Infinity>() << "\n";
    std::cout << "Dual residual: " << (admm_dual - osqp_dual).template lpNorm<Eigen::Infinity>() << "\n";
    */

    /** test filter */
    /**
    LSFilter<RobotOCP::scalar_t> filter;
    filter.add(1, 3);
    filter.add(3, 2.5);
    filter.add(4, 2);
    filter.add(5, 1);
    filter.print();

    std::cout << "Is (6 , 3) acceptable: " << filter.is_acceptable(6, 3) << "\n";
    std::cout << "Is (2 , 1) acceptable: " << filter.is_acceptable(2, 1) << "\n";

    filter.add(0.5, 2);
    //filter.add(0.5, 0.5);
    filter.print();
    */

    return EXIT_SUCCESS;
}

