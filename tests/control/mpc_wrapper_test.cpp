#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "control/mpc_wrapper.hpp"
#include "polynomials/splines.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#include "control/simple_robot_model.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/admm.hpp"
#include "solvers/qp_preconditioners.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, SPARSE>
{
public:
    ~RobotOCP() = default;

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

    void set_Q_coeff(const scalar_t& coeff)
    {
        Q.diagonal() << coeff, coeff, coeff;
    }
};

/** create solver */
template<typename Problem, typename QPSolver> class MySolver;

template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::DUAL_SIZE, typename Problem::scalar_t>>
class MySolver : public SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;

    //EIGEN_STRONG_INLINE const typename Base::_Problem& get_problem() const noexcept { return this->problem; }
    EIGEN_STRONG_INLINE Problem& get_problem() noexcept { return this->problem; }


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

using preconditioner_t = polympc::RuizEquilibration<RobotOCP::scalar_t, RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::MATRIXFMT>;

// segment experiment
/**
class Lox
{
public:
    Lox()
    {
        vector = vec_t::LinSpaced(10,0,9);
        std::cout << vector.transpose() << "\n";
    }
    using vec_t = Eigen::Matrix<double, 10, 1>;
    using segment_t = Eigen::Matrix<double, 5, 1>;
    vec_t vector;

    inline vec_t get_vector() const noexcept
    {
        std::cout << "vector: " << vector.transpose() << "\n";
        return vector;
    }

    inline segment_t get_segment() const noexcept
    {
        std::cout << "segment: " << vector.template head<5>().transpose() << "\n";
        return vector.template head<5>();
    }

    inline void get_segment(segment_t& seg) const noexcept
    {
        seg = vector.template segment<5>(0);
    }

    inline double* get_segemnt_ptr() noexcept
    {
        return vector.template head<5>().data();
    }

};
*/





int main(void)
{
    /**
    Lox var;
    auto tmp = var.get_segment();
    Lox::segment_t tmp2;
    var.get_segment(tmp2);

    double *vector  = var.get_vector().data();
    double *seg = var.get_segemnt();

    for (int i =0; i <  5; ++i)
        std::cout << seg[i] << " ";
    std::cout << std::endl;
    for (int i =0; i <  5; ++i)
        std::cout << vector[i] << " ";
    std::cout << std::endl;
    */

    using mpc_t = MPC<RobotOCP, MySolver, box_admm_solver>;
    mpc_t mpc;
    mpc.ocp().set_Q_coeff(2.0);
    mpc.settings().max_iter = 20;
    mpc.settings().line_search_max_iter = 10;
    mpc.set_time_limits(0, 2);
    // problem data
    mpc_t::static_param p; p << 2.0;          // robot wheel base
    mpc_t::state_t x0; x0 << 0.5, 0.5, 0.5;   // initial condition
    mpc_t::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
    mpc_t::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control

    mpc.set_static_parameters(p);
    mpc.control_bounds(lbu, ubu);
    mpc.initial_conditions(x0);

    polympc::time_point start = polympc::get_time();
    mpc.solve();
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "MPC status: " << mpc.info().status.value << "\n";
    std::cout << "Num iterations: " << mpc.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    std::cout << "Solution X: " << mpc.solution_x().transpose() << "\n";
    std::cout << "Solution U: " << mpc.solution_u().transpose() << "\n";

    auto segment = mpc.solution_x();
    mpc_t::scalar_t *lox;
    lox = mpc.solution_x().data(); //segment.data(); //mpc.solver().primal_solution().template head<10>().data();

    std::cout << "lox: \n";
    for(int i = 0; i < 10; ++i)
        std::cout << lox[i] << " ";
    std::cout << std::endl;

    // warm started iteration
    x0 << 0.3, 0.4, 0.5;
    mpc.initial_conditions(x0, x0);

    start = polympc::get_time();
    mpc.solve();
    stop = polympc::get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << mpc.info().status.value << "\n";
    std::cout << "Num iterations: " << mpc.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    std::cout << "Solution X: \n" << mpc.solution_x_reshaped() << "\n";
    std::cout << "Solution U: \n" << mpc.solution_u_reshaped() << "\n";

    // sample x solution at collocation points [0, 5, 10]
    std::cout << "x[0]: " << mpc.solution_x_at(0).transpose() << "\n";
    std::cout << "x[5]: " << mpc.solution_x_at(5).transpose() << "\n";
    std::cout << "x[10]: " << mpc.solution_x_at(10).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    //sample control at collocation points
    std::cout << "u[0]: " << mpc.solution_u_at(0).transpose() << "\n";
    std::cout << "u[1]: " << mpc.solution_u_at(1).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    // sample state at time 't'
    std::cout << "x(0.0): " << mpc.solution_x_at(0.0).transpose() << "\n";
    std::cout << "x(0.5): " << mpc.solution_x_at(0.5).transpose() << "\n";

    std::cout << " ------------------------------------------------ \n";

    //  sample control at time 't'
    std::cout << "u(0.0): " << mpc.solution_u_at(0.0).transpose() << "\n";
    std::cout << "u(0.5): " << mpc.solution_u_at(0.5).transpose() << "\n";

    return EXIT_SUCCESS;
}
