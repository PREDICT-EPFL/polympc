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

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ ParkingOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 1, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class ParkingOCP : public ContinuousOCP<ParkingOCP, Approximation, DENSE>
{
public:
    ~ParkingOCP() = default;

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = p(0) * u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = p(0) * u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = p(0) * u(0) * sin(u(1)) / d(0);

        polympc::ignore_unused_var(t);
    }

    /**
    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = R.toDenseMatrix().template cast<T>();

        lagrange = x.dot(Qm * x) + u.dot(Rm * u);
    }
    */

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        mayer = p(0);

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
    }
};

/** create solver */
template<typename Problem, typename QPSolver> class Solver;
template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ,
                                             typename Problem::scalar_t, Problem::MATRIXFMT, linear_solver_traits<ParkingOCP::MATRIXFMT>::default_solver>>
class Solver : public SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;


    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }
};


int main(void)
{
    using mpc_t = MPC<ParkingOCP, Solver>;
    mpc_t mpc;
    mpc.settings().max_iter = 20;
    mpc.settings().line_search_max_iter = 10;

    // problem data
    mpc_t::static_param p; p << 2.0;          // robot wheel base
    mpc_t::state_t x0; x0 << 0.5, 0.5, 0.5;   // initial condition
    mpc_t::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
    mpc_t::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control
    mpc_t::parameter_t lbp; lbp << 0;         // lower bound on time
    mpc_t::parameter_t ubp; ubp << 10;        // upper bound on time
    mpc_t::state_t lbx_f; lbx_f << 0, 0, 0;   // lower bound on final position
    mpc_t::state_t ubx_f = lbx_f;             // upper bound  =  lower bound

    mpc.set_static_parameters(p);
    mpc.control_bounds(lbu, ubu);
    mpc.parameters_bounds(lbp, ubp);
    mpc.final_state_bounds(lbx_f, ubx_f);
    mpc.initial_conditions(x0);

    polympc::time_point start = polympc::get_time();
    mpc.solve();
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "MPC status: " << mpc.info().status.value << "\n";
    std::cout << "Num iterations: " << mpc.info().iter << "\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
    std::cout << "Solution X: \n" << mpc.solution_x_reshaped() << "\n";

    return EXIT_SUCCESS;
}
