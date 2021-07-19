#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"
#include "solvers/ipopt_interface.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#include "control/simple_robot_model.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, DENSE>
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

int main(void)
{
    IpoptInterface<RobotOCP> solver;
    //solver.get_problem().set_Q_coeff(1.0);
    //solver.get_problem().set_time_limits(0, 2); // another way to set optimisation horizon
    solver.settings().SetIntegerValue("max_iter", 10);
    solver.parameters()(0) = 2.0;
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

    std::cout << "Solve status: " << solver.info().status << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
              << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
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

    std::cout << "Solve status: " << solver.info().status << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
              << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    return EXIT_SUCCESS;
}
