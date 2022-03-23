#include <iostream>
#include <iomanip>
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"
#include "solvers/ipopt_interface.hpp"

using namespace Eigen;
using namespace polympc;

#define POLY_ORDER 5
#define NUM_SEG 2

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ PendulumOCP, /*NX*/ 2, /*NU*/ 1, /*NP*/ 0, /*ND*/ 0, /*NG*/0, /*TYPE*/ double)

class PendulumOCP : public polympc::ContinuousOCP<PendulumOCP, Approximation, polympc::DENSE>
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

int main()
{
    polympc::IpoptInterface<PendulumOCP> solver;
    solver.get_problem().set_time_limits(0, 3); // another way to set optimisation horizon
    solver.settings().SetIntegerValue("print_level", 5);
    Eigen::Matrix<PendulumOCP::scalar_t, 2, 1> init_cond; init_cond << M_PI, 0;
    Eigen::Matrix<PendulumOCP::scalar_t, 1, 1> ub; ub <<  1;
    Eigen::Matrix<PendulumOCP::scalar_t, 1, 1> lb; lb << -1;

    solver.upper_bound_x().tail(11) = ub.replicate(11, 1);
    solver.lower_bound_x().tail(11) = lb.replicate(11, 1);

    solver.upper_bound_x().segment(20, 2) = init_cond;
    solver.lower_bound_x().segment(20, 2) = init_cond;

    polympc::time_point start = polympc::get_time();
    solver.solve();
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Solve status: " << solver.info().status << "\n";
    std::cout << "Num iterations: " << solver.info().iter << "\n";
    std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
              << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) / 1000 << " ms\n";
    std::cout << "Size of the solver: " << sizeof (solver) << "\n";
    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    return EXIT_SUCCESS;
}
