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

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const static_parameter_t &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / d(0);
    }
};


int main(void)
{
    using namespace polympc;

    RobotOCP robot_nlp;
    RobotOCP::nlp_variable_t var = RobotOCP::nlp_variable_t::Ones();
    RobotOCP::nlp_constraints_t constr;
    RobotOCP::nlp_eq_jacobian_t eq_jac;
    RobotOCP::static_parameter_t p; p(0) = 2.0;

    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    for(int i = 0; i < test_NUM_EXP; ++i)
        robot_nlp.equalities_linerised(var, p, constr, eq_jac);
        //robot_nlp.equalities(var, p, constr);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Collocation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) / test_NUM_EXP << " [microseconds]" << "\n";

    std::cout << "Constraint new: " << constr(0) << "\n";
    std::cout << "Size of NLP:" << sizeof (robot_nlp) << "\n";

    /**
    start = get_time();
    robot_nlp.equalities_linerised(var, p, constr, eq_jac);
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Linearisation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) << " [microseconds]" << "\n";
              */


    Eigen::IOFormat fmt(3);
    std::cout << "Constraint \n" << constr.transpose() << "\n";
    //std::cout << eq_jac.template leftCols<29>().format(fmt) << "\n";
    std::cout << eq_jac(0,0) << "\n";

    return EXIT_SUCCESS;
}
