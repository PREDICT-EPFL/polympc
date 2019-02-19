#include "polynomials/ebyshev.hpp"
#include "control/nmpc.hpp"
#include <iomanip>


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


template <typename _Scalar>
struct MobileRobot2
{
    MobileRobot2(){}
    ~MobileRobot2(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 3, 1>;
    using Control    = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    void operator() (const State &state, const Control &control, const Parameters &param, State &value) const
    {
        value[0] = control[0] * cos(state[2]) * cos(control[1]);
        value[1] = control[0] * sin(state[2]) * cos(control[1]);
        value[2] = control[0] * sin(control[1]) / param[0];
    }

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedA> &value) const
    {
        value[0] = control[0] * cos(state[2]) * cos(control[1]);
        value[1] = control[0] * sin(state[2]) * cos(control[1]);
        value[2] = control[0] * sin(control[1]) / param[0];
    }
};




int main(void)
{
    using chebyshev = Chebyshev<3>;
    using collocation = polympc::ode_collocation<MobileRobot2<double>, chebyshev, 2>;

    collocation ps_ode;
    collocation::var_t x = collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    collocation::constr_t y;
    collocation::jacobian_t A;
    collocation::constr_t b;

    //ps_ode(x, y);

    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    ps_ode.linearized(x, A, b);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    std::cout << "Constraint: " << b.transpose() << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    /** compute linearized PS constraints */

    std::cout << "Jacobian: \n" << A.template rightCols<7>() << "\n";

    return 0;
}
