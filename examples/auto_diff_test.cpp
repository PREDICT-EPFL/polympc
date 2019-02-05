#include "eigen3/Eigen/Core"
#include "eigen3/unsupported/Eigen/AutoDiff"
#include <iostream>
#include <chrono>
#include <iomanip>
#include "casadi/casadi.hpp"

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



template <typename Scalar>
struct MobileRobot
{
    /*
   * Definitions required by AutoDiffJacobian.
   */
    typedef Eigen::Matrix<Scalar, 3, 1> InputType;
    typedef Eigen::Matrix<Scalar, 3, 1> ValueType;

    /*
   * Implementation
   */
    MobileRobot() : V(1.0), phi(0.0) {}
    Scalar V, phi;
    template <typename X, typename Y>
    void operator() (const X &state, Y *_output) const
    {
        Y &output = *_output;
        /* Implementation... */
        const Scalar L = Scalar(1);

        output[0] = V * sin(state[2]) * cos(phi);
        output[1] = V * cos(state[2]) * cos(phi);
        output[2] = V * sin(phi) / L;
    }
};


template <typename Scalar, typename Functor>
struct Residual
{
    Residual(const Scalar &_t = Scalar(1), const Scalar &_dt = 0.1) : t(_t), dt(_dt){}
    Scalar t, dt;

    Functor f;

    template <typename T1, typename T2, typename T3>
    void operator() (const T1 &x0, const T2 &u, T3 &integral)
    {
        eigen_assert(u.size() != 2);
        f.V   = 1.0;//u[0];
        f.phi = M_PI_4; u[1];

        /** initialize the integral */
        integral = T3(0);

        Scalar time = Scalar(0);
        T1 x = x0, xdot;
        integral = x0.dot(x0);
        while(time <= t)
        {
            f(x, &xdot);
            x += dt * xdot;
            integral += x.dot(x);
            time += dt;
        }
    }
};


int main(void)
{
    Eigen::Matrix<double,3,1> x0;
    x0 << 1.0, 1.0, M_PI_4;
    Eigen::Matrix<double,2,1>u;
    u << 1.0, M_PI_4;
    Eigen::Matrix<double,3,3> jacobian;
    Eigen::Matrix<double,3,1> output;

    MobileRobot<double> lox;
    lox.V  = 10;
    lox.phi = M_PI_4;

    Eigen::AutoDiffJacobian<MobileRobot<double>> AD_jacobian(lox);
    Eigen::Matrix<double, 3, 1> dx;
    dx << 0.1, 0.1, 0.05;
    std::chrono::time_point<std::chrono::system_clock> start = get_time();

    for(int i = 0; i <= 100; ++i)
    {
        x0 += dx;
        AD_jacobian(x0, &output, &jacobian);
    }

    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    std::cout << "Input: \n" << x0 << "\n";
    std::cout << "Ouput: \n" << output << "\n";
    std::cout << "Jacobian: \n" << jacobian << "\n";

    /** check computations with CAsadi */
    casadi::SX x = casadi::SX::sym("x",3);
    casadi::SX x_dot = lox.V * cos(x[2]) * cos(lox.phi);
    casadi::SX y_dot = lox.V * sin(x[2]) * cos(lox.phi);
    casadi::SX theta_dot = lox.V * sin(lox.phi) / 1.0;
    casadi::SX f_sym = casadi::SX::vertcat({x_dot, y_dot, theta_dot});

    casadi::Function f = casadi::Function("f", {x}, {f_sym});
    casadi::SX Jacx_sym = casadi::SX::jacobian(f_sym,x);
    casadi::Function Jacx = casadi::Function("Jac",{x},{Jacx_sym});

    Jacx.generate("Jacx");

    casadi::DM x_init = casadi::DM({1.0, 1.0, M_PI_4});
    casadi::DM dx_dm = casadi::DM({ 0.1, 0.1, 0.05});
    casadi::DM result;
    start = get_time();
    for(int i = 0; i <= 100; ++i)
    {
        x_init += dx_dm;
        result = Jacx({x_init})[0];
    }
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    std::cout << "Input: \n" << x_init << "\n";
    std::cout << "Jacobian: \n" << result << "\n";


    /** Second example : Residual differentiation */
    Residual<double, MobileRobot<double>> residual;
    double res;
    x0[0] = 1.0; x0[1] = 1.0; x0[2] = M_PI_4;
    residual(x0,u,res);

    std::cout << "Numerical integral: \n";
    std::cout << "Res: " << res << "\n";

    using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<double, 5, 1>>;
    Eigen::Matrix<ADScalar, 3, 1> Ax0;
    Eigen::Matrix<ADScalar, 2, 1> Au;

    /** initialize values */
    Ax0(0).value() = x0[0]; Ax0(1).value() = x0[1]; Ax0(2).value() = x0[2];
    Au(0).value() = u[0]; Au(1).value() = u[1];

    /** initialize derivatives */
    int div_size = Ax0.size() + Au.size();
    int derivative_idx = 0;
    for(int i = 0; i < Ax0.size(); ++i)
    {
        Ax0(i).derivatives() =  Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        derivative_idx++;
    }

    for(int i = 0; i < Au.size(); ++i)
    {
        Au(0).derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        derivative_idx++;
    }

    ADScalar Ares;
    residual(Ax0, Au, Ares);
    std::cout << "AD result: " << Ares << "\n";


    return 0;
}
