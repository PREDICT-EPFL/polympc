#include "Eigen/Core"
#include "unsupported/Eigen/AutoDiff"
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


/** No Jacobian example */
template <typename Scalar>
struct MobileRobot2
{
    MobileRobot2(){}
    ~MobileRobot2(){}

    template <typename X, typename U, typename Y>
    void operator() (const X &state, const U &control, Y &output)
    {
        const Scalar L = Scalar(1);

        output[0] = control[0] * sin(state[2]) * cos(control[1]);
        output[1] = control[0] * cos(state[2]) * cos(control[1]);
        output[2] = control[0] * sin(control[1]) / L;
    }
};


template <typename Scalar, typename Functor>
struct Residual
{
    Residual(const Scalar &_t = Scalar(1), const Scalar &_dt = 0.05) : t(_t), dt(_dt){}
    Scalar t, dt;

    Functor f;

    template <typename T1, typename T2, typename T3>
    void operator() (const T1 &x0, const T2 &u, T3 &integral)
    {
        //eigen_assert(u.size() != 2);

        /** initialize the integral */
        integral = T3(0);

        Scalar time = Scalar(0);
        T1 x = x0, xdot;
        integral = x0.dot(x0);

        while(time <= t)
        {
            f(x, u, xdot);
            x += static_cast<T3>(dt) * xdot;
            /** @bug : dirty hack with static cast :(( */
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
    casadi::SX x_dot = lox.V * cos(x(2)) * cos(lox.phi);
    casadi::SX y_dot = lox.V * sin(x(2)) * cos(lox.phi);
    casadi::SX theta_dot = lox.V * sin(lox.phi) / 1.0;
    casadi::SX f_sym = casadi::SX::vertcat({x_dot, y_dot, theta_dot});

    casadi::Function f = casadi::Function("f", {x}, {f_sym});
    casadi::SX Jacx_sym = casadi::SX::jacobian(f_sym,x);
    casadi::Function Jacx = casadi::Function("Jac",{x},{Jacx_sym});

    Jacx.generate("Jacx");

    casadi::DM x_init = casadi::DM(std::vector<double>{1.0, 1.0, M_PI_4});
    casadi::DM dx_dm = casadi::DM(std::vector<double>{ 0.1, 0.1, 0.05});
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
    Residual<double, MobileRobot2<double>> residual;
    double res;
    x0[0] = 1.0; x0[1] = 1.0; x0[2] = M_PI_4;
    u[0] = 1.0; u[1] = M_PI_4;
    residual(x0,u,res);

    std::cout << "Numerical integral: \n";
    std::cout << "Res: " << res << "\n";

    using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<double, 5, 1>>;
    using outer_deriv_type = Eigen::Matrix<ADScalar, 5, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outer_deriv_type>;
    Eigen::Matrix<outerADScalar, 3, 1> Ax0;
    Eigen::Matrix<outerADScalar, 2, 1> Au;

    /** initialize values */
    for(int i = 0; i < Ax0.SizeAtCompileTime; ++i) Ax0(i).value().value() = x0[i];
    for(int i = 0; i < Au.SizeAtCompileTime; ++i) Au(i).value().value() = u[i];

    /** initialize derivatives */
    int div_size = Ax0.size() + Au.size();
    int derivative_idx = 0;
    for(int i = 0; i < Ax0.size(); ++i)
    {
        Ax0(i).value().derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        Ax0(i).derivatives() =  Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        // initialize hessian matrix to zero
        for(int idx=0; idx<div_size; idx++)
        {
            Ax0(i).derivatives()(idx).derivatives()  = Eigen::Matrix<double, 5, 1>::Zero();
        }
        derivative_idx++;
    }

    for(int i = 0; i < Au.size(); ++i)
    {
        Au(i).value().derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        Au(i).derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        for(int idx=0; idx<div_size; idx++)
        {
            Au(i).derivatives()(idx).derivatives()  = Eigen::Matrix<double, 5, 1>::Zero();
        }
        derivative_idx++;
    }

    outerADScalar Ares;
    residual(Ax0, Au, Ares);
    std::cout << "AD result: " << Ares.value().value() << "\n";
    std::cout << "AD derivatives: " << Ares.value().derivatives().transpose() << "\n";

    //Ax0(0).value() = 100.0;
    //residual(Ax0, Au, Ares);
    //std::cout << "AD derivatives: " << Ares.derivatives().transpose() << "\n";


    start = get_time();
    for(int i = 0; i < 100; ++i)
    {
        residual(Ax0, Au, Ares);
        Ax0(0).value().value() += dx[0];
        Ax0(1).value().value() += dx[1];
        Ax0(2).value().value() += dx[2];
    }
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "AD derivatives: " << Ares.value().derivatives().transpose() << "\n";
    std::cout << "Eigen Gradient time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    /** @note 10x slower than using simple AD */


    /** compute the Hessian */
    /** allocate hessian */
    Eigen::Matrix<double, 5, 5> hessian = Eigen::Matrix<double, 5, 5>::Zero();
    for(int i = 0; i < Ares.derivatives().SizeAtCompileTime; ++i)
    {
        hessian.middleRows(i,1) = Ares.derivatives()(i).derivatives().transpose();
    }
    std::cout << "Hessian: " << "\n" <<  hessian << "\n";


    return 0;
}
