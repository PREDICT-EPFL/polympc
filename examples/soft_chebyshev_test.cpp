#include "chebyshev_soft.hpp"
#include "mobile_robot.hpp"
#include "iomanip"

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

static constexpr int POLY_ORDER = 5;
static constexpr int NUM_SEGMENTS = 2;

int main()
{
    SoftChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, 3, 2, 0, 0> cheb;
    MobileRobot robot;
    casadi::Function ode = robot.getDynamics();

    casadi::SX varx = cheb.VarX();
    casadi::SX varu = cheb.VarU();
    casadi::SX opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    casadi::SX G = cheb.CollocateDynamics(ode, -1.0, 1.0);

    casadi::Function G_fun = casadi::Function("dynamics", {opt_var}, {G});
    casadi::DM x_init = casadi::DM::ones(opt_var.size1());
    casadi::DMVector result;


    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    result = G_fun(x_init);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    //std::cout << "Casadi constraint: " << result[0].T() << "\n";

    /** compute gradient */
    casadi::SX G_gradient = casadi::SX::gradient(G, opt_var);
    /** Augmented Jacobian */
    casadi::Function G_gradient_fun = casadi::Function("gradient",{opt_var}, {G_gradient});

    start = get_time();
    casadi::DM jac = G_gradient_fun(x_init)[0];
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Gradient evaluation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    /** Hessian evaluation */
    casadi::SX G_hessian = casadi::SX::hessian(G, opt_var);
    casadi::Function G_hessian_fun = casadi::Function("L_grad",{opt_var}, {G_hessian});

    start = get_time();
    casadi::DM G_val_hess = G_hessian_fun({x_init})[0];
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Cost time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    return 0;
}
