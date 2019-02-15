#include "chebyshev.hpp"
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

int main()
{
    Chebyshev<casadi::SX, 3, 2, 3, 2, 0> cheb;
    //std::cout << "Nodes: " << cheb.CPoints() << "\n";
    //std::cout << "Weights: " << cheb.QWeights() << "\n";
    //std::cout << "_D : \n" << cheb.D() << "\n";

    using namespace casadi;
    SX v = SX::sym("v");
    SX phi = SX::sym("phi");
    SX x = SX::sym("x");
    SX y = SX::sym("y");
    SX theta = SX::sym("theta");

    SX state = SX::vertcat({x,y,theta});
    SX control = SX::vertcat({v, phi});

    SX x_dot = v * cos(theta) * cos(phi);
    SX y_dot = v * sin(theta) * cos(phi);
    SX th_dot = v * sin(theta) / 2.0;

    SX dynamics = SX::vertcat({x_dot, y_dot, th_dot});
    Function robot = Function("robot", {state, control}, {dynamics});

    casadi::SX varx = cheb.VarX();
    casadi::SX varu = cheb.VarU();
    casadi::SX opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    SX G = cheb.CollocateDynamics(robot, -1.0, 1.0);

    Function G_fun = Function("constraint", {opt_var}, {G});
    DM x_init = DM::ones(opt_var.size1());
    DMVector result; //= G_fun(x_init);


    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    result = G_fun(x_init);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    std::cout << "Casadi constraint: " << result[0].T() << "\n";

}
