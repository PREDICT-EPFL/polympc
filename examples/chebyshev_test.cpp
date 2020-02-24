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

static constexpr int POLY_ORDER = 5;
static constexpr int NUM_SEGMENTS = 2;

int main()
{
    Chebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, 3, 2, 0> cheb;
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
    SX th_dot = v * sin(phi) / 2.0;

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
    //result = G_fun(x_init);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    //std::cout << "Casadi constraint: " << result[0].T() << "\n";


    /** compute Jacobian */
    casadi::SX G_jacobian = casadi::SX::jacobian(G, opt_var);
    /** Augmented Jacobian */
    casadi::Function AugJacobian = casadi::Function("aug_jacobian",{opt_var}, {G_jacobian});

    start = get_time();
    DM jac = AugJacobian(x_init)[0];
    stop = get_time();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Jacobian time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";
    std::cout << "Jacobian: \n" << jac(casadi::Slice(0, jac.size1()), casadi::Slice(jac.size2() - 6, jac.size2())) << "\n";

    casadi::Function Mayer = casadi::Function("mayer",{state},{casadi::SX::dot(state, state)});
    casadi::Function Lagrange = casadi::Function("lagrange",{state, control}, {casadi::SX::dot(state, state) + casadi::SX::dot(control, control)});

    casadi::SX L = cheb.CollocateCost(Mayer, Lagrange, -1, 1);
    casadi::SX L_grad = casadi::SX::gradient(L, opt_var);
    casadi::SX L_hess = casadi::SX::hessian(L, opt_var);

    /** evaluate L */
    casadi::Function L_f = casadi::Function("L",{opt_var},{L});
    casadi::Function L_f_grad = casadi::Function("L_grad",{opt_var}, {L_grad});
    casadi::Function L_f_hess = casadi::Function("L_grad",{opt_var}, {L_hess});

    start = get_time();
    casadi::DM L_val = L_f({x_init})[0];
    casadi::DM L_val_grad = L_f_grad({x_init})[0];
    casadi::DM L_val_hess = L_f_hess({x_init})[0];
    stop = get_time();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Cost time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    std::cout << "Cost function value: " << L_val << "\n";
    std::cout << "Cost function gradient: " << L_val_grad << "\n";
    std::cout << "Cost function hessian: \n" << L_val_hess << "\n";


    /** differentiation test */
    casadi::SX poly_x = pow(x,2) + 1;
    casadi::Function poly = casadi::Function("poly", {state, control}, {poly_x});

    casadi::SX deriv = cheb.DifferentiateFunction(poly);
    casadi::Function deriv_f = casadi::Function("deriv", {varx}, {deriv});
    casadi::DM lin_space = casadi::DM({0, 0.0477, 0.1727, 0.3273, 0.4523, 0.5000, 0.5477, 0.6727, 0.8273, 0.9523, 1.0000});
    casadi::DM points = casadi::DM::zeros(3, NUM_SEGMENTS * POLY_ORDER + 1);
    points(0, casadi::Slice()) = pow(lin_space,2);
    std::cout << "x: \n" << points << "\n";
    std::cout << "Derivative: " << deriv_f(casadi::DMVector{casadi::DM::vec(points)})[0] << "\n";
}
