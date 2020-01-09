#include "integrator.h"

using namespace casadi;


class KinematicBicycle {
public:
    KinematicBicycle();
    ~KinematicBicycle() {}

    casadi::Function getDynamics() { return NumDynamics; }
    casadi::Function getParamDynamics() { return ParamDynamics; }

private:
    casadi::SX state;
    casadi::SX control;
    casadi::SX parameters;
    casadi::SX Dynamics;

    casadi::Function ParamDynamics;
    casadi::Function NumDynamics;
};

KinematicBicycle::KinematicBicycle() {
    SX x = SX::sym("x");
    SX y = SX::sym("y");
    SX theta = SX::sym("theta");
    state = SX::vertcat({x, y, theta});

    SX v = SX::sym("v");
    SX phi = SX::sym("phi");
    control = SX::vertcat({v, phi});

    SX L = SX::sym("L");
    parameters = SX::vertcat({L});

    DM L_num = 2.8;
    DM parameters_num = DM::vertcat({L_num});
    /** Dynamic equations */
    Dynamics = SX::vertcat({v * cos(theta) * cos(phi), v * sin(theta) * cos(phi), v * sin(phi) / L});
    ParamDynamics = Function("ParamDynamics", {state, control, parameters}, {Dynamics});
    NumDynamics = Function("Dynamics", {state, control}, {ParamDynamics(SXVector({state, control, parameters_num}))[0]});
}

int main(void)
{
    /** kite object */
    KinematicBicycle car;
    Function ode = car.getDynamics();

    double tf = 2.0;
    casadi::DMDict props;
    props["scale"] = 0;
    casadi::Dict solver_options;
    solver_options["ipopt.linear_solver"] = "mumps";

    const int NUM_SEGMENTS = 2;
    const int POLY_ORDER = 5;

    PSODESolver<POLY_ORDER,NUM_SEGMENTS,3,2>ps_solver(ode, tf, props, solver_options);

    /** solve the problem */
    DMDict ps_sol;

    DM init_state = DM::vertcat({0.0, 10.0, 0.0});
    DM control = DM::vertcat({19.9198, 0.1139, 20.1185, -0.0368, 19.8883, 0.0565, 20.3789, -0.1054, 20.2985, 0.3564, 0.1000, 0.0981, 29.3983,
                              -0.5047,   25.2032,    0.3547,   22.8910,    0.5236,   48.0521,   -0.5236,   50.0000,   -0.5236});

    bool FULL = true;
    ps_sol = ps_solver.solve_trajectory(init_state, control, FULL);
    std::cout << "PS:" << ps_sol.at("x") << "\n";

    casadi::DM points = casadi::DM::vertcat({0, 0.0477, 0.1727, 0.3273, 0.4523, 0.5000});
    casadi::DM v      = casadi::DM::vertcat({19.9198, 20.1185, 19.8883, 20.3789, 20.2985, 0.1000});
    casadi::DM phi    = casadi::DM::vertcat({0.1139, -0.0368, 0.0565, -0.1054, 0.3564, 0.0981});
    casadi::Function f_v   = polymath::lagrange_interpolant(points, v);
    casadi::Function f_phi = polymath::lagrange_interpolant(points, phi);

    std::cout << f_v(casadi::DM{0.1})[0] << " " << f_phi(casadi::DM({0.1}))[0] << "\n";

    //Eigen::VectorXd points(4);
    //points << 1.0, 0.5, -0.5, -1.0;
    //Eigen::VectorXd poly = polymath::lagrange_interpolant(points, points);
}



