#include "mobile_robot.hpp"

using namespace casadi;

MobileRobot::MobileRobot(const MobileRobotProperties &props)
{
    SX x     = SX::sym("x");
    SX y     = SX::sym("y");
    SX theta = SX::sym("theta");
    state = SX::vertcat({x, y, theta});

    SX v   = SX::sym("v");
    SX phi = SX::sym("phi");
    control = SX::vertcat({v, phi});

    /** Dynamic equations */
    double L = 1.0;
    Dynamics = SX::vertcat({v * cos(theta) * cos(phi), v * sin(theta) * cos(phi), v * sin(phi) / L});
    NumDynamics = Function("Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    OutputMap = Function("Map",{state}, {state});
}

MobileRobot::MobileRobot()
{
    SX x     = SX::sym("x");
    SX y     = SX::sym("y");
    SX theta = SX::sym("theta");
    state = SX::vertcat({x, y, theta});

    SX v   = SX::sym("v");
    SX phi = SX::sym("phi");
    control = SX::vertcat({v, phi});

    /** Dynamic equations */
    double L = 1.0;
    Dynamics = SX::vertcat({v * cos(theta) * cos(phi), v * sin(theta) * cos(phi), v * sin(phi) / L});
    NumDynamics = Function("Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    OutputMap = Function("Map",{state}, {state});
}
