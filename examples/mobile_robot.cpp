#include "mobile_robot.hpp"

using namespace casadi;

MobileRobot::MobileRobot(const MobileRobotProperties &props)
{
    SX x     = SX::sym("x");
    SX y     = SX::sym("y");
    SX theta = SX::sym("theta");
    state = SX::vertcat({x, y, theta});

    SX v   = SX::sym("v");
    SX omega = SX::sym("omega");
    control = SX::vertcat({v, omega});

    /** Dynamic equations */
    Dynamics = SX::vertcat({v * cos(theta), v * sin(theta), omega});
    NumDynamics = Function("Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    SX H = SX::zeros(2,3);
    H(0,0) = 1; H(1,1) = 1;
    OutputMap = Function("Map",{state}, {SX::mtimes(H, state)});
}

MobileRobot::MobileRobot()
{
    SX x     = SX::sym("x");
    SX y     = SX::sym("y");
    SX theta = SX::sym("theta");
    state = SX::vertcat({x, y, theta});

    SX v   = SX::sym("v");
    SX omega = SX::sym("omega");
    control = SX::vertcat({v, omega});

    /** Dynamic equations */
    Dynamics = SX::vertcat({v * cos(theta), v * sin(theta), omega});
    NumDynamics = Function("Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    SX H = SX::zeros(2,3);
    H(0,0) = 1; H(1,1) = 1;
    OutputMap = Function("Map",{state}, {SX::mtimes(H, state)});
}
