// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
