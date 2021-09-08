// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MOBILE_ROBOT_HPP
#define MOBILE_ROBOT_HPP

#include "casadi/casadi.hpp"
#include <chrono>
#include "sys/stat.h"
#include <fstream>


struct MobileRobotProperties
{
    MobileRobotProperties(const double &_wb = 1.0) : wheel_base(_wb){}
    ~MobileRobotProperties(){}
    double wheel_base;
};

/** Relatively simple mobile model : tricycle */
class MobileRobot
{
public:
    MobileRobot(const MobileRobotProperties &props);
    MobileRobot();
    ~MobileRobot(){}

    casadi::Function getDynamics(){return NumDynamics;}
    casadi::Function getOutputMapping(){return OutputMap;}
private:
    casadi::SX state;
    casadi::SX control;
    casadi::SX Dynamics;

    casadi::Function NumDynamics;
    casadi::Function OutputMap;
};



#endif // MOBILE_ROBOT_HPP
