// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef KITE_H
#define KITE_H

#include "casadi/casadi.hpp"
#include <chrono>
#include "sys/stat.h"
#include <fstream>


struct SimpleKinematicKiteProperties
{
    double tether_length;
    double wind_speed;
    double gliding_ratio;
};

/** Relatively simple kite model : tricycle on a sphere */
class SimpleKinematicKite
{
public:
    SimpleKinematicKite(const SimpleKinematicKiteProperties &KiteProps);
    SimpleKinematicKite();
    virtual ~SimpleKinematicKite(){}

    casadi::Function getDynamics(){return NumDynamics;}
    casadi::Function getOutputMapping(){return OutputMap;}
private:
    casadi::SX state;
    casadi::SX control;
    casadi::SX Dynamics;

    casadi::Function NumDynamics;
    casadi::Function OutputMap;
};


#endif // KITE_H
