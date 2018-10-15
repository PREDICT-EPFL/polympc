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
