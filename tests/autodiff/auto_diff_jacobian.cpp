// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "autodiff/AutoDiffScalar.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <type_traits>


template<typename scalar_t>
using state_t = Eigen::Matrix<scalar_t, 14, 1>;

template<typename T>
void flight_dynamics(const state_t<T>& x, state_t<T>& xdot, const double &t)
{
    /** constants */
    const double dry_mass = 41.558;
    Eigen::Matrix<double, 3, 1> rocket_control; rocket_control << 0, 0, 2000;
    Eigen::Matrix<double, 3, 1> I_inv; I_inv << 1.0/47, 1.0/47, 1.0/2;
    Eigen::Matrix<double, 3, 1> total_torque; total_torque << 0, 0, 0;
    const double Isp = 213;

    // -------------- Simulation variables -----------------------------
    T g0 = 3.986e14/pow(6371e3+x(2), 2);  // Earth gravity in [m/s^2]
    T mass = dry_mass + x(13);                  // Instantaneous mass of the rocket in [kg]

    // Orientation of the rocket with quaternion
    Eigen::Quaternion<T> attitude(x(9), x(6), x(7), x(8));
    attitude.normalize();
    Eigen::Matrix<T, 3, 3> rot_matrix = attitude.toRotationMatrix();

    // Force in inertial frame: gravity
    Eigen::Matrix<T, 3, 1> gravity; gravity << 0, 0, g0*mass;

    // Total force in inertial frame [N]
    Eigen::Matrix<T, 3, 1> total_force;  total_force << 0,0,0;

    // Angular velocity omega in quaternion format to compute quaternion derivative
    Eigen::Quaternion<T> omega_quat(0.0, x(10), x(11), x(12));

    // -------------- Differential equation ---------------------

    // Position variation is speed
    xdot.head(3) = x.segment(3,3);

    // Speed variation is Force/mass
    xdot.segment(3,3) = total_force/mass;

    // Quaternion variation is 0.5*w◦q
    xdot.segment(6, 4) =  0.5*(omega_quat*attitude).coeffs();

    // Angular speed variation is Torque/Inertia
    xdot.segment(10, 3) = rot_matrix*(total_torque.cwiseProduct(I_inv));

    // Mass variation is proportional to total thrust
    xdot(13) = -rocket_control.norm()/(Isp*g0);
}

int main(void)
{
    Eigen::Matrix<double, 14, 14> jacobian;
    state_t<double> output; // xdot

    using ADScalar = Eigen::AutoDiffScalar<state_t<double>>;
    using ad_state_t = state_t<ADScalar>;
    ad_state_t x, xdot;

    // set linearisation point and seed derivatives
    x = state_t<double>::Zero();
    /** initialize derivatives */
    int div_size = x.size();
    int derivative_idx = 0;
    for(int i = 0; i < x.size(); ++i)
    {
        x(i).derivatives() =  state_t<double>::Unit(div_size, derivative_idx);
        derivative_idx++;
    }

    // propagate through the function
    flight_dynamics<ADScalar>(x,xdot,0);

    // obtain the value and jacobian
    for(int i = 0; i < x.size(); i++)
    {
        output(i) = xdot(i).value();
        jacobian.row(i) = xdot(i).derivatives();
    }

    std::cout << "Jacoabian: \n" << jacobian << "\n";
    std::cout << "xdot: " << output.transpose() << "\n";

    return EXIT_SUCCESS;
}
