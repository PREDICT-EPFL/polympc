// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "kite.h"

using namespace casadi;

SimpleKinematicKite::SimpleKinematicKite(const SimpleKinematicKiteProperties &KiteProps)
{
    SX theta = SX::sym("theta");
    SX phi   = SX::sym("phi");
    SX gamma = SX::sym("gamma");
    state = SX::vertcat({theta, phi, gamma});

    SX u_g   = SX::sym("u_gamma");
    control = SX::vertcat({u_g});

    double L  = KiteProps.tether_length;
    double E  = KiteProps.gliding_ratio;
    double ws = KiteProps.wind_speed;

    /** assume constant tether flight reel-out(in) speed = 0*/
    double z = 0;

    SX vw = SX::vertcat({ws,0,0});
    SX M = SX::diag(SX::vertcat({(1/L), (1/L * cos(theta))}));
    /** try matrix definition with vertcat here instaed */

    SX R_GN = SX::zeros(3,3);
    R_GN(0,0) = -sin(theta) * cos(phi);  R_GN(0,1) = -sin(theta);  R_GN(0,2) = -cos(theta) * cos(phi);
    R_GN(1,0) = -sin(theta) * sin(phi);  R_GN(1,1) = cos(theta);   R_GN(1,2) = -cos(theta) * sin(phi);
    R_GN(2,2) = cos(theta);              R_GN(2,1) = 0;            R_GN(2,2) = -sin(theta);

    SX Rb_NK = SX::zeros(2,2);
    SX R_NK    = SX::eye(3);
    Rb_NK(0,0) = cos(gamma);  Rb_NK(0,1) = -sin(gamma);
    Rb_NK(1,0) = sin(gamma);  Rb_NK(1,1) = cos(gamma);
    R_NK(Slice(0,2), Slice(0,2)) = Rb_NK;

    SX EM = SX::zeros(2,3);
    EM(0,0) = 1; EM(0,2) = -E;

    /** multiplications */
    SX MRb_NKE = SX::mtimes(SX::mtimes(M, Rb_NK), EM);
    SX MRb_NKERNR = SX::mtimes(SX::mtimes(MRb_NKE, R_NK.T()), R_GN.T());
    SX qdot = SX::mtimes(MRb_NKERNR, vw) - SX::mtimes(Rb_NK, SX::vertcat({E * z, 0}));
    Dynamics = SX::vertcat({qdot, u_g});
    NumDynamics = Function("SKK_Dynamics", {state, control}, {Dynamics});
}

SimpleKinematicKite::SimpleKinematicKite()
{
    SX theta = SX::sym("theta");
    SX phi   = SX::sym("phi");
    SX gamma = SX::sym("gamma");
    state = SX::vertcat({theta, phi, gamma});

    SX u_g   = SX::sym("u_gamma");
    control = SX::vertcat({u_g});

    double L  = 5;
    double E  = 5;
    double ws = 3;

    /** assume constant tether flight reel-out(in) speed = 0*/
    double z = 0;

    SX vw = SX::vertcat({ws,0,0});
    SX M = SX::diag(SX::vertcat({(1/L), (1/L * cos(theta))}));
    /** try matrix definition with vertcat here instaed */

    SX R_GN = SX::zeros(3,3);
    R_GN(0,0) = -sin(theta) * cos(phi);  R_GN(0,1) = -sin(theta);  R_GN(0,2) = -cos(theta) * cos(phi);
    R_GN(1,0) = -sin(theta) * sin(phi);  R_GN(1,1) = cos(theta);   R_GN(1,2) = -cos(theta) * sin(phi);
    R_GN(2,2) = cos(theta);              R_GN(2,1) = 0;            R_GN(2,2) = -sin(theta);

    SX Rb_NK = SX::zeros(2,2);
    SX R_NK    = SX::eye(3);
    Rb_NK(0,0) = cos(gamma);  Rb_NK(0,1) = -sin(gamma);
    Rb_NK(1,0) = sin(gamma);  Rb_NK(1,1) = cos(gamma);
    R_NK(Slice(0,2), Slice(0,2)) = Rb_NK;

    SX EM = SX::zeros(2,3);
    EM(0,0) = 1; EM(0,2) = -E;

    /** multiplications */
    SX MRb_NKE = SX::mtimes(SX::mtimes(M, Rb_NK), EM);
    SX MRb_NKERNR = SX::mtimes(SX::mtimes(MRb_NKE, R_NK.T()), R_GN.T());
    SX qdot = SX::mtimes(MRb_NKERNR, vw) - SX::mtimes(Rb_NK, SX::vertcat({E * z, 0}));
    Dynamics = SX::vertcat({qdot, u_g});
    NumDynamics = Function("SKK_Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    SX H = SX::zeros(2,3);
    H(0,0) = 1; H(1,1) = 1;
    OutputMap = Function("Map",{state}, {SX::mtimes(H, state)});
}
