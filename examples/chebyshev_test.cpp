// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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

    /** constraints and jacobian */
    SX G = cheb.CollocateDynamics(robot, 0.0, 2.0);
    Function G_fun = Function("constraint", {opt_var}, {G});
    casadi::SX G_jacobian = casadi::SX::jacobian(G, opt_var);
    casadi::Function AugJacobian = casadi::Function("aug_jacobian",{opt_var}, {G_jacobian});

    casadi::Function Mayer = casadi::Function("mayer",{state},{casadi::SX::dot(state, state)});
    casadi::Function Lagrange = casadi::Function("lagrange",{state, control}, {casadi::SX::dot(state, state) + casadi::SX::dot(control, control)});

    /** Cost, Gradient and Hessian */
    casadi::SX C = cheb.CollocateCost(Mayer, Lagrange, -1, 1);
    casadi::SX C_grad = casadi::SX::gradient(C, opt_var);
    casadi::SX C_hess = casadi::SX::hessian(C, opt_var);
    casadi::Function C_f = casadi::Function("C",{opt_var},{C});
    casadi::Function C_f_grad = casadi::Function("C_grad",{opt_var}, {C_grad});
    casadi::Function C_f_hess = casadi::Function("C_hess",{opt_var}, {C_hess});

    /** Lagrangian, Gradient and Hessian */
    casadi::SX lam = casadi::SX::sym("lam", G.size1());
    casadi::SX L      = C + casadi::SX::mtimes(G.T(), lam);
    casadi::SX L_grad = casadi::SX::gradient(L, opt_var);
    casadi::SX L_hess = casadi::SX::hessian(L, opt_var);
    casadi::Function L_f = casadi::Function("L",{opt_var, lam},{L});
    casadi::Function L_f_grad = casadi::Function("L_grad",{opt_var, lam},{L_grad});
    casadi::Function L_f_hess = casadi::Function("L_hess",{opt_var, lam},{L_hess});

    DM x_init   = DM::ones(opt_var.size1());
    DM lam_init = DM::ones(lam.size1());
    DMVector result; //= G_fun(x_init);


    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    //result = G_fun(x_init);
    result = L_f_hess(DMVector{x_init, lam_init});
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) << " [microseconds]" << "\n";

    casadi::DM hessian = result[0];
    //std::cout << "Hessian: \n" << hessian(casadi::Slice(), casadi::Slice(30, 55)) << "\n";

    std::cout << "Size of: " << sizeof (cheb) << "\n";

    result = L_f(DMVector{x_init, lam_init});
    std::cout << "lagrangian: " << result[0] << "\n";


    /** differentiation test */
    casadi::SX poly_x = pow(x,2) + 1;
    casadi::Function poly = casadi::Function("poly", {state, control}, {poly_x});

    casadi::SX deriv = cheb.DifferentiateFunction(poly);
    casadi::Function deriv_f = casadi::Function("deriv", {varx}, {deriv});
    casadi::DM lin_space = casadi::DM(std::vector<double>{0, 0.0477, 0.1727, 0.3273, 0.4523, 0.5000, 0.5477, 0.6727, 0.8273, 0.9523, 1.0000});
    casadi::DM points = casadi::DM::zeros(3, NUM_SEGMENTS * POLY_ORDER + 1);
    points(0, casadi::Slice()) = pow(lin_space,2);
    //std::cout << "x: \n" << points << "\n";
    //std::cout << "Derivative: " << deriv_f(casadi::DMVector{casadi::DM::vec(points)})[0] << "\n";
}
