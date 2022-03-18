// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "chebyshev_soft.hpp"
#include "mobile_robot.hpp"
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
    using SoftCheb = polympc::SoftChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, 3, 2, 0, 0>;

    polympc::SoftChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, 3, 2, 0, 0> cheb;
    MobileRobot robot;
    casadi::Function ode = robot.getDynamics();

    casadi::SX varx = cheb.VarX();
    casadi::SX varu = cheb.VarU();
    casadi::SX opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    casadi::SX G = cheb.CollocateDynamics(ode, -1.0, 1.0);

    casadi::Function G_fun = casadi::Function("dynamics", {opt_var}, {G});
    casadi::DM x_init = casadi::DM::ones(opt_var.size1());
    casadi::DMVector result;


    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    result = G_fun(x_init);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    //std::cout << "Casadi constraint: " << result[0].T() << "\n";

    /** compute gradient */
    casadi::SX G_gradient = casadi::SX::gradient(G, opt_var);
    /** Augmented Jacobian */
    casadi::Function G_gradient_fun = casadi::Function("gradient",{opt_var}, {G_gradient});

    start = get_time();
    casadi::DM jac = G_gradient_fun(x_init)[0];
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Gradient evaluation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    /** Hessian evaluation */
    casadi::SX G_hessian = casadi::SX::hessian(G, opt_var);
    casadi::Function G_hessian_fun = casadi::Function("L_grad",{opt_var}, {G_hessian});

    start = get_time();
    casadi::DM G_val_hess = G_hessian_fun({x_init})[0];
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Casadi Cost time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";


    /** check the integration routine */
    casadi::SXDict NLP;
    casadi::Function NLP_Solver;
    casadi::Dict OPTS;
    casadi::DMDict ARG;

    NLP["x"] = opt_var;
    NLP["f"] = G;

    /** default solver options */
    OPTS["ipopt.linear_solver"]         = "mumps";
    OPTS["ipopt.print_level"]           = 5;
    OPTS["ipopt.tol"]                   = 1e-4;
    OPTS["ipopt.acceptable_tol"]        = 1e-4;

    NLP_Solver = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    casadi::DM lbg = -casadi::DM::inf(varx.size1());
    casadi::DM ubg =  casadi::DM::inf(varx.size1());
    casadi::DM control = casadi::DM::vertcat({1.0, 0.1});
    casadi::DM lbg_u = casadi::DM::repmat(control, NUM_SEGMENTS * POLY_ORDER + 1, 1);
    casadi::DM ubg_u = lbg_u;

    lbg = casadi::DM::vertcat({lbg, lbg_u});
    ubg = casadi::DM::vertcat({ubg, ubg_u});

    /** set initial conditions */
    casadi::DM init_cond = casadi::DM::vertcat({0.0, 0.0, 0.0});
    lbg(casadi::Slice(SoftCheb::_X_END_IDX - 3, SoftCheb::_X_END_IDX)) = init_cond;
    ubg(casadi::Slice(SoftCheb::_X_END_IDX - 3, SoftCheb::_X_END_IDX)) = init_cond;

    ARG["lbx"] = lbg;
    ARG["ubx"] = ubg;

    casadi::DMDict res = NLP_Solver(ARG);
    auto NLP_X = res.at("x");

    std::cout << "solution: \n" << NLP_X << "\n";

    return 0;
}
