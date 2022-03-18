// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "integration/integrator.h"
#include "integration/chebyshev_integrator.hpp"
#include "kite.h"
#include <chrono>
#include <iomanip>

using namespace casadi;

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

int main(void)
{
    /** kite properties */
    SimpleKinematicKiteProperties kite_props;
    kite_props.gliding_ratio = 5;
    kite_props.tether_length = 5;
    kite_props.wind_speed = 1.1;

    /** kite object */
    SimpleKinematicKite kite(kite_props);
    Function ode = kite.getDynamics();

    /** compare three ode solvers */
    Dict opts;
    opts["tf"]         = 5.0;
    opts["tol"]        = 1e-5;
    opts["method"]     = polympc::IntType::RK4;
    polympc::ODESolver rk4_solver(ode, opts);

    opts["method"] = polympc::IntType::CVODES;
    polympc::ODESolver cvodes_solver(ode, opts);

    double tf = 5.0;
    casadi::DMDict props;
    props["scale"] = 0;
    props["P"] = casadi::DM::diag(casadi::DM(std::vector<double>{0.1, 1/3.0, 1/3.0, 1/2.0, 1/5.0, 1/2.0, 1/3.0, 1/3.0, 1/3.0, 1.0, 1.0, 1.0, 1.0}));
    props["R"] = casadi::DM::diag(casadi::DM(std::vector<double>{1/0.15, 1/0.2618, 1/0.2618}));

    casadi::Dict solver_options;
    solver_options["ipopt.linear_solver"] = "mumps";

    polympc::PSODESolver<10, 10, 3, 1, 0>ps_solver(ode, tf, props, solver_options);

    /** solve the problem */
    DM rk4_sol, cheb_sol, cv_sol, ps_sol;

    DM init_state = DM::vertcat({0.25,0.35,0.78});
    DM control = DM::vertcat({0.1});
    DM parameter = casadi::DM(); //dummy parameter

    std::chrono::time_point<std::chrono::system_clock> start = get_time();

    rk4_sol  = rk4_solver.solve(init_state, control, parameter, tf);
    std::chrono::time_point<std::chrono::system_clock> rk4_stop = get_time();

    cv_sol   = cvodes_solver.solve(init_state, control, parameter, tf);
    std::chrono::time_point<std::chrono::system_clock> cv_stop = get_time();

    bool FULL = true;
    ps_sol = ps_solver.solve(init_state, control, FULL);
    std::chrono::time_point<std::chrono::system_clock> ps_stop = get_time();

    auto rk4_duration = std::chrono::duration_cast<std::chrono::microseconds>(rk4_stop - start);
    auto cv_duration = std::chrono::duration_cast<std::chrono::microseconds>(cv_stop - rk4_stop);
    auto ps_duration = std::chrono::duration_cast<std::chrono::microseconds>(ps_stop - cv_stop);

    std::cout << "RK4 solve time: " << std::setprecision(6)
              << static_cast<double>(rk4_duration.count()) * 1e-6 << " [seconds]" << "\n";
    std::cout << "CVODES solve time: " << std::setprecision(6)
              << static_cast<double>(cv_duration.count()) * 1e-6 << " [seconds]" << "\n";
    std::cout << "PS solve time: " << std::setprecision(6)
              << static_cast<double>(ps_duration.count()) * 1e-6 << " [seconds]" << "\n";

    std::cout << "RK4: " <<  rk4_sol << "\n";
    std::cout << "CVODES: " <<  cv_sol << "\n";
    std::cout << "Chebyshev CasADi:" << ps_sol(Slice(0, 3)) << "\n";
}

