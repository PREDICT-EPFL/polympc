// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "generic_ocp.hpp"

static constexpr int NX = 3; // number of system states
static constexpr int NU = 2; // number of input signals
static constexpr int NP = 0; // number of unknown parameters (can be optimised)
static constexpr int ND = 0; // number of user specified parameters (changed excusively by the user)

static constexpr int POLY_ORDER = 5;
static constexpr int NUM_SEGMENTS = 2;

using Approximation  = Chebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>;     // standard collocation

class RobotOCP : public GenericOCP<RobotOCP, Approximation>
{
public:
    /** constructor inheritance */
    using GenericOCP::GenericOCP;
    ~RobotOCP()  = default;

    casadi::Dict solver_options;

    static constexpr double t_start = 0.0;
    static constexpr double t_final = 1.0;

    /**
     * x - state
     * u - control
     * p - optimised parameters
     * d - static parameters
     */

    casadi::SX mayer_term_impl(const casadi::SX &x, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX XWeight = casadi::SX::eye(NX);
        casadi::SX UWeight = casadi::SX::eye(NU);
        return casadi::SX::dot(x, casadi::SX::mtimes(XWeight, x));
    }

    casadi::SX lagrange_term_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX XWeight = casadi::SX::eye(NX);
        casadi::SX UWeight = casadi::SX::eye(NU);
        return casadi::SX::dot(x, casadi::SX::mtimes(XWeight, x)) + casadi::SX::dot(u, casadi::SX::mtimes(UWeight, u));
    }

    casadi::SX system_dynamics_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX xdot = casadi::SX::vertcat({u(0) * cos(x(2)) * cos(u(1)),
                                               u(0) * sin(x(2)) * cos(u(1)),
                                               u(0) * sin(u(1)) / 1.0});

        return xdot;
    }

    casadi::SX inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX();
    }

    casadi::SX final_inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX();
    }

};


int main(void)
{
    casadi::Dict user_options;
    user_options["ipopt.print_level"] = 5;

    casadi::Dict mpc_options;
    mpc_options["mpc.scaling"] = false;
    mpc_options["mpc.scale_x"] = std::vector<double>{1,1,1};
    mpc_options["mpc.scale_u"] = std::vector<double>{1,1};

    RobotOCP lox(user_options, mpc_options);
    casadi::DM lbx = casadi::DM::vertcat({-10, -10, -10});
    casadi::DM ubx = casadi::DM::vertcat({ 10,  10,  10});

    casadi::DM lbu = casadi::DM::vertcat({-5.0, -0.5});
    casadi::DM ubu = casadi::DM::vertcat({ 5.0,  0.5});

    lox.set_state_box_constraints(lbx, ubx);
    lox.set_control_box_constraints(lbu,ubu);

    casadi::DM x0 = casadi::DM::vertcat({1.0, 1.0, 0.0});

    casadi::DM X0 = casadi::DM::repmat(x0, 11);
    casadi::DM U0 = casadi::DM::repmat(casadi::DM(std::vector<double>{-0.5, -0.1}), 11);
    casadi::DM INIT_X = casadi::DM::vertcat({X0, U0});

    lox.solve(x0, x0, INIT_X);

    casadi::DM solution = lox.get_optimal_control();
    std::cout << "Optimal Control: " << solution << "\n";

    casadi::DM trajectory = lox.get_optimal_trajectory();
    std::cout << " \n Optimal Trajectory " << trajectory << "\n";

    lox.generate_nlp_code();

    return EXIT_SUCCESS;
}
