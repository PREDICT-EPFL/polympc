// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "gtest/gtest.h"
#include "solvers/box_admm.hpp"
#include "solvers/admm.hpp"
#include "solvers/osqp_interface.hpp"
#include "solvers/qp_preconditioners.hpp"

/** OSQP tests */

TEST(QPSolverSparseTest, osqpSimpleQP)
{
    using scalar = double;
    Eigen::SparseMatrix<scalar> H(2,2);
    Eigen::SparseMatrix<scalar> A(1,2);
    Eigen::Matrix<scalar, 1,1> Al, Au;
    Eigen::Matrix<scalar, 2,1> h, xl, xu, solution;

    H.reserve(2); H.insert(0,0) = 4; H.insert(0,1) = 1; H.insert(1,0) = 1; H.insert(1,1) = 2;
    A.reserve(1); A.insert(0,0) = 1; A.insert(0,1) = 1;
    Al << 1;
    Au << 1;
    xl << 0, 0;
    xu << 0.7, 0.7;
    h << 1, 1;
    solution << 0.3, 0.7;

    polympc::OSQP<2, 1, scalar> prob;

    prob.settings().max_iter = 150;
    prob.settings().adaptive_rho = false;

    prob.solve(H, h, A, Al, Au, xl, xu);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.info().iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}

TEST(QPSolverSparseTest, osqpSimpleQPTwice)
{
    using scalar = double;
    Eigen::SparseMatrix<scalar> H(2,2);
    Eigen::SparseMatrix<scalar> A(1,2);
    Eigen::Matrix<scalar, 1,1> Al, Au;
    Eigen::Matrix<scalar, 2,1> h, xl, xu, solution;

    H.reserve(2); H.insert(0,0) = 4; H.insert(0,1) = 1; H.insert(1,0) = 1; H.insert(1,1) = 2;
    A.reserve(1); A.insert(0,0) = 1; A.insert(0,1) = 1;
    Al << 1;
    Au << 1;
    xl << 0, 0;
    xu << 0.7, 0.7;
    h << 1, 1;
    solution << 0.3, 0.7;

    polympc::OSQP<2, 1, scalar> prob;

    prob.settings().max_iter = 150;
    prob.settings().adaptive_rho = false;

    prob.solve(H, h, A, Al, Au, xl, xu);

    prob.settings().reuse_pattern = true;
    prob.settings().warm_start = true;
    prob.solve(H, h, A, Al, Au, xl, xu);

    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.info().iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}

TEST(QPSolverSparseTest, osqpAdaptiveRho)
{
    using scalar = double;
    Eigen::SparseMatrix<scalar> H(2,2);
    Eigen::SparseMatrix<scalar> A(1,2);
    Eigen::Matrix<scalar, 1,1> Al, Au;
    Eigen::Matrix<scalar, 2,1> h, xl, xu, solution;

    H.reserve(2); H.insert(0,0) = 4; H.insert(0,1) = 1; H.insert(1,0) = 1; H.insert(1,1) = 2;
    A.reserve(1); A.insert(0,0) = 1; A.insert(0,1) = 1;
    Al << 1;
    Au << 1;
    xl << 0, 0;
    xu << 0.7, 0.7;
    h << 1, 1;
    solution << 0.3, 0.7;

    polympc::OSQP<2, 1, scalar> prob;

    prob.settings().warm_start = false;
    prob.settings().max_iter = 1000;
    prob.settings().rho = 0.1;

    // solve whithout adaptive rho
    prob.settings().adaptive_rho = false;
    prob.solve(H,h,A,Al,Au,xl,xu);
    int prev_iter = prob.info().iter;

    // solve with adaptive rho
    prob.settings().adaptive_rho = true;
    prob.settings().reuse_pattern = true;
    prob.settings().warm_start = true;
    prob.settings().adaptive_rho_interval = 10;
    prob.solve(H,h,A,Al,Au,xl,xu);

    auto info = prob.info();
    EXPECT_LT(info.iter, prob.settings().max_iter);
    EXPECT_LE(info.iter, prev_iter); // adaptive rho should improve :)
    EXPECT_EQ(info.status, SOLVED);
}
