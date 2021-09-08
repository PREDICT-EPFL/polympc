// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "polynomials/ebyshev.hpp"
#include "control/ode_collocation.hpp"
#include "control/continuous_ocp.hpp"
#include "polynomials/splines.hpp"
#include "utils/helpers.hpp"
#include <iomanip>
#include <iostream>
#include <chrono>
#include "control/simple_robot_model.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1



int main(void)
{
    using namespace polympc;

    using chebyshev = Chebyshev<test_POLY_ORDER>;
    using collocation = polympc::ode_collocation<MobileRobot<double>, chebyshev, test_NUM_SEG>;

    collocation ps_ode;
    collocation::var_t x = collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    collocation::constr_t y;
    collocation::jacobian_t A;
    collocation::constr_t b;

    time_point start = get_time();
    for(int i = 0; i < test_NUM_EXP; ++i)
        ps_ode.linearized(x, A, b);
        //ps_ode(x, b);

    //ps_ode.linearized(x, A, b);
    time_point stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "ODE Collocation time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) / test_NUM_EXP << " [microseconds]" << "\n";

    std::cout << "Constraint old: " << b.transpose() << "\n";
    std::cout << "size of old:" << sizeof (ps_ode) << "\n";

    /**
    start = get_time();
    ps_ode.linearized(x, A, b);
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    */

    std::cout << "Jacobian time: " << std::setprecision(9)
              << static_cast<double>(duration.count())  << " [microseconds]" << "\n";

    Eigen::SparseMatrix<double> As = A.sparseView();

    /** compute linearized PS constraints */
    std::cout << "Size: " << As.size() << "\n";
    std::cout << "NNZ: " << As.nonZeros() << "\n";
    Eigen::IOFormat fmt(3);
    //std::cout << A.template leftCols<29>().format(fmt) << "\n";
    std::cout << A(0,0) << "\n";
    Eigen::ColPivHouseholderQR< collocation::jacobian_t > lu(A);
    std::cout << "Jacobian: \n" << lu.rank() << "\n";

    /**
    std::cout << "Diff_MAT: \n" << ps_ode.m_DiffMat << "\n";
    Eigen::SparseMatrix<double> SpA = ps_ode.m_DiffMat.sparseView();
    std::cout << "Matrix size: " << ps_ode.m_DiffMat.size() << "\n";
    std::cout << "NNZ: " << SpA.nonZeros() << "\n";
    */

    return 0;
}
