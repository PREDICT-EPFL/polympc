// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "polynomials/ebyshev.hpp"
#include "control/sparse_ode_collocation.hpp"
#include <iomanip>
#include <iostream>
#include <chrono>
#include "control/simple_robot_model.hpp"
#include "utils/helpers.hpp"


int main(void)
{
    using namespace polympc;

    using Scalar = double;
    using chebyshev = Chebyshev<3,GAUSS_LOBATTO,Scalar>;
    using collocation = polympc::sparse_ode_collocation<MobileRobot<Scalar>, chebyshev, 2>;

    collocation ps_ode;
    collocation::var_t x = collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    collocation::constr_t y;
    collocation::jacobian_t A(collocation::constr_t::RowsAtCompileTime, collocation::var_t::RowsAtCompileTime);
    collocation::constr_t b;

    //ps_ode(x, y);
    Eigen::SparseMatrix<Scalar> A1, A2;

    ps_ode.linearized(x, A, b);
    time_point start = get_time();
    ps_ode.linearized(x, A, b);
    //ps_ode(x, b);
    time_point stop = get_time();

    std::cout << "Constraint: " << b.transpose() << "\n";
    std::cout << "Size: " << A.size() << "\n";
    std::cout << "NNZ: " << A.nonZeros() << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    //std::cout << "Diff_M: \n" << ps_ode.m_DiffMat << "\n";

    std::cout << "Jacobian: \n" << A.template leftCols<10>() << "\n";

    return 0;
}
