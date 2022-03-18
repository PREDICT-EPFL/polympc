// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace polympc {

template <typename qp_t>
void print_qp(qp_t qp)
{
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "]", "[", "]");
    std::cout << "P = " << qp.P.format(fmt) << std::endl;
    std::cout << "q = " << qp.q.transpose().format(fmt) << std::endl;
    std::cout << "A = " << qp.A.format(fmt) << std::endl;
    std::cout << "l = " << qp.l.transpose().format(fmt) << std::endl;
    std::cout << "u = " << qp.u.transpose().format(fmt) << std::endl;
}

template <typename Mat>
bool is_psd(Mat &h)
{
    Eigen::EigenSolver<Mat> eigensolver(h);
    for (int i = 0; i < eigensolver.eigenvalues().RowsAtCompileTime; i++) {
        double v = eigensolver.eigenvalues()(i).real();
        if (v < 0) {
            return false;
        }
    }
    return true;
}

} // polympc namespace

#endif /* UTILS_HPP */
