// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"

/** class to store Optimal Control problems */
namespace polympc {

template<typename _Dynamics, typename _Lagrange, typename _Mayer>
class OCProblem
{
public:
    OCProblem(){};
    ~OCProblem(){}

    using Dynamics = _Dynamics;
    using Lagrange = _Lagrange;
    using Mayer    = _Mayer;

    enum
    {
        NX_D = Dynamics::State::RowsAtCompileTime,
        NU_D = Dynamics::Control::RowsAtCompileTime,
        NP_D = Dynamics::Parameters::RowsAtCompileTime,
    };
};

}

#endif // PROBLEM_HPP
