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
