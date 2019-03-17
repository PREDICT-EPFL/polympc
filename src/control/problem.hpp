#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"

/** class to store Optimal Control problems */
namespace polympc {

struct EmptyFunctor
{
    EmptyFunctor(){}
    ~EmptyFunctor(){}
};


template<typename _Dynamics, typename _Lagrange, typename _Mayer, typename _InequalityConstraints = EmptyFunctor, typename _EqualityConstraints = EmptyFunctor>
class OCProblem
{
public:
    OCProblem();
    ~OCProblem(){}

    using Dynamics = _Dynamics;
    using Lagrange = _Lagrange;
    using Mayer    = _Mayer;
    using InequalityConstraints = _InequalityConstraints;
    using EqualityConstraints = _EqualityConstraints;

    enum
    {
        NX_D = Dynamics::State::RowsAtCompileTime,
        NU_D = Dynamics::Control::RowsAtCompileTime,
        NP_D = Dynamics::Parameters::RowsAtCompileTime,
    };

    Dynamics m_f;
    Lagrange m_lagrange;
    Mayer    m_mayer;
    EqualityConstraints m_eq_h;
    InequalityConstraints m_ineq_g;
};

template<typename Dynamics, typename Lagrange, typename Mayer, typename InequalityConstraints, typename EqualityConstraints>
OCProblem<Dynamics, Lagrange, Mayer, InequalityConstraints, EqualityConstraints>::OCProblem()
{
    std::cout << "Problem size: " << NX_D << "\n";
}


}

#endif // PROBLEM_HPP
