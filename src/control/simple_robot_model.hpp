#ifndef SIMPLE_ROBOT_MODEL_HPP
#define SIMPLE_ROBOT_MODEL_HPP

#include "Eigen/Dense"

template <typename _Scalar = double>
struct MobileRobot
{
    MobileRobot(){}
    ~MobileRobot(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 3, 1>;
    using Control    = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &value) const
    {
        value[0] = control[0] * cos(state[2]) * cos(control[1]);
        value[1] = control[0] * sin(state[2]) * cos(control[1]);
        value[2] = control[0] * sin(control[1]) / param[0];
    }
};

template<typename _Scalar = double>
struct Lagrange
{
    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;
    Eigen::Matrix<Scalar, Control::RowsAtCompileTime, Control::RowsAtCompileTime> R;

    Lagrange(){
        Q << 0.5, 0, 0,
             0, 0.5, 0,
             0, 0, 0.01;
        R << 1, 0,
             0, 0.001;
    }
    ~Lagrange(){}


    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename CostT>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, CostT &value) const
    {
        value = state.dot(Q * state) + control.dot(R * control);
    }

};

template<typename _Scalar = double>
struct Mayer
{
    Mayer(){
        Q << 20, 0, 0,
             0, 20, 0,
             0, 0, 10;
    }
    ~Mayer(){}

    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;

    template<typename StateT, typename CostT>
    void operator() (const Eigen::MatrixBase<StateT> &state, CostT &value) const
    {
        using ScalarT = typename Eigen::MatrixBase<StateT>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state);
    }
};

#endif // SIMPLE_ROBOT_MODEL_HPP
