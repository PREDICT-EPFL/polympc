#ifndef SIMPLE_ROBOT_MODEL_HPP
#define SIMPLE_ROBOT_MODEL_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"

template <typename _Scalar = double>
struct MobileRobot
{
    MobileRobot(){}
    ~MobileRobot(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 3, 1>;
    using Control    = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    void operator() (const State &state, const Control &control, const Parameters &param, State &value) const
    {
        value[0] = control[0] * cos(state[2]) * cos(control[1]);
        value[1] = control[0] * sin(state[2]) * cos(control[1]);
        value[2] = control[0] * sin(control[1]) / param[0];
    }

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedA> &value) const
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

    Lagrange(){Q << 0.00001, 0, 0, 0, 0.00001, 0, 0, 0, 0.000001; R << 0.0001, 0, 0, 0.00001;}
    ~Lagrange(){}


    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename CostT>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, CostT &value) const
    {
        //value = state.dot(Q * state) + control.dot(R * control);
        /** @note: does not work with second derivatives without explicit cast ???*/
        using ScalarT = typename Eigen::MatrixBase<DerivedA>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state) + control.dot(R. template cast<ScalarT>() * control);
    }

};

template<typename _Scalar = double>
struct Mayer
{
    Mayer(){Q << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0.3;}
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
