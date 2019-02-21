#include "control/cost_collocation.hpp"
#include "polynomials/ebyshev.hpp"
#include "eigen3/Eigen/Dense"

template<typename _Scalar = double>
struct Lagrange
{
    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;
    Eigen::Matrix<Scalar, Control::RowsAtCompileTime, Control::RowsAtCompileTime> R;

    Lagrange(){Q << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0; R << 1.0, 0, 0, 1.0;}
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
    Mayer(){Q << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;}
    ~Mayer(){}

    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;

    template<typename StateT, typename CostT>
    void operator() (const Eigen::MatrixBase<StateT> &state, CostT &value) const
    {
        value = state.dot(Q * state);
    }
};

int main(void)
{
    using chebyshev = Chebyshev<3>;
    using cost_collocation = polympc::cost_collocation<Lagrange<double>, Mayer<double>, chebyshev, 2>;

    cost_collocation::var_t x = cost_collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    cost_collocation cost_f;
    cost_collocation::Scalar value;
    cost_f(x, value);

    std::cout << "Value: " << value << "\n";

    return 0;
}
