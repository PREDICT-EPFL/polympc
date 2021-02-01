#include "control/cost_collocation.hpp"
#include "polynomials/ebyshev.hpp"
#include "Eigen/Dense"
#include <chrono>
#include <iostream>
#include <iomanip>

typedef std::chrono::time_point<std::chrono::system_clock> time_point;
time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}



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
        //value = state.dot(Q * state) + control.dot(R * control);
        /** @note: does not work with second derivatives without explicit cast ???*/
        using ScalarT = typename Eigen::MatrixBase<DerivedA>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state) + control.dot(R. template cast<ScalarT>() * control);
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
        using ScalarT = typename Eigen::MatrixBase<StateT>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state);
    }

};

int main(void)
{
    using namespace polympc;

    using chebyshev = Chebyshev<5>;
    using cost_collocation = polympc::cost_collocation<Lagrange<double>, Mayer<double>, chebyshev, 2>;

    cost_collocation::var_t x = cost_collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    cost_collocation cost_f;
    cost_collocation::var_t gradient;
    cost_collocation::hessian_t hessian;
    cost_collocation::Scalar value;

    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    //cost_f(x, value);
    //cost_f.value_gradient(x,value,gradient);
    cost_f.value_gradient_hessian(x, value, gradient, hessian);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    Eigen::IOFormat fmt(3);
    std::cout << "Cost: " << value << "\n";
    std::cout << "Gradient: " << gradient.transpose().format(fmt) << "\n";
    std::cout << "Hessian: \n" << hessian.template rightCols<33>().format(fmt) << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    std::cout << "HAS_LAGRANGE " << cost_collocation::HAS_LAGRANGE << "\n";
    std::cout << "HAS_MAYER " << cost_collocation::HAS_MAYER << "\n";

    return 0;
}
