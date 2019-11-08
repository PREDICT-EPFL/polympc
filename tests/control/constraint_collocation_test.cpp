#include <control/constraints_collocation.hpp>
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


struct Function
{
   /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &output) const
    {
        output[0] = state.dot(state) + pow(control[0], 2);
        output[1] = 5 * state[0] + control[0];
    }

};

struct Function2
{
   /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &output) const
    {
        output = control;
    }

};


int main(void)
{
    using chebyshev = Chebyshev<3>;
    using constraint_collocation = polympc::constraint_collocation<2, 1, 1, chebyshev, 2>;

    constraint_collocation::var_t x = constraint_collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    constraint_collocation c_col;
    Function f;
    Function2 f2;

    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    auto value  = c_col.generic_function<Function, 2>(f, x);
    auto value2 = c_col.generic_function<Function2, 1>(f2, x);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    std::cout  << value.transpose() << "\n";
    std::cout  << value2.transpose() << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    return 0;
}
