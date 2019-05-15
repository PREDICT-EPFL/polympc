#include "polynomials/ebyshev.hpp"
#include "control/ode_collocation.hpp"
#include <iomanip>
#include <iostream>
#include <chrono>
#include "control/simple_robot_model.hpp"

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



int main(void)
{
    using chebyshev = Chebyshev<3>;
    using collocation = polympc::ode_collocation<MobileRobot<double>, chebyshev, 2>;


    collocation ps_ode;
    collocation::var_t x = collocation::var_t::Ones();
    x[x.SizeAtCompileTime - 1] = 2.0;

    collocation::constr_t y;
    collocation::jacobian_t A;
    collocation::constr_t b;

    //ps_ode(x, y);

    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    ps_ode.linearized(x, A, b);
    Eigen::SparseMatrix<double> As = A.sparseView();
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    std::cout << "Constraint: " << b.transpose() << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eigen time: " << std::setprecision(9)
              << static_cast<double>(duration.count()) * 1e-3 << " [milliseconds]" << "\n";

    /** compute linearized PS constraints */
    std::cout << "Size: " << As.size() << "\n";
    std::cout << "NNZ: " << As.nonZeros() << "\n";
    std::cout << "Jacobian: \n" << A.template leftCols<10>() << "\n";
    Eigen::ColPivHouseholderQR< collocation::jacobian_t > lu(A);
    std::cout << "Jacobian: \n" << lu.rank() << "\n";

    /**
    std::cout << "Diff_MAT: \n" << ps_ode.m_DiffMat << "\n";
    Eigen::SparseMatrix<double> SpA = ps_ode.m_DiffMat.sparseView();
    std::cout << "Matrix size: " << ps_ode.m_DiffMat.size() << "\n";
    std::cout << "NNZ: " << SpA.nonZeros() << "\n";
    */

    return 0;
}
