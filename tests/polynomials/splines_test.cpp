#include "polynomials/splines.hpp"
#include "polynomials/ebyshev.hpp"
#include "polynomials/legendre.hpp"

int main(void)
{
    using namespace polympc;

    using Chebyshev = polympc::Chebyshev<5, GAUSS_LOBATTO, double>;
    using Legendre  = Legendre< 5, GAUSS_LOBATTO, float>;

    using ChebyshevSpline = Spline<Chebyshev, 4>;
    using LegendreSpline  = Spline<Legendre, 5>;

    /** compute Chebyshev values */
    ChebyshevSpline::nodes_t cheb_nodes   = ChebyshevSpline::compute_nodes();
    ChebyshevSpline::diff_mat_t cheb_diff = ChebyshevSpline::compute_diff_matrix();
    ChebyshevSpline::q_weights_t cheb_weights = ChebyshevSpline::compute_int_weights();

    std::cout << "Chebyshev nodes: " << cheb_nodes.transpose() << "\n";
    std::cout << "Chebyshev quadrature weights: " << cheb_weights.transpose() << "\n";
    std::cout << "Chebyshev diff matrix: \n" << cheb_diff << "\n \n";

    /** compute Chebyshev values */
    LegendreSpline::nodes_t leg_nodes   = LegendreSpline::compute_nodes();
    LegendreSpline::diff_mat_t leg_diff = LegendreSpline::compute_diff_matrix();
    LegendreSpline::q_weights_t leg_weights = LegendreSpline::compute_int_weights();

    std::cout << "Legendre nodes: " << leg_nodes.transpose() << "\n";
    std::cout << "Legendre quadrature weights: " << leg_weights.transpose() << "\n";
    std::cout << "Legendre diff matrix: \n" << leg_diff << "\n";

    return EXIT_SUCCESS;
}
