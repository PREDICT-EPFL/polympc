#ifndef POLYMPC_QUADRATURES_LEGENDRE_GAUSS_LOBATTO_HPP
#define POLYMPC_QUADRATURES_LEGENDRE_GAUSS_LOBATTO_HPP

#include "polynomials/polynomial_math.hpp"

namespace polympc {

template<int PolyOrder, typename Scalar = double>
class LegendreGaussLobatto
{
public:
    enum
    {
        POLY_ORDER = PolyOrder,
        NUM_NODES = PolyOrder + 1
    };

    using scalar_t = Scalar;

    using q_weights_t = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using nodes_t     = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using diff_mat_t  = Eigen::Matrix<scalar_t, NUM_NODES, NUM_NODES>;
    /** Legendre basis */
    using basis_t = Eigen::Matrix<scalar_t, POLY_ORDER + 1, POLY_ORDER + 1>;

    /** compute basis */
    static basis_t     compute_basis() noexcept;
    /** compute nodal points */
    static nodes_t     compute_nodes() noexcept;
    /** compute Clenshaw-Curtis quadrature weights */
    static q_weights_t compute_int_weights() noexcept;
    /** generate differentiation matrix */
    static diff_mat_t  compute_diff_matrix() noexcept;
};

template<int PolyOrder, typename Scalar>
typename LegendreGaussLobatto<PolyOrder, Scalar>::basis_t
LegendreGaussLobatto<PolyOrder, Scalar>::compute_basis() noexcept
{
    basis_t Ln = basis_t::Zero();
    /** the first basis polynomial is L0(x) = 1 */
    Ln(0,0) = 1;
    /** the second basis polynomial is L1(x) = x */
    Ln(1,1) = 1;

    /** compute recurrent coefficients */
    nodes_t a = nodes_t::Zero();
    nodes_t c = nodes_t::Zero();
    nodes_t x = nodes_t::Zero(); // p(x) = x
    x[1] = 1;
    for(int n = 0; n <= PolyOrder; ++n)
    {
        a(n) = scalar_t(2 * n + 1) / (n + 1);
        c(n) = scalar_t(n) / (n + 1);
    }

    /** create polynomial basis */
    for(int n = 1; n < POLY_ORDER; ++n)
        Ln.col(n+1)= a(n) * polymath::poly_mul(Ln.col(n), x) - c(n) * Ln.col(n-1);

    return Ln;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussLobatto<PolyOrder, Scalar>::nodes_t
LegendreGaussLobatto<PolyOrder, Scalar>::compute_nodes() noexcept
{
    basis_t Ln = compute_basis();
    /** Legendre (LGL) collocation points for the interval [-1, 1]*/
    /** compute roots of LN_dot(x) polynomial - extremas of LN(x) */
    nodes_t LN_dot = polymath::poly_diff(Ln.col(PolyOrder));
    scalar_t eps = std::numeric_limits<scalar_t>::epsilon();

    /** prepare the polynomial for the solver */
    for(int i = 0; i < PolyOrder; ++i)
    {
        if(std::fabs(LN_dot[i]) <= eps)
            LN_dot[i] = scalar_t(0);
    }

    Eigen::PolynomialSolver<scalar_t, PolyOrder - 1> root_finder;
    root_finder.compute(polymath::segment<nodes_t, PolyOrder>(LN_dot, 0)); // remove the last zero

    nodes_t nodes = nodes_t::Zero();
    nodes[0] = -1; nodes[PolyOrder] = 1;

    polymath::segment<nodes_t, PolyOrder - 1>(nodes, 1) = root_finder.roots().real();

    /** sort the nodes in the ascending order */
    std::sort(nodes.data(), nodes.data() + nodes.size());
    return nodes;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussLobatto<PolyOrder, Scalar>::q_weights_t
LegendreGaussLobatto<PolyOrder, Scalar>::compute_int_weights() noexcept
{
    const basis_t Ln = compute_basis();
    const nodes_t nodes = compute_nodes();

    q_weights_t weights  = q_weights_t::Zero();
    const scalar_t coeff = scalar_t(2) / (PolyOrder * (PolyOrder + 1));
    for(int i = 0; i <= PolyOrder; ++i)
    {
        scalar_t LN_xi = Eigen::poly_eval(Ln.col(PolyOrder), nodes[i]);
        weights[i] = coeff / (LN_xi * LN_xi);
    }
    return weights;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussLobatto<PolyOrder, Scalar>::diff_mat_t
LegendreGaussLobatto<PolyOrder, Scalar>::compute_diff_matrix() noexcept
{
    diff_mat_t D  = diff_mat_t::Zero();
    nodes_t nodes = compute_nodes();
    basis_t Ln    = compute_basis();

    /** use formula from Canuto-Quarteroni */
    D(0,0) = -(POLY_ORDER + 1) * POLY_ORDER / static_cast<scalar_t>(4);
    D(POLY_ORDER, POLY_ORDER) = -D(0,0);

    for(Eigen::Index i = 0; i < NUM_NODES; ++i)
    {
        for (Eigen::Index j = 0; j < NUM_NODES; ++j)
        {
            if(i != j)
                D(i, j) = (Eigen::poly_eval(Ln.col(POLY_ORDER), nodes[i]) / Eigen::poly_eval(Ln.col(POLY_ORDER), nodes(j)))
                          * (1 / (nodes(i) - nodes(j)));
        }
    }
    return D;
}

} // polympc namespace

#endif //POLYMPC_QUADRATURES_LEGENDRE_GAUSS_LOBATTO_HPP
