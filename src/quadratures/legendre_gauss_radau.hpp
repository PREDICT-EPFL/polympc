#ifndef POLYMPC_QUADRATURES_LEGENDRE_GAUSS_RADAU_HPP
#define POLYMPC_QUADRATURES_LEGENDRE_GAUSS_RADAU_HPP

#include "polynomials/polynomial_math.hpp"

namespace polympc {

template<int PolyOrder, typename Scalar = double>
class LegendreGaussRadau
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
    /** compute and evaluate derivative of lagrange polynomial */
    static scalar_t    compute_d_lagrange(const nodes_t &nodes, int j, scalar_t x) noexcept;
    /** generate differentiation matrix */
    static diff_mat_t  compute_diff_matrix() noexcept;
};

template<int PolyOrder, typename Scalar>
typename LegendreGaussRadau<PolyOrder, Scalar>::basis_t
LegendreGaussRadau<PolyOrder, Scalar>::compute_basis() noexcept
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
typename LegendreGaussRadau<PolyOrder, Scalar>::nodes_t
LegendreGaussRadau<PolyOrder, Scalar>::compute_nodes() noexcept
{
    basis_t Ln = compute_basis();
    /** Legendre (LGL) collocation points for the interval [-1, 1]*/
    /** compute roots of LN-1 + LN */
    nodes_t Ln_sum = Ln.col(PolyOrder - 1) + Ln.col(PolyOrder);
    scalar_t eps = std::numeric_limits<scalar_t>::epsilon();

    /** prepare the polynomial for the solver */
    for(int i = 0; i < PolyOrder; ++i)
    {
        if(std::fabs(Ln_sum[i]) <= eps)
            Ln_sum[i] = scalar_t(0);
    }

    Eigen::PolynomialSolver<scalar_t, PolyOrder> root_finder;
    root_finder.compute(Ln_sum);

    nodes_t nodes = nodes_t::Zero();
    nodes[PolyOrder] = 1;

    polymath::segment<nodes_t, PolyOrder>(nodes, 0) = root_finder.roots().real();

    /** sort the nodes in the ascending order */
    std::sort(nodes.data(), nodes.data() + nodes.size());
    return nodes;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussRadau<PolyOrder, Scalar>::q_weights_t
LegendreGaussRadau<PolyOrder, Scalar>::compute_int_weights() noexcept
{
    const basis_t Ln = compute_basis();
    const nodes_t nodes = compute_nodes();

    q_weights_t weights  = q_weights_t::Zero();
    const scalar_t coeff = scalar_t(1) / (PolyOrder * PolyOrder);
    weights[0] = scalar_t(2) * coeff;
    for(int i = 1; i <= PolyOrder; ++i)
    {
        scalar_t LN_xi = Eigen::poly_eval(Ln.col(PolyOrder), nodes[i]);
        weights[i] = coeff * (scalar_t(1) - nodes[i]) / (LN_xi * LN_xi);
    }
    return weights;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussRadau<PolyOrder, Scalar>::scalar_t
LegendreGaussRadau<PolyOrder, Scalar>::compute_d_lagrange(const nodes_t &nodes, int j, scalar_t x) noexcept
{
    scalar_t dL = scalar_t(0);

    for (Eigen::Index i = 0; i < NUM_NODES; ++i)
    {
        if (i == j) continue;
        scalar_t L = scalar_t(1);
        for (Eigen::Index l = 0; l < NUM_NODES; ++l)
        {
            if (l == j) continue;
            if (i != l)
            {
                L *= (x - nodes(l)) / (nodes(j) - nodes(l));
            }
        }
        dL += L / (nodes(j) - nodes(i));
    }

    return dL;
}

template<int PolyOrder, typename Scalar>
typename LegendreGaussRadau<PolyOrder, Scalar>::diff_mat_t
LegendreGaussRadau<PolyOrder, Scalar>::compute_diff_matrix() noexcept
{
    diff_mat_t D  = diff_mat_t::Zero();
    nodes_t nodes = compute_nodes();

    for(Eigen::Index i = 0; i < NUM_NODES; ++i)
    {
        for (Eigen::Index j = 0; j < NUM_NODES; ++j)
        {
            D(i, j) = compute_d_lagrange(nodes, j, nodes(i));
        }
    }

    return D;
}

} // polympc namespace

#endif // POLYMPC_QUADRATURES_LEGENDRE_GAUSS_RADAU_HPP
