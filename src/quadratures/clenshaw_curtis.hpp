#ifndef POLYMPC_QUADRATURES_CLENSHAW_CURTIS_HPP
#define POLYMPC_QUADRATURES_CLENSHAW_CURTIS_HPP

#include "polynomials/polynomial_math.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace polympc {

template<int PolyOrder, typename Scalar = double>
class ClenshawCurtis
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

    /** compute nodal points */
    static nodes_t     compute_nodes() noexcept;
    /** compute Clenshaw-Curtis quadrature weights */
    static q_weights_t compute_int_weights() noexcept;
    /** generate differentiation matrix */
    static diff_mat_t  compute_diff_matrix() noexcept;
};

template<int PolyOrder, typename Scalar>
typename ClenshawCurtis<PolyOrder, Scalar>::nodes_t
ClenshawCurtis<PolyOrder, Scalar>::compute_nodes() noexcept
{
    nodes_t grid = nodes_t::LinSpaced(NUM_NODES, 0, PolyOrder);
    return (grid * (M_PI / PolyOrder)).array().cos().reverse();
}

template<int PolyOrder, typename Scalar>
typename ClenshawCurtis<PolyOrder, Scalar>::q_weights_t
ClenshawCurtis<PolyOrder, Scalar>::compute_int_weights() noexcept
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    nodes_t theta = nodes_t::LinSpaced(NUM_NODES, 0, PolyOrder);
    theta *= (M_PI / PolyOrder);

    q_weights_t w = q_weights_t::Zero(NUM_NODES, 1);
    using tmp_vtype = Eigen::Matrix<scalar_t, PolyOrder - 1, 1>;
    tmp_vtype v = tmp_vtype::Ones(PolyOrder - 1, 1);

    if ( PolyOrder % 2 == 0 )
    {
        w[0]         = static_cast<scalar_t>(1 / (std::pow(PolyOrder, 2) - 1));
        w[PolyOrder] = w[0];

        for(int k = 1; k <= PolyOrder / 2 - 1; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * polymath::segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<scalar_t>(2.0 / (4 * std::pow(k, 2) - 1)) * vk;
        }
        tmp_vtype vk = Eigen::cos((PolyOrder * polymath::segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
        v -= vk / (std::pow(PolyOrder, 2) - 1);
    }
    else
    {
        w[0] = static_cast<scalar_t>(1 / std::pow(PolyOrder, 2));
        w[PolyOrder] = w[0];
        for (int k = 1; k <= (PolyOrder - 1) / 2; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * polymath::segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<scalar_t>(2.0 / (4 * pow(k, 2) - 1)) * vk;
        }
    }

    polymath::segment<q_weights_t, PolyOrder - 1>(w, 1) =  static_cast<scalar_t>(2.0 / PolyOrder) * v;
    return w;
}

template<int PolyOrder, typename Scalar>
typename ClenshawCurtis<PolyOrder, Scalar>::diff_mat_t
ClenshawCurtis<PolyOrder, Scalar>::compute_diff_matrix() noexcept
{
    nodes_t grid = nodes_t::LinSpaced(NUM_NODES, 0, POLY_ORDER).reverse();
    nodes_t c    = nodes_t::Ones(); c[0] = scalar_t(2); c[POLY_ORDER] = scalar_t(2);
    c = (Eigen::pow(scalar_t(-1), grid.array()).matrix()).asDiagonal() * c;

    nodes_t nodes = compute_nodes();
    diff_mat_t XM = nodes.template replicate<1, NUM_NODES>();
    diff_mat_t dX = XM - XM.transpose();

    diff_mat_t Dn = (c * (c.cwiseInverse()).transpose()).array() * (dX + diff_mat_t::Identity()).cwiseInverse().array();
    diff_mat_t diag_D = (Dn.rowwise().sum()).asDiagonal();

    return Dn - diag_D;
}

} // polympc namespace

#endif //POLYMPC_QUADRATURES_CLENSHAW_CURTIS_HPP
