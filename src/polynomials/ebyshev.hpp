#ifndef EBYSHEV_HPP
#define EBYSHEV_HPP

#include "polynomial_math.hpp"

namespace polympc {

using namespace polymath;

template<int PolyOrder, collocation_scheme Qtype = GAUSS_LOBATTO, typename _Scalar = double>
class Chebyshev
{
public:

    enum
    {
        POLY_ORDER = PolyOrder,
        NUM_NODES = PolyOrder + 1
    };

public:
    /** constructor */
    Chebyshev();
    ~Chebyshev(){}

    using scalar_t = _Scalar;

    using q_weights_t   = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using nodes_t       = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using diff_mat_t    = Eigen::Matrix<scalar_t, NUM_NODES, NUM_NODES>;

    /** some getters */
    diff_mat_t D() const {return _D;}
    q_weights_t QWeights() const {return _QuadWeights;}
    q_weights_t CCQWeights() const {return _CCQuadWeights;}
    nodes_t CPoints() const {return _Nodes;}
    q_weights_t NFactors() const {return _NormFactors;}

    /** numerical integration of an arbitrary function */
    template<class Integrand>
    scalar_t integrate(const scalar_t &t0= -1, const scalar_t &tf = 1);

    /** Evaluate Chebyshev polynomial of order n*/
    scalar_t eval(const scalar_t &arg, const int &n){return Tn(arg, n);}
    static scalar_t eval(const scalar_t &arg){return std::cos(PolyOrder * std::acos(arg));}

    /** Evaluate density function associated with Chebyshev basis */
    static scalar_t weight(const scalar_t &arg){return 1.0 / std::sqrt(1 - std::pow(arg, 2));}

    /** generate differentiation matrix */
    static diff_mat_t  compute_diff_matrix() noexcept;
    /** compute nodal points */
    static nodes_t     compute_nodes() noexcept;
    /** compute Clenshaw-Curtis quadrature weights */
    static q_weights_t compute_int_weights() noexcept;
    /** compute Chebyshev quadrature weights */
    static q_weights_t compute_quad_weights() noexcept;
    /** compute normalization factors */
    static q_weights_t compute_norm_factors() noexcept;

private:

    /** private members */
    /** Diff matrix */
    diff_mat_t _D;
    /** Collocation points */
    nodes_t _Nodes;
    /** Clenshaw-Curtis Quadrature weights */
    q_weights_t _QuadWeights;
    /** Chebyshev quadrature weights */
    q_weights_t _CCQuadWeights;
    /** Normalisation factors */
    q_weights_t _NormFactors;

    /** Evaluate Chebyshev polynomial of order n*/
    inline scalar_t Tn(const scalar_t &arg, const int &n){return std::cos(n * std::acos(arg));}
};

/** @brief constructor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
Chebyshev<PolyOrder, Qtype, Scalar>::Chebyshev()
{
    EIGEN_STATIC_ASSERT(Qtype == GAUSS_LOBATTO, "Sorry :( Only GAUSS_LOBATTO quadrature points available at the moment!");
    /** initialize pseudopsectral scheme */
    _Nodes         = compute_nodes();
    _QuadWeights   = compute_quad_weights();
    _CCQuadWeights = compute_int_weights();

    _NormFactors = compute_norm_factors();
    _D           = compute_diff_matrix();
}

/** @brief : compute nodal points for the Chebyshev collocation scheme */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::nodes_t
Chebyshev<PolyOrder, Qtype, Scalar>::compute_nodes() noexcept
{
    nodes_t grid = nodes_t::LinSpaced(PolyOrder + 1, 0, PolyOrder);
    return (grid * (M_PI / PolyOrder)).array().cos();
}


/** @brief : compute Clenshaw-Curtis quadrature weights */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::q_weights_t
Chebyshev<PolyOrder, Qtype, Scalar>::compute_int_weights() noexcept
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    nodes_t theta = nodes_t::LinSpaced(PolyOrder + 1, 0, PolyOrder);
    theta *= (M_PI / PolyOrder);

    q_weights_t w = q_weights_t::Zero(PolyOrder + 1, 1);
    using tmp_vtype = Eigen::Matrix<Scalar, PolyOrder - 1, 1>;
    tmp_vtype v = tmp_vtype::Ones(PolyOrder - 1, 1);

    if ( PolyOrder % 2 == 0 )
    {
        w[0]         = static_cast<Scalar>(1 / (std::pow(PolyOrder, 2) - 1));
        w[PolyOrder] = w[0];

        for(int k = 1; k <= PolyOrder / 2 - 1; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<Scalar>(2.0 / (4 * std::pow(k, 2) - 1)) * vk;
        }
        tmp_vtype vk = Eigen::cos((PolyOrder * segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
        v -= vk / (std::pow(PolyOrder, 2) - 1);
    }
    else
    {
        w[0] = static_cast<Scalar>(1 / std::pow(PolyOrder, 2));
        w[PolyOrder] = w[0];
        for (int k = 1; k <= (PolyOrder - 1) / 2; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * segment<q_weights_t, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<Scalar>(2.0 / (4 * pow(k, 2) - 1)) * vk;
        }
    }

    segment<q_weights_t, PolyOrder - 1>(w, 1) =  static_cast<Scalar>(2.0 / PolyOrder) * v;
    return w;
}

/** @brief : compute Chebyshev quadrature weights */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::q_weights_t
Chebyshev<PolyOrder, Qtype, Scalar>::compute_quad_weights() noexcept
{
    q_weights_t w = q_weights_t::Constant(static_cast<Scalar>(M_PI / PolyOrder));
    w[0] *= 0.5; w[PolyOrder] *= 0.5;
    return w;
}

/** @brief : compute Chebyshev normalisation factors: c = 1/ck */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::q_weights_t
Chebyshev<PolyOrder, Qtype, Scalar>::compute_norm_factors() noexcept
{
    q_weights_t w = q_weights_t::Constant(static_cast<Scalar>(Scalar(2) / M_PI));
    w[0] *= Scalar(0.5);
    return w;
}

/** @brief : Compute integrals using CC-quadrature rule */
template<int PolyOrder, collocation_scheme Qtype,typename Scalar>
template<class Integrand>
Scalar Chebyshev<PolyOrder, Qtype, Scalar>::integrate(const Scalar &t0, const Scalar &tf)
{
    Scalar integral = 0;
    Integrand f;
    const Scalar t_scale = (tf - t0) / 2;
    const Scalar t_delta = (tf + t0) / 2;
    for(int i = 0; i <= PolyOrder; ++i)
    {
        integral += f(t_scale * _Nodes[i] + t_delta) * _CCQuadWeights[i];
    }
    return t_scale * integral;
}


template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::diff_mat_t
Chebyshev<PolyOrder, Qtype, Scalar>::compute_diff_matrix() noexcept
{
    nodes_t grid = nodes_t::LinSpaced(NUM_NODES, 0, POLY_ORDER);
    nodes_t c    = nodes_t::Ones(); c[0] = Scalar(2); c[POLY_ORDER] = Scalar(2);
    c = (Eigen::pow(Scalar(-1), grid.array()).matrix()).asDiagonal() * c;

    nodes_t nodes = compute_nodes();
    diff_mat_t XM = nodes.template replicate<1, NUM_NODES>();
    diff_mat_t dX = XM - XM.transpose();

    diff_mat_t Dn = (c * (c.cwiseInverse()).transpose()).array() * (dX + diff_mat_t::Identity()).cwiseInverse().array();
    diff_mat_t diag_D = (Dn.rowwise().sum()).asDiagonal();

    return Dn - diag_D;
}


} // polympc namespace

#endif // EBYSHEV_HPP
