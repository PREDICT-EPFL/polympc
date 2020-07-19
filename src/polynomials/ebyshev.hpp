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

    using Scalar = _Scalar;

    using q_weights_t   = Eigen::Matrix<Scalar, NUM_NODES, 1>;
    using nodes_t       = Eigen::Matrix<Scalar, NUM_NODES, 1>;
    using diff_mat_t    = Eigen::Matrix<Scalar, NUM_NODES, NUM_NODES>;

    /** some getters */
    diff_mat_t D(){return _D;}
    q_weights_t QWeights(){return _QuadWeights;}
    q_weights_t CCQWeights(){return _CCQuadWeights;}
    nodes_t CPoints(){return _Nodes;}
    q_weights_t NFactors(){return _NormFactors;}

    /** numerical integration of an arbitrary function */
    template<class Integrand>
    Scalar integrate(const Scalar &t0= -1, const Scalar &tf = 1);

    /** Evaluate Chebyshev polynomial of order n*/
    Scalar eval(const Scalar &arg, const int &n){return Tn(arg, n);}

    /** Evaluate density function associated with Chebyshev basis */
    static Scalar weight(const Scalar &arg){return 1.0 / std::sqrt(1 - std::pow(arg, 2));}

private:

    /** generate Differentiation matrix */
    diff_mat_t DiffMatrix();
    /** compute nodal points */
    nodes_t CollocPoints();
    /** compute Clenshaw-Curtis quadrature weights */
    q_weights_t CCQuadWeights();
    /** compute Chebyshev quadrature points */
    q_weights_t QuadWeights();
    /** compute normalization factors */
    q_weights_t NormFactors();

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
    inline Scalar Tn(const Scalar &arg, const int &n){return std::cos(n * std::acos(arg));}
};

/** @brief constructor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
Chebyshev<PolyOrder, Qtype, Scalar>::Chebyshev()
{
    EIGEN_STATIC_ASSERT(Qtype == GAUSS_LOBATTO, "Sorry :( Only GAUSS_LOBATTO quadrature points available at the moment!");
    /** initialize pseudopsectral scheme */
    _Nodes = CollocPoints();
    //std::cout << "Nodal points: " << _Nodes.transpose() << "\n";
    _QuadWeights = QuadWeights();
    //std::cout << "Quadrature weights: " << _QuadWeights.transpose() << "\n";
    _CCQuadWeights = CCQuadWeights();

    _NormFactors = NormFactors();
    _D           = DiffMatrix();
    //_ComD        = CompDiffMatrix();
}

/** @brief : compute nodal points for the Chebyshev collocation scheme */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::nodes_t
Chebyshev<PolyOrder, Qtype, Scalar>::CollocPoints()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    nodes_t grid = nodes_t::LinSpaced(PolyOrder + 1, 0, PolyOrder);
    return (grid * (M_PI / PolyOrder)).array().cos();
}

/** @brief : compute Clenshaw-Curtis quadrature weights */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::q_weights_t
Chebyshev<PolyOrder, Qtype, Scalar>::CCQuadWeights()
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
Chebyshev<PolyOrder, Qtype, Scalar>::QuadWeights()
{
    q_weights_t w = q_weights_t::Constant(static_cast<Scalar>(M_PI / PolyOrder));
    w[0] *= 0.5; w[PolyOrder] *= 0.5;
    return w;
}

/** @brief : compute Chebyshev normalisation factors: c = 1/ck */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::q_weights_t
Chebyshev<PolyOrder, Qtype, Scalar>::NormFactors()
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


/** @brief : compute Chebyshev differentiation matrix (Trefethen)*/
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::diff_mat_t
Chebyshev<PolyOrder, Qtype, Scalar>::DiffMatrix()
{
    nodes_t grid = nodes_t::LinSpaced(NUM_NODES, 0, POLY_ORDER);
    nodes_t c    = nodes_t::Ones(); c[0] = Scalar(2); c[POLY_ORDER] = Scalar(2);
    c = (Eigen::pow(Scalar(-1), grid.array()).matrix()).asDiagonal() * c;

    diff_mat_t XM = _Nodes.template replicate<1, NUM_NODES>();
    diff_mat_t dX = XM - XM.transpose();

    diff_mat_t Dn = (c * (c.cwiseInverse()).transpose()).array() * (dX + diff_mat_t::Identity()).cwiseInverse().array();
    diff_mat_t diag_D = (Dn.rowwise().sum()).asDiagonal();

    return Dn - diag_D;
}


} // polympc namespace

#endif // EBYSHEV_HPP
