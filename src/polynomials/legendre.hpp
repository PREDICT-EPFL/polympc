#ifndef LEGENDRE_HPP
#define LEGENDRE_HPP

#include "polynomial_math.hpp"

namespace polympc {

using namespace polymath;

template<int PolyOrder, collocation_scheme Qtype = GAUSS_LOBATTO, typename _Scalar = double>
class Legendre
{
public:

    enum
    {
        POLY_ORDER = PolyOrder,
        NUM_NODES  = PolyOrder + 1
    };

public:
    /** constructor */
    Legendre();
    ~Legendre(){}

    using scalar_t = _Scalar;
    using q_weights_t = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using nodes_t     = Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using diff_mat_t  = Eigen::Matrix<scalar_t, NUM_NODES, NUM_NODES>;
    using tensor_t    = Eigen::TensorFixedSize<scalar_t, Eigen::Sizes<NUM_NODES, NUM_NODES, NUM_NODES>>;
    /** Legendre basis */
    using basis_t = Eigen::Matrix<scalar_t, POLY_ORDER + 1, POLY_ORDER + 1>;

    /** some getters */
    diff_mat_t  D() const {return _D;}
    q_weights_t QWeights() const {return _QuadWeights;}
    nodes_t     CPoints()  const {return _Nodes;}
    q_weights_t NFactors() const {return _NormFactors;}
    tensor_t    getGalerkinTensor() const {return _Galerkin;}

    /** Evaluate Legendre polynomial of order n @bug put protectin against n > PolyOrder*/
    scalar_t eval(const scalar_t &arg, const int &n){return Ln(arg, n);}

    /** numerical integration of an arbitrary function using LGL quadratures*/
    template<class Integrand>
    scalar_t integrate(const scalar_t &t0= -1, const scalar_t &tf = 1);

    /** Evaluate density function associated with Chebyshev basis */
    static scalar_t weight(const scalar_t &arg){return static_cast<scalar_t>(1);}

    /** compute basis */
    static basis_t compute_basis();
    /** generate differentiation matrix */
    static diff_mat_t  compute_diff_matrix();
    /** compute nodal points */
    static nodes_t     compute_nodes();
    /** compute quadrature weights */
    static q_weights_t compute_int_weights();
    /** compute quadrature weights */
    static q_weights_t compute_quad_weights();
    /** compute normalization factors */
    static q_weights_t compute_norm_factors();

private:

    /** generate Differentiation matrix */
    diff_mat_t DiffMatrix();
    /** compute nodal points */
    nodes_t CollocPoints();
    /** compute clenshaw-Curtis quadrature weights */
    q_weights_t QuadWeights();
    /** compute normalization factors */
    q_weights_t NormFactors();

    /** private members */
    /** Diff matrix */
    diff_mat_t _D = diff_mat_t::Zero();
    /** Collocation points */
    nodes_t _Nodes = nodes_t::Zero();
    /** Quadrature weights */
    q_weights_t _QuadWeights = q_weights_t::Zero();
    /** Normalization factors */
    q_weights_t _NormFactors = q_weights_t::Zero();

    basis_t _Ln = basis_t::Zero();
    void generate_legendre_basis();

    /** Tensor to hold Galerkin product */
    tensor_t _Galerkin;
    //Eigen::DynamicSGroup symmetry; // NOT EFFICIENT
    void compute_galerkin_tensor();

    scalar_t poly_eval(const int &order, const scalar_t &arg) {return Eigen::poly_eval(_Ln.col(order), arg); }

    /** Evaluate Lengendre polynomial of order n*/
    scalar_t Ln(const scalar_t &arg, const int &n){return Eigen::poly_eval(_Ln.col(n), arg);}
};

/** @brief constructor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
Legendre<PolyOrder, Qtype, Scalar>::Legendre()
{
    EIGEN_STATIC_ASSERT(Qtype == GAUSS_LOBATTO, "Sorry :( Only GAUSS_LOBATTO quadrature points available at the moment!");

    /** initialize pseudopsectral scheme */
    _Ln    = compute_basis();
    _Nodes = compute_nodes();
    _D     = compute_diff_matrix();

    _QuadWeights = compute_quad_weights();
    _NormFactors = compute_norm_factors();

    //compute_galerkin_tensor();
}

/** @brief : compute nodal points for the Legendre collocation scheme */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::nodes_t
Legendre<PolyOrder, Qtype, Scalar>::compute_nodes()
{
    basis_t Ln = compute_basis();
    /** Legendre (LGL) collocation points for the interval [-1, 1]*/
    /** compute roots of LN_dot(x) polynomial - extremas of LN(x) */
    nodes_t LN_dot = poly_diff(Ln.col(PolyOrder));
    scalar_t eps = std::numeric_limits<scalar_t>::epsilon();

    /** prepare the polynomial for the solver */
    for(int i = 0; i < PolyOrder; ++i)
    {
        if(std::fabs(LN_dot[i]) <= eps)
            LN_dot[i] = scalar_t(0);
    }

    Eigen::PolynomialSolver<scalar_t, PolyOrder - 1> root_finder;
    root_finder.compute(segment<nodes_t, PolyOrder>(LN_dot, 0)); // remove the last zero

    nodes_t nodes = nodes_t::Zero();
    nodes[0] = -1; nodes[PolyOrder] = 1;

    segment<nodes_t, PolyOrder - 1>(nodes, 1) = root_finder.roots().real();

    /** sort the nodes in the ascending order */
    std::sort(nodes.data(), nodes.data() + nodes.size());
    return nodes;
}

/** compute differentiation matrix */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::diff_mat_t
Legendre<PolyOrder, Qtype, Scalar>::compute_diff_matrix()
{
    diff_mat_t D  = diff_mat_t::Zero();
    nodes_t nodes = compute_nodes();
    basis_t Ln    = compute_basis();

    /** use formula from Canuto-Quarteroni */
    D(0,0) = (POLY_ORDER + 1) * POLY_ORDER / static_cast<scalar_t>(4);
    D(POLY_ORDER, POLY_ORDER) = -D(0,0);

    for(Eigen::Index i = 0; i < NUM_NODES; ++i)
    {
        for (Eigen::Index j = 0; j < NUM_NODES; ++j)
        {
            if(i != j)
                D(i,j) = (Eigen::poly_eval(Ln.col(POLY_ORDER), nodes[i]) / Eigen::poly_eval(Ln.col(POLY_ORDER), nodes(j)))
                        * (1 / (nodes(i) - nodes(j)));
        }
    }
    return D;
}

/** @brief : compute LGL quadrature weights */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::q_weights_t
Legendre<PolyOrder, Qtype, Scalar>::compute_quad_weights()
{
    const basis_t Ln = compute_basis();
    const nodes_t nodes = compute_nodes();
    /** Chebyshev collocation points for the interval [-1, 1]*/
    q_weights_t weights  = q_weights_t::Zero();
    const scalar_t coeff = scalar_t(2) / (PolyOrder * (PolyOrder + 1));
    for(int i = 0; i <= PolyOrder; ++i)
    {
        scalar_t LN_xi = Eigen::poly_eval(Ln.col(PolyOrder), nodes[i]);
        weights[i] = coeff / std::pow(LN_xi, 2);
    }
    return weights;
}

template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::q_weights_t
Legendre<PolyOrder, Qtype, Scalar>::compute_int_weights()
{
    return compute_quad_weights();
}

/** @brief : compute LGL normalization factors c = 1 / ck */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::q_weights_t
Legendre<PolyOrder, Qtype, Scalar>::compute_norm_factors()
{
    q_weights_t factors = q_weights_t::Zero();
    for(int k = 0; k < PolyOrder; ++k)
    {
        factors[k] = (scalar_t(2) * k + 1) / scalar_t(2);
    }
    factors[PolyOrder] =  PolyOrder / scalar_t(2);
    return factors;
}

/** @brief : Compute integrals using LGL-quadrature rule */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
template<class Integrand>
Scalar Legendre<PolyOrder, Qtype, Scalar>::integrate(const Scalar &t0, const Scalar &tf)
{
    Scalar integral = 0;
    Integrand f;
    const Scalar t_scale = (tf - t0) / 2;
    const Scalar t_delta = (tf + t0) / 2;
    for(int i = 0; i <= PolyOrder; ++i)
    {
        integral += f(t_scale * _Nodes[i] + t_delta) * _QuadWeights[i];
    }
    return t_scale * integral;
}

/** @brief : Compute Legendre basis*/
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::basis_t
Legendre<PolyOrder, Qtype, Scalar>::compute_basis()
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
        Ln.col(n+1)= a(n) * poly_mul(Ln.col(n), x) - c(n) * Ln.col(n-1);

    return Ln;
}


/** @brief : Compute Galerkin Tensor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
void Legendre<PolyOrder, Qtype, Scalar>::compute_galerkin_tensor()
{
    /** naive implementation */
    for(int k = 0; k <= PolyOrder; ++k){
        for(int i = 0; i <= PolyOrder; ++i){
            for(int j = 0; j <= PolyOrder; ++j){
                scalar_t inner_prod = 0;
                for(int n = 0; n <= PolyOrder; ++n)
                {
                    // compute projection //
                    inner_prod += poly_eval(i, _Nodes[n]) * poly_eval(j, _Nodes[n]) * poly_eval(k, _Nodes[n]) * _QuadWeights[n];
                }
                _Galerkin(i, j, k) = _NormFactors[k] * inner_prod;
            }
        }
    }

}

} // polympc namespace


#endif // LEGENDRE_HPP
