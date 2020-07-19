#ifndef LEGENDRE_HPP
#define LEGENDRE_HPP

#include "polynomial_math.hpp"

namespace polympc {

using namespace polymath;

template<int PolyOrder, collocation_scheme Qtype = GAUSS_LOBATTO, typename _Scalar = double>
class Legendre
{
public:

    enum{
        POLY_ORDER = PolyOrder,
        NUM_NODES  = PolyOrder + 1
    };

public:
    /** constructor */
    Legendre();
    ~Legendre(){}

    using Scalar = _Scalar;

    using q_weights_t = Eigen::Matrix<Scalar, NUM_NODES, 1>;
    using nodes_t     = Eigen::Matrix<Scalar, NUM_NODES, 1>;
    using diff_mat_t  = Eigen::Matrix<Scalar, NUM_NODES, NUM_NODES>;
    using tensor_t    = Eigen::TensorFixedSize<Scalar, Eigen::Sizes<NUM_NODES, NUM_NODES, NUM_NODES>>;

    /** some getters */
    diff_mat_t D(){return _D;}
    q_weights_t QWeights(){return _QuadWeights;}
    nodes_t CPoints(){return _Nodes;}
    q_weights_t NFactors(){return _NormFactors;}
    tensor_t getGalerkinTensor(){return _Galerkin;}

    /** Evaluate Legendre polynomial of order n @bug put protectin against n > PolyOrder*/
    Scalar eval(const Scalar &arg, const int &n){return Ln(arg, n);}

    /** numerical integration of an arbitrary function using LGL quadratures*/
    template<class Integrand>
    Scalar integrate(const Scalar &t0= -1, const Scalar &tf = 1);

    /** Evaluate density function associated with Chebyshev basis */
    static Scalar weight(const Scalar &arg){return static_cast<Scalar>(1);}

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

    /** Legendre basis */
    using LnBASIS = Eigen::Matrix<Scalar, PolyOrder + 1, PolyOrder + 1>;
    LnBASIS _Ln = LnBASIS::Zero();
    void generate_legendre_basis();

    /** Tensor to hold Galerkin product */
    tensor_t _Galerkin;
    //Eigen::DynamicSGroup symmetry; // NOT EFFICIENT
    void compute_galerkin_tensor();

    Scalar poly_eval(const int &order, const Scalar &arg) {return Eigen::poly_eval(_Ln.col(order), arg); }

    /** Evaluate Lengendre polynomial of order n*/
    Scalar Ln(const Scalar &arg, const int &n){return Eigen::poly_eval(_Ln.col(n), arg);}
};

/** @brief constructor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
Legendre<PolyOrder, Qtype, Scalar>::Legendre()
{
    EIGEN_STATIC_ASSERT(Qtype == GAUSS_LOBATTO, "Sorry :( Only GAUSS_LOBATTO quadrature points available at the moment!");

    /** initialize pseudopsectral scheme */
    generate_legendre_basis();
    //std::cout << "Polynomial basis: \n" << _Ln << "\n";

    _Nodes = CollocPoints();
    //std::cout << "Nodal points: " << _Nodes.transpose() << "\n";

    _QuadWeights = QuadWeights();
    //std::cout << "Quadrature weights: " << _QuadWeights.transpose() << "\n";

    _NormFactors = NormFactors();
    //std::cout << "Normalization factors: " << _NormFactors.transpose() << "\n";

    //compute_galerkin_tensor();
}

/** @brief : compute nodal points for the Legendre collocation scheme */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::nodes_t
Legendre<PolyOrder, Qtype, Scalar>::CollocPoints()
{
    /** Legendre (LGL) collocation points for the interval [-1, 1]*/
    /** compute roots of LN_dot(x) polynomial - extremas of LN(x) */
    nodes_t LN_dot = poly_diff(_Ln.col(PolyOrder));
    Scalar eps = std::numeric_limits<Scalar>::epsilon();

    /** prepare the polynomial for the solver */
    for(int i = 0; i < PolyOrder; ++i)
    {
        if(std::fabs(LN_dot[i]) <= eps)
            LN_dot[i] = Scalar(0);
    }

    Eigen::PolynomialSolver<Scalar, PolyOrder-1> root_finder;
    root_finder.compute(segment<nodes_t, PolyOrder>(LN_dot, 0)); // remove the last zero

    nodes_t nodes = nodes_t::Zero();
    nodes[0] = -1; nodes[PolyOrder] = 1;

    segment<nodes_t, PolyOrder - 1>(nodes, 1) = root_finder.roots().real();

    /** sort the nodes in the ascending order */
    std::sort(nodes.data(), nodes.data() + nodes.size());
    return nodes;
}

/** @brief : compute LGL quadrature weights */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::q_weights_t
Legendre<PolyOrder, Qtype, Scalar>::QuadWeights()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    q_weights_t weights = q_weights_t::Zero();
    const Scalar coeff = Scalar(2) / (PolyOrder * (PolyOrder + 1));
    for(int i = 0; i <= PolyOrder; ++i)
    {
        Scalar LN_xi = Eigen::poly_eval(_Ln.col(PolyOrder), _Nodes[i]);
        weights[i] = coeff / std::pow(LN_xi, 2);
    }
    return weights;
}

/** @brief : compute LGL normalization factors c = 1 / ck */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
typename Legendre<PolyOrder, Qtype, Scalar>::q_weights_t
Legendre<PolyOrder, Qtype, Scalar>::NormFactors()
{
    q_weights_t factors = q_weights_t::Zero();
    for(int k = 0; k < PolyOrder; ++k)
    {
        factors[k] = (Scalar(2) * k + 1) / Scalar(2);
    }
    factors[PolyOrder] =  PolyOrder / Scalar(2);
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
void Legendre<PolyOrder, Qtype, Scalar>::generate_legendre_basis()
{
    /** the first basis polynomial is L0(x) = 1 */
    _Ln(0,0) = 1;
    /** the second basis polynomial is L1(x) = x */
    _Ln(1,1) = 1;

    /** compute recurrent coefficients */
    nodes_t a = nodes_t::Zero();
    nodes_t c = nodes_t::Zero();
    nodes_t x = nodes_t::Zero(); // p(x) = x
    x[1] = 1;
    for(int n = 0; n <= PolyOrder; ++n)
    {
        a[n] = Scalar(2 * n + 1) / (n + 1);
        c[n] = Scalar(n) / (n + 1);
    }

    /** create polynomial basis */
    /** @note had to put volatile here since the compiler messes copies of Ln
     * columns here in Release mode */
    for(volatile int n = 1; n < PolyOrder; ++n)
    {
        _Ln.col(n+1) = a[n] * poly_mul(_Ln.col(n), x) - c[n] * _Ln.col(n-1);
    }
}

/** @brief : Compute Galerkin Tensor */
template<int PolyOrder, collocation_scheme Qtype, typename Scalar>
void Legendre<PolyOrder, Qtype, Scalar>::compute_galerkin_tensor()
{
    /** naive implementation */
    for(int k = 0; k <= PolyOrder; ++k){
        for(int i = 0; i <= PolyOrder; ++i){
            for(int j = 0; j <= PolyOrder; ++j){
                Scalar inner_prod = 0;
                for(int n = 0; n <= PolyOrder; ++n)
                {
                    // compute projection //
                    inner_prod += poly_eval(i, _Nodes[n]) * poly_eval(j, _Nodes[n]) * poly_eval(k, _Nodes[n]) * _QuadWeights[n];
                }
                _Galerkin(i, j, k) = _NormFactors[k] * inner_prod;
            }
            //std::cout << "\n";
        }
        //std::cout << "\n \n";
    }

}

} // polympc namespace


#endif // LEGENDRE_HPP
