#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include "iostream"

/** @brief : Orthogonal projections */
/** a struct to store Chebyshev projections */
template<typename Basis>
class Projection
{
public:
    using Scalar = typename Basis::Scalar;
    using coeff_t = typename Basis::q_weights_t;
    using nodes_t = typename Basis::nodes_t;

    /** Coefficients of the expansion */
    coeff_t coeff;

    template<typename Function>
    Projection(const Function &f, const Scalar &a = Scalar(-1), const Scalar &b = Scalar(1))
    {
        m_nodes = m_basis_func.CPoints();
        m_quad_weights = m_basis_func.QWeights();
        m_norm_factors = m_basis_func.NFactors();
        project(f, a, b);
    }
    ~Projection(){}

    /** compute orthogonal projection */
    template<typename Function>
    void project(const Function &f, const Scalar &a = Scalar(-1), const Scalar &b = Scalar(1))
    {
        t_scale = (b - a) / 2;
        t_delta = (b + a) / 2;

        for(int n = 0; n <= PolyOrder; ++n)
        {
            Scalar inner_prod = 0;
            for(int i = 0; i <= PolyOrder; ++i)
            {
                inner_prod += f(t_scale * m_nodes[i] + t_delta) * m_basis_func.eval(m_nodes[i], n) * m_quad_weights[i];
            }
            coeff[n] = m_norm_factors[n] * inner_prod;
        }
    }

    /** evaluate projection */
    Scalar eval(const Scalar &arg)
    {
        Scalar val = 0;
        for(int i = 0; i <= PolyOrder; ++i)
        {
            Scalar _arg = (arg - t_delta) / t_scale;
            val += coeff[i] * m_basis_func.eval(_arg, i);
        }
        return val;
    }

private:
    Basis m_basis_func;

    int PolyOrder = (Basis::q_weights_t::RowsAtCompileTime - 1);
    Scalar t_scale = Scalar(1), t_delta = Scalar(0);

    nodes_t m_nodes;
    coeff_t m_quad_weights;
    coeff_t m_norm_factors;
};


/** @brief : Compute orthogonal projection onto the Chebyshev basis using Chebyshev-Gauss quadrature*/
/**
template<int PolyOrder, q_type Qtype, typename Scalar>
template<class Function>
typename Legendre<PolyOrder, Qtype, Scalar>::Projection
Legendre<PolyOrder, Qtype, Scalar>::project(const Scalar &t0, const Scalar &tf)
{
    Projection proj;
    Function f;
    const Scalar t_scale = (tf - t0) / 2;
    const Scalar t_delta = (tf + t0) / 2;

    for(int n = 0; n <= PolyOrder; ++n)
    {
        Scalar fn;
        Scalar inner_prod = 0;
        for(int i = 0; i <= PolyOrder; ++i)
        {
            inner_prod += f(t_scale * _Nodes[i] + t_delta) * Eigen::poly_eval(_Ln.col(n), _Nodes[i]) * _QuadWeights[i];
        }
        fn = ((2.0 * n + 1) / 2.0) * inner_prod;
        proj.coeff[n] = fn;
    }
    proj.coeff[PolyOrder] *= (PolyOrder / (2.0 * PolyOrder + 1));
    proj.t_scale = t_scale;
    proj.t_delta = t_delta;

    return proj;
}
*/


#endif // PROJECTION_HPP
