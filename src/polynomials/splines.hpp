#ifndef SPLINES_HPP
#define SPLINES_HPP

namespace polympc {


template<typename Polynomial, int NumSegments>
class Spline
{
public:
    Spline() = default;
    ~Spline() = default;

    enum
    {
        POLY_ORDER   = Polynomial::POLY_ORDER,
        NUM_SEGMENTS = NumSegments,
        NUM_NODES    = POLY_ORDER * NUM_SEGMENTS + 1
    };

    using scalar_t    = typename Polynomial::scalar_t;
    using q_weights_t = typename Polynomial::q_weights_t;
    using nodes_t     = typename Polynomial::nodes_t;
    using diff_mat_t  = typename Polynomial::diff_mat_t;

    static diff_mat_t  compute_diff_matrix()  {return Polynomial::compute_diff_matrix();}
    static q_weights_t compute_int_weights()  {return Polynomial::compute_int_weights();}
    static nodes_t     compute_nodes()        {return Polynomial::compute_nodes();}
    static q_weights_t compute_quad_weights() {return Polynomial::compute_quad_weights();}
    static q_weights_t compute_norm_factors() {return Polynomial::compute_norm_factors();}
 };

} // polympc namespace

#endif // SPLINES_HPP
