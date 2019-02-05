#ifndef EBYSHEV_HPP
#define EBYSHEV_HPP

#include "iostream"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"

enum q_type: unsigned char{GAUSS, GAUSS_RADAU, GAUSS_LOBATTO};

template<typename Derived, int size>
Eigen::VectorBlock<Derived, size>
segment(Eigen::MatrixBase<Derived>& v, int start)
{
  return Eigen::VectorBlock<Derived, size>(v.derived(), start);
}
template<typename Derived, int size>
const Eigen::VectorBlock<const Derived, size>
segment(const Eigen::MatrixBase<Derived>& v, int start)
{
  return Eigen::VectorBlock<const Derived, size>(v.derived(), start);
}


/** --------------------------------------------------------------- */
/** --------------------------------------------------------------- */

template<int PolyOrder, q_type Qtype = GAUSS_LOBATTO, typename Scalar = double>
class Chebyshev
{
private:
    static constexpr int NUM_NODES = PolyOrder + 1;

public:
    /** constructor */
    Chebyshev();
    ~Chebyshev(){}

    /** datat types */
    //using XTYPE = Eigen::Matrix<Scalar, NUM_NODES * NX, 1>;
    //using UTYPE = Eigen::Matrix<Scalar, NUM_NODES * NU, 1>;
    //using PTYPE = Eigen::Matrix<Scalar, NUM_NODES * NP, 1>;

    using QUADWEIGHTYPE = Eigen::Matrix<Scalar, PolyOrder + 1, 1>;
    using NODESVECTYPE  = Eigen::Matrix<Scalar, PolyOrder + 1, 1>;
    using DIFFMTYPE = Eigen::Matrix<Scalar, NUM_NODES, NUM_NODES>;
    //using COMPDIFFMTYPE = Eigen::Matrix<Scalar, NUM_NODES * NX, NUM_NODES * NX>;

    /** some getters */
    DIFFMTYPE D(){return _D;}
    QUADWEIGHTYPE QWeights(){return _QuadWeights;}
    NODESVECTYPE CPoints(){return _Nodes;}

    /** @brief : Orthogonal projections */
    /** a struct to store Chebyshev projections */
    struct Projection
    {
        QUADWEIGHTYPE coeff;
        Scalar t_scale, t_delta;
        Scalar eval(const Scalar &arg)
        {
            Scalar val = 0;
            for(int i = 0; i <= PolyOrder; ++i)
            {
                Scalar _arg = (arg - t_delta) / t_scale;
                val += coeff[i] * Tn(_arg, i);
            }
            return val;
        }
    };

    /** Chebyshev transform using orthogonal projection */
    template<class Function>
    Projection project(const Scalar &t0 = -1, const Scalar &tf = 1);

    /** numerical integration of an arbitrary function */
    template<class Integrand>
    Scalar integrate(const Scalar &t0= -1, const Scalar &tf = 1);

    /** Evaluate Chebyshev polynomial of order n*/
    static Scalar Tn(const Scalar &arg, const int &n){return std::cos(n * std::acos(arg));}

    /** Evaluate density function associated with Chebyshev basis */
    static Scalar weight(const Scalar &arg){return 1.0 / std::sqrt(1 - std::pow(arg, 2));}

private:

    /** generate Differentiation matrix */
    DIFFMTYPE DiffMatrix();
    /** compute nodal points */
    NODESVECTYPE CollocPoints();
    /** compute clenshaw-Curtis quadrature weights */
    QUADWEIGHTYPE QuadWeights();

    /** private members */
    /** Diff matrix */
    DIFFMTYPE _D;
    /** Collocation points */
    NODESVECTYPE _Nodes;
    /** Quadrature weights */
    QUADWEIGHTYPE _QuadWeights;

    /** variables to store interpolation coefficients */
    // XTYPE _X;
    // UTYPE _U;
    // PTYPE _P;
};

/** @brief constructor */
template<int PolyOrder, q_type Qtype, typename Scalar>
Chebyshev<PolyOrder, Qtype, Scalar>::Chebyshev()
{
    EIGEN_STATIC_ASSERT(Qtype == GAUSS_LOBATTO, "Sorry :( Only GAUSS_LOBATTO quadrature points available at the moment!");
    /** initialize pseudopsectral scheme */
    _Nodes = CollocPoints();
    std::cout << "Nodal points: " << _Nodes.transpose() << "\n";
    _QuadWeights = QuadWeights();
    std::cout << "Quadrature weights: " << _QuadWeights.transpose() << "\n";
    //_D           = DiffMatrix();
    //_ComD        = CompDiffMatrix();

    /** initialize coefficients */
    //_X = XTYPE::Zero();
    //_U = UTYPE::Zero();
    //_P = PTYPE::Zero();
}

/** @brief : compute nodal points for the Chebyshev collocation scheme */
template<int PolyOrder, q_type Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::NODESVECTYPE
Chebyshev<PolyOrder, Qtype, Scalar>::CollocPoints()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    NODESVECTYPE grid = NODESVECTYPE::LinSpaced(PolyOrder + 1, 0, PolyOrder);
    return (grid * (M_PI / PolyOrder)).array().cos();
}

/** @brief : compute Clenshaw-Curtis quadrature weights */
template<int PolyOrder, q_type Qtype, typename Scalar>
typename Chebyshev<PolyOrder, Qtype, Scalar>::QUADWEIGHTYPE
Chebyshev<PolyOrder, Qtype, Scalar>::QuadWeights()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    NODESVECTYPE theta = NODESVECTYPE::LinSpaced(PolyOrder + 1, 0, PolyOrder);
    theta *= (M_PI / PolyOrder);

    QUADWEIGHTYPE w = QUADWEIGHTYPE::Zero(PolyOrder + 1, 1);
    using tmp_vtype = Eigen::Matrix<Scalar, PolyOrder - 1, 1>;
    tmp_vtype v = tmp_vtype::Ones(PolyOrder - 1, 1);

    if ( PolyOrder % 2 == 0 )
    {
        w[0]         = static_cast<Scalar>(1 / (std::pow(PolyOrder, 2) - 1));
        w[PolyOrder] = w[0];

        for(int k = 1; k <= PolyOrder / 2 - 1; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * segment<QUADWEIGHTYPE, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<Scalar>(2.0 / (4 * std::pow(k, 2) - 1)) * vk;
        }
        tmp_vtype vk = Eigen::cos((PolyOrder * segment<QUADWEIGHTYPE, PolyOrder - 1>(theta, 1)).array());
        v -= vk / (std::pow(PolyOrder, 2) - 1);
    }
    else
    {
        w[0] = static_cast<Scalar>(1 / std::pow(PolyOrder, 2));
        w[PolyOrder] = w[0];
        for (int k = 1; k <= (PolyOrder - 1) / 2; ++k)
        {
            tmp_vtype vk = Eigen::cos((2 * k * segment<QUADWEIGHTYPE, PolyOrder - 1>(theta, 1)).array());
            v -= static_cast<Scalar>(2.0 / (4 * pow(k, 2) - 1)) * vk;
        }
    }

    segment<QUADWEIGHTYPE, PolyOrder - 1>(w, 1) =  static_cast<Scalar>(2.0 / PolyOrder) * v;
    return w;
}

/** @brief : Compute integrals using CC-quadrature rule */
template<int PolyOrder, q_type Qtype,typename Scalar>
template<class Integrand>
Scalar Chebyshev<PolyOrder, Qtype, Scalar>::integrate(const Scalar &t0, const Scalar &tf)
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


/** @brief : Compute orthogonal projection onto the Chebyshev basis using Chebyshev-Gauss quadrature*/
template<int PolyOrder, q_type Qtype, typename Scalar>
template<class Function>
typename Chebyshev<PolyOrder, Qtype, Scalar>::Projection
Chebyshev<PolyOrder, Qtype, Scalar>::project(const Scalar &t0, const Scalar &tf)
{
    Projection proj;
    Function f;
    QUADWEIGHTYPE w = QUADWEIGHTYPE::Constant(static_cast<Scalar>(M_PI / PolyOrder));
    w[0] *= 0.5; w[PolyOrder] *= 0.5;

    const Scalar t_scale = (tf - t0) / 2;
    const Scalar t_delta = (tf + t0) / 2;

    for(int n = 0; n <= PolyOrder; ++n)
    {
        Scalar fn;
        Scalar inner_prod = 0;
        for(int i = 0; i <= PolyOrder; ++i)
        {
            inner_prod += f(t_scale * _Nodes[i] + t_delta) * Tn(_Nodes[i], n) * w[i];
        }
        fn = (2.0 / M_PI) * inner_prod;
        proj.coeff[n] = fn;
    }
    proj.coeff[0] *= 0.5;
    proj.t_scale = t_scale;
    proj.t_delta = t_delta;

    return proj;
}

#endif // EBYSHEV_HPP
