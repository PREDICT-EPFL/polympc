#ifndef LEGENDRE_HPP
#define LEGENDRE_HPP

#include "iostream"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/unsupported/Eigen/Polynomials"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "eigen3/unsupported/Eigen/CXX11/TensorSymmetry"

/** utility functions */
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

/** very partivular product of two polynomials that the maximum order */
/** works with static matrices only */
template<typename Derived, typename DerivedB>
typename Derived::PlainObject poly_mul(const Eigen::MatrixBase<Derived> &p1, const Eigen::MatrixBase<DerivedB> &p2)
{
    typename Derived::PlainObject product;
    const int p1_size = Derived::RowsAtCompileTime;
    const int p2_size = Derived::RowsAtCompileTime;
    static_assert(p1_size == p2_size, "poly_mul: polynomials should be of the same order!");

    using Scalar = typename Derived::Scalar;
    Scalar eps = std::numeric_limits<Scalar>::epsilon();

    /** detect nonzeros */
    int nnz_p1, nnz_p2;
    for(int i = 0; i < p1_size; ++i)
    {
        if(std::fabs(p1[i]) >= eps)
            nnz_p1 = i;
        if(std::fabs(p2[i]) >= eps)
            nnz_p2 = i;
    }

    for(int i = 0; i <= nnz_p2; ++i)
    {
        for(int j = 0; j <= nnz_p1; ++j)
        {
            /** truncate higher orders if neccessary */
            if( (i+j) == p1_size )
                break;
            product[i+j] = p1[j] * p2[i];
        }
    }

    return product;
}


/** differentiate polynomial */
/** works for static matrices only */
template<typename Derived>
typename Derived::PlainObject poly_diff(const Eigen::MatrixBase<Derived> &p)
{
    typename Derived::Scalar max_order = Derived::RowsAtCompileTime;
    using DerivedObj = typename Derived::PlainObject;
    const DerivedObj ord = DerivedObj::LinSpaced(max_order, 0, max_order - 1);
    DerivedObj derivative = DerivedObj::Zero();

    for(int i = 0; i < max_order - 1; ++i)
        derivative[i] = ord[i+1] * p[i+1];

    return derivative;
}


/** ---------------------------------------------------------------------*/


template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar = double>
class Legendre
{
private:
    static constexpr int NUM_NODES = PolyOrder + NumSegments + 1;

public:
    /** constructor */
    Legendre();
    ~Legendre(){}

    /** datat types */
    using XTYPE = Eigen::Matrix<Scalar, NUM_NODES * NX, 1>;
    using UTYPE = Eigen::Matrix<Scalar, NUM_NODES * NU, 1>;
    using PTYPE = Eigen::Matrix<Scalar, NUM_NODES * NP, 1>;

    using QUADWEIGHTYPE = Eigen::Matrix<Scalar, PolyOrder + 1, 1>;
    using NODESVECTYPE  = Eigen::Matrix<Scalar, PolyOrder + 1, 1>;
    using DIFFMTYPE = Eigen::Matrix<Scalar, NUM_NODES, NUM_NODES>;
    using COMPDIFFMTYPE = Eigen::Matrix<Scalar, NUM_NODES * NX, NUM_NODES * NX>;
    using MTENSOR = Eigen::TensorFixedSize<Scalar, Eigen::Sizes<PolyOrder + 1, PolyOrder + 1, PolyOrder + 1>>;

    /** some getters */
    DIFFMTYPE D(){return _D;}
    COMPDIFFMTYPE CompD(){return _ComD;}
    QUADWEIGHTYPE QWeights(){return _QuadWeights;}
    NODESVECTYPE CPoints(){return _Nodes;}
    NODESVECTYPE NFactors(){return _NormFactors;}
    MTENSOR getGalerkinTensor(){return _Galerkin;}

    /** @brief : Orthogonal projections */
    /** a struct to store Legendre projections */
    struct Projection : public Legendre
    {
        QUADWEIGHTYPE coeff;
        Scalar t_scale, t_delta;
        Scalar eval(const Scalar &arg)
        {
            Scalar val = 0;
            for(int i = 0; i <= PolyOrder; ++i)
            {
                Scalar _arg = (arg - t_delta) / t_scale;
                val += coeff[i] * Ln(_arg, i);
            }
            return val;
        }
    };

    /** Evaluate Lengendre polynomial of order n*/
    Scalar Ln(const Scalar &arg, const int &n){return Eigen::poly_eval(_Ln.col(n), arg);}

    /** Legendre transform using orthogonal projection */
    template<class Function>
    Projection project(const Scalar &t0 = -1, const Scalar &tf = 1);

    /** numerical integration of an arbitrary function using LGL quadratures*/
    template<class Integrand>
    Scalar integrate(const Scalar &t0= -1, const Scalar &tf = 1);

    /** Evaluate density function associated with Chebyshev basis */
    static Scalar weight(const Scalar &arg){return static_cast<Scalar>(1);}

private:

    /** generate Differentiation matrix */
    DIFFMTYPE DiffMatrix();
    /** compute nodal points */
    NODESVECTYPE CollocPoints();
    /** compute clenshaw-Curtis quadrature weights */
    QUADWEIGHTYPE QuadWeights();
    /** compute normalization factors */
    NODESVECTYPE NormFactors();
    /** compute composite differentiation matrix */
    COMPDIFFMTYPE CompDiffMatrix();

    /** private members */
    /** Diff matrix */
    DIFFMTYPE _D;
    /** Composite diff matrix */
    COMPDIFFMTYPE _ComD;
    /** Collocation points */
    NODESVECTYPE _Nodes;
    /** Quadrature weights */
    QUADWEIGHTYPE _QuadWeights;
    /** Normalization factors */
    NODESVECTYPE _NormFactors;

    /** variables to store interpolation coefficients */
    XTYPE _X;
    UTYPE _U;
    PTYPE _P;

    /** Legendre basis */
    using LnBASIS = Eigen::Matrix<Scalar, PolyOrder + 1, PolyOrder + 1>;
    LnBASIS _Ln;
    void generate_legendre_basis();

    /** Tensor to hold Galerkin product */
    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<PolyOrder + 1, PolyOrder + 1, PolyOrder + 1>> _Galerkin;
    //Eigen::DynamicSGroup symmetry; // NOT EFFICIENT
    void compute_galerkin_tensor();
    Scalar poly_eval(const int &order, const Scalar &arg) {return Eigen::poly_eval(_Ln.col(order), arg); }

    /**
    BaseClass VarX(){return _X;}
    BaseClass VarU(){return _U;}
    BaseClass VarP(){return _P;}

    BaseClass CollocateDynamics(casadi::Function &dynamics, const double &t0, const double &tf);
    BaseClass CollocateCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                            const double &t0, const double &tf);
    BaseClass CollocateIdCost(casadi::Function &IdCost, casadi::DM data, const double &t0, const double &tf);

    typedef std::function<BaseClass(BaseClass, BaseClass, BaseClass)> functor;
    */
    /** right hand side function of the ODE */
    /**
    functor _ode;
    double _t0, _tf;
    functor CollocateDynamics2(const functor &dynamics, const double &t0, const double &tf);
    BaseClass collocate_dynamics(const BaseClass &X, const BaseClass &U, const BaseClass &P);

private:

    /** helper functions */
};

/** @brief constructor */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::Legendre()
{
    /** initialize pseudopsectral scheme */
    generate_legendre_basis();
    //std::cout << "Polynomial basis: \n" << _Ln << "\n";

    _Nodes = CollocPoints();
    //std::cout << "Nodal points: " << _Nodes.transpose() << "\n";

    _QuadWeights = QuadWeights();
    //std::cout << "Quadrature weights: " << _QuadWeights.transpose() << "\n";

    _NormFactors = NormFactors();
    //std::cout << "Normalization factors: " << _NormFactors.transpose() << "\n";

    compute_galerkin_tensor();
    //Eigen::TensorFixedSize<Scalar, Eigen::Sizes<PolyOrder + 1, PolyOrder + 1>> lox = _Galerkin.chip(5, 2);
    //Eigen::DynamicSGroup symmetry; // NOT EFFICIENT = _Galerkin.chip(1, 2);
    //std::cout << lox << "\n \n";
    //std::cout << "Galerkin: " << _Galerkin.dimension(0) << " x " << _Galerkin.dimension(1) << " x " << _Galerkin.dimension(2) << "\n";

    //_D           = DiffMatrix();
    //_ComD        = CompDiffMatrix();

    /** initialize coefficients */
    _X = XTYPE::Zero();
    _U = UTYPE::Zero();
    _P = PTYPE::Zero();

    std::cout << "CONSTRUCTOR CALL \n";
}

/** @brief : compute nodal points for the Legendre collocation scheme */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
typename Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::NODESVECTYPE
Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::CollocPoints()
{
    /** Legendre (LGL) collocation points for the interval [-1, 1]*/
    /** compute roots of LN_dot(x) polynomial - extremas of LN(x) */
    NODESVECTYPE LN_dot = poly_diff(_Ln.col(PolyOrder));
    Scalar eps = std::numeric_limits<Scalar>::epsilon();

    /** prepare the polynomial for the solver */
    for(int i = 0; i < PolyOrder; ++i)
    {
        if(std::fabs(LN_dot[i]) <= eps)
            LN_dot[i] = static_cast<Scalar>(0);
    }

    Eigen::PolynomialSolver<Scalar, PolyOrder-1> root_finder;
    root_finder.compute(segment<NODESVECTYPE, PolyOrder>(LN_dot, 0)); // remove the last zero

    NODESVECTYPE nodes = NODESVECTYPE::Zero();
    nodes[0] = -1; nodes[PolyOrder] = 1;

    segment<NODESVECTYPE, PolyOrder - 1>(nodes, 1) = root_finder.roots().real();

    /** sort the nodes in the ascending order */
    std::sort(nodes.data(), nodes.data() + nodes.size());
    return nodes;
}

/** @brief : compute LGL quadrature weights */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
typename Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::QUADWEIGHTYPE
Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::QuadWeights()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    QUADWEIGHTYPE weights = QUADWEIGHTYPE::Zero();
    const Scalar coeff = static_cast<Scalar>(2) / (PolyOrder * (PolyOrder + 1));
    for(int i = 0; i <= PolyOrder; ++i)
    {
        Scalar LN_xi = Eigen::poly_eval(_Ln.col(PolyOrder), _Nodes[i]);
        weights[i] = coeff / std::pow(LN_xi, 2);
    }
    return weights;
}

/** @brief : compute LGL normalization factors */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
typename Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::NODESVECTYPE
Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::NormFactors()
{
    NODESVECTYPE factors = NODESVECTYPE::Zero();
    for(int k = 0; k < PolyOrder; ++k)
    {
        factors[k] = 2.0 / (2 * k + 1);
    }
    factors[PolyOrder] = 2.0 / (PolyOrder);
    return factors;
}

/** @brief : Compute integrals using LGL-quadrature rule */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
template<class Integrand>
Scalar Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::integrate(const Scalar &t0, const Scalar &tf)
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
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
template<class Function>
typename Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::Projection
Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::project(const Scalar &t0, const Scalar &tf)
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


/** @brief : Compute Legendre basis*/
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
void Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::generate_legendre_basis()
{
    _Ln = LnBASIS::Zero();
    /** the first basis polynomial is L0(x) = 1 */
    _Ln(0,0) = 1;
    /** the second basis polynomial is L1(x) = x */
    _Ln(1,1) = 1;

    /** compute recurrent coefficients */
    NODESVECTYPE a = NODESVECTYPE::Zero();
    NODESVECTYPE c = NODESVECTYPE::Zero();
    NODESVECTYPE x = NODESVECTYPE::Zero(); // p(x) = x
    x[1] = 1;
    for(int n = 0; n <= PolyOrder; ++n)
    {
        a[n] = static_cast<Scalar>(2 * n + 1) / (n + 1);
        c[n] = static_cast<Scalar>(n) / (n + 1);
    }

    /** create polynomial basis */
    for(int n = 1; n <= PolyOrder - 1; ++n)
        _Ln.col(n+1) = a[n] * poly_mul(_Ln.col(n), x) - c[n] * _Ln.col(n-1);
}


/** @brief : Compute Galerkin Tensor */
template<int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         typename Scalar>
void Legendre<PolyOrder, NumSegments, NX, NU, NP, Scalar>::compute_galerkin_tensor()
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
                if(k == PolyOrder)
                    _Galerkin(i, j, k) = (PolyOrder / 2.0) * inner_prod;
                else
                    _Galerkin(i, j, k) = ((2.0 * k + 1) / 2.0) * inner_prod;
                //std::cout << _Galerkin(i, j, k) << " " ;
            }
            //std::cout << "\n";
        }
        //std::cout << "\n \n";
    }

}

#endif // LEGENDRE_HPP
