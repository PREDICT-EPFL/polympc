#ifndef CHEBYSHEV_HPP
#define CHEBYSHEV_HPP

#include "polymath.h"

template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
class Chebyshev
{
public:
    /** constructor */
    Chebyshev();
    virtual ~Chebyshev(){}

    BaseClass D(){return _D;}
    BaseClass CompD(){return _ComD;}
    BaseClass CPoints(){return _Points;}
    BaseClass QWeights(){return _QuadWeights;}

    BaseClass VarX(){return _X;}
    BaseClass VarU(){return _U;}
    BaseClass VarP(){return _P;}

    BaseClass CollocateDynamics(casadi::Function &dynamics, const double &t0, const double &tf);
    BaseClass CollocateCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                            const double &t0, const double &tf);
    BaseClass CollocateParametricCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                                      const double &t0, const double &tf);
    BaseClass CollocateIdCost(casadi::Function &IdCost, casadi::DM data, const double &t0, const double &tf);
    BaseClass CollocateFunction(casadi::Function &_Function);
    BaseClass DifferentiateFunction(casadi::Function &_Function, const int order = 1);

    typedef std::function<BaseClass(BaseClass, BaseClass, BaseClass)> functor;
    /** right hand side function of the ODE */
    functor _ode;
    double _t0, _tf;
    functor CollocateDynamics2(const functor &dynamics, const double &t0, const double &tf);
    BaseClass collocate_dynamics(const BaseClass &X, const BaseClass &U, const BaseClass &P);

    enum
    {
        _NX = NX,
        _NU = NU,
        _NP = NP,

        _POLY_ORDER = PolyOrder,
        _NUM_SEGMENTS = NumSegments,
        _NUM_COLLOC_PTS_X = NumSegments * PolyOrder + 1,
        _NUM_COLLOC_PTS_U = NumSegments * PolyOrder + 1,

        _X_START_IDX = 0,
        _X_END_IDX = _NUM_COLLOC_PTS_X * NX,
        _U_START_IDX = _NUM_COLLOC_PTS_X * NX,
        _U_END_IDX = _U_START_IDX + _NUM_COLLOC_PTS_U * NU
    };

private:

    /** generate Differentiation matrix */
    BaseClass DiffMatrix();
    /** generate Chebyshev collocation points */
    BaseClass CollocPoints();
    /** generate Clenshaw-Curtis quadrature weights */
    BaseClass QuadWeights();
    /** generate Composite Differentiation matrix for vector of dimension DIM */
    BaseClass CompDiffMatrix(const int &DIM = NX);

    /** Diff matrix */
    BaseClass _D;
    /** Composite diff matrix */
    BaseClass _ComD;
    /** Collocation points */
    BaseClass _Points;
    /** Quadrature weights */
    BaseClass _QuadWeights;

    /** helper functions */
    BaseClass range(const uint &first, const uint &last, const uint &step);

    /** state in terms of Chebyshev coefficients */
    BaseClass _X;
    /** control in terms of Chebyshev coefficients */
    BaseClass _U;
    /** vector of parameters */
    BaseClass _P;
};

/** @brief constructor */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::Chebyshev()
{
    /** initialize pseudopsectral scheme */
    _Points      = CollocPoints();
    _D           = DiffMatrix();
    _QuadWeights = QuadWeights();
    _ComD        = CompDiffMatrix();

    /** create discretized states and controls */
    _X = casadi::SX::sym("X", (NumSegments * PolyOrder + 1) * NX );
    _U = casadi::SX::sym("U", (NumSegments * PolyOrder + 1) * NU );
    _P = casadi::SX::sym("P", NP);
}

/** @brief range */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::range(const uint &first, const uint &last, const uint &step)
{
    int numel = std::floor((last - first) / step);
    BaseClass _range;
    _range.reserve(numel);
    int idx = 0;
    for (uint value = first; value <= last; ++value)
    {
        _range(idx) = 0;
    }
    return _range;
}

/** @brief compute collocation points */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocPoints()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    auto grid_int = polymath::range<double>(0, PolyOrder);
    /** cast grid to Casadi type */
    BaseClass grid(grid_int);
    BaseClass X = cos(grid * (M_PI / PolyOrder));
    return X;
}

/** @brief compute differentiation matrix / ref {L. Trefethen "Spectral Methods in Matlab"}*/
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::DiffMatrix()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    auto grid_int = polymath::range<double>(0, PolyOrder);
    /** cast grid to Casadi type */
    BaseClass grid(grid_int);
    BaseClass cpoints = cos(grid * (M_PI / PolyOrder));

    /** Diff Matrix */
    BaseClass c = BaseClass::vertcat({2, BaseClass::ones(PolyOrder - 1, 1), 2});
    c = BaseClass::mtimes(BaseClass::diag( pow(-1, grid)), c);
    BaseClass XM = BaseClass::repmat(cpoints, 1, PolyOrder + 1);
    BaseClass dX = XM - XM.T();
    BaseClass Dn  = BaseClass::mtimes(c, (1 / c).T() ) / (dX + (BaseClass::eye(PolyOrder + 1)));      /** off-diagonal entries */

    return Dn - BaseClass::diag( BaseClass::sum1(Dn.T() ));               /**  diagonal entries */
}

/** @brief compute weights for Clenshaw-Curtis quadrature / ref {L. Trefethen "Spectral Methods in Matlab"}*/
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::QuadWeights()
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    auto grid_int = polymath::range<double>(0, PolyOrder);
    /** cast grid to Casadi type */
    BaseClass theta(grid_int);
    theta = theta * (M_PI / PolyOrder);

    BaseClass w = BaseClass::zeros(1, PolyOrder + 1);
    BaseClass v = BaseClass::ones(PolyOrder - 1, 1);

    if ( PolyOrder % 2 == 0 )
    {
        w(0)         = 1 / (pow(PolyOrder, 2) - 1);
        w(PolyOrder) = w(0);

        for(int k = 1; k <= PolyOrder / 2 - 1; ++k)
        {
            v = v - 2 * cos(2 * k * theta(casadi::Slice(1, PolyOrder))) / (4 * pow(k, 2) - 1);
        }
        v = v - cos( PolyOrder * theta(casadi::Slice(1, PolyOrder))) / (pow(PolyOrder, 2) - 1);
    }
    else
    {
        w(0) = 1 / std::pow(PolyOrder, 2);
        w(PolyOrder) = w(0);
        for (int k = 1; k <= (PolyOrder - 1) / 2; ++k)
        {
            v = v - 2 * cos(2 * k * theta(casadi::Slice(1, PolyOrder))) / (4 * pow(k, 2) - 1);
        }
    }
    w(casadi::Slice(1, PolyOrder)) =  2 * v / PolyOrder;
    return w;
}

/** @brief compute composite differentiation matrix */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CompDiffMatrix(const int &DIM)
{
    int comp_rows = NumSegments * PolyOrder + 1;
    int comp_cols = NumSegments * PolyOrder + 1;

    BaseClass CompDiff = BaseClass::zeros(comp_rows, comp_cols);
    BaseClass D        = DiffMatrix();
    BaseClass D0       = D;
    BaseClass E        = BaseClass::eye(DIM);

    if(NumSegments < 2)
    {
        CompDiff = D0;
    }
    else
    {
        /** insert first matrix */
        CompDiff(casadi::Slice(CompDiff.size1() - D0.size1(), CompDiff.size1()),
                 casadi::Slice(CompDiff.size2() - D0.size2(), CompDiff.size2())) = D0;
        /** fill in diagonal terms */
        for(int k = 0; k < (NumSegments - 1) * PolyOrder; k += PolyOrder)
        {
            CompDiff(casadi::Slice(k, k + PolyOrder), casadi::Slice(k, k + PolyOrder + 1)) =
                    D(casadi::Slice(0, PolyOrder), casadi::Slice(0, D.size2()));
        }
    }

    return BaseClass::kron(CompDiff, E);
}


/** @brief collocate differential constraints */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateDynamics(casadi::Function &dynamics,
                                                                                      const double &t0, const double &tf)
{
    /** evaluate RHS at the collocation points */
    int DIMX = _X.size1();
    BaseClass F_XU = BaseClass::zeros(DIMX);
    casadi::SXVector tmp;
    int j = 0;
    double t_scale = (tf - t0) / (2 * NumSegments);

    for (int i = 0; i <= DIMX - NX; i += NX)
    {
        if(NP == 0)
        {
            tmp = dynamics(casadi::SXVector{_X(casadi::Slice(i, i + NX)),
                                            _U(casadi::Slice(j, j + NU)) });
        }
        else
        {
            tmp = dynamics(casadi::SXVector{_X(casadi::Slice(i, i + NX)),
                                            _U(casadi::Slice(j, j + NU)),
                                            _P});
        }

            F_XU(casadi::Slice(i, i + NX)) = t_scale * tmp[0];
            j += NU;
    }

    BaseClass G_XU = BaseClass::mtimes(_ComD, _X) - F_XU;
    return G_XU;
}

/** @brief collocate performance index */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateCost(casadi::Function &MayerTerm,
                                                                                  casadi::Function &LagrangeTerm,
                                                                                  const double &t0, const double &tf)
{
    casadi::SXVector value;
    BaseClass Mayer    = {0};
    BaseClass Lagrange = {0};

    /** collocate Mayer term */
    if(!MayerTerm.is_null())
    {
        value = MayerTerm(casadi::SXVector{_X(casadi::Slice(0, NX))});
        Mayer = value[0];
    }

    /** collocate Lagrange term */
    if(!LagrangeTerm.is_null())
    {
        /** for each segment */
        double t_scale = (tf - t0) / (2 * NumSegments);
        for (int k = 0; k < NumSegments; ++k)
        {
            BaseClass local_int = {0};
            int j = k * NU * PolyOrder;
            int m = 0;
            for (int i = k * NX * PolyOrder; i <= (k + 1) * NX * PolyOrder; i += NX)
            {
                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(i, i + NX)), _U(casadi::Slice(j, j + NU))});
                local_int += _QuadWeights(m) * value[0];
                j += NU;
                ++m;
            }
            //std::cout << "Local Integral : [ " << k << " ] : " << local_int << "\n";
            Lagrange += t_scale * local_int;
        }
    }

    return Mayer + Lagrange;
}

/** @brief collocate parametric performance index */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateParametricCost(casadi::Function &MayerTerm,
                                                                                            casadi::Function &LagrangeTerm,
                                                                                            const double &t0, const double &tf)
{
    casadi::SXVector value;
    BaseClass Mayer    = {0};
    BaseClass Lagrange = {0};

    /** collocate Mayer term */
    if(!MayerTerm.is_null())
    {
        value = MayerTerm(casadi::SXVector{_X(casadi::Slice(0, NX)),
                                           _P});
        Mayer = value[0];
    }

    /** collocate Lagrange term */
    if(!LagrangeTerm.is_null())
    {
        /** for each segment */
        double t_scale = (tf - t0) / (2 * NumSegments);
        for (int k = 0; k < NumSegments; ++k)
        {
            BaseClass local_int = {0};
            int j = k * NU * PolyOrder;
            int m = 0;
            for (int i = k * NX * PolyOrder; i <= (k + 1) * NX * PolyOrder; i += NX)
            {
                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(i, i + NX)), _U(casadi::Slice(j, j + NU)), _P});
                local_int += _QuadWeights(m) * value[0];
                j += NU;
                ++m;
            }
            //std::cout << "Local Integral : [ " << k << " ] : " << local_int << "\n";
            Lagrange += t_scale * local_int;
        }
    }

    return Mayer + Lagrange;
}

/** Collocate Identification cost function */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateIdCost(casadi::Function &IdCost,
                                                                                    casadi::DM data,
                                                                                    const double &t0, const double &tf)
{
    if ( (data.size1() != NX) || (data.size2() != (NumSegments * PolyOrder + 1)) )
    {
        std::cout << "CollocateIdCost: Inconsistent data size! \n";
        return casadi::SX(0);
    }

    /** collocate Integral cost */
    BaseClass IntCost = {0};
    casadi::SXVector value;
    casadi::DM _data = casadi::DM::vec(data);
    int size_x = _X.size1();

    if(!IdCost.is_null())
    {
        /** for each segment */
        double t_scale = (tf - t0) / (2 * NumSegments);
        for (int k = 0; k < NumSegments; ++k)
        {
            BaseClass local_int = {0};
            int m = 0;
            for (int i = k * NX * PolyOrder; i <= (k + 1) * NX * PolyOrder; i += NX)
            {
                int idx = size_x - i;
                value = IdCost(casadi::SXVector{_X(casadi::Slice(i, i + NX)), _data(casadi::Slice(idx, idx - NX)) });

                local_int += _QuadWeights[m] * value[0];
                ++m;
            }
            IntCost += t_scale * local_int;
        }
    }
    return IntCost;
}

/** set up collocation function */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
std::function<BaseClass(BaseClass, BaseClass, BaseClass)> Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>
                                                                                       ::CollocateDynamics2(const functor &dynamics,
                                                                                       const double &t0, const double &tf)
{
    _ode = dynamics;
    _t0 = t0;
    _tf = tf;

    return std::bind(&Chebyshev::collocate_dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}


template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::collocate_dynamics(const BaseClass &X,
                                                                                       const BaseClass &U, const BaseClass &P)
{
    /** evaluate RHS at the collocation points */
    int DIMX = X.size1();
    BaseClass F_XU = BaseClass::zeros(DIMX);
    BaseClass tmp;
    int j = 0;
    double t_scale = (_tf - _t0) / (2 * NumSegments);

    for (int i = 0; i <= DIMX - NX; i += NX)
    {
            tmp = _ode( X(casadi::Slice(i, i + NX)), U(casadi::Slice(j, j + NU)), P );

            F_XU(casadi::Slice(i, i + NX)) = t_scale * tmp;
            j += NU;
    }

    BaseClass G_XU = BaseClass::mtimes(_ComD, _X) - F_XU;
    return G_XU;
}

/** experimental */

template<class BaseClass>
BaseClass lox(std::function<BaseClass(BaseClass, BaseClass, BaseClass)> my_func, const BaseClass &X,
              const BaseClass &U, const BaseClass &P)
{
    return my_func(X, U, P);
}


/** Collocate an arbitrary function */
template<class BaseClass,
        int PolyOrder,
        int NumSegments,
        int NX,
        int NU,
        int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateFunction(casadi::Function &_Function)
{
    /** evaluate function at the collocation points */
    const int n_f_out = _Function.nnz_out();
    casadi::SXVector tmp;
    const int n_colloc = (NumSegments * PolyOrder + 1);
    BaseClass f_colloc = BaseClass::zeros(n_colloc * n_f_out);
    int i_x, i_u, i_f;
    for(int i = 0; i < n_colloc; i++)
    {
        i_x = i * NX;
        i_u = i * NU;
        i_f = i * n_f_out;
        if(_NP == 0)
        {
            tmp = _Function(casadi::SXVector{_X(casadi::Slice(i_x, i_x + NX)),
                                             _U(casadi::Slice(i_u, i_u + NU))});
        }
        else
        {
            tmp = _Function(casadi::SXVector{_X(casadi::Slice(i_x, i_x + NX)),
                                             _U(casadi::Slice(i_u, i_u + NU)),
                                             _P});
        }

        f_colloc(casadi::Slice(i_f, i_f + n_f_out)) = tmp[0];
    }
    return f_colloc;
}


/** Differentiate an arbitrary function */
template<class BaseClass,
        int PolyOrder,
        int NumSegments,
        int NX,
        int NU,
        int NP>
BaseClass Chebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::DifferentiateFunction(casadi::Function &_Function, const int order)
{
    /** evaluate function at the collocation points */
    int n_f_out = _Function.nnz_out();
    BaseClass Derivative = CollocateFunction(_Function);
    BaseClass Diff = CompDiffMatrix(n_f_out);

    for(uint i = 0; i < order; ++i)
        Derivative = BaseClass::mtimes(Diff, Derivative);

    return Derivative;
}

#endif // CHEBYSHEV_HPP
