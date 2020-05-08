#ifndef CHEBYSHEV_MS_HPP
#define CHEBYSHEV_MS_HPP

#include "polymath.h"

template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND = 0>
class MSChebyshev
{
public:
    /** constructor */
    MSChebyshev();
    virtual ~MSChebyshev(){}

    BaseClass D(){return _D;}
    BaseClass CompD(){return _ComD;}
    BaseClass CPoints(){return _Points;}
    BaseClass QWeights(){return _QuadWeights;}

    BaseClass VarX(){return _X;}
    BaseClass VarU(){return _U;}
    BaseClass VarP(){return _P;}
    BaseClass VarD(){return _DT;}

    BaseClass CollocateDynamics(casadi::Function &dynamics, const double &t0, const double &tf);
    BaseClass CollocateCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                            const double &t0, const double &tf);
    BaseClass CollocateParametricCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                                      const double &t0, const double &tf);
    BaseClass CollocateFunction(casadi::Function &_Function);
    BaseClass DifferentiateFunction(casadi::Function &_Function, const int order = 1);

    double _t0, _tf;

    enum
    {
        _NX = NX,
        _NU = NU,
        _NP = NP,
        _ND = ND,

        _POLY_ORDER       = PolyOrder,
        _NUM_SEGMENTS     = NumSegments,
        _NUM_COLLOC_PTS_X = _NUM_SEGMENTS * (_POLY_ORDER + 1),
        _NUM_COLLOC_PTS_U = _NUM_SEGMENTS,

        _X_START_IDX = 0,
        _X_END_IDX   = _NUM_COLLOC_PTS_X * _NX,
        _U_START_IDX = _X_END_IDX,
        _U_END_IDX   = _U_START_IDX + _NUM_COLLOC_PTS_U * _NU,
        _P_START_IDX = _U_END_IDX,
        _P_END_IDX   = _P_START_IDX + _ND
    };

private:

    /** generate Differentiation matrix */
    BaseClass DiffMatrix();
    /** generate Chebyshev collocation points */
    BaseClass CollocPoints();
    /** generate Clenshaw-Curtis quadrature weights */
    BaseClass QuadWeights();
    /** generate Composite Differentiation matrix for vector of dimension DIM (=NX) */
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
    /** vector of static parameters */
    BaseClass _DT;
};

/** @brief constructor */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::MSChebyshev()
{
    /** initialize pseudospectral scheme */
    _Points      = CollocPoints();
    _D           = DiffMatrix();
    _QuadWeights = QuadWeights();
    _ComD        = CompDiffMatrix();

    /** create discretized states and controls */
    _X  = casadi::SX::sym("X", _NUM_COLLOC_PTS_X * _NX );
    _U  = casadi::SX::sym("U", _NUM_COLLOC_PTS_U * _NU );
    _P  = casadi::SX::sym("P", _NP);
    _DT = casadi::SX::sym("D", _ND);
}

/** @brief range */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::range(const uint &first, const uint &last, const uint &step)
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
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocPoints()
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
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::DiffMatrix()
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
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::QuadWeights()
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
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CompDiffMatrix(const int &DIM)
{
    int comp_rows = _NUM_COLLOC_PTS_X;
    int comp_cols = _NUM_COLLOC_PTS_X;

    BaseClass CompDiff = BaseClass::zeros(comp_rows, comp_cols);
    BaseClass D        = DiffMatrix();
    BaseClass E        = BaseClass::eye(DIM);

    if(_NUM_SEGMENTS < 2)
    {
        CompDiff = D;
    }
    else
    {
        /** construct a simple diagonal matrix */
        CompDiff = BaseClass::diagcat(casadi::SXVector(3, D));
    }

    return BaseClass::kron(CompDiff, E);
}

/** @brief collocate differential constraints */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocateDynamics(casadi::Function &dynamics,
                                                                                      const double &t0, const double &tf)
{
    /** evaluate RHS at the collocation points */
    BaseClass F_XU = BaseClass::zeros(_NUM_COLLOC_PTS_X * _NX);
    casadi::SXVector tmp;
    double t_scale = (tf - t0) / (2 * _NUM_SEGMENTS);

    for(int j = 0; j < _NUM_SEGMENTS; j++)
    {
        for (int i = 0; i < (_POLY_ORDER + 1); i++)
        {
            int start = j * (_POLY_ORDER + 1) * _NX + (i * _NX);
            int end   = start + _NX;

            if((_NP == 0) and (_ND == 0))
                tmp = dynamics(casadi::SXVector{_X(casadi::Slice(start, end)), _U(casadi::Slice(j*_NU, j*_NU + _NU)) });
            else
                tmp = dynamics(casadi::SXVector{_X(casadi::Slice(start, end)), _U(casadi::Slice(j*_NU, j*_NU + _NU)), _P, _DT});

            F_XU(casadi::Slice(start, end)) = t_scale * tmp[0];
        }
    }

    BaseClass G_XU = BaseClass::mtimes(_ComD, _X) - F_XU;

    /** add more equality constraints */
    for(int j = 0; j < (_NUM_SEGMENTS - 1); j++)
    {
        int start = (j * (_POLY_ORDER + 1) * _NX) + (_POLY_ORDER * _NX);
        int end   = start + _NX;

        G_XU = BaseClass::vertcat({G_XU, _X(casadi::Slice(start, end)) - _X(casadi::Slice(end, end + _NX))});
    }

    return G_XU;
}

/** @brief collocate integral cost */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocateCost(casadi::Function &MayerTerm,
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
        double t_scale = (tf - t0) / (2 * _NUM_SEGMENTS);
        for (int j = 0; j < _NUM_SEGMENTS; ++j)
        {
            BaseClass local_int = {0};
            for (int i = 0; i < (_POLY_ORDER + 1); ++i)
            {
                int start = j * (_POLY_ORDER + 1) * _NX + (i * _NX);
                int end   = start + _NX;

                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(start, end)), _U(casadi::Slice(j*_NU, j*_NU + _NU))});
                local_int += _QuadWeights(i) * value[0];
            }

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
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocateParametricCost(casadi::Function &MayerTerm,
                                                                                            casadi::Function &LagrangeTerm,
                                                                                            const double &t0, const double &tf)
{
    casadi::SXVector value;
    BaseClass Mayer    = {0};
    BaseClass Lagrange = {0};

    /** collocate Mayer term */
    if(!MayerTerm.is_null())
    {
        value = MayerTerm(casadi::SXVector{_X(casadi::Slice(0, NX)), _P, _DT});
        Mayer = value[0];
    }

    /** collocate Lagrange term */
    if(!LagrangeTerm.is_null())
    {
        /** for each segment */
        double t_scale = (tf - t0) / (2 * NumSegments);
        for (int j = 0; j < _NUM_SEGMENTS; ++j)
        {
            BaseClass local_int = {0};
            for (int i = 0; i < (_POLY_ORDER + 1); i++)
            {
                int start = j * (_POLY_ORDER + 1) * _NX + (i * _NX);
                int end   = start + _NX;

                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(start, end)), _U(casadi::Slice(j*_NU, j*_NU + _NU)), _P, _DT});
                local_int += _QuadWeights(i) * value[0];
            }

            Lagrange += t_scale * local_int;
        }
    }

    return Mayer + Lagrange;
}


/** Collocate an arbitrary function */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocateFunction(casadi::Function &_Function)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", _NX);
    casadi::SX u_test = casadi::SX::sym("u_test", _NU);
    casadi::SX p_test = casadi::SX::sym("p_test", _NP);
    casadi::SX d_test = casadi::SX::sym("d_test", _ND);
    casadi::SX res_test;
    if((_NP == 0) and (_ND == 0))
        res_test = _Function(casadi::SXVector{x_test, u_test})[0];
    else
        res_test = _Function(casadi::SXVector{x_test, u_test, p_test, d_test})[0];

    bool depends_on_state   = casadi::SX::depends_on(res_test, x_test);
    bool depends_on_control = casadi::SX::depends_on(res_test, u_test);

    /** evaluate function at the collocation points */
    int NC = _Function.nnz_out();
    int DIMC = depends_on_state ? _NUM_COLLOC_PTS_X * NC : _NUM_COLLOC_PTS_U * NC;
    BaseClass F_XU = BaseClass::zeros(DIMC);
    casadi::SXVector tmp;
    int k = 0;

    for(int j = 0; j < _NUM_SEGMENTS; j++)
    {
        if(!depends_on_state)
        {
            int idx_start_u = j * _NU;
            int idx_end_u = idx_start_u + _NU;

            if((_NP == 0) and (_ND == 0))
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(0, NX)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(0, NX)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                 _P, _DT});
            }

            F_XU(casadi::Slice(k, k + NC)) = tmp[0];
            k += NC;
        }
        else
        {
            for (int i = 0; i < (_POLY_ORDER + 1); i++)
            {
                int idx_start_x = j * (_POLY_ORDER + 1) * _NX + (i * _NX);
                int idx_end_x   = idx_start_x + _NX;

                int idx_start_u = j * _NU;
                int idx_end_u = idx_start_u + _NU;

                if((_NP == 0) and (_ND == 0))
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                     _P, _DT});
                }

                F_XU(casadi::Slice(k, k + NC)) = tmp[0];
                k += NC;
            }
        }
    }

    return F_XU;
}


/** Collocate an arbitrary function */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP,
         int ND>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::DifferentiateFunction(casadi::Function &_Function, const int order)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", _NX);
    casadi::SX u_test = casadi::SX::sym("u_test", _NU);
    casadi::SX p_test = casadi::SX::sym("p_test", _NP);
    casadi::SX d_test = casadi::SX::sym("d_test", _ND);
    casadi::SX res_test;
    if((_NP == 0) and (_ND == 0))
        res_test = _Function(casadi::SXVector{x_test, u_test})[0];
    else
        res_test = _Function(casadi::SXVector{x_test, u_test, p_test, d_test})[0];

    bool depends_on_state   = casadi::SX::depends_on(res_test, x_test);
    bool depends_on_control = casadi::SX::depends_on(res_test, u_test);

    /** evaluate function at the collocation points */
    int NC = _Function.nnz_out();
    int DIMC = depends_on_state ? _NUM_COLLOC_PTS_X * NC : _NUM_COLLOC_PTS_U * NC;
    BaseClass F_XU = BaseClass::zeros(DIMC);
    casadi::SXVector tmp;
    int k = 0;

    for(int j = 0; j < _NUM_SEGMENTS; j++)
    {
        if(!depends_on_state)
        {
            int idx_start_u = j * _NU;
            int idx_end_u = idx_start_u + _NU;

            if((_NP == 0) and (_ND == 0))
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(0, NX)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(0, NX)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                 _P, _DT});
            }

            F_XU(casadi::Slice(k, k + NC)) = tmp[0];
            k += NC;
        }
        else
        {
            for (int i = 0; i < (_POLY_ORDER + 1); i++)
            {
                int idx_start_x = j * (_POLY_ORDER + 1) * _NX + (i * _NX);
                int idx_end_x   = idx_start_x + _NX;

                int idx_start_u = j * _NU;
                int idx_end_u = idx_start_u + _NU;

                if((_NP == 0) and (_ND == 0))
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                     _P, _DT});
                }

                F_XU(casadi::Slice(k, k + NC)) = tmp[0];
                k += NC;
            }
        }
    }

    if(depends_on_state)
    {
        BaseClass Diff = CompDiffMatrix(NC);
        BaseClass Derivative = F_XU;
        for(unsigned i = 0; i < order; ++i)
            Derivative = BaseClass::mtimes(Diff, Derivative);

        return Derivative;
    }
    else
    {
        /** use ZOH assumption to estimate gradient */
        casadi_assert(order == 1, "Derivative of this order does not exist for the piecewise constant control parametrisation");
        BaseClass Derivative = BaseClass::zeros((_NUM_COLLOC_PTS_U - 1) * _NU, 1);
        for(int i = 0; i < (_NUM_COLLOC_PTS_U - 1) * _NU; i += _NU)
            Derivative(casadi::Slice(i, i + _NU)) = _U(casadi::Slice(i + _NU, i + 2*_NU)) - _U(casadi::Slice(i, i + _NU));

        return Derivative;
    }
}

#endif // CHEBYSHEV_MS_HPP
