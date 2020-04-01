#ifndef CHEBYSHEV_MS_HPP
#define CHEBYSHEV_MS_HPP

#include "polymath.h"

template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
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

        _POLY_ORDER = PolyOrder,
        _NUM_SEGMENTS = NumSegments,
        _NUM_COLLOC_PTS_X = NumSegments * PolyOrder + 1,
        _NUM_COLLOC_PTS_U = NumSegments,

        _X_START_IDX = 0,
        _X_END_IDX   = _NUM_COLLOC_PTS_X * _NX,
        _U_START_IDX = _NUM_COLLOC_PTS_X * _NX,
        _U_END_IDX   = _U_START_IDX + _NUM_COLLOC_PTS_U * _NU
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
};

/** @brief constructor */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::MSChebyshev()
{
    /** initialize pseudospectral scheme */
    _Points      = CollocPoints();
    _D           = DiffMatrix();
    _QuadWeights = QuadWeights();
    _ComD        = CompDiffMatrix();

    /** create discretized states and controls */
    _X = casadi::SX::sym("X", (NumSegments * PolyOrder + 1) * NX );
    _U = casadi::SX::sym("U", (NumSegments) * NU );
    _P = casadi::SX::sym("P", NP);
}

/** @brief range */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::range(const uint &first, const uint &last, const uint &step)
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocPoints()
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::DiffMatrix()
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::QuadWeights()
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CompDiffMatrix(const int &DIM)
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateDynamics(casadi::Function &dynamics,
                                                                                      const double &t0, const double &tf)
{
    /** evaluate RHS at the collocation points */
    int DIMX = _X.size1();
    BaseClass F_XU = BaseClass::zeros(DIMX);
    casadi::SXVector tmp;
    double t_scale = (tf - t0) / (2 * NumSegments);

    for(int j = 0; j < NumSegments; j++)
    {
        for (int i = 0; i < (PolyOrder + 1); i++)
        {
            int idx_start_x = j * PolyOrder * NX + (i * NX);
            int idx_end_x   = idx_start_x + NX;

            int idx_start_u = j * NU;
            int idx_end_u = idx_start_u + NU;

            if(NP == 0)
            {
                tmp = dynamics(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                _U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = dynamics(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                _P});
            }

            F_XU(casadi::Slice(idx_start_x, idx_end_x)) = t_scale * tmp[0];
        }
    }

    BaseClass G_XU = BaseClass::mtimes(_ComD, _X) - F_XU;
    return G_XU;
}

/** @brief collocate integral cost */
template<class BaseClass,
         int PolyOrder,
         int NumSegments,
         int NX,
         int NU,
         int NP>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateCost(casadi::Function &MayerTerm,
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
            int j = k * NU;
            int m = 0;
            for (int i = k * NX * PolyOrder; i <= (k + 1) * NX * PolyOrder; i += NX)
            {
                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(i, i + NX)), _U(casadi::Slice(j, j + NU))});
                local_int += _QuadWeights(m) * value[0];
                ++m;
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
         int NP>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateParametricCost(casadi::Function &MayerTerm,
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
            int j = k * NU;
            int m = 0;
            for (int i = k * NX * PolyOrder; i <= (k + 1) * NX * PolyOrder; i += NX)
            {
                value = LagrangeTerm(casadi::SXVector{_X(casadi::Slice(i, i + NX)), _U(casadi::Slice(j, j + NU)), _P});
                local_int += _QuadWeights(m) * value[0];
                ++m;
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
         int NP>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::CollocateFunction(casadi::Function &_Function)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", NX);
    casadi::SX u_test = casadi::SX::sym("u_test", NU);
    casadi::SX p_test = casadi::SX::sym("p_test", NP);
    casadi::SX res_test = _Function(casadi::SXVector{x_test, u_test, p_test})[0];
    bool depends_on_state = casadi::SX::depends_on(res_test, x_test);

    /** evaluate function at the collocation points */
    int NC = _Function.nnz_out();
    int DIMC = depends_on_state ? (NumSegments * PolyOrder + 1) * NC : NumSegments * NC;
    BaseClass F_XU = BaseClass::zeros(DIMC);
    casadi::SXVector tmp;
    int k = 0;

    for(int j = 0; j < NumSegments; j++)
    {
        if(!depends_on_state)
        {
            int idx_start_x = 0;
            int idx_end_x   = idx_start_x + NX;

            int idx_start_u = j * NU;
            int idx_end_u = idx_start_u + NU;

            if(NP == 0)
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                 _P});
            }

            F_XU(casadi::Slice(k, k + NC)) = tmp[0];
            k += NC;
        }
        else
        {

            for (int i = 0; i < (PolyOrder + 1); i++)
            {
                int idx_start_x = j * PolyOrder * NX + (i * NX);
                int idx_end_x   = idx_start_x + NX;

                int idx_start_u = j * NU;
                int idx_end_u = idx_start_u + NU;

                if(NP == 0)
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                     _P});
                }

                if( ((i % PolyOrder) != 0) || (i == 0))
                {
                    F_XU(casadi::Slice(k, k + NC)) = tmp[0];
                    k += NC;
                }

                // hack add last point
                if((j == (NumSegments - 1)) && (i == PolyOrder))
                    F_XU(casadi::Slice(k, k + NC)) = tmp[0];
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
         int NP>
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP>::DifferentiateFunction(casadi::Function &_Function, const int order)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", NX);
    casadi::SX u_test = casadi::SX::sym("u_test", NU);
    casadi::SX p_test = casadi::SX::sym("p_test", NP);
    casadi::SX res_test = _Function(casadi::SXVector{x_test, u_test, p_test})[0];
    bool depends_on_state = casadi::SX::depends_on(res_test, x_test);

    /** evaluate function at the collocation points */
    int NC = _Function.nnz_out();
    int DIMC = depends_on_state ? (NumSegments * PolyOrder + 1) * NC : NumSegments * NC;
    BaseClass F_XU = BaseClass::zeros(DIMC);
    casadi::SXVector tmp;
    int k = 0;

    for(int j = 0; j < NumSegments; j++)
    {
        if(!depends_on_state)
        {
            int idx_start_x = 0;
            int idx_end_x   = idx_start_x + NX;

            int idx_start_u = j * NU;
            int idx_end_u = idx_start_u + NU;

            if(NP == 0)
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                 _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                 _P});
            }

            F_XU(casadi::Slice(k, k + NC)) = tmp[0];
            k += NC;
        }
        else
        {

            for (int i = 0; i < (PolyOrder + 1); i++)
            {
                int idx_start_x = j * PolyOrder * NX + (i * NX);
                int idx_end_x   = idx_start_x + NX;

                int idx_start_u = j * NU;
                int idx_end_u = idx_start_u + NU;

                if(NP == 0)
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = _Function(casadi::SXVector{_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                     _U(casadi::Slice(idx_start_u, idx_end_u)),
                                                     _P});
                }

                if( ((i % PolyOrder) != 0) || (i == 0))
                {
                    F_XU(casadi::Slice(k, k + NC)) = tmp[0];
                    k += NC;
                }

                // hack add last point
                if((j == (NumSegments - 1)) && (i == PolyOrder))
                    F_XU(casadi::Slice(k, k + NC)) = tmp[0];
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
        /** @badcode : write a custom assert */
        assert(order == 1 && "Derivative of this order does not exist for the piecewise constant control parametrisation");
        BaseClass Derivative = BaseClass::zeros(_NUM_COLLOC_PTS_U * _NU, 1);
        for(unsigned i = 0; i < (_NUM_COLLOC_PTS_U - 1) * _NU; i += NU)
            Derivative(casadi::Slice(i, i + _NU)) = _U(casadi::Slice(i + _NU, (i + 1) * _NU)) - _U(casadi::Slice(i, i + _NU));

        return Derivative;
    }
}

#endif // CHEBYSHEV_MS_HPP
