// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    virtual ~MSChebyshev() = default;

    BaseClass D() const {return m_D;}
    BaseClass CompD() const {return m_ComD;}
    BaseClass CPoints() const {return m_Points;}
    BaseClass QWeights() const {return m_QuadWeights;}

    BaseClass VarX() const {return m_X;}
    BaseClass VarU() const {return m_U;}
    BaseClass VarP() const {return m_P;}
    BaseClass VarD() const {return m_DT;}

    BaseClass CollocateDynamics(casadi::Function &dynamics, const double &t0, const double &tf);
    BaseClass CollocateCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                            const double &t0, const double &tf);
    BaseClass CollocateParametricCost(casadi::Function &MayerTerm, casadi::Function &LagrangeTerm,
                                      const double &t0, const double &tf);
    BaseClass CollocateFunction(casadi::Function &a_Function);
    BaseClass DifferentiateFunction(casadi::Function &a_Function, const int order = 1);

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
    BaseClass m_D;
    /** Composite diff matrix */
    BaseClass m_ComD;
    /** Collocation points */
    BaseClass m_Points;
    /** Quadrature weights */
    BaseClass m_QuadWeights;

    /** helper functions */
    BaseClass range(const uint &first, const uint &last, const uint &step);

    /** state in terms of Chebyshev coefficients */
    BaseClass m_X;
    /** control in terms of Chebyshev coefficients */
    BaseClass m_U;
    /** vector of parameters */
    BaseClass m_P;
    /** vector of static parameters */
    BaseClass m_DT;
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
    m_Points      = CollocPoints();
    m_D           = DiffMatrix();
    m_QuadWeights = QuadWeights();
    m_ComD        = CompDiffMatrix();

    /** create discretized states and controls */
    m_X  = casadi::SX::sym("X", _NUM_COLLOC_PTS_X * _NX );
    m_U  = casadi::SX::sym("U", _NUM_COLLOC_PTS_U * _NU );
    m_P  = casadi::SX::sym("P", _NP);
    m_DT = casadi::SX::sym("D", _ND);
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
        CompDiff = BaseClass::diagcat(casadi::SXVector(NumSegments, D));
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

            if((_NP == 0) && (_ND == 0))
                tmp = dynamics(casadi::SXVector{m_X(casadi::Slice(start, end)), m_U(casadi::Slice(j*_NU, j*_NU + _NU)) });
            else
                tmp = dynamics(casadi::SXVector{m_X(casadi::Slice(start, end)), m_U(casadi::Slice(j*_NU, j*_NU + _NU)), m_P, m_DT});

            F_XU(casadi::Slice(start, end)) = t_scale * tmp[0];
        }
    }

    BaseClass G_XU = BaseClass::mtimes(m_ComD, m_X) - F_XU;

    /** add more equality constraints */
    for(int j = 0; j < (_NUM_SEGMENTS - 1); j++)
    {
        int start = (j * (_POLY_ORDER + 1) * _NX) + (_POLY_ORDER * _NX);
        int end   = start + _NX;

        G_XU = BaseClass::vertcat({G_XU, m_X(casadi::Slice(start, end)) - m_X(casadi::Slice(end, end + _NX))});
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
        value = MayerTerm(casadi::SXVector{m_X(casadi::Slice(0, NX))});
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

                value = LagrangeTerm(casadi::SXVector{m_X(casadi::Slice(start, end)), m_U(casadi::Slice(j*_NU, j*_NU + _NU))});
                local_int += m_QuadWeights(i) * value[0];
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
        value = MayerTerm(casadi::SXVector{m_X(casadi::Slice(0, NX)), m_P, m_DT});
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

                value = LagrangeTerm(casadi::SXVector{m_X(casadi::Slice(start, end)), m_U(casadi::Slice(j*_NU, j*_NU + _NU)), m_P, m_DT});
                local_int += m_QuadWeights(i) * value[0];
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::CollocateFunction(casadi::Function &a_Function)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", _NX);
    casadi::SX u_test = casadi::SX::sym("u_test", _NU);
    casadi::SX p_test = casadi::SX::sym("p_test", _NP);
    casadi::SX d_test = casadi::SX::sym("d_test", _ND);
    casadi::SX res_test;
    if((_NP == 0) && (_ND == 0))
        res_test = a_Function(casadi::SXVector{x_test, u_test})[0];
    else
        res_test = a_Function(casadi::SXVector{x_test, u_test, p_test, d_test})[0];

    /** evaluate function at the collocation points */
    bool depends_on_state   = casadi::SX::depends_on(res_test, x_test);
    int NC = a_Function.nnz_out();
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

            if((_NP == 0) && (_ND == 0))
            {
                tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(0, NX)),
                                                 m_U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(0, NX)),
                                                  m_U(casadi::Slice(idx_start_u, idx_end_u)),
                                                  m_P, m_DT});
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

                if((_NP == 0) && (_ND == 0))
                {
                    tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                      m_U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                      m_U(casadi::Slice(idx_start_u, idx_end_u)),
                                                      m_P, m_DT});
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
BaseClass MSChebyshev<BaseClass, PolyOrder, NumSegments, NX, NU, NP, ND>::DifferentiateFunction(casadi::Function &a_Function, const int order)
{
    /** check if the function depends on X */
    casadi::SX x_test = casadi::SX::sym("x_test", _NX);
    casadi::SX u_test = casadi::SX::sym("u_test", _NU);
    casadi::SX p_test = casadi::SX::sym("p_test", _NP);
    casadi::SX d_test = casadi::SX::sym("d_test", _ND);
    casadi::SX res_test;
    if((_NP == 0) && (_ND == 0))
        res_test = a_Function(casadi::SXVector{x_test, u_test})[0];
    else
        res_test = a_Function(casadi::SXVector{x_test, u_test, p_test, d_test})[0];

    /** evaluate function at the collocation points */
    bool depends_on_state   = casadi::SX::depends_on(res_test, x_test);
    int NC = a_Function.nnz_out();
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

            if((_NP == 0) && (_ND == 0))
            {
                tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(0, NX)),
                                                  m_U(casadi::Slice(idx_start_u, idx_end_u)) });
            }
            else
            {
                tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(0, NX)),
                                                  m_U(casadi::Slice(idx_start_u, idx_end_u)),
                                                  m_P, m_DT});
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

                if((_NP == 0) && (_ND == 0))
                {
                    tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                      m_U(casadi::Slice(idx_start_u, idx_end_u)) });
                }
                else
                {
                    tmp = a_Function(casadi::SXVector{m_X(casadi::Slice(idx_start_x, idx_end_x)),
                                                      m_U(casadi::Slice(idx_start_u, idx_end_u)),
                                                      m_P, m_DT});
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
            Derivative(casadi::Slice(i, i + _NU)) = m_U(casadi::Slice(i + _NU, i + 2*_NU)) - m_U(casadi::Slice(i, i + _NU));

        return Derivative;
    }
}

#endif // CHEBYSHEV_MS_HPP
