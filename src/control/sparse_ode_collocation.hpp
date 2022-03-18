// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_ODE_COLLOCATION_HPP
#define SPARSE_ODE_COLLOCATION_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/KroneckerProduct"
#include "unsupported/Eigen/AutoDiff"

#include <iostream>


namespace polympc {

template <typename Dynamics, typename Polynomial, int NumSegments = 1>
class sparse_ode_collocation
{
public:

    using Scalar = typename Dynamics::Scalar;
    using diff_mat_t = typename Polynomial::diff_mat_t;
    using weights_t  = typename Polynomial::q_weights_t;
    using nodes_t    = typename Polynomial::nodes_t;

    enum
    {
        NX = Dynamics::State::RowsAtCompileTime,
        NU = Dynamics::Control::RowsAtCompileTime,
        NP = Dynamics::Parameters::RowsAtCompileTime,
        POLY_ORDER = Polynomial::POLY_ORDER,
        NUM_NODES = POLY_ORDER + 1,

        VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX,
        VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU,
        VARP_SIZE = NP
    };

    /** composite differentiation matrix */
    using comp_diff_mat_t = Eigen::SparseMatrix<Scalar>; //Eigen::Matrix<Scalar, VARX_SIZE, VARX_SIZE>;

    sparse_ode_collocation(const Dynamics &_f){}
    sparse_ode_collocation();
    ~sparse_ode_collocation(){}

    /** type to store optimization variable var = [x, u, p] */
    using var_t     = Eigen::Matrix<Scalar, VARX_SIZE + VARU_SIZE + VARP_SIZE, 1>;
    using constr_t  = Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    void operator() (const var_t &var, constr_t &constr_value,
                     const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1) ) const;

    /** linearized approximation */
    using jacobian_t = Eigen::SparseMatrix<Scalar>; //Eigen::Matrix<Scalar, constr_t::RowsAtCompileTime, var_t::RowsAtCompileTime>;
    using local_jacobian_t = Eigen::Matrix<Scalar, NX, NX + NU>;
    using Derivatives = Eigen::Matrix<Scalar, NX + NU, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    /** AD variables */
    using ADx_t = Eigen::Matrix<ADScalar, NX, 1>;
    using ADu_t = Eigen::Matrix<ADScalar, NU, 1>;
    ADx_t m_ADx, m_ADy;
    ADu_t m_ADu;

    void linearized(const var_t &var, jacobian_t &A, constr_t &b,
                    const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1));
    void _linearized(const var_t &var, jacobian_t &A, constr_t &b,
                     const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1));
    void _linearized_same_pattern(const var_t &var, jacobian_t &A, constr_t &b,
                                  const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1));

    void initialize_derivatives();
    void compute_inner_nnz();

public:
    Dynamics m_f;
    Polynomial m_basis_f;

    comp_diff_mat_t m_DiffMat, m_DiffMat_Ext;
    Eigen::Matrix<int, var_t::RowsAtCompileTime, 1> m_jac_inner_nnz = Eigen::Matrix<int, var_t::RowsAtCompileTime, 1>::Zero();
    void compute_diff_matrix();
};



template <typename Dynamics, typename Polynomial, int NumSegments>
sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::sparse_ode_collocation()
{
    compute_diff_matrix();
    compute_inner_nnz();
    initialize_derivatives();
}


template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::compute_diff_matrix()
{
    diff_mat_t D = m_basis_f.D();
    Eigen::SparseMatrix<Scalar> E(NX, NX); //= Eigen::Matrix<Scalar, NX, NX>::Identity().sparseView();
    E.setIdentity();

    if(NumSegments < 2)
    {
        m_DiffMat = Eigen::KroneckerProductSparse<diff_mat_t, Eigen::SparseMatrix<Scalar>>(D,E);
        return;
    }
    else
    {
        Eigen::Matrix<Scalar, NumSegments * POLY_ORDER + 1, NumSegments * POLY_ORDER + 1> DM =
                Eigen::Matrix<Scalar, NumSegments * POLY_ORDER + 1, NumSegments * POLY_ORDER + 1>::Zero();
        DM.template bottomRightCorner<NUM_NODES, NUM_NODES>() = D;
        for(int k = 0; k < (NumSegments - 1) * POLY_ORDER; k += POLY_ORDER)
            DM.template block<NUM_NODES - 1, NUM_NODES>(k, k) = D.template topLeftCorner<NUM_NODES - 1, NUM_NODES>();

        Eigen::SparseMatrix<Scalar> SpDM = DM.sparseView();
        m_DiffMat = Eigen::KroneckerProductSparse<Eigen::SparseMatrix<Scalar>, Eigen::SparseMatrix<Scalar>>(SpDM,E);

        m_DiffMat_Ext.resize(constr_t::RowsAtCompileTime, var_t::RowsAtCompileTime);
        m_DiffMat_Ext.setZero();
        m_DiffMat_Ext.template leftCols<VARX_SIZE>() = m_DiffMat;
        m_DiffMat_Ext.makeCompressed();

        return;
    }
}

/** Evaluate differential constraint */
template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::operator()(const var_t &var, constr_t &constr_value,
                                                                    const Scalar &t0, const Scalar &tf) const
{
    constr_t value;
    Eigen::Matrix<Scalar, NX, 1> f_res;
    Scalar t_scale = (tf - t0) / (2 * NumSegments);

    int n = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        m_f(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
            var.template segment<NP>(VARX_SIZE + VARU_SIZE), f_res);

        value. template segment<NX>(k) = f_res;
        n += NU;
    }

    constr_value = m_DiffMat * var.template head<VARX_SIZE>() - t_scale * value;
}

template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::initialize_derivatives()
{
    int deriv_num = NX + NU;
    int deriv_idx = 0;

    for(int i = 0; i < NX; i++)
    {
        m_ADx[i].derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
    for(int i = 0; i < NU; i++)
    {
        m_ADu(i).derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
}

/** estimate number of nonzeros in Jacobian */
template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::compute_inner_nnz()
{
    int *inner_nnz = m_DiffMat.innerNonZeroPtr();
    Eigen::Map<Eigen::Matrix<int, VARX_SIZE, 1>> mi(inner_nnz);
    m_jac_inner_nnz. template head<VARX_SIZE>() = mi;
    m_jac_inner_nnz. template head<VARX_SIZE>() += Eigen::Matrix<int, VARX_SIZE, 1>::Constant(NX-1);
    m_jac_inner_nnz. template segment<VARU_SIZE>(VARX_SIZE) = Eigen::Matrix<int, VARU_SIZE, 1>::Constant(NX);
}

/** compute linearization of diferential constraints */
template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::linearized(const var_t &var, jacobian_t &A, constr_t &b,
                                                                           const Scalar &t0, const Scalar &tf)
{
    if(A.nonZeros() != m_jac_inner_nnz.sum())
        _linearized(var, A, b, t0, tf);
    else
        _linearized_same_pattern(var, A, b, t0, tf);
}

template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::_linearized(const var_t &var, jacobian_t &A, constr_t &b,
                                                                           const Scalar &t0, const Scalar &tf)
{
    A.reserve(m_jac_inner_nnz);
    /** compute jacoabian of dynamics */
    local_jacobian_t jac;

    constr_t value;
    Scalar t_scale = (tf - t0) / (2 * NumSegments);

    /** initialize AD veriables */
    int n = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        /** @note is it an embedded cast ??*/
        for(int i = 0; i < NX; i++)
            m_ADx(i).value() = var.template segment<NX>(k)(i);

        for(int i = 0; i < NU; i++)
            m_ADu(i).value() = var.template segment<NU>(n + VARX_SIZE)(i);

        m_f(m_ADx, m_ADu,
            var.template segment<NP>(VARX_SIZE + VARU_SIZE), m_ADy);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            value. template segment<NX>(k)(i) = m_ADy(i).value();
            jac.row(i) = m_ADy(i).derivatives();
        }

        /** insert block jacobian */
        for(int j = 0; j < NX; ++j)
        {
            for(int m = 0; m < NX; ++m)
                A.insert(j + k, m + k) = -t_scale * jac(j, m);

            for(int m = 0; m < NU; ++m)
                A.insert(j + k, m + n + VARX_SIZE) = -t_scale * jac(j, m + NX);
        }

        n += NU;
    }

    b = m_DiffMat * var.template head<VARX_SIZE>() - t_scale * value;

    A.template leftCols<VARX_SIZE>() = m_DiffMat + A.template leftCols<VARX_SIZE>();
    A.makeCompressed();
}


template <typename Dynamics, typename Polynomial, int NumSegments>
void sparse_ode_collocation<Dynamics, Polynomial, NumSegments>::_linearized_same_pattern(const var_t &var, jacobian_t &A, constr_t &b,
                                                                                         const Scalar &t0, const Scalar &tf)
{
    /** compute jacoabian of dynamics */
    local_jacobian_t jac;

    constr_t value;
    Scalar t_scale = (tf - t0) / (2 * NumSegments);

    /** initialize AD veriables */
    int n = 0;
    int nnz_count = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        /** @note is it an embedded cast ??*/
        for(int i = 0; i < NX; i++)
            m_ADx(i).value() = var.template segment<NX>(k)(i);

        for(int i = 0; i < NU; i++)
            m_ADu(i).value() = var.template segment<NU>(n + VARX_SIZE)(i);

        m_f(m_ADx, m_ADu,
            var.template segment<NP>(VARX_SIZE + VARU_SIZE), m_ADy);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            value. template segment<NX>(k)(i) = m_ADy(i).value();
            jac.row(i) = m_ADy(i).derivatives();
        }

        /** set block jacobian */
        if(nnz_count > POLY_ORDER)
            nnz_count = 1; // set to one to glue blocks

        for(int j = 0; j < NX; ++j)
        {
            for(int m = 0; m < NX; ++m)
                A.valuePtr()[A.outerIndexPtr()[k + j] + m + nnz_count]  = -t_scale * jac(m,j);
        }

        for(int j = 0; j < NU; ++j)
        {
            for (int m = 0; m < NX; ++m)
                A.valuePtr()[A.outerIndexPtr()[j + n + VARX_SIZE] + m] = -t_scale * jac(m, j + NX);
        }
        ++nnz_count;

        n += NU;
    }

    b = m_DiffMat * var.template head<VARX_SIZE>() - t_scale * value;
    A.diagonal() = m_DiffMat.diagonal() + A.diagonal();
}

} // polympc namespace

#endif // SPARSE_ODE_COLLOCATION_HPP
