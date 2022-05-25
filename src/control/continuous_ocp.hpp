// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CONTINUOUS_OCP_HPP
#define CONTINUOUS_OCP_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/KroneckerProduct>
#include "autodiff/AutoDiffScalar.h"
#include "utils/helpers.hpp"
#include "solvers/bfgs.hpp"
#include "iostream"
#include <type_traits>

/** define the macro for forward declarations */
#define POLYMPC_FORWARD_DECLARATION( cNAME, cNX, cNU, cNP, cND, cNG, TYPE ) \
class cNAME;                                        \
template<>                                          \
struct polympc_traits<cNAME>                        \
{                                                   \
public:                                             \
    using Scalar = TYPE;                            \
    enum { NX = cNX, NU = cNU, NP = cNP, ND = cND, NG = cNG}; \
};                                                  \

/** define derived class traits */
template<typename Derived> struct polympc_traits;
template<typename T> struct polympc_traits<const T> : polympc_traits<T> {};

/** forward declare base class */
template<typename OCP, typename Approximation, int MatrixFormat> class ContinuousOCP;


template<typename OCP, typename Approximation, int MatrixFormat = DENSE>
class ContinuousOCP
{
public:
    ContinuousOCP()
    {
        EIGEN_STATIC_ASSERT((MatrixFormat == DENSE) || (MatrixFormat == SPARSE), "MatrixFormat bit is either 0 [DENSE] or 1 [SPARSE]");

        /** compute time nodes */
        set_time_limits(0, 1);

        /** seed derivatives */
        seed_derivatives();
        compute_diff_composite_matrix();
        estimate_jac_inner_nnz();
        estimate_hes_inner_nnz();
        estimate_ineq_jac_inner_nnz();

        // resize matrices for sparce computations
        allocate_jacobians();
    }
    ~ContinuousOCP() = default;

    static const bool CollocLast = false;

    enum
    {
        /** OCP dimensions */
        NX = polympc_traits<OCP>::NX,
        NU = polympc_traits<OCP>::NU,
        NP = polympc_traits<OCP>::NP,
        ND = polympc_traits<OCP>::ND,
        NG = polympc_traits<OCP>::NG,

        /** Collocation dimensions */
        NUM_NODES    = Approximation::NUM_NODES,
        POLY_ORDER   = Approximation::POLY_ORDER,
        NUM_SEGMENTS = Approximation::NUM_SEGMENTS,
        VARX_SIZE  = NX * NUM_NODES,
        VARU_SIZE  = NU * NUM_NODES,
        VARP_SIZE  = NP,
        VARD_SIZE  = ND,

        /** NLP dimensions */
        VAR_SIZE  = VARX_SIZE + VARU_SIZE + VARP_SIZE,
        NUM_EQ    = CollocLast ? VARX_SIZE : VARX_SIZE - NX,
        NUM_INEQ  = NG * NUM_NODES,
        NUM_BOX   = VAR_SIZE,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,

        /** Various flags */
        is_sparse = (MatrixFormat == SPARSE) ? 1 : 0,
        is_dense  = is_sparse ? 0 : 1,
        MATRIXFMT = MatrixFormat
    };

    /** define types*/
    /** state */
    template<typename scalar_t>
    using state_t = Eigen::Matrix<scalar_t, NX, 1>;

    /** control */
    template<typename scalar_t>
    using control_t = Eigen::Matrix<scalar_t, NU, 1>;

    /** parameters */
    template<typename scalar_t>
    using parameter_t = Eigen::Matrix<scalar_t, NP, 1>;

    /** inequality constraints */
    template<typename scalar_t>
    using constraint_t = Eigen::Matrix<scalar_t, NG, 1>;

    /** static parameters */
    using scalar_t = typename polympc_traits<OCP>::Scalar;
    using static_parameter_t = Eigen::Matrix<scalar_t, ND, 1>;
    using time_t   = typename Eigen::Matrix<scalar_t, NUM_NODES, 1>;
    using nodes_t  = typename Approximation::nodes_t;

    /** AD variables */
    using derivatives_t = Eigen::Matrix<scalar_t, NX + NU + NP, 1>;
    using ad_scalar_t   = Eigen::AutoDiffScalar<derivatives_t>;
    using second_derivatives_t = Eigen::Matrix<ad_scalar_t, NX + NU + NP, 1>;
    using ad_state_t   = Eigen::Matrix<ad_scalar_t, NX, 1>;
    using ad_control_t = Eigen::Matrix<ad_scalar_t, NU, 1>;
    using ad_param_t   = Eigen::Matrix<ad_scalar_t, NP, 1>;
    using ad_ineq_t    = Eigen::Matrix<ad_scalar_t, NG, 1>;
    ad_state_t m_ad_x, m_ad_y;
    ad_control_t m_ad_u;
    ad_param_t m_ad_p;
    ad_ineq_t  m_ad_g;
    ad_scalar_t m_ad_cost;

    using ad2_scalar_t = Eigen::AutoDiffScalar<second_derivatives_t>;
    Eigen::Matrix<ad2_scalar_t, NX, 1> m_ad2_x;
    Eigen::Matrix<ad2_scalar_t, NU, 1> m_ad2_u;
    Eigen::Matrix<ad2_scalar_t, NP, 1> m_ad2_p;
    Eigen::Matrix<ad2_scalar_t, NG, 1> m_ad2_g;
    ad2_scalar_t m_ad2_cost;

    /** time limits - do not make constant */
    scalar_t t_start{0};
    scalar_t t_stop{1};
    EIGEN_STRONG_INLINE void set_time_limits(const scalar_t& t0, const scalar_t& tf) noexcept
    {
        t_start = t0;
        t_stop = tf;

        /** update time nodes */
        const scalar_t t_length = (t_stop - t_start) / (NUM_SEGMENTS);
        const scalar_t t_shift  = t_length / 2;
        for(Eigen::Index i = 0; i < NUM_SEGMENTS; ++i)
            time_nodes.template segment<POLY_ORDER + 1>(i * POLY_ORDER) =  (t_length/2) * m_nodes +
                    (t_start + t_shift + i * t_length) * Approximation::nodes_t::Ones();
    }

    /** compute collocation parameters */
    const typename Approximation::diff_mat_t  m_D     = Approximation::compute_diff_matrix();
    const typename Approximation::nodes_t     m_nodes = Approximation::compute_nodes();
    const typename Approximation::q_weights_t m_quad_weights = Approximation::compute_int_weights();
    time_t time_nodes = time_t::Zero();

    /** NLP variables */
    using nlp_variable_t         = typename dense_matrix_type_selector<scalar_t, VAR_SIZE, 1>::type;
    using nlp_constraints_t      = typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, 1>::type;
    using nlp_eq_constraints_t   = typename dense_matrix_type_selector<scalar_t, NUM_EQ, 1>::type;
    using nlp_ineq_constraints_t = typename dense_matrix_type_selector<scalar_t, NUM_INEQ, 1>::type;
    // choose to allocate sparse or dense jacoabian and hessian
    using nlp_eq_jacobian_t = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, NUM_EQ, VAR_SIZE>::type>::type;
    using nlp_ineq_jacobian_t = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                typename dense_matrix_type_selector<scalar_t, NUM_INEQ, VAR_SIZE>::type>::type;
    using nlp_jacobian_t    = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>::type>::type;
    using nlp_hessian_t     = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, VAR_SIZE, VAR_SIZE>::type>::type;
    using nlp_cost_t        = scalar_t;
    using nlp_dual_t        = typename dense_matrix_type_selector<scalar_t, DUAL_SIZE, 1>::type;

    // temporary matrices for equality and ineqquality Jacobians
    typename std::conditional<MATRIXFMT == SPARSE, nlp_eq_jacobian_t, void*>::type m_Je;
    typename std::conditional<MATRIXFMT == SPARSE, nlp_ineq_jacobian_t, void*>::type m_Ji;

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                                    const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                                    const scalar_t &t, Eigen::Ref<constraint_t<T>> g) const noexcept
    {
        static_cast<const OCP*>(this)->inequality_constraints_impl(x,u,p,d,t,g);
    }
    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                                    const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                                    const scalar_t &t, Eigen::Ref<constraint_t<T>> g) noexcept
    {
        static_cast<OCP*>(this)->inequality_constraints_impl(x,u,p,d,t,g);
    }
    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                                         const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                                         const scalar_t &t, Eigen::Ref<constraint_t<T>> g) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(g);
    }

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void dynamics(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                         const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                         const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        static_cast<const OCP*>(this)->dynamics_impl(x,u,p,d,t,xdot);
    }

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void mayer_term(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                        const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                        const scalar_t &t, T &mayer) const noexcept
    {
        static_cast<const OCP*>(this)->mayer_term_impl(x,u,p,d,t,mayer);
    }
    template<typename T>
    EIGEN_STRONG_INLINE void mayer_term(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                        const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                        const scalar_t &t, T &mayer) noexcept
    {
        static_cast<OCP*>(this)->mayer_term_impl(x,u,p,d,t,mayer);
    }
    template<typename T>
    EIGEN_STRONG_INLINE void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                             const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                                             const scalar_t &t, T &mayer) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(mayer);
    }

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void lagrange_term(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                              const scalar_t &t, T &lagrange) const noexcept
    {
        static_cast<const OCP*>(this)->lagrange_term_impl(x,u,p,d,t,lagrange);
    }

    template<typename T>
    EIGEN_STRONG_INLINE void lagrange_term(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p,const Eigen::Ref<const static_parameter_t> d,
                              const scalar_t &t, T &lagrange) noexcept
    {
        static_cast<OCP*>(this)->lagrange_term_impl(x,u,p,d,t,lagrange);
    }
    template<typename T>
    EIGEN_STRONG_INLINE void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                                const scalar_t &t, T &lagrange) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(lagrange);
    }

    /** seed edrivatives */
    void seed_derivatives();

    /** in case of sparse representation, estimate the upper bound amount of non-zero elements */
    // store number of nonzeros per column in Jacobian and Hessian
    Eigen::VectorXi m_jac_inner_nnz, m_ineq_jac_inner_nnz;
    Eigen::VectorXi m_hes_inner_nnz;
    Eigen::SparseMatrix<scalar_t> m_DiffMat; // store sparse differentiation matrix for sparse implementation

    template <int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type compute_diff_composite_matrix() const noexcept {}

    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type estimate_jac_inner_nnz() const noexcept {}

    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type estimate_ineq_jac_inner_nnz() const noexcept {}

    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type estimate_hes_inner_nnz() const noexcept {}


    template <int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type compute_diff_composite_matrix() noexcept
    {
        Eigen::SparseMatrix<scalar_t> E(NX, NX);
        E.setIdentity();
        E.makeCompressed();

        if(NUM_SEGMENTS < 2)
        {
            if (CollocLast)
            {
                m_DiffMat = Eigen::KroneckerProductSparse<typename Approximation::diff_mat_t, Eigen::SparseMatrix<scalar_t>>(m_D, E);
            }
            else
            {
                using diff_mat_tilde_t = Eigen::Matrix<scalar_t, POLY_ORDER, POLY_ORDER + 1>;
                m_DiffMat = Eigen::KroneckerProductSparse<diff_mat_tilde_t, Eigen::SparseMatrix<scalar_t>>(m_D.template topLeftCorner<POLY_ORDER, POLY_ORDER + 1>(), E);
            }
        }
        else
        {
            const unsigned num_nodes = CollocLast ? NUM_NODES : NUM_NODES - 1;
            Eigen::Matrix<scalar_t, num_nodes, NUM_NODES> DM;
            DM.setZero();

            for(int k = 0; k < (NUM_SEGMENTS - 1) * POLY_ORDER; k += POLY_ORDER)
            {
                DM.template block<POLY_ORDER, POLY_ORDER + 1>(k, k) = m_D.template topLeftCorner<POLY_ORDER, POLY_ORDER + 1>();
            }

            if (CollocLast)
            {
                DM.template bottomRightCorner<POLY_ORDER + 1, POLY_ORDER + 1>() = m_D;
            } else
            {
                DM.template bottomRightCorner<POLY_ORDER, POLY_ORDER + 1>() = m_D.template topLeftCorner<POLY_ORDER, POLY_ORDER + 1>();
            }

            Eigen::SparseMatrix<scalar_t> SpDM = DM.sparseView().pruned(std::numeric_limits<scalar_t>::epsilon());
            SpDM.makeCompressed();

            m_DiffMat = Eigen::KroneckerProductSparse<Eigen::SparseMatrix<scalar_t>, Eigen::SparseMatrix<scalar_t>>(SpDM, E);
            m_DiffMat.makeCompressed();

        }
    }

    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type estimate_jac_inner_nnz() noexcept
    {
        m_jac_inner_nnz = Eigen::VectorXi::Zero(VAR_SIZE);
        // dynamics entries
        if (CollocLast)
        {
            // all points are collocated
            m_jac_inner_nnz. template head<VARX_SIZE + VARU_SIZE>() = Eigen::VectorXi::Constant(VARX_SIZE + VARU_SIZE, NX);
        }
        else
        {
            // last points are not collocated
            m_jac_inner_nnz. template head<VARX_SIZE - NX>() = Eigen::VectorXi::Constant(VARX_SIZE - NX, NX);
            m_jac_inner_nnz. template segment<VARU_SIZE - NU>(VARX_SIZE) = Eigen::VectorXi::Constant(VARU_SIZE - NU, NX);
        }
        m_jac_inner_nnz. template tail<VARP_SIZE>() = Eigen::VectorXi::Constant(VARP_SIZE, VARX_SIZE);

        // add diff matrix entries
        for(Eigen::Index i = 0; i < NUM_NODES; i++)
        {
            // only POLY_ORDER - 1 because there is 1 nnz overlap with dynamics
            m_jac_inner_nnz. template segment<NX>(i * NX) += Eigen::VectorXi::Constant(NX, POLY_ORDER - 1);
            if((i != 0) && (i != (NUM_NODES-1)) && (i % POLY_ORDER) == 0)
               m_jac_inner_nnz. template segment<NX>(i * NX) += Eigen::VectorXi::Constant(NX, POLY_ORDER);
        }
        if (CollocLast)
        {
            // last collocation node
            m_jac_inner_nnz. template segment<(POLY_ORDER + 1) * NX>((NUM_NODES - (POLY_ORDER + 1)) * NX) +=
                    Eigen::VectorXi::Ones((POLY_ORDER + 1) * NX);
        }
        else
        {
            // add back 1 nnz for last node
            m_jac_inner_nnz. template segment<NX>(VARX_SIZE - NX) += Eigen::VectorXi::Constant(NX, 1);
        }
    }

    // estimate number of non-zeros in Hessian
    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type estimate_hes_inner_nnz() noexcept
    {
        m_hes_inner_nnz = Eigen::VectorXi::Constant(VAR_SIZE, NX + NU + NP);
        if(NP > 0)
            m_hes_inner_nnz. template tail<NP>().setConstant(VAR_SIZE);
    }

    //estimate number of non-zeros in inequality constraints Jacoabian
    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type estimate_ineq_jac_inner_nnz() noexcept
    {
        m_ineq_jac_inner_nnz = Eigen::VectorXi::Zero(VAR_SIZE);
        m_ineq_jac_inner_nnz. template head<VARX_SIZE + VARU_SIZE>() = Eigen::VectorXi::Constant(VARX_SIZE + VARU_SIZE, NG);
        m_ineq_jac_inner_nnz. template tail<VARP_SIZE>() = Eigen::VectorXi::Constant(VARP_SIZE, NUM_INEQ);
    }

    /** interface functions to comply with ipopt */
    EIGEN_STRONG_INLINE int nnz_jacobian() const noexcept
    {
        if(MatrixFormat == SPARSE)
            return m_jac_inner_nnz.sum() + m_ineq_jac_inner_nnz.sum();
        else
            return (NUM_EQ + NUM_INEQ) * VAR_SIZE;
    }

    // number of nonzeros in lower-triangular part of the hessian
    EIGEN_STRONG_INLINE int nnz_lag_hessian() const noexcept
    {
        if(MatrixFormat == SPARSE)
            return (m_hes_inner_nnz.sum() + VAR_SIZE) / 2;
        else
            return (VAR_SIZE * (VAR_SIZE + 1)) / 2;
    }


    template <int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type allocate_jacobians() const noexcept {}

    template <int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type allocate_jacobians() noexcept
    {
        m_Je.resize(NUM_EQ, VAR_SIZE);
        m_Ji.resize(NUM_INEQ, VAR_SIZE);
    }

    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    block_insert_sparse(Eigen::SparseMatrix<scalar_t>& dst, const Eigen::Index &row_offset,
                        const Eigen::Index &col_offset, const Eigen::SparseMatrix<scalar_t>& src) const noexcept
    {
        // assumes enough spase is allocated in the dst matrix
        for(Eigen::Index k = 0; k < src.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(src, k); it; ++it)
                dst.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
    }

    template<class InputIt, class Size, int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    copy_n_sparse(InputIt first, Size count, Eigen::SparseMatrix<scalar_t>& dst, const Eigen::Index &col, const Eigen::Index &offset)
    {
#ifndef EIGEN_NO_DEBUG
        // assert col exists
        eigen_assert(col < dst.outerSize());
        // uncompressed
        if (dst.innerNonZeroPtr() != 0)
        {
            // assert not writing outside memory
            eigen_assert(count + offset <= dst.innerNonZeroPtr()[col]);
        }
            // compressed
        else
        {
            // assert not writing outside memory
            eigen_assert(dst.outerIndexPtr()[col] + offset < dst.nonZeros());
        }
#endif
        std::copy_n(first, count, dst.valuePtr() + dst.outerIndexPtr()[col] + offset);
    }

    template<class InputIt, class Size, int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    add_n_sparse(InputIt first, Size count, Eigen::SparseMatrix<scalar_t>& dst, const Eigen::Index &col, const Eigen::Index &offset)
    {
#ifndef EIGEN_NO_DEBUG
        // assert col exists
        eigen_assert(col < dst.outerSize());
        // uncompressed
        if (dst.innerNonZeroPtr() != 0)
        {
            // assert not writing outside memory
            eigen_assert(count + offset <= dst.innerNonZeroPtr()[col]);
        }
            // compressed
        else
        {
            // assert not writing outside memory
            eigen_assert(dst.outerIndexPtr()[col] + offset < dst.nonZeros());
        }
#endif
        std::transform(dst.valuePtr() + dst.outerIndexPtr()[col] + offset,
                       dst.valuePtr() + dst.outerIndexPtr()[col] + offset + count,
                       first, dst.valuePtr() + dst.outerIndexPtr()[col] + offset,
                       std::plus<scalar_t>());
    }

    /** @brief
     *
     */
    /**
    template<typename T>
    EIGEN_STRONG_INLINE void final_inequality_constraints(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                                                          const static_parameter_t &d, const scalar_t &t, Eigen::Ref<constraint_t<T>> h) const noexcept
    {
        static_cast<OCP*>(this)->final_inequality_constraints_impl(x,u,p,d,t,h);
    } */

    /** equality constraint */
    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var,
                                        const Eigen::Ref<const static_parameter_t>& p,
                                        Eigen::Ref<nlp_eq_constraints_t> constraint) const noexcept;

    /** equality constraint */
    EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var,
                                          const Eigen::Ref<const static_parameter_t>& p,
                                          Eigen::Ref<nlp_ineq_constraints_t> constraint) const noexcept;

    /** generic (combined) constraint */
    EIGEN_STRONG_INLINE void constraints(const Eigen::Ref<const nlp_variable_t>& var,
                                         const Eigen::Ref<const static_parameter_t>& p,
                                         Eigen::Ref<nlp_constraints_t> constraint) const noexcept;

    /** linearise equality constraints */
    //void equalities_linerised(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                          Eigen::Ref<nlp_constraints_t> constraint, Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                                    Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                    Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                                     Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                     nlp_eq_jacobian_t& jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                      const Eigen::Ref<const static_parameter_t>& p,
                                                                      Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                      Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                       const Eigen::Ref<const static_parameter_t>& p,
                                                                       Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                       nlp_ineq_jacobian_t& jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type constraints_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                                     Eigen::Ref<nlp_constraints_t> constraint,
                                                                     Eigen::Ref<nlp_jacobian_t> jacobian) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type constraints_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                      const Eigen::Ref<const static_parameter_t>& p,
                                                                      Eigen::Ref<nlp_constraints_t> constraint,
                                                                      nlp_jacobian_t& jacobian) noexcept;

    /** sparse linearisation */
    void _equalities_linearised_sparse(const Eigen::Ref<const nlp_variable_t>& var,
                                       const Eigen::Ref<const static_parameter_t>& p,
                                       Eigen::Ref<nlp_eq_constraints_t> constraint,
                                       nlp_eq_jacobian_t& jacobian) noexcept;

    /** sparse linearisation with pattern unchanged: very dangerous but extremely efficient */
    void _equalities_linearised_sparse_update(const Eigen::Ref<const nlp_variable_t>& var,
                                              const Eigen::Ref<const static_parameter_t>& p,
                                              Eigen::Ref<nlp_eq_constraints_t> constraint,
                                              nlp_eq_jacobian_t& jacobian) noexcept;

    /** sparse linearisation */
    template <int N = NUM_INEQ> typename std::enable_if< (N > 0) >::type
    _inequalities_linearised_sparse(const Eigen::Ref<const nlp_variable_t>& var,
                                    const Eigen::Ref<const static_parameter_t>& p,
                                    Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                    nlp_ineq_jacobian_t &jacobian) noexcept;

    template <int N = NUM_INEQ> typename std::enable_if< (N == 0) >::type
    _inequalities_linearised_sparse(const Eigen::Ref<const nlp_variable_t>& var,
                                    const Eigen::Ref<const static_parameter_t>& p,
                                    Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                    nlp_ineq_jacobian_t &jacobian) const noexcept
    {
        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(constraint);
        polympc::ignore_unused_var(jacobian);
    }

    /** sparse linearisation with pattern unchanged: very dangerous but extremely efficient */
    template <int N = NUM_INEQ> typename std::enable_if< (N > 0) >::type
    _inequalities_linearised_sparse_update(const Eigen::Ref<const nlp_variable_t>& var,
                                           const Eigen::Ref<const static_parameter_t>& p,
                                           Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                           nlp_ineq_jacobian_t& jacobian) noexcept;

    template <int N = NUM_INEQ> typename std::enable_if< (N == 0) >::type
    _inequalities_linearised_sparse_update(const Eigen::Ref<const nlp_variable_t>& var,
                                           const Eigen::Ref<const static_parameter_t>& p,
                                           Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                           nlp_ineq_jacobian_t& jacobian) const noexcept
    {
        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(constraint);
        polympc::ignore_unused_var(jacobian);
    }

    template<int N = NUM_INEQ> typename std::enable_if< (N==0) >::type
    _inequalities_linearised_dense(const Eigen::Ref<const nlp_variable_t>& var,
                                   const Eigen::Ref<const static_parameter_t>& p,
                                   Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                   Eigen::Ref<nlp_ineq_jacobian_t> jacobian) const noexcept
    {
        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(constraint);
        polympc::ignore_unused_var(jacobian);
    }

    template<int N = NUM_INEQ> typename std::enable_if< (N > 0) >::type
    _inequalities_linearised_dense(const Eigen::Ref<const nlp_variable_t>& var,
                                   const Eigen::Ref<const static_parameter_t>& p,
                                   Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                   Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        jacobian = nlp_ineq_jacobian_t::Zero(NUM_INEQ, VAR_SIZE);
        Eigen::Matrix<scalar_t, NG, NX + NU + NP> jac;

        m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
        for(int k = 0; k < NUM_NODES; ++k)
        {
            m_ad_x = var.template segment<NX>(k * NX);
            m_ad_u = var.template segment<NU>(k * NU + VARX_SIZE);

            inequality_constraints<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(k), m_ad_g);

            // compute value and first derivatives
            for(int i = 0; i < NG; i++)
            {
                constraint. template segment<NG>(k * NG)(i) = m_ad_g(i).value();
                jac.row(i) = m_ad_g(i).derivatives();
            }

            // insert block jacobian
            jacobian.template block<NG, NX>(k * NG, k * NX) = jac.template leftCols<NX>();
            jacobian.template block<NG, NU>(k * NG, k * NU + VARX_SIZE) = jac.template block<NG, NU>(0, NX);
            jacobian.template block<NG, NP>(k * NG, VARX_SIZE + VARU_SIZE) = jac.template rightCols<NP>();
        }
    }


    /** compute cost */
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept;
    void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                       scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                                    scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                    Eigen::Ref<nlp_hessian_t> cost_hessian) noexcept;

    /** sparse hessian calculation */
    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                                     scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                     nlp_hessian_t& cost_hessian) noexcept;

    void _cost_grad_hess_sparse(const Eigen::Ref<const nlp_variable_t>& var,
                                const Eigen::Ref<const static_parameter_t>& p,
                                scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                nlp_hessian_t& cost_hessian) noexcept;

    void _cost_grad_hess_sparse_update(const Eigen::Ref<const nlp_variable_t>& var,
                                       const Eigen::Ref<const static_parameter_t>& p,
                                       scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                       nlp_hessian_t& cost_hessian) noexcept;


    /** compute Lagrangian */
    EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                    const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian) noexcept;

    EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                    const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_constraints_t> g) noexcept;

    /** Lagrangian gradient */
    void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type
    lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                        const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                        Eigen::Ref<nlp_variable_t> cost_gradient,
                        Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept;

    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type
    lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                        const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                        Eigen::Ref<nlp_variable_t> cost_gradient,
                        Eigen::Ref<nlp_constraints_t> g, nlp_jacobian_t& jac_g) noexcept;

    /** lagrangian hessian */
    void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                     const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                     Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept;


    /** dense implementation */
    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type
    lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                Eigen::Ref<nlp_hessian_t> lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
    {
        lagrangian_gradient_hessian(var, p, lam, _lagrangian, lag_gradient, lag_hessian, cost_gradient, g, jac_g, scalar_t(1.0));
    }


    template<int T = MatrixFormat>
    typename std::enable_if<T == DENSE>::type
    lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                Eigen::Ref<nlp_hessian_t> lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g, const scalar_t& cost_scale) noexcept;

    /** sparse implementation */
    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type
    lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                nlp_hessian_t& lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                Eigen::Ref<nlp_constraints_t> g, nlp_jacobian_t& jac_g) noexcept
    {
        lagrangian_gradient_hessian(var, p, lam, _lagrangian, lag_gradient, lag_hessian, cost_gradient, g, jac_g, scalar_t(1.0));
    }

    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type
    lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                nlp_hessian_t& lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                Eigen::Ref<nlp_constraints_t> g, nlp_jacobian_t& jac_g,  const scalar_t& cost_scale) noexcept;

    /** Symmetric Rank 1 update preserving the sparsity pattern */
    template<int T = MatrixFormat>
    typename std::enable_if<T == SPARSE>::type
    hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t> s, const Eigen::Ref<const nlp_variable_t> y) const noexcept;

    /** Plain damped BFGS  update */
    template<int T = MatrixFormat>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t> s, const Eigen::Ref<const nlp_variable_t> y) const noexcept
    {
        BFGS_update(hessian, s, y);
    }

};

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::seed_derivatives()
{
    const int deriv_num = NX + NU + NP;
    int deriv_idx = 0;

    for(int i = 0; i < NX; i++)
    {
        m_ad_y(i).derivatives()  = derivatives_t::Zero();
        m_ad_x(i).derivatives()  = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_x(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_x(i).value().derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        for(int idx = 0; idx < deriv_num; idx++)
        {
            m_ad2_x(i).derivatives()(idx).derivatives()  = derivatives_t::Zero();
        }
        deriv_idx++;
    }
    for(int i = 0; i < NU; i++)
    {
        m_ad_u(i).derivatives()  = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_u(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_u(i).value().derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        for(int idx = 0; idx < deriv_num; idx++)
        {
            m_ad2_u(i).derivatives()(idx).derivatives()  = derivatives_t::Zero();
        }
        deriv_idx++;
    }
    for(int i = 0; i < NP; i++)
    {
        m_ad_p(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_p(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_p(i).value().derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        for(int idx = 0; idx < deriv_num; idx++)
        {
            m_ad2_p(i).derivatives()(idx).derivatives()  = derivatives_t::Zero();
        }
        deriv_idx++;
    }
    for(int idx = 0; idx < deriv_num; idx++)
    {
        m_ad2_cost.derivatives()(idx).derivatives()  = derivatives_t::Zero();
    }
    m_ad2_cost.value().derivatives() = derivatives_t::Zero();
}


template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::equalities(const Eigen::Ref<const nlp_variable_t>& var,
                                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                                 Eigen::Ref<nlp_eq_constraints_t> constraint) const noexcept
{
    state_t<scalar_t> f_res; f_res.setZero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();

    //DX.transposeInPlace();

    int n = 0;
    int t = 0;
    for(int k = 0; k < (CollocLast ? VARX_SIZE : VARX_SIZE - NX); k += NX)
    {
        dynamics<scalar_t>(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
                           var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(t), f_res);
        constraint. template segment<NX>(k) = DX.transpose().col(t);
        constraint. template segment<NX>(k).noalias() -= t_scale * f_res;
        n += NU;
        ++t;
    }
}

// evaluate constraints
template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::inequalities(const Eigen::Ref<const nlp_variable_t>& var,
                                                                   const Eigen::Ref<const static_parameter_t>& p,
                                                                   Eigen::Ref<nlp_ineq_constraints_t> constraint) const noexcept
{
    constraint_t<scalar_t> g_res; g_res.setZero();
    for (int k = 0; k < NUM_NODES; ++k)
    {
        inequality_constraints<scalar_t>(var.template segment<NX>(k * NX), var.template segment<NU>(k * NU + VARX_SIZE),
                                         var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(k), g_res);

        constraint.template segment<NG>(k * NG) = g_res;
    }
}

// evaluate combined constraints
template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::constraints(const Eigen::Ref<const nlp_variable_t>& var,
                                                                  const Eigen::Ref<const static_parameter_t>& p,
                                                                  Eigen::Ref<nlp_constraints_t> constraint) const noexcept
{
    this->equalities(var, p, constraint.template head<NUM_EQ>());
    this->inequalities(var, p, constraint.template tail<NUM_INEQ>()); // why???
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::equalities_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                       const Eigen::Ref<const static_parameter_t> &p,
                                                                       Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                       Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
{
    jacobian = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
    /** compute jacoabian of dynamics */
    Eigen::Matrix<scalar_t, NX, NX + NU + NP> jac;
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
    {
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();
    }

    /** add Diff matrix to the Jacobian */
    Eigen::DiagonalMatrix<scalar_t, NX> I; I.setIdentity();
    for(int s = 0; s < NUM_SEGMENTS; s++)
    {
        for(int i = 0; i < POLY_ORDER; i++)
        {
            for(int j = 0; j < POLY_ORDER + 1; j++)
            {
                int shift = s * POLY_ORDER * NX;
                jacobian.template block<NX, NX>(shift + i * NX, shift + j * NX) = m_D(i, j) * I;
            }
        }
    }

    // last node
    if (CollocLast)
    {
        int shift = (NUM_SEGMENTS - 1) * POLY_ORDER * NX;
        int i = POLY_ORDER;
        for(int j = 0; j < POLY_ORDER + 1; j++)
        {
            jacobian.template block<NX, NX>(shift + i * NX, shift + j * NX) = m_D(i, j) * I;
        }
    }

    /** initialize AD veriables */
    int n = 0;
    int t = 0;
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < (CollocLast ? VARX_SIZE : VARX_SIZE - NX); k += NX)
    {
        m_ad_x = var.template segment<NX>(k);
        m_ad_u = var.template segment<NU>(n + VARX_SIZE);

        dynamics<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(t), m_ad_y);

        /** compute value and first derivatives */
        for(int i = 0; i < NX; i++)
        {
            constraint. template segment<NX>(k)(i) = -t_scale * m_ad_y(i).value();
            jac.row(i) = m_ad_y(i).derivatives();
        }
        constraint. template segment<NX>(k) += DX.transpose().col(t);
        /** finish the computation of the constraint */


        /** insert block jacobian */
        jacobian.template block<NX, NX>(k, k).noalias() -= t_scale * jac.template leftCols<NX>();
        jacobian.template block<NX, NU>(k, n + VARX_SIZE).noalias() -= t_scale * jac.template block<NX, NU>(0, NX);
        jacobian.template block<NX, NP>(k, VARX_SIZE + VARU_SIZE).noalias() -= t_scale * jac.template rightCols<NP>();

        n += NU;
        t++;
    }

}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::equalities_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                       const Eigen::Ref<const static_parameter_t> &p,
                                                                       Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                       nlp_eq_jacobian_t &jacobian) noexcept
{
    if(jacobian.nonZeros() != m_jac_inner_nnz.sum())
        _equalities_linearised_sparse(var, p, constraint, jacobian);
    else
        _equalities_linearised_sparse_update(var, p, constraint, jacobian);
}

/** sparse linearisation */
template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::_equalities_linearised_sparse(const Eigen::Ref<const nlp_variable_t> &var,
                                                                                    const Eigen::Ref<const static_parameter_t> &p,
                                                                                    Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                                    nlp_eq_jacobian_t &jacobian) noexcept
{
    eigen_assert(jacobian.outerSize() == VAR_SIZE);
    jacobian.reserve(m_jac_inner_nnz);
    /** compute jacoabian of dynamics */
    Eigen::Matrix<scalar_t, NX, NX + NU + NP> jac = Eigen::Matrix<scalar_t, NX, NX + NU + NP>::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
    {
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();
    }

    /** initialize AD veriables */
    int n = 0;
    int t = 0;
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < (CollocLast ? VARX_SIZE : VARX_SIZE - NX); k += NX)
    {
        m_ad_x = var.template segment<NX>(k);
        m_ad_u = var.template segment<NU>(n + VARX_SIZE);

        dynamics<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(t), m_ad_y);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            constraint. template segment<NX>(k)(i) = -t_scale * m_ad_y(i).value();
            jac.row(i) = m_ad_y(i).derivatives();
        }

        constraint. template segment<NX>(k) += DX.transpose().col(t);
        /** finish the computation of the constraint */

        /** insert block jacobian */
        for(int j = 0; j < NX; ++j)
        {
            for(int m = 0; m < NX; ++m)
                jacobian.insert(j + k, m + k) = -t_scale * jac(j, m);

            for(int m = 0; m < NU; ++m)
                jacobian.insert(j + k, m + n + VARX_SIZE) = -t_scale * jac(j, m + NX);

            for(int m = 0; m < NP; ++m)
                jacobian.insert(j + k, VARX_SIZE + VARU_SIZE + m) = -t_scale * jac(j, NX + NU + m);
        }

        n += NU;
        t++;
    }

    jacobian.template leftCols<VARX_SIZE>() += m_DiffMat;
}


template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::_equalities_linearised_sparse_update(const Eigen::Ref<const nlp_variable_t> &var,
                                                                                           const Eigen::Ref<const static_parameter_t> &p,
                                                                                           Eigen::Ref<nlp_eq_constraints_t> constraint,
                                                                                           nlp_eq_jacobian_t &jacobian) noexcept
{
    eigen_assert(jacobian.outerSize() == VAR_SIZE);
    /** compute jacoabian of dynamics */
    Eigen::Matrix<scalar_t, NX, NX + NU + NP> jac = Eigen::Matrix<scalar_t, NX, NX + NU + NP>::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

    /** @badcode: redo with Reshaped expression later*/
    Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
    Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

    for(int i = 0; i < NUM_SEGMENTS; ++i)
    {
        DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();
    }

    /** initialize AD veriables */
    int n = 0;
    int t = 0;
    int nnz_count = 0;
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < (CollocLast ? VARX_SIZE : VARX_SIZE - NX); k += NX)
    {
        m_ad_x = var.template segment<NX>(k);
        m_ad_u = var.template segment<NU>(n + VARX_SIZE);

        dynamics<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(t), m_ad_y);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            constraint. template segment<NX>(k)(i) = -t_scale * m_ad_y(i).value();
            jac.row(i) = m_ad_y(i).derivatives();
        }
        constraint. template segment<NX>(k) += DX.transpose().col(t);
        /** finish the computation of the constraint */
        jac *= -t_scale;

        /** insert block jacobian */
        if(nnz_count > POLY_ORDER)
            nnz_count = 1; // set to one to glue blocks

        /** copy state sensitivities */
        for(int j = 0; j < NX; ++j)
            copy_n_sparse(jac.col(j).data(), NX, jacobian, k + j, nnz_count);

        /** copy control sensitivities */
        for(int j = 0; j < NU; ++j)
            copy_n_sparse(jac.col(j + NX).data(), NX, jacobian, j + n + VARX_SIZE, 0);

        for(int j = 0; j < NP; ++j)
            copy_n_sparse(jac.col(j + NX + NU).data(), NX, jacobian, j + VARX_SIZE + VARU_SIZE, t * NX);

        ++nnz_count;

        n += NU;
        t++;
    }

    // instead of overwriting m_DiffMat completely we just update the changed parts on the diagonal
    jacobian.diagonal() += m_DiffMat.diagonal();
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::inequalities_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                         const Eigen::Ref<const static_parameter_t> &p,
                                                                         Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                         Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
{
    _inequalities_linearised_dense(var, p, constraint, jacobian);
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::inequalities_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                         const Eigen::Ref<const static_parameter_t> &p,
                                                                         Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                         nlp_ineq_jacobian_t &jacobian) noexcept
{
    if(jacobian.nonZeros() != m_ineq_jac_inner_nnz.sum())
        _inequalities_linearised_sparse(var, p, constraint, jacobian);
    else
        _inequalities_linearised_sparse_update(var, p, constraint, jacobian);
}

//sparse linearisation
template<typename OCP, typename Approximation, int MatrixFormat>
template<int N> typename std::enable_if<(N > 0)>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::_inequalities_linearised_sparse(const Eigen::Ref<const nlp_variable_t> &var,
                                                                                 const Eigen::Ref<const static_parameter_t> &p,
                                                                                 Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                                 nlp_ineq_jacobian_t &jacobian) noexcept
{
    eigen_assert(jacobian.outerSize() == VAR_SIZE);
    jacobian.reserve(m_ineq_jac_inner_nnz);
    /** compute jacoabian of inequality constraints */
    Eigen::Matrix<scalar_t, NG, NX + NU + NP> jac;

    /** initialize AD veriables */
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < NUM_NODES; ++k)
    {
        m_ad_x = var.template segment<NX>(k * NX);
        m_ad_u = var.template segment<NU>(k * NU + VARX_SIZE);

        inequality_constraints<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(k), m_ad_g);

        /** compute value and first derivatives */
        for(int i = 0; i < NG; i++)
        {
            constraint. template segment<NG>(k * NG)(i) = m_ad_g(i).value();
            jac.row(i) = m_ad_g(i).derivatives(); // !!
        }

        /** insert block jacobian */
        for(int j = 0; j < NG; ++j)
        {
            for(int m = 0; m < NX; ++m)
                jacobian.insert(j + k * NG, m + k * NX) = jac(j, m);

            for(int m = 0; m < NU; ++m)
                jacobian.insert(j + k * NG, m + k * NU + VARX_SIZE) = jac(j, m + NX);

            for(int m = 0; m < NP; ++m)
                jacobian.insert(j + k * NG, VARX_SIZE + VARU_SIZE + m) = jac(j, NX + NU + m);
        }
    }
}

template<typename OCP, typename Approximation, int MatrixFormat>
template <int N> typename std::enable_if<(N > 0)>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::_inequalities_linearised_sparse_update(const Eigen::Ref<const nlp_variable_t> &var,
                                                                                        const Eigen::Ref<const static_parameter_t> &p,
                                                                                        Eigen::Ref<nlp_ineq_constraints_t> constraint,
                                                                                        nlp_ineq_jacobian_t &jacobian) noexcept
{
    eigen_assert(jacobian.outerSize() == VAR_SIZE);
    /** compute jacoabian of dynamics */
    Eigen::Matrix<scalar_t, NX, NX + NU + NP> jac;

    /** initialize AD veriables */
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    for(int k = 0; k < NUM_NODES; ++k)
    {
        m_ad_x = var.template segment<NX>(k * NX);
        m_ad_u = var.template segment<NU>(k * NU + VARX_SIZE);

        inequality_constraints<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(k), m_ad_g);

        /** compute value and first derivatives */
        for(int i = 0; i < NG; i++)
        {
            constraint. template segment<NG>(k * NG)(i) = m_ad_g(i).value();
            jac.row(i) = m_ad_g(i).derivatives();
        }

        /** copy state sensitivities */
        for(int j = 0; j < NX; ++j)
            copy_n_sparse(jac.col(j).data(), NG, jacobian, j + k * NX, 0);

        /** copy control sensitivities */
        for(int j = 0; j < NU; ++j)
            copy_n_sparse(jac.col(j + NX).data(), NG, jacobian, j + k * NU + VARX_SIZE, 0);

        /** @bug: iterate over NP columns */
        for(int j = 0; j < NP; ++j)
            copy_n_sparse(jac.col(j + NX + NU).data(), NG, jacobian, j + VARX_SIZE + VARU_SIZE, k * NG);
    }
}


// linearise combined constraints
template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::constraints_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                        const Eigen::Ref<const static_parameter_t> &p,
                                                                        Eigen::Ref<nlp_constraints_t> constraint,
                                                                        Eigen::Ref<nlp_jacobian_t> jacobian) noexcept
{
    this->equalities_linearised(var, p, constraint.template head<NUM_EQ>(), jacobian.topRows(NUM_EQ));
    this->inequalities_linearised(var, p, constraint.template tail<NUM_INEQ>(), jacobian.bottomRows(NUM_INEQ)); // why???
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::constraints_linearised(const Eigen::Ref<const nlp_variable_t> &var,
                                                                         const Eigen::Ref<const static_parameter_t> &p,
                                                                         Eigen::Ref<nlp_constraints_t> constraint,
                                                                         nlp_jacobian_t &jacobian) noexcept
{
    this->equalities_linearised(var, p, constraint.template head<NUM_EQ>(), m_Je);
    this->inequalities_linearised(var, p, constraint.template tail<NUM_INEQ>(), m_Ji);

    // check if we need to allocate memory (first function entry)
    if(jacobian.nonZeros() != (m_jac_inner_nnz.sum() + m_ineq_jac_inner_nnz.sum()) )
    {
        jacobian.resize(NUM_EQ + NUM_INEQ, VAR_SIZE);
        jacobian.reserve(m_jac_inner_nnz + m_ineq_jac_inner_nnz);
        block_insert_sparse(jacobian, 0, 0, m_Je);
        block_insert_sparse(jacobian, NUM_EQ, 0, m_Ji);
    }
    else
    {
        // copy Je and Ji blocks to jac_g
        if(NUM_EQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Je.valuePtr() + m_Je.outerIndexPtr()[k], m_Je.innerNonZeroPtr()[k], jacobian, k, 0);

        if(NUM_INEQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Ji.valuePtr() + m_Ji.outerIndexPtr()[k], m_Ji.innerNonZeroPtr()[k],
                              jacobian, k, m_Je.innerNonZeroPtr()[k]);
    }
}


/** cost computation */
template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::cost(const Eigen::Ref<const nlp_variable_t>& var,
                                             const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept
{
    cost = scalar_t(0);
    scalar_t cost_i = scalar_t(0);
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    int t = 0;
    /** not generic for other collocation schemes : redo*/
    for (int s = 0; s < NUM_SEGMENTS; s++)
    {
        int shift = s * POLY_ORDER;
        t = 0;
        for(int k = 0; k < POLY_ORDER + 1; k++ )
        {
            lagrange_term<scalar_t>(var.template segment<NX>((k + shift) * NX), var.template segment<NU>((k + shift) * NU + VARX_SIZE),
                                    var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(t + shift), cost_i);
            cost += t_scale * m_quad_weights(k) * cost_i;
            ++t;
        }
    }

    cost_i = scalar_t(0);
    mayer_term<scalar_t>(var.template segment<NX>(VARX_SIZE - NX), var.template segment<NU>(VARX_SIZE + VARU_SIZE - NU),
                         var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(NUM_NODES - 1), cost_i);
    cost += cost_i;
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::cost_gradient(const Eigen::Ref<const nlp_variable_t> &var,
                                                                    const Eigen::Ref<const static_parameter_t> &p,
                                                                    scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
{
    cost = scalar_t(0);
    cost_gradient.setZero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    int t = 0;
    /** not generic for other collocation schemes : redo*/
    m_ad_p = var.template segment<NP>(VARX_SIZE + VARU_SIZE);
    m_ad_cost.value() = 0;
    for (int s = 0; s < NUM_SEGMENTS; s++)
    {
        int shift = s * POLY_ORDER;
        t = 0;
        for(int k = 0; k < POLY_ORDER + 1; k++ )
        {
            m_ad_x = var.template segment<NX>((k + shift) * NX);
            m_ad_u = var.template segment<NU>((k + shift) * NU + VARX_SIZE);
            lagrange_term<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(t + shift), m_ad_cost);
            cost += t_scale * m_quad_weights(k) * m_ad_cost.value();
            cost_gradient. template segment<NX>((k + shift) * NX).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad_cost.derivatives(). template head<NX>();
            cost_gradient. template segment<NU>((k + shift) * NU + VARX_SIZE).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad_cost.derivatives(). template segment<NU>(NX);
            cost_gradient. template tail<NP>().noalias() +=
                    t_scale * m_quad_weights(k) * m_ad_cost.derivatives(). template tail<NP>();
            ++t;
        }
    }

    m_ad_cost = ad_scalar_t(0);
    m_ad_x = var.template segment<NX>(VARX_SIZE - NX);
    m_ad_u = var.template segment<NU>(VARX_SIZE + VARU_SIZE - NU);
    mayer_term<ad_scalar_t>(m_ad_x, m_ad_u, m_ad_p, p, time_nodes(NUM_NODES - 1), m_ad_cost);
    cost += m_ad_cost.value();
    cost_gradient. template segment<NX>(VARX_SIZE - NX).noalias() += m_ad_cost.derivatives().template head<NX>();
    cost_gradient. template segment<NU>(VARX_SIZE + VARU_SIZE - NU).noalias() += m_ad_cost.derivatives(). template segment<NU>(NX);
    cost_gradient. template tail<NP>().noalias() += m_ad_cost.derivatives(). template tail<NP>();
}



template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                            const Eigen::Ref<const static_parameter_t>& p,
                                                                            scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                            Eigen::Ref<nlp_hessian_t> cost_hessian) noexcept
{
    cost = scalar_t(0);
    cost_gradient.setZero();
    cost_hessian.setZero();

    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> ::Zero();

    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    int t = 0;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    m_ad2_cost.value().value() = 0;

    for (int s = 0; s < NUM_SEGMENTS; s++)
    {
        int shift = s * POLY_ORDER;
        t = 0;
        for(int k = 0; k < POLY_ORDER + 1; k++ )
        {
            // set x values
            for(int i = 0; i < NX; i++)
                m_ad2_x(i).value().value() = var.template segment<NX>((k + shift) * NX)(i);
            // set u values
            for(int i = 0; i < NU; i++)
                m_ad2_u(i).value().value() = var.template segment<NU>((k + shift) * NU + VARX_SIZE)(i);

            lagrange_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(t + shift), m_ad2_cost);
            cost += t_scale * m_quad_weights(k) * m_ad2_cost.value().value();

            cost_gradient. template segment<NX>((k + shift) * NX).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template head<NX>();
            cost_gradient. template segment<NU>((k + shift) * NU + VARX_SIZE).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template segment<NU>(NX);
            cost_gradient. template tail<NP>().noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template tail<NP>();

            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i) = m_ad2_cost.derivatives()(i).derivatives();
            }
            //hes.transposeInPlace();
            scalar_t coeff =  t_scale * m_quad_weights(k);

            cost_hessian.template block<NX, NX>((k + shift) * NX, (k + shift) * NX).noalias() +=
                    coeff * hes.template topLeftCorner<NX, NX>();
            cost_hessian.template block<NX, NU>((k + shift) * NX, (k + shift) * NU + VARX_SIZE).noalias() +=
                    coeff * hes. template block<NX, NU>(0, NX);
            cost_hessian.template block<NU, NX>((k + shift) * NU + VARX_SIZE, (k + shift) * NX).noalias() +=
                    coeff * hes.template block<NU, NX>(NX, 0);
            cost_hessian.template block<NU, NU>((k + shift) * NU + VARX_SIZE, (k + shift) * NU + VARX_SIZE).noalias() +=
                    coeff * hes.template block<NU, NU>(NX, NX);

            cost_hessian.template block<NX, NP>((k + shift) * NX, VARX_SIZE + VARU_SIZE).noalias() +=
                    coeff * hes.template block<NX, NP>(0, NX + NU);
            cost_hessian.template block<NU, NP>((k + shift) * NU + VARX_SIZE, VARX_SIZE + VARU_SIZE).noalias() +=
                    coeff * hes.template block<NU, NP>(NX, NX + NU);
            cost_hessian.template block<NP, NX>(VARX_SIZE + VARU_SIZE, (k + shift) * NX).noalias() +=
                    coeff * hes.template block<NP, NX>(NX + NU, 0);

            cost_hessian.template block<NP, NU>(VARX_SIZE + VARU_SIZE, (k + shift) * NU + VARX_SIZE).noalias() +=
                    coeff * hes.template block<NP, NU>(NX + NU, NX);

            cost_hessian.template bottomRightCorner<NP, NP>().noalias() +=
                    coeff * hes.template bottomRightCorner<NP, NP>();
            ++t;
        }
    }

    /** Mayer term */
    ad2_scalar_t mayer_cost(0);
    for(int i = 0; i < NX; i++)
        m_ad2_x(i).value().value() = var.template segment<NX>(VARX_SIZE - NX)(i);
    for(int i = 0; i < NU; i++)
        m_ad2_u(i).value().value() = var.template segment<NU>(VARX_SIZE + VARU_SIZE - NU)(i);

    mayer_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(NUM_NODES - 1), mayer_cost);
    cost += mayer_cost.value().value();
    cost_gradient. template segment<NX>(VARX_SIZE - NX).noalias() +=  mayer_cost.value().derivatives(). template head<NX>();
    cost_gradient. template segment<NU>(VARX_SIZE + VARU_SIZE - NU).noalias() += mayer_cost.value().derivatives(). template segment<NU>(NX);
    cost_gradient. template tail<NP>().noalias() += mayer_cost.value().derivatives(). template tail<NP>();

    for(int i = 0; i < NX + NU + NP; ++i)
    {
        hes.col(i) = mayer_cost.derivatives()(i).derivatives();
    }

    /** @bug: do we really need this transpose()? */
    //hes.transposeInPlace();

    /** diagonal  */
    cost_hessian.template block<NX, NX>(VARX_SIZE - NX, VARX_SIZE - NX) += hes.template topLeftCorner<NX, NX>();
    cost_hessian.template block<NU, NU>(VARX_SIZE + VARU_SIZE - NU, VARX_SIZE + VARU_SIZE - NU) += hes.template block<NU, NU>(NX, NX);
    cost_hessian.template bottomLeftCorner<NP, NP>() += hes.template bottomLeftCorner<NP, NP>();

    /** dxdu */
    cost_hessian.template block<NX, NU>(VARX_SIZE - NX, VARX_SIZE + VARU_SIZE - NU) += hes.template block<NX, NU>(0, NX);
    cost_hessian.template block<NU, NX>(VARX_SIZE + VARU_SIZE - NU, VARX_SIZE - NX) += hes.template block<NU, NX>(NX, 0);

    /** dxdp */
    cost_hessian.template block<NX, NP>(VARX_SIZE - NX, VARX_SIZE + VARU_SIZE) += hes.template block<NX, NP>(0, NX + NU);
    cost_hessian.template block<NP, NX>(VARX_SIZE + VARU_SIZE, VARX_SIZE - NX) += hes.template block<NP, NX>(NX + NU, 0);

    /** dudp */
    cost_hessian.template block<NU, NP>(VARX_SIZE + VARU_SIZE - NU, VARX_SIZE + VARU_SIZE) += hes.template block<NU, NP>(NX, NX + NU);
    cost_hessian.template block<NP, NU>(VARX_SIZE + VARU_SIZE, VARX_SIZE + VARU_SIZE - NU) += hes.template block<NP, NU>(NX + NU, NX);
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                       const Eigen::Ref<const static_parameter_t>& p,
                                                                       scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                       nlp_hessian_t& cost_hessian) noexcept
{
    if(cost_hessian.nonZeros() != m_hes_inner_nnz.sum())
        _cost_grad_hess_sparse(var, p, cost, cost_gradient, cost_hessian);
    else
        _cost_grad_hess_sparse_update(var, p, cost, cost_gradient, cost_hessian);
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::_cost_grad_hess_sparse(const Eigen::Ref<const nlp_variable_t>& var,
                                                                             const Eigen::Ref<const static_parameter_t>& p,
                                                                             scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                             nlp_hessian_t& cost_hessian) noexcept
{
    eigen_assert(cost_hessian.outerSize() == VAR_SIZE);
    cost = scalar_t(0);
    cost_gradient.setZero();
    cost_hessian.reserve(m_hes_inner_nnz);

    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> ::Zero();
    Eigen::Matrix<scalar_t, NP, NP> hes_pp = Eigen::Matrix<scalar_t, NP, NP>::Zero();

    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    int t = 0;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    m_ad2_cost.value().value() = 0;
    for (int s = 0; s < NUM_SEGMENTS; s++)
    {
        int shift = s * POLY_ORDER;
        t = 0;
        for(int k = 0; k < POLY_ORDER + 1; k++ )
        {
            // set x values
            for(int i = 0; i < NX; i++)
                m_ad2_x(i).value().value() = var.template segment<NX>((k + shift) * NX)(i);
            // set u values
            for(int i = 0; i < NU; i++)
                m_ad2_u(i).value().value() = var.template segment<NU>((k + shift) * NU + VARX_SIZE)(i);

            lagrange_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(t + shift), m_ad2_cost);
            cost += t_scale * m_quad_weights(k) * m_ad2_cost.value().value();

            cost_gradient. template segment<NX>((k + shift) * NX).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template head<NX>();
            cost_gradient. template segment<NU>((k + shift) * NU + VARX_SIZE).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template segment<NU>(NX);
            cost_gradient. template tail<NP>().noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template tail<NP>();

            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i) = m_ad2_cost.derivatives()(i).derivatives();
            }
            //hes.transposeInPlace();
            scalar_t coeff =  t_scale * m_quad_weights(k);
            hes *= coeff;

            hes_pp += hes.template bottomRightCorner<NP, NP>();

            /** complex logic to glue segments */
            if((s > 0) && (t == 0))
            {
                // add values
                /** dx^2 */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data(), NX, cost_hessian, (k + shift) * NX + j, 0);
                }
                /** dxdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data(), NX, cost_hessian, (k + shift) * NU + VARX_SIZE + j, 0);
                }
                /** dxdp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data(), NX, cost_hessian, VARX_SIZE + VARU_SIZE + j, (k + shift) * NX);
                }
                /** du^2 */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data() + NX, NU, cost_hessian, (k + shift) * NU + VARX_SIZE + j, NX);
                }
                /** dudx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data() + NX, NU, cost_hessian, (k + shift) * NX + j, NX);
                }
                /** dudp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data() + NX, NU, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + (k + shift) * NU);
                }
                /** dp^2 */
                /**
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data() + NX + NU, NP, cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);
                }*/

                /** dpdx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data() + NX + NU, NP, cost_hessian, (k + shift) * NX + j, NX + NU);
                }
                /** dpdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data() + NX + NU, NP, cost_hessian, (k + shift) * NU + j + VARX_SIZE, NX + NU);
                }
            } else
            {
                // insert new values
                /** dx^2 */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    for(Eigen::Index r = 0; r < NX; ++r)
                        cost_hessian.insert((k + shift) * NX + r, (k + shift) * NX + j) = hes(r, j);
                }

                /** dxdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    for(Eigen::Index r = 0; r < NX; ++r)
                        cost_hessian.insert((k + shift) * NX + r, (k + shift) * NU + VARX_SIZE + j) = hes(r, j + NX);
                }

                /** dxdp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    for(Eigen::Index r = 0; r < NX; ++r)
                        cost_hessian.insert((k + shift) * NX + r, VARX_SIZE + VARU_SIZE + j) = hes(r, j + NX + NP);
                }

                /** du^2 */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    for(Eigen::Index r = 0; r < NU; ++r)
                        cost_hessian.insert((k + shift) * NU + VARX_SIZE + r, (k + shift) * NU + VARX_SIZE + j) = hes(r + NX, j + NX);
                }

                /** dudx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    for(Eigen::Index r = 0; r < NU; ++r)
                        cost_hessian.insert((k + shift) * NU + VARX_SIZE + r, (k + shift) * NX + j) = hes(r + NX, j);
                }

                /** dudp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    for(Eigen::Index r = 0; r < NU; ++r)
                        cost_hessian.insert((k + shift) * NU + VARX_SIZE + r, VARX_SIZE + VARU_SIZE + j) = hes(r + NX, j + NX + NU);
                }

                /** dpdx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    for(Eigen::Index r = 0; r < NP; ++r)
                        cost_hessian.insert(VARX_SIZE + VARU_SIZE + r, (k + shift) * NX + j) = hes(r + NX + NU, j);
                }

                /** dpdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    for(Eigen::Index r = 0; r < NP; ++r)
                        cost_hessian.insert(VARX_SIZE + VARU_SIZE + r, (k + shift) * NU + VARX_SIZE + j) = hes(r + NX + NU, j + NU);
                }

            }

            ++t;
        }
    }

    // Insert the last block
    /** dp^2 */
    for(Eigen::Index j = 0; j < NP; ++j)
        for(Eigen::Index r = 0; r < NP; ++r)
            cost_hessian.insert(VARX_SIZE + VARU_SIZE + r, VARX_SIZE + VARU_SIZE + j) = hes_pp(r, j);

    /** Mayer term */
    ad2_scalar_t mayer_cost(0);
    for(int i = 0; i < NX; i++)
        m_ad2_x(i).value().value() = var.template segment<NX>(VARX_SIZE - NX)(i);
    for(int i = 0; i < NU; i++)
        m_ad2_u(i).value().value() = var.template segment<NU>(VARX_SIZE + VARU_SIZE - NU)(i);

    mayer_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(NUM_NODES - 1), mayer_cost);
    cost += mayer_cost.value().value();
    cost_gradient. template segment<NX>(VARX_SIZE - NX).noalias() += mayer_cost.value().derivatives(). template head<NX>();
    cost_gradient. template segment<NU>(VARX_SIZE + VARU_SIZE - NU).noalias() += mayer_cost.value().derivatives(). template segment<NU>(NX);
    cost_gradient. template tail<NP>().noalias() += mayer_cost.value().derivatives(). template tail<NP>();

    for(int i = 0; i < NX + NU + NP; ++i)
    {
        hes.col(i) = mayer_cost.derivatives()(i).derivatives();
    }
    //hes.transposeInPlace();

    // add values
    /** dx^2 */
    for(Eigen::Index j = 0; j < NX; ++j)
    {
        add_n_sparse(hes.col(j).data(), NX, cost_hessian, VARX_SIZE - NX + j, 0);
    }
    /** dxdu */
    for(Eigen::Index j = 0; j < NU; ++j)
    {
        add_n_sparse(hes.col(j + NX).data(), NX, cost_hessian, VARX_SIZE + VARU_SIZE - NU + j, 0);
    }
    /** dxdp */
    for(Eigen::Index j = 0; j < NP; ++j)
    {
        add_n_sparse(hes.col(j + NX + NU).data(), NX, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE - NX);
    }
    /** du^2 */
    for(Eigen::Index j = 0; j < NU; ++j)
    {
        add_n_sparse(hes.col(j + NX).data(), NU, cost_hessian, VARX_SIZE + VARU_SIZE - NU + j, NX);
    }
    /** dudx */
    for(Eigen::Index j = 0; j < NX; ++j)
    {
        add_n_sparse(hes.col(j).data() + NX, NU, cost_hessian, VARX_SIZE - NX + j, NX);
    }
    /** dudp */
    for(Eigen::Index j = 0; j < NP; ++j)
    {
        add_n_sparse(hes.col(j + NX + NU).data() + NX, NU, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + VARU_SIZE - NU);
    }
    /** dp^2 */
    for(Eigen::Index j = 0; j < NP; ++j)
    {
        add_n_sparse(hes.col(j + NX + NU).data() + NX + NU, NP, cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);
    }
    /** dpdx */
    for(Eigen::Index j = 0; j < NX; ++j)
    {
        add_n_sparse(hes.col(j).data() + NX + NU, NP, cost_hessian, VARX_SIZE - NX + j, NX + NU);
    }
    /** dpdu */
    for(Eigen::Index j = 0; j < NU; ++j)
    {
        add_n_sparse(hes.col(j + NX).data() + NX + NU, NP, cost_hessian, VARX_SIZE + VARU_SIZE - NU + j, NX + NU);
    }
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::_cost_grad_hess_sparse_update(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                                                    scalar_t &cost, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                                    nlp_hessian_t& cost_hessian) noexcept
{
    eigen_assert(cost_hessian.outerSize() == VAR_SIZE);
    cost = scalar_t(0);
    cost_gradient.setZero();

    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> ::Zero();
    Eigen::Matrix<scalar_t, NP, NP> hes_pp = Eigen::Matrix<scalar_t, NP, NP>::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    int t = 0;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    m_ad2_cost.value().value() = 0;
    for (int s = 0; s < NUM_SEGMENTS; s++)
    {
        int shift = s * POLY_ORDER;
        t = 0;
        for(int k = 0; k < POLY_ORDER + 1; k++ )
        {
            // set x values
            for(int i = 0; i < NX; i++)
                m_ad2_x(i).value().value() = var.template segment<NX>((k + shift) * NX)(i);
            // set u values
            for(int i = 0; i < NU; i++)
                m_ad2_u(i).value().value() = var.template segment<NU>((k + shift) * NU + VARX_SIZE)(i);

            lagrange_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(t + shift), m_ad2_cost);
            cost += t_scale * m_quad_weights(k) * m_ad2_cost.value().value();

            cost_gradient. template segment<NX>((k + shift) * NX).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template head<NX>();
            cost_gradient. template segment<NU>((k + shift) * NU + VARX_SIZE).noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template segment<NU>(NX);
            cost_gradient. template tail<NP>().noalias() +=
                    t_scale * m_quad_weights(k) * m_ad2_cost.value().derivatives(). template tail<NP>();

            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i) = m_ad2_cost.derivatives()(i).derivatives();
            }
            //hes.transposeInPlace();
            scalar_t coeff =  t_scale * m_quad_weights(k);
            hes *= coeff;
            hes_pp += hes.template bottomRightCorner<NP, NP>();

            /** complex condition to glue segments */
            if((s > 0) && (t == 0))
            {
                // add values
                /** dx^2 */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data(), NX, cost_hessian, (k + shift) * NX + j, 0);
                }
                /** dxdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data(), NX, cost_hessian, (k + shift) * NU + VARX_SIZE + j, 0);
                }
                /** dxdp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data(), NX, cost_hessian, VARX_SIZE + VARU_SIZE + j, (k + shift) * NX);
                }
                /** du^2 */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data() + NX, NU, cost_hessian, (k + shift) * NU + VARX_SIZE + j, NX);
                }
                /** dudx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data() + NX, NU, cost_hessian, (k + shift) * NX + j, NX);
                }
                /** dudp */
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data() + NX, NU, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + (k + shift) * NU);
                }
                /** dp^2 */
                /**
                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    add_n_sparse(hes.col(j + NX + NU).data() + NX + NU, NP, cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);
                } */
                /** dpdx */
                for(Eigen::Index j = 0; j < NX; ++j)
                {
                    add_n_sparse(hes.col(j).data() + NX + NU, NP, cost_hessian, (k + shift) * NX + j, NX + NU);
                }
                /** dpdu */
                for(Eigen::Index j = 0; j < NU; ++j)
                {
                    add_n_sparse(hes.col(j + NX).data() + NX + NU, NP, cost_hessian, (k + shift) * NU + j + VARX_SIZE, NX + NU);
                }
            } else
            {
                /** copy content by columns where possible*/
                for (Eigen::Index j = 0; j < NX; ++j)
                    copy_n_sparse(hes.col(j).data(), NX + NU + NP, cost_hessian, (k + shift) * NX + j, 0);

                for (Eigen::Index j = 0; j < NU; ++j)
                    copy_n_sparse(hes.col(j + NX).data(), NX + NU + NP,
                                  cost_hessian, (k + shift) * NU + VARX_SIZE + j, 0);

                for(Eigen::Index j = 0; j < NP; ++j)
                {
                    copy_n_sparse(hes.col(j + NX + NU).data(), NX,
                                  cost_hessian, VARU_SIZE + VARX_SIZE + j, (k + shift) * NX);
                    copy_n_sparse(hes.col(j + NX + NU).data() + NX, NU,
                                  cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + (k + shift) * NU);
                }

            }

            ++t;
        }
    }

    // copy pp portion
    for(Eigen::Index j = 0; j < NP; ++j)
        copy_n_sparse(hes_pp.col(j).data(), NP,
                      cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);

    /** Mayer term */
    ad2_scalar_t mayer_cost(0);
    for(int i = 0; i < NX; i++)
        m_ad2_x(i).value().value() = var.template segment<NX>(VARX_SIZE - NX)(i);
    for(int i = 0; i < NU; i++)
        m_ad2_u(i).value().value() = var.template segment<NU>(VARX_SIZE + VARU_SIZE - NU)(i);

    mayer_term<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(NUM_NODES - 1), mayer_cost);
    cost += mayer_cost.value().value();
    cost_gradient. template segment<NX>(VARX_SIZE - NX).noalias() +=  mayer_cost.value().derivatives(). template head<NX>();
    cost_gradient. template segment<NU>(VARX_SIZE + VARU_SIZE - NU).noalias() += mayer_cost.value().derivatives(). template segment<NU>(NX);
    cost_gradient. template tail<NP>().noalias() += mayer_cost.value().derivatives(). template tail<NP>();

    for(int i = 0; i < NX + NU + NP; ++i)
    {
        hes.col(i) = mayer_cost.derivatives()(i).derivatives();
    }
    //hes.transposeInPlace();

    // add values by columns when possible
    for(Eigen::Index j = 0; j < NX; ++j)
    {
        add_n_sparse(hes.col(j).data(), NX + NU + NP, cost_hessian, VARX_SIZE - NX + j, 0);
    }

    for(Eigen::Index j = 0; j < NU; ++j)
    {
        add_n_sparse(hes.col(j + NX).data(), NX + NU + NP, cost_hessian, VARX_SIZE + VARU_SIZE - NU + j, 0);
    }


    for(Eigen::Index j = 0; j < NP; ++j)
    {
        /** dxdp */
        add_n_sparse(hes.col(j + NX + NU).data(), NX, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE - NX);

        /** dudp */
        add_n_sparse(hes.col(j + NX + NU).data() + NX, NU, cost_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + VARU_SIZE - NU);

        /** dp^2 */
        add_n_sparse(hes.col(j + NX + NU).data() + NX + NU, NP, cost_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);
    }

}


template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) noexcept
{
    /** @bug: Lagrangian is computed wrongly - fix */
    nlp_eq_constraints_t c;
    nlp_ineq_constraints_t g;
    this->cost(var, p, _lagrangian);
    this->equalities(var, p, c);
    this->inequalities(var, p, g);
    _lagrangian += c.dot(lam.template head<NUM_EQ>()) + g.dot(lam.template segment<NUM_INEQ>(NUM_EQ)) + var.dot(lam.template tail<NUM_BOX>());
    /** @note: Lagrangian here is incorrect: since we're missing [(lam-)' * lbg + lam+ * ubg]. In general
     * we do not need Lagrangian itself for optimisation itself, so this function can be safely skipped (optimise later)*/
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                 Eigen::Ref<nlp_constraints_t> g) noexcept
{
    /** @bug: Lagrangian is computed wrongly - fix */
    this->cost(var, p, _lagrangian);
    this->equalities(var, p, g.template head<NUM_EQ>());
    this->inequalities(var, p, g.template tail<NUM_INEQ>());
    _lagrangian += g.dot(lam.template head<NUM_EQ + NUM_INEQ>()) + var.dot(lam.template tail<NUM_BOX>());
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient(const Eigen::Ref<const nlp_variable_t> &var,
                                                                          const Eigen::Ref<const static_parameter_t> &p,
                                                                          const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                                          Eigen::Ref<nlp_variable_t> lag_gradient) noexcept
{
    nlp_eq_constraints_t c;
    nlp_ineq_constraints_t g;
    nlp_eq_jacobian_t jac_c;
    nlp_ineq_jacobian_t jac_g;
    this->cost_gradient(var, p, _lagrangian, lag_gradient);
    this->equalities_linerised(var, p, c, jac_c);
    this->inequalities_linearised(var, p, g, jac_g);
    //_lagrangian += c.dot(lam.template head<NUM_EQ>()); // do not compute at all??
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() += jac_c.transpose() * lam.template head<NUM_EQ>();
    lag_gradient.noalias() += jac_g.transpose() * lam.template segment<NUM_INEQ>(NUM_EQ);
    lag_gradient += lam.template tail<VAR_SIZE>();
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                                     const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                     Eigen::Ref<nlp_variable_t> lag_gradient,
                                                                     Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                     Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
{
    this->cost_gradient(var, p, _lagrangian, cost_gradient);
    this->equalities_linearised(var, p, g.template head<NUM_EQ>(), jac_g.topRows(NUM_EQ));
    this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), jac_g.bottomRows(NUM_INEQ)); // why???
    //_lagrangian += g.dot(lam.template head<NUM_EQ>());
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                                     const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                     Eigen::Ref<nlp_variable_t> lag_gradient,
                                                                     Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                     Eigen::Ref<nlp_constraints_t> g, nlp_jacobian_t& jac_g) noexcept
{
    this->cost_gradient(var, p, _lagrangian, cost_gradient);
    this->equalities_linearised(var, p, g.template head<NUM_EQ>(), m_Je);
    this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), m_Ji); // why???

    // check if we need to allocate memory (first function entry)
    if(jac_g.nonZeros() != (m_jac_inner_nnz.sum() + m_ineq_jac_inner_nnz.sum()) )
    {
        jac_g.resize(NUM_EQ + NUM_INEQ, VAR_SIZE);
        jac_g.reserve(m_jac_inner_nnz + m_ineq_jac_inner_nnz);
        block_insert_sparse(jac_g, 0, 0, m_Je);
        block_insert_sparse(jac_g, NUM_EQ, 0, m_Ji);
    }
    else
    {
        // copy Je and Ji blocks to jac_g
        if(NUM_EQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Je.valuePtr() + m_Je.outerIndexPtr()[k], m_Je.innerNonZeroPtr()[k], jac_g, k, 0);

        if(NUM_INEQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Ji.valuePtr() + m_Ji.outerIndexPtr()[k], m_Ji.innerNonZeroPtr()[k],
                              jac_g, k, m_Je.innerNonZeroPtr()[k]);
    }

    //_lagrangian += g.dot(lam.template head<NUM_EQ>());
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();
}

template<typename OCP, typename Approximation, int MatrixFormat>
void ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                                    const Eigen::Ref<const nlp_dual_t>& lam,
                                                                    scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                                                    Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept
{
    nlp_eq_constraints_t c;
    nlp_ineq_constraints_t g;
    nlp_eq_jacobian_t jac_c;
    nlp_ineq_jacobian_t jac_g;
    this->cost_gradient_hessian(var, p, _lagrangian, lag_gradient, lag_hessian);
    this->equalities_linerised(var, p, c, jac_c);
    this->inequalities_linearised(var, p, g, jac_g);
    //_lagrangian += c.dot(lam.template head<NUM_EQ>()); // do not compute at all??
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() += jac_c.transpose() * lam.template head<NUM_EQ>();
    lag_gradient.noalias() += jac_g.transpose() * lam.template segment<NUM_INEQ>(NUM_EQ);
    lag_gradient += lam.template tail<VAR_SIZE>();

    /** hessian part */
    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP>::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    Eigen::Matrix<ad2_scalar_t, NX, 1> ad2_xdot;
    Eigen::Matrix<ad2_scalar_t, NG, 1> ad2_g;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    for(int k = 0; k < NUM_NODES; k++)
    {
        for(int i = 0; i < NX; i++)
            m_ad2_x(i).value().value() = var.template segment<NX>(k * NX)(i);
        for(int i = 0; i < NU; i++)
            m_ad2_u(i).value().value() = var.template segment<NU>(k * NU + VARX_SIZE)(i);

        // hessian accumulator
        hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP>::Zero();

        //dynamics contribution
        dynamics<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, static_cast<ad2_scalar_t>(time_nodes(k)), ad2_xdot);

        for(int n = 0; n < NX; n++)
        {
            scalar_t coeff = -lam(n + k * NX) * t_scale;
            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i).noalias() += coeff * ad2_xdot(n).derivatives()(i).derivatives();
            }
        }

        // constraints contribution
        inequality_constraints<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, static_cast<ad2_scalar_t>(time_nodes(k)), ad2_g);
        for(int n = 0; n < NG; n++)
        {
            scalar_t coeff = lam(n + k * NG + NUM_EQ);
            for(int i = 0; i < NX + NU + NP; ++i)
                hes.col(i).noalias() += coeff * ad2_g(n).derivatives()(i).derivatives();
        }
        //hes.transposeInPlace(); // if 2nd derivative is not continuous

        /** append Lagrangian Hessian */
        lag_hessian.template block<NX, NX>(k * NX, k * NX) += hes.template topLeftCorner<NX, NX>();
        lag_hessian.template block<NU, NU>(k * NU + VARX_SIZE, k * NU + VARX_SIZE) += hes.template block<NU, NU>(NX, NX);
        lag_hessian.template bottomRightCorner<NP, NP>() += hes.template bottomRightCorner<NP, NP>();

        lag_hessian.template block<NX, NU>(k * NX, k * NU + VARX_SIZE) += hes. template block<NX, NU>(0, NX);
        lag_hessian.template block<NU, NX>(k * NU + VARX_SIZE, k * NX) += hes.template block<NU, NX>(NX, 0);

        lag_hessian.template block<NX, NP>(k * NX, VARX_SIZE + VARU_SIZE) += hes.template block<NX, NP>(0, NX + NU);
        lag_hessian.template block<NP, NX>(VARX_SIZE + VARU_SIZE, k * NX) += hes.template block<NP, NX>(NX + NU, 0);

        lag_hessian.template block<NU, NP>(k * NU + VARX_SIZE, VARX_SIZE + VARU_SIZE) += hes.template block<NU, NP>(NX, NX + NU);
        lag_hessian.template block<NP, NU>(VARX_SIZE + VARU_SIZE, k * NU + VARX_SIZE) += hes.template block<NP, NU>(NX + NU, NX);
    }
}


template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == DENSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var,
                                                                             const Eigen::Ref<const static_parameter_t> &p,
                                 const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient,
                                 Eigen::Ref<nlp_hessian_t> lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                 Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g, const scalar_t& cost_scale) noexcept
{
    this->cost_gradient_hessian(var, p, _lagrangian, cost_gradient, lag_hessian);
    this->equalities_linearised(var, p, g.template head<NUM_EQ>(), jac_g.topRows(NUM_EQ));
    this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), jac_g.bottomRows(NUM_INEQ));
    //_lagrangian += c.dot(lam.template head<NUM_EQ>()); // do not compute at all??

    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();

    /** hessian part */
    if(cost_scale != scalar_t(1.0))
        lag_hessian.noalias() = cost_scale * lag_hessian;

    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP>::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    Eigen::Matrix<ad2_scalar_t, NX, 1> ad2_xdot;
    Eigen::Matrix<ad2_scalar_t, NG, 1> ad2_g;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    for(int k = 0; k < NUM_NODES; k++)
    {
        for(int i = 0; i < NX; i++)
            m_ad2_x(i).value().value() = var.template segment<NX>(k * NX)(i);
        for(int i = 0; i < NU; i++)
            m_ad2_u(i).value().value() = var.template segment<NU>(k * NU + VARX_SIZE)(i);

        // hessian accumulator
        hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP>::Zero();

        //dynamics contribution
        dynamics<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, static_cast<ad2_scalar_t>(time_nodes(k)), ad2_xdot);

        for(int n = 0; n < NX; n++)
        {
            scalar_t coeff = -lam(n + k * NX) * t_scale;
            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i).noalias() += coeff * ad2_xdot(n).derivatives()(i).derivatives();
            }
        }

        // constraints contribution
        inequality_constraints<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(k), ad2_g);
        for(int n = 0; n < NG; n++)
        {
            scalar_t coeff = lam(n + k * NG + NUM_EQ);
            for(int i = 0; i < NX + NU + NP; ++i)
                hes.col(i).noalias() += coeff * ad2_g(n).derivatives()(i).derivatives();
        }
        //hes.transposeInPlace(); // if 2nd derivative is not continuous

        /** append Lagrangian Hessian */
        lag_hessian.template block<NX, NX>(k * NX, k * NX) += hes.template topLeftCorner<NX, NX>();
        lag_hessian.template block<NU, NU>(k * NU + VARX_SIZE, k * NU + VARX_SIZE) += hes.template block<NU, NU>(NX, NX);
        lag_hessian.template bottomRightCorner<NP, NP>() += hes.template bottomRightCorner<NP, NP>();

        lag_hessian.template block<NX, NU>(k * NX, k * NU + VARX_SIZE) += hes. template block<NX, NU>(0, NX);
        lag_hessian.template block<NU, NX>(k * NU + VARX_SIZE, k * NX) += hes.template block<NU, NX>(NX, 0);

        lag_hessian.template block<NX, NP>(k * NX, VARX_SIZE + VARU_SIZE) += hes.template block<NX, NP>(0, NX + NU);
        lag_hessian.template block<NP, NX>(VARX_SIZE + VARU_SIZE, k * NX) += hes.template block<NP, NX>(NX + NU, 0);

        lag_hessian.template block<NU, NP>(k * NU + VARX_SIZE, VARX_SIZE + VARU_SIZE) += hes.template block<NU, NP>(NX, NX + NU);
        lag_hessian.template block<NP, NU>(VARX_SIZE + VARU_SIZE, k * NU + VARX_SIZE) += hes.template block<NP, NU>(NX + NU, NX);
    }
}

template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                             Eigen::Ref<nlp_variable_t> lag_gradient,
                                                                             nlp_hessian_t& lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                             Eigen::Ref<nlp_constraints_t> g, nlp_jacobian_t &jac_g,
                                                                             const scalar_t& cost_scale) noexcept
{
    this->cost_gradient_hessian(var, p, _lagrangian, cost_gradient, lag_hessian);
    this->equalities_linearised(var, p, g.template head<NUM_EQ>(), m_Je);
    this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), m_Ji);

    // check if we need to allocate memory (first function entry)
    if(jac_g.nonZeros() != (m_jac_inner_nnz.sum() + m_ineq_jac_inner_nnz.sum()) )
    {
        jac_g.resize(NUM_EQ + NUM_INEQ, VAR_SIZE);
        jac_g.reserve(m_jac_inner_nnz + m_ineq_jac_inner_nnz);
        block_insert_sparse(jac_g, 0, 0, m_Je);
        block_insert_sparse(jac_g, NUM_EQ, 0, m_Ji);
    }
    else
    {
        // copy Je and Ji blocks to jac_g
        if(NUM_EQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Je.valuePtr() + m_Je.outerIndexPtr()[k], m_Je.innerNonZeroPtr()[k], jac_g, k, 0);

        if(NUM_INEQ > 0)
            for(Eigen::Index k = 0; k < VAR_SIZE; ++k)
                copy_n_sparse(m_Ji.valuePtr() + m_Ji.outerIndexPtr()[k], m_Ji.innerNonZeroPtr()[k],
                              jac_g, k, m_Je.innerNonZeroPtr()[k]);
    }


    //_lagrangian += c.dot(lam.template head<NUM_EQ>()); // do not compute at all??
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();

    /** hessian part */
    if(cost_scale != scalar_t(1.0))
        lag_hessian = cost_scale * lag_hessian;

    Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP> ::Zero();
    const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);
    Eigen::Matrix<ad2_scalar_t, NX, 1> ad2_xdot;
    Eigen::Matrix<ad2_scalar_t, NG, 1> ad2_g;

    for(int i = 0; i < NP; i++)
        m_ad2_p(i).value().value() = var.template tail<NP>()(i);

    for(int k = 0; k < NUM_NODES; k++)
    {
        for(int i = 0; i < NX; i++)
            m_ad2_x(i).value().value() = var.template segment<NX>(k * NX)(i);
        for(int i = 0; i < NU; i++)
            m_ad2_u(i).value().value() = var.template segment<NU>(k * NU + VARX_SIZE)(i);

        // hessian accumulator
        hes = Eigen::Matrix<scalar_t, NX + NU + NP, NX + NU + NP>::Zero();

        //dynamics contribution
        dynamics<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, static_cast<ad2_scalar_t>(time_nodes(k)), ad2_xdot);

        for(int n = 0; n < NX; n++)
        {
            scalar_t coeff = -lam(n + k * NX) * t_scale;
            for(int i = 0; i < NX + NU + NP; ++i)
            {
                hes.col(i).noalias() += coeff * ad2_xdot(n).derivatives()(i).derivatives();
            }
        }

        // constraints contribution
        inequality_constraints<ad2_scalar_t>(m_ad2_x, m_ad2_u, m_ad2_p, p, time_nodes(k), ad2_g);
        for(int n = 0; n < NG; n++)
        {
            scalar_t coeff = lam(n + k * NG + NUM_EQ);
            for(int i = 0; i < NX + NU + NP; ++i)
                hes.col(i).noalias() += coeff * ad2_g(n).derivatives()(i).derivatives();
        }
        //hes.transposeInPlace(); // if 2nd derivative is not continuous

        // add values by columns when possible
        for(Eigen::Index j = 0; j < NX; ++j)
        {
            add_n_sparse(hes.col(j).data(), NX + NU + NP, lag_hessian, k * NX + j, 0);
        }

        for(Eigen::Index j = 0; j < NU; ++j)
        {
            add_n_sparse(hes.col(j + NX).data(), NX + NU + NP, lag_hessian, k * NU + VARX_SIZE + j, 0);
        }

        for(Eigen::Index j = 0; j < NP; ++j)
        {
            /** dxdp */
            add_n_sparse(hes.col(j + NX + NU).data(), NX, lag_hessian, VARX_SIZE + VARU_SIZE + j, k * NX);

            /** dudp */
            add_n_sparse(hes.col(j + NX + NU).data() + NX, NU, lag_hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + k * NU);

            /** dp^2 */
            add_n_sparse(hes.col(j + NX + NU).data() + NX + NU, NP, lag_hessian, VARU_SIZE + VARX_SIZE + j, VARX_SIZE + VARU_SIZE);
        }

    }
}

// sparsity preserving block BFGS update
template<typename OCP, typename Approximation, int MatrixFormat>
template<int T>
typename std::enable_if<T == SPARSE>::type
ContinuousOCP<OCP, Approximation, MatrixFormat>::hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t> s,
                                                                     const Eigen::Ref<const nlp_variable_t> y) const noexcept
{
    //std::cout << "s: " << s.transpose() << "\n";
    //std::cout << "y: " << y.transpose() << "\n";

    const nlp_variable_t v = hessian * s;
    const scalar_t scaling = s.dot(v);
    const scalar_t scaling_inv = scalar_t(1) / scaling;
    const scalar_t sy  = s.dot(y);
    const scalar_t sy_inv = scalar_t(1) / sy;
    nlp_variable_t r;

    // create temporaries to store
    Eigen::Matrix<scalar_t, NX, NX> hes_xx;
    Eigen::Matrix<scalar_t, NU, NU> hes_uu;
    Eigen::Matrix<scalar_t, NU, NX> hes_ux;
    Eigen::Matrix<scalar_t, NX, NU> hes_xu;

    /** first rank update */
    for(Eigen::Index k = 0; k < NUM_NODES; ++k)
    {
        hes_xx.noalias() = -scaling_inv * v.template segment<NX>(k * NX) * v.template segment<NX>(k * NX).transpose();
        hes_uu.noalias() = -scaling_inv * v.template segment<NU>(k * NU + VARX_SIZE) * v.template segment<NU>(k * NU + VARX_SIZE).transpose();
        hes_ux.noalias() = -scaling_inv * v.template segment<NU>(k * NU + VARX_SIZE) * v.template segment<NX>(k * NX).transpose();

        //damping
        if(sy >= 0.2 * scaling)
        {
            hes_xx.noalias() += sy_inv * y.template segment<NX>(k * NX) * y.template segment<NX>(k * NX).transpose();
            hes_uu.noalias() += sy_inv * y.template segment<NU>(k * NU + VARX_SIZE) * y.template segment<NU>(k * NU + VARX_SIZE).transpose();
            hes_ux.noalias() += sy_inv * y.template segment<NU>(k * NU + VARX_SIZE) * y.template segment<NX>(k * NX).transpose();
        }
        else
        {
            const scalar_t theta = 0.8 * scaling / (scaling - sy);
            r.noalias() = theta * y + (1 - theta) * v;
            const scalar_t sr_inv = scalar_t(1) / (s.dot(r));

            hes_xx.noalias() += sr_inv * r.template segment<NX>(k * NX) * r.template segment<NX>(k * NX).transpose();
            hes_uu.noalias() += sr_inv * r.template segment<NU>(k * NU + VARX_SIZE) * r.template segment<NU>(k * NU + VARX_SIZE).transpose();
            hes_ux.noalias() += sr_inv * r.template segment<NU>(k * NU + VARX_SIZE) * r.template segment<NX>(k * NX).transpose();
        }

        hes_xu = hes_ux.transpose();

        // add values by columns when possible
        for(Eigen::Index j = 0; j < NX; ++j)
        {
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j],
                           hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j] + NX,
                           hes_xx.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j],
                           std::plus<scalar_t>());

            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j] + NX,
                           hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j] + NX + NU,
                           hes_ux.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[k * NX + j] + NX,
                           std::plus<scalar_t>());

//            add_n_sparse(hes_xx.col(j).data(), NX, hessian, k * NX + j, 0);
//            add_n_sparse(hes_ux.col(j).data(), NU, hessian, k * NX + j, NX);
        }

        for(Eigen::Index j = 0; j < NU; ++j)
        {
            // add_n_sparse(hes_xu.col(j).data(), NX, hessian, k * NU + VARX_SIZE + j, 0);
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j],
                           hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j] + NX,
                           hes_xu.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j],
                           std::plus<scalar_t>());

            // add_n_sparse(hes_uu.col(j).data(), NU, hessian, k * NU + VARX_SIZE + j, NX);
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j] + NX,
                           hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j] + NX + NU,
                           hes_uu.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[k * NU + VARX_SIZE + j] + NX,
                           std::plus<scalar_t>());
        }
    }

    /** update parameter derivatives */
    if(NP > 0)
    {
        Eigen::Matrix<scalar_t, NP, NP> hes_pp;
        const int np_size = NP > 0 ? NP : 1; // workaround to avoid Eigen static assert triggering
        Eigen::Matrix<scalar_t, VARX_SIZE + VARU_SIZE, np_size> hes_ap;

        hes_pp.noalias() = -scaling_inv * v.template tail<NP>() * v.template tail<NP>().transpose();
        hes_ap.noalias() = -scaling_inv * v.template head<VARX_SIZE + VARU_SIZE>() * v.template tail<np_size>().transpose();

        // damping
        if(sy >= 0.2 * scaling)
        {
            hes_pp.noalias() += sy_inv * y.template tail<NP>() * y.template tail<NP>().transpose();
            hes_ap.noalias() += sy_inv * y.template head<VARX_SIZE + VARU_SIZE>() * y.template tail<np_size>().transpose();
        }
        else
        {
            const scalar_t theta = 0.8 * scaling / (scaling - sy);
            r.noalias() = theta * y + (1 - theta) * v;
            const scalar_t sr_inv = scalar_t(1) / (s.dot(r));

            hes_pp.noalias() += sr_inv * r.template tail<NP>() * r.template tail<NP>().transpose();
            hes_ap.noalias() += sr_inv * r.template head<VARX_SIZE + VARU_SIZE>() * r.template tail<np_size>().transpose();
        }


        for(Eigen::Index j = 0; j < NP; ++j)
        {
            // add_n_sparse(hes_ap.col(j).data(), VARX_SIZE + VARU_SIZE, hessian, VARX_SIZE + VARU_SIZE + j, 0);
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j],
                    hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j] + VARX_SIZE + VARU_SIZE,
                    hes_ap.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j],
                    std::plus<scalar_t>());

            // add_n_sparse(hes_pp.col(j).data(), NP, hessian, VARX_SIZE + VARU_SIZE + j, VARX_SIZE + VARU_SIZE);
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j] + VARX_SIZE + VARU_SIZE,
                    hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j] + VARX_SIZE + VARU_SIZE + NP,
                    hes_pp.col(j).data(), hessian.valuePtr() + hessian.outerIndexPtr()[VARX_SIZE + VARU_SIZE + j] + VARX_SIZE + VARU_SIZE,
                    std::plus<scalar_t>());
        }

        Eigen::Matrix<scalar_t, np_size, 1> tmp;
        for(Eigen::Index j = 0; j < VARX_SIZE + VARU_SIZE; ++j)
        {
            tmp = hes_ap.row(j);
            // add_n_sparse(tmp.data(), NP, hessian, j, NX + NU);
            std::transform(hessian.valuePtr() + hessian.outerIndexPtr()[j] + NX + NU,
                           hessian.valuePtr() + hessian.outerIndexPtr()[j] + NX + NU + NP,
                           tmp.data(), hessian.valuePtr() + hessian.outerIndexPtr()[j] + NX + NU,
                           std::plus<scalar_t>());
        }
    }
}



#endif // CONTINUOUS_OCP_HPP
