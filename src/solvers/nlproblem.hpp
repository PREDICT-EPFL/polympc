// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef NLPROBLEM_HPP
#define NLPROBLEM_HPP

#include "utils/helpers.hpp"
#include "autodiff/AutoDiffScalar.h"

#include <iostream>

template<int I, int U>
struct more
{
    static constexpr bool ret = I > U ? true : false;
};


/** define the macro for forward declarations */
#define POLYMPC_FORWARD_NLP_DECLARATION( cNAME, cNX, cNE, cNI, cNP, TYPE ) \
class cNAME;                                        \
template<>                                          \
struct nlp_traits<cNAME>                            \
{                                                   \
public:                                             \
    using Scalar = TYPE;                            \
    enum { NX = cNX, NE = cNE, NI = cNI, NP = cNP}; \
};                                                  \

/** define derived class traits */
template<typename Derived> struct nlp_traits;
template<typename T> struct nlp_traits<const T> : nlp_traits<T> {};

/** forward declare base class */
template<typename Derived, int MatrixFormat> class ProblemBase;

template<typename Derived, int MatrixFormat = DENSE>
class ProblemBase
{
public:

    ProblemBase()
    {
        seed_derivatives();
    }

    ~ProblemBase() = default;


    enum
    {
        /** problem dimensions */
        VAR_SIZE  = nlp_traits<Derived>::NX,
        NUM_EQ    = nlp_traits<Derived>::NE,
        NUM_INEQ  = nlp_traits<Derived>::NI,
        NUM_BOX   = nlp_traits<Derived>::NX,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,

        /** Various flags */
        is_sparse = (MatrixFormat == SPARSE) ? 1 : 0,
        is_dense  = is_sparse ? 0 : 1,
        MATRIXFMT = MatrixFormat
    };

    using scalar_t = typename nlp_traits<Derived>::Scalar;

    /** optimisation variable */
    template<typename T>
    using variable_t = Eigen::Matrix<T, VAR_SIZE, 1>;

    /** constraints types */
    template<typename T>
    using constraint_t = Eigen::Matrix<T, NUM_EQ + NUM_INEQ, 1>;
    template<typename T>
    using eq_constraint_t   = Eigen::Matrix<T, NUM_EQ, 1>;
    template<typename T>
    using ineq_constraint_t = Eigen::Matrix<T, NUM_INEQ, 1>;

    /** parameters */
    template<typename T>
    using parameter_t = Eigen::Matrix<T, nlp_traits<Derived>::NP, 1>;

    /** AD variables */
    using derivatives_t = Eigen::Matrix<scalar_t, VAR_SIZE, 1>;
    using ad_scalar_t   = Eigen::AutoDiffScalar<derivatives_t>;
    using ad_var_t      = Eigen::Matrix<ad_scalar_t, VAR_SIZE, 1>;
    using ad2_scalar_t  = Eigen::AutoDiffScalar<ad_var_t>;
    using ad_eq_t       = Eigen::Matrix<ad_scalar_t, NUM_EQ, 1>;
    using ad_ineq_t     = Eigen::Matrix<ad_scalar_t, NUM_INEQ, 1>;

    ad_var_t  m_ad_var, m_ad_y;
    ad_eq_t   m_ad_eq;
    ad_ineq_t m_ad_ineq;
    ad_scalar_t m_ad_cost;
    Eigen::Matrix<ad2_scalar_t, VAR_SIZE, 1> m_ad2_var;
    ad2_scalar_t m_ad2_cost;

    /** seed edrivatives */
    void seed_derivatives();

    /** NLP variables */
    using nlp_variable_t         = typename dense_matrix_type_selector<scalar_t, VAR_SIZE, 1>::type;
    using nlp_eq_constraints_t   = typename dense_matrix_type_selector<scalar_t, NUM_EQ, 1>::type;
    using nlp_ineq_constraints_t = typename dense_matrix_type_selector<scalar_t, NUM_INEQ, 1>::type;
    using nlp_constraints_t      = typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, 1>::type;
    // choose to allocate sparse or dense jacoabian and hessian
    using nlp_eq_jacobian_t      = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, NUM_EQ, VAR_SIZE>::type>::type;
    using nlp_ineq_jacobian_t    = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, NUM_INEQ, VAR_SIZE>::type>::type;

    using nlp_jacobian_t    = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>::type>::type;

    using nlp_hessian_t     = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, VAR_SIZE, VAR_SIZE>::type>::type;
    using nlp_cost_t        = scalar_t;
    using nlp_dual_t        = typename dense_matrix_type_selector<scalar_t, DUAL_SIZE, 1>::type;
    using static_parameter_t = typename dense_matrix_type_selector<scalar_t, nlp_traits<Derived>::NP, 1>::type;

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        static_cast<const Derived*>(this)->cost_impl(x, p, cost);
    }

    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept
    {
        //this->cost<scalar_t>(var,p,cost);
        static_cast<const Derived*>(this)->cost_impl(var, p, cost);
    }

    // default empty implementation
    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x,
                                       const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept {}

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                  Eigen::Ref<eq_constraint_t<T>> constraint) const noexcept
    {
        static_cast<const Derived*>(this)->equality_constraints_impl(x, p, constraint);
    }

    // default implementation
    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<eq_constraint_t<T>> constraint) const noexcept {}

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                          Eigen::Ref<ineq_constraint_t<T>> constraint) const noexcept
    {
        static_cast<const Derived*>(this)->inequality_constraints_impl(x, p, constraint);
    }

    // default implementation
    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                               Eigen::Ref<ineq_constraint_t<T>> constraint) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(constraint);
    }

    /** interface functions to comply with ipopt*/
    /** @bug: how to generalise for Sparse? */
    EIGEN_STRONG_INLINE int nnz_jacobian() const noexcept
    {
        return (NUM_EQ + NUM_INEQ) * VAR_SIZE;
    }

    // number of nonzeros in lower-triangular part of the Hessian
    EIGEN_STRONG_INLINE int nnz_lag_hessian() const noexcept
    {
        return (VAR_SIZE * (VAR_SIZE + 1)) / 2;
    }

    /**  NLP interface functions */
    /** @brief:
     *
     */
    EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                           scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient) noexcept
    {
        static_cast<Derived*>(this)->cost_gradient_impl(var, p, _cost, _cost_gradient);
    }
    // default implementation
    EIGEN_STRONG_INLINE void cost_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient) noexcept;


    EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                   scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept
    {
        static_cast<Derived*>(this)->cost_gradient_hessian_impl(var, p, _cost, _cost_gradient, hessian);
    }
    // default
    EIGEN_STRONG_INLINE void cost_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                      scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept;

    /** @brief
     *
     */
    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                              Eigen::Ref<nlp_eq_constraints_t> _equalities) const noexcept
    {
        static_cast<const Derived*>(this)->equalities_impl(var, p, _equalities);
    }
    // default
    EIGEN_STRONG_INLINE void equalities_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                             Eigen::Ref<nlp_eq_constraints_t> _equalities) const noexcept;


    EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                Eigen::Ref<nlp_ineq_constraints_t> _inequalities) const noexcept
    {
        static_cast<const Derived*>(this)->inequalities_impl(var, p, _inequalities);
    }
    // default
    EIGEN_STRONG_INLINE void inequalities_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                               Eigen::Ref<nlp_ineq_constraints_t> _inequalities) const noexcept;


    EIGEN_STRONG_INLINE void constraints(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                         Eigen::Ref<nlp_constraints_t> _constraints) const noexcept
    {
        static_cast<const Derived*>(this)->constraints_impl(var, p, _constraints);
    }

    // default
    EIGEN_STRONG_INLINE void constraints_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                              Eigen::Ref<nlp_constraints_t> _constraints) const noexcept;


    /** linearisation of equalities */
    EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                   const Eigen::Ref<const static_parameter_t>& p,
                                                   Eigen::Ref<nlp_eq_constraints_t> equalities,
                                                   Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        static_cast<Derived*>(this)->equalities_linearised_impl(var, p, equalities, jacobian);
    }
    // default
    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE < 1 >::type equalities_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                           const Eigen::Ref<const static_parameter_t>& p,
                                                                                           Eigen::Ref<nlp_eq_constraints_t> equalities,
                                                                                           Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        /** @badcode : remove setting to zero? */
        jacobian   = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
        equalities = nlp_eq_constraints_t::Zero(NUM_EQ);

        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
    }

    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE >= 1 >::type equalities_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                            const Eigen::Ref<const static_parameter_t>& p,
                                                                                            Eigen::Ref<nlp_eq_constraints_t> equalities,
                                                                                            Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        jacobian   = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
        equalities = nlp_eq_constraints_t::Zero(NUM_EQ);

        m_ad_var = var;
        equality_constraints<ad_scalar_t>(m_ad_var, p, m_ad_eq);

        /** compute value and first derivatives */
        for(int i = 0; i < NUM_EQ; i++)
        {
            equalities(i) = m_ad_eq(i).value();
            jacobian.row(i) = m_ad_eq(i).derivatives();
        }
    }

    /** linearisation of inequalities */
    EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                     const Eigen::Ref<const static_parameter_t>& p,
                                                     Eigen::Ref<nlp_ineq_constraints_t> inequalities,
                                                     Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        static_cast<Derived*>(this)->inequalities_linearised_impl(var, p, inequalities, jacobian);
    }

    template<int NI = NUM_INEQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NI < 1 >::type inequalities_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                             const Eigen::Ref<const static_parameter_t>& p,
                                                                                             Eigen::Ref<nlp_ineq_constraints_t> inequalities,
                                                                                             Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        /** @badcode : remove setting to zero? */
        //jacobian   = nlp_eq_jacobian_t::Zero(NUM_INEQ, VAR_SIZE);
        //inequalities = nlp_eq_constraints_t::Zero(NUM_INEQ);

        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
    }

    template<int NI = NUM_INEQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NI >= 1 >::type inequalities_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                              const Eigen::Ref<const static_parameter_t>& p,
                                                                                              Eigen::Ref<nlp_ineq_constraints_t> inequalities,
                                                                                              Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        //jacobian   = nlp_ineq_jacobian_t::Zero(NUM_INEQ, VAR_SIZE);
        //inequalities = nlp_ineq_constraints_t::Zero(NUM_INEQ);

        m_ad_var = var;
        inequality_constraints<ad_scalar_t>(m_ad_var, p, m_ad_ineq);

        // compute value and first derivatives
        for(int i = 0; i < NUM_INEQ; i++)
        {
            inequalities(i) = m_ad_ineq(i).value();
            jacobian.row(i) = m_ad_ineq(i).derivatives();
        }
    }

    EIGEN_STRONG_INLINE void constraints_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                    Eigen::Ref<nlp_constraints_t> constraints,
                                                    Eigen::Ref<nlp_jacobian_t> jacobian) noexcept
    {
        static_cast<Derived*>(this)->constraints_linearised_impl(var, p, constraints, jacobian);
    }

    // default implementation
    EIGEN_STRONG_INLINE void constraints_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                         const Eigen::Ref<const static_parameter_t>& p,
                                                         Eigen::Ref<nlp_constraints_t> constraints,
                                                         Eigen::Ref<nlp_jacobian_t> jacobian) noexcept
    {

        nlp_eq_jacobian_t eq_jac;
        nlp_ineq_jacobian_t ineq_jac;

        if(NUM_EQ > 0)
        {
            this->equalities_linearised(var, p, constraints.template head<NUM_EQ>(), eq_jac);
            jacobian.template topRows<NUM_EQ>() = eq_jac;
        }
        if(NUM_INEQ > 0)
        {
            this->inequalities_linearised(var, p, constraints.template tail<NUM_INEQ>(), ineq_jac);
            jacobian.template bottomRows<NUM_INEQ>() = ineq_jac;
        }
    }


    /** @brief:
     *
     */
    EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                        const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept
    {
        static_cast<const Derived*>(this)->lagrangian_impl(var, p, lam, _lagrangian);
    }
    // default
    EIGEN_STRONG_INLINE void lagrangian_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept;


    /** @brief:
     *
     */
    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept
    {
        static_cast<Derived*>(this)->lagrangian_gradient_impl(var, p, lam, _lagrangian, _lag_gradient);
    }
    // default
    EIGEN_STRONG_INLINE void lagrangian_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                      const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                      Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept;

    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                 Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
    {
        static_cast<Derived*>(this)->lagrangian_gradient_impl(var, p, lam, _lagrangian, lag_gradient, cost_gradient, g, jac_g);
    }
    // default
    EIGEN_STRONG_INLINE void lagrangian_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                      const Eigen::Ref<const static_parameter_t>& p,
                                                      const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                      Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                      Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept;


    /** @brief:
     *
     */
    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p,
                                                         const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                         Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                         Eigen::Ref<nlp_variable_t> cost_gradient,
                                                         Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
    {
        static_cast<Derived*>(this)->lagrangian_gradient_hessian_impl(var, p, lam, _lagrangian, lag_gradient,
                                                                      lag_hessian, cost_gradient, g, jac_g, scalar_t(1.0));
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p,
                                                         const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                         Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                         Eigen::Ref<nlp_variable_t> cost_gradient,
                                                         Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g,
                                                         const scalar_t cost_scale) noexcept
    {
        static_cast<Derived*>(this)->lagrangian_gradient_hessian_impl(var, p, lam, _lagrangian, lag_gradient,
                                                                      lag_hessian, cost_gradient, g, jac_g, cost_scale);
    }
    //default
    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p,
                                                              const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                              Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                              Eigen::Ref<nlp_variable_t> cost_gradient,
                                                              Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g,
                                                              const scalar_t cost_scale) noexcept;


};

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::seed_derivatives()
{
    const int deriv_num = VAR_SIZE;
    int deriv_idx = 0;

    for(int i = 0; i < VAR_SIZE; i++)
    {
        m_ad_var(i).derivatives()  = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_var(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_var(i).value().derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        for(int idx = 0; idx < deriv_num; idx++)
        {
            m_ad2_var(i).derivatives()(idx).derivatives()  = derivatives_t::Zero();
        }
        deriv_idx++;
    }
}

// compute cost gradient
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::cost_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                            scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
{
    _cost = scalar_t(0);
    cost_gradient = nlp_variable_t::Zero(VAR_SIZE);

    m_ad_cost = ad_scalar_t(0);
    m_ad_var = var;
    cost<ad_scalar_t>(m_ad_var, p, m_ad_cost);
    _cost += m_ad_cost.value();
    cost_gradient.noalias() += m_ad_cost.derivatives();
}

//compute cost, gradient and hessian
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::cost_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                    const Eigen::Ref<const static_parameter_t>& p,
                                                                    scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient,
                                                                    Eigen::Ref<nlp_hessian_t> hessian) noexcept
{
    _cost = scalar_t(0);
    _cost_gradient = nlp_variable_t::Zero(VAR_SIZE);
    hessian  = nlp_hessian_t::Zero(VAR_SIZE, VAR_SIZE);
    m_ad2_cost.value().value() = 0;

    // set variable values
    for(int i = 0; i < VAR_SIZE; i++)
        m_ad2_var(i).value().value() = var(i);

    // compute cost, gradient, hessian
    cost<ad2_scalar_t>(m_ad2_var, p, m_ad2_cost);
    _cost = m_ad2_cost.value().value();
    _cost_gradient =  m_ad2_cost.value().derivatives();

    for(int i = 0; i < VAR_SIZE; ++i)
        hessian.col(i) = m_ad2_cost.derivatives()(i).derivatives();
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                         const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept
{
    /** create temporary */
    nlp_eq_constraints_t c;
    nlp_ineq_constraints_t g;
    this->cost(var, p, _lagrangian);
    this->equalities(var, p, c);
    this->inequalities(var, p, g);
    _lagrangian += c.dot(lam.template head<NUM_EQ>()) + g.dot(lam.template segment<NUM_INEQ>(NUM_EQ)) + var.dot(lam.template tail<NUM_BOX>());
    /** @note: Lagrangian here is incorrect: since we're missing [(lam-)' * lbg + lam+ * ubg]. In general
     * we do not need Lagrangian itself for optimisation itself, so this function can be safely skipped (optimise later)*/
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                  const Eigen::Ref<const static_parameter_t>& p,
                                                                  const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                  Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept
{
    nlp_eq_constraints_t c;
    nlp_ineq_constraints_t g;
    nlp_eq_jacobian_t jac_c;
    nlp_ineq_jacobian_t jac_g;
    this->cost_gradient(var, p, _lagrangian, _lag_gradient);
    this->equalities_linerised(var, p, c, jac_c);
    this->inequalities_linearised(var, p, g, jac_g);
    //_lagrangian += c.dot(lam.template head<NUM_EQ>()); // do not compute at all??
    /** @badcode: replace with block products ???*/
    _lag_gradient.noalias() += jac_c.transpose() * lam.template head<NUM_EQ>();
    _lag_gradient.noalias() += jac_g.transpose() * lam.template segment<NUM_INEQ>(NUM_EQ);
    _lag_gradient += lam.template tail<VAR_SIZE>();
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                                  const Eigen::Ref<const static_parameter_t>& p,
                                                                  const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                                  Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                  Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
{
    nlp_eq_jacobian_t eq_jac;
    nlp_ineq_jacobian_t ineq_jac;

    this->cost_gradient(var, p, _lagrangian, cost_gradient);
    //this->equalities_linearised(var, p, g.template head<NUM_EQ>(), jac_g.topRows(NUM_EQ));
    //this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), jac_g.bottomRows(NUM_INEQ)); // why???
    if(NUM_EQ > 0)
    {
        this->equalities_linearised(var, p, g.template head<NUM_EQ>(), eq_jac);
        jac_g.template topRows<NUM_EQ>() = eq_jac;
    }
    if(NUM_INEQ > 0)
    {
        this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), ineq_jac);
        jac_g.template bottomRows<NUM_INEQ>() = ineq_jac;
    }
    //_lagrangian += g.dot(lam.template head<NUM_EQ>());
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t> &var,
                                                                          const Eigen::Ref<const static_parameter_t> &p,
                                                                          const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                                          Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                                          Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                          Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g,
                                                                          const scalar_t cost_scale) noexcept
{
    /** @bug: excessive copying-> do separate implementation for big Jacobian */
    nlp_eq_jacobian_t eq_jac;
    nlp_ineq_jacobian_t ineq_jac;

    this->cost_gradient_hessian(var, p, _lagrangian, cost_gradient, lag_hessian);
    //this->equalities_linearised(var, p, g.template head<NUM_EQ>(), jac_g.topRows(NUM_EQ));
    //this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), jac_g.bottomRows(NUM_INEQ)); // why???
    if(NUM_EQ > 0)
    {
        this->equalities_linearised(var, p, g.template head<NUM_EQ>(), eq_jac);
        jac_g.template topRows<NUM_EQ>() = eq_jac;
    }
    if(NUM_INEQ > 0)
    {
        this->inequalities_linearised(var, p, g.template tail<NUM_INEQ>(), ineq_jac); // why???
        jac_g.template bottomRows<NUM_INEQ>() = ineq_jac;
    }
    //_lagrangian += g.dot(lam.template head<NUM_EQ>());

    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ + NUM_INEQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();

    /** hessian part */
    if(cost_scale != scalar_t(1.0))
        lag_hessian.noalias() = cost_scale * lag_hessian;


    Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE> hes = Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE>::Zero();
    Eigen::Matrix<ad2_scalar_t, NUM_EQ, 1> ad2_eq;
    Eigen::Matrix<ad2_scalar_t, NUM_INEQ, 1> ad2_ineq;

    for(int i = 0; i < VAR_SIZE; i++)
        m_ad2_var(i).value().value() = var(i);

    //process equalities
    equality_constraints<ad2_scalar_t>(m_ad2_var, p, ad2_eq);

    for(int n = 0; n < NUM_EQ; ++n)
    {
        for(int i = 0; i < VAR_SIZE; ++i)
        {
            hes.col(i) = ad2_eq(n).derivatives()(i).derivatives();
        }
        // do we really need it?
        hes.transposeInPlace();

        lag_hessian.noalias() += lam(n) * hes;
    }

    //process inequalities
    inequality_constraints<ad2_scalar_t>(m_ad2_var, p, ad2_ineq);

    //for(int n = NUM_EQ; n < NUM_EQ + NUM_INEQ; ++n)
    for(int n = 0; n < NUM_INEQ; ++n)
    {
        for(int i = 0; i < VAR_SIZE; ++i)
        {
            hes.col(i) = ad2_ineq(n).derivatives()(i).derivatives();
        }
        // do we really need it?
        hes.transposeInPlace();

        lag_hessian.noalias() += lam(n + NUM_EQ) * hes;
    }
}

// evaluate equality constraints
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::equalities_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                         Eigen::Ref<nlp_eq_constraints_t> _equalities) const noexcept
{
    equality_constraints<scalar_t>(var, p, _equalities);
}

// evaluate inequality constraints
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::inequalities_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                           Eigen::Ref<nlp_ineq_constraints_t> _inequalities) const noexcept
{
    inequality_constraints<scalar_t>(var, p, _inequalities);
}


// evaluate generic constraints
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::constraints_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                          Eigen::Ref<nlp_constraints_t> _constraints) const noexcept
{
    equality_constraints<scalar_t>(var, p, _constraints.template head<NUM_EQ>());
    inequality_constraints<scalar_t>(var, p, _constraints.template tail<NUM_INEQ>());
}


#endif // NLPROBLEM_HPP
