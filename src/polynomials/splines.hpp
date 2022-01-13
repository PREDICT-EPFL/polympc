// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPLINES_HPP
#define SPLINES_HPP

#include "Eigen/Core"
#include "unsupported/Eigen/Polynomials"
#include "autodiff/AutoDiffScalar.h"

#include <iostream>

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


/** Automatically differentiable Equidistant Cubic Spline */
template<typename Scalar = double>
class EquidistantCubicSpline
{
public:
    using coeffs_matrix_type = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>;

    EquidistantCubicSpline() = default;
    EquidistantCubicSpline(const coeffs_matrix_type &coeffs, const Scalar &segm_length) : m_coeffs(coeffs), m_length(segm_length) {}
    ~EquidistantCubicSpline() = default;

    // we need two different implementation for first and second derivatives
    template<typename DerType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_scalar<typename DerType::Scalar>::value, int>::type
    floor_(const Eigen::AutoDiffScalar<DerType> &x) const noexcept {return std::floor(x.value());}

    template<typename DerType>
    EIGEN_STRONG_INLINE typename std::enable_if<!std::is_scalar<typename DerType::Scalar>::value, int>::type
    floor_(const Eigen::AutoDiffScalar<DerType> &x) const noexcept {return std::floor(x.value().value());}

    template<typename scalar_t>
    EIGEN_STRONG_INLINE int floor_(const scalar_t &x) const noexcept {return std::floor(x);}

    template<typename scalar_t>
    EIGEN_STRONG_INLINE scalar_t eval(const scalar_t &x) const noexcept
    {
        // compute the segment index
        Eigen::Index idx = (Eigen::Index)floor_( x / static_cast<scalar_t>(m_length));

// due to some weird Windows redefinitions
#if defined _WIN32  || defined _WIN64
        #undef min
        #undef max
#endif
        idx = std::max(Eigen::Index(0), std::min(idx, m_coeffs.cols())); // clip idx stay within the spline bounds

        // evaluate polynomial using Horner's method
        return m_coeffs.col(idx)(0) + x * (m_coeffs.col(idx)(1) + x * (m_coeffs.col(idx)(2) + x * m_coeffs.col(idx)(3)));
    }

    EIGEN_STRONG_INLINE const coeffs_matrix_type& coefficients() const noexcept {return m_coeffs;}
    EIGEN_STRONG_INLINE coeffs_matrix_type& coefficients() noexcept {return m_coeffs;}

    EIGEN_STRONG_INLINE const Scalar& segment_length() const noexcept {return m_length;}
    EIGEN_STRONG_INLINE Scalar& segment_length() noexcept {return m_length;}

private:
    coeffs_matrix_type m_coeffs;
    Scalar m_length{Scalar(1)};
};


class LagrangeSpline
{
public:
    template<typename Derived, typename Derived2>
    static void compute_lagrange_basis(const Eigen::MatrixBase<Derived>& nodes, Eigen::MatrixBase<Derived2>& basis) noexcept
    {
        typename Derived::PlainObject polynomial(nodes.rows()); polynomial.setZero();
        basis = Eigen::MatrixBase<Derived2>::Zero(nodes.rows(), nodes.rows());

        for(unsigned i = 0; i < basis.rows(); ++i)
        {
            Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> roots(nodes.rows() - 1);
            unsigned count = 0;
            for(unsigned j = 0; j < nodes.rows(); ++j)
            {    if(i != j)
                {
                    roots(count) = nodes(j);
                    count++;
                }
            }
            Eigen::roots_to_monicPolynomial(roots, polynomial);
            basis.row(i) = polynomial / Eigen::poly_eval(polynomial, nodes(i));
        }
    }

    // evaluation of a multidimensional polynomial
    template<typename DerVal, typename DerBasis>
    static auto eval(const typename DerVal::Scalar& arg, const Eigen::MatrixBase<DerVal>& values,
                     const Eigen::MatrixBase<DerBasis>& basis) -> Eigen::Matrix<typename DerVal::Scalar, DerVal::RowsAtCompileTime, 1>
    {
        typename DerVal::PlainObject interpolant = values * basis;
        using result_t = Eigen::Matrix<typename DerVal::Scalar, DerVal::RowsAtCompileTime, 1>;
        result_t res = result_t::Zero(values.rows(), 1);
        for(int i = 0; i < interpolant.rows(); i++)
            res(i) = Eigen::poly_eval(interpolant.row(i), arg);

        return res;
    }
};


} // polympc namespace

#endif // SPLINES_HPP
