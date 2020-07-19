#ifndef POLYNOMIAL_MATH_HPP
#define POLYNOMIAL_MATH_HPP

#include "iostream"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/Polynomials"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/TensorSymmetry"


namespace polympc {

namespace polymath {

enum collocation_scheme {GAUSS, GAUSS_RADAU, GAUSS_LOBATTO};

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


} // polymath namespace

} // polympc namespace

#endif // POLYNOMIAL_MATH_HPP
