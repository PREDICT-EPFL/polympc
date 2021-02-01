#ifndef SR1_HPP
#define SR1_HPP

#include "utils/helpers.hpp"

/** Safe guarded SR1 update
 * Implements "Formula 6.24" form Numerical Optimization by Nocedal.
 *
 * @param[in,out]   B hessian matrix, is updated by this function
 * @param[in]       s step vector (x - x_prev)
 * @param[in]       y gradient change (grad - grad_prev)
 */
template <typename Mat, typename Vec>
void SR1_update(Eigen::MatrixBase<Mat>& B, const Eigen::MatrixBase<Vec>& s, const Eigen::MatrixBase<Vec>& y)
{
    using Scalar = typename Mat::Scalar;
    typename Vec::PlainObject v = y;
    v.noalias() -= B * s;

    const Scalar scaling = v.dot(s);
    const Scalar eps = Scalar(1e-6); // as suggested by Nocedal
    // safeguard
    if(std::abs(scaling) >= eps * s.template lpNorm<2>() * v.template lpNorm<2>())
        B.noalias() += v * v.transpose() / scaling;

    //else: leave matrix unchanged B_{k+1} = B_{k}
}



#endif // SR1_HPP
