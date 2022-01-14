#ifndef QP_PRECONDITIONERS_HPP
#define QP_PRECONDITIONERS_HPP

#include "utils/helpers.hpp"
#include <iostream>

/**
template<typename Derived, int N, int M, typename Scalar>
class GenericDiagonalPreconditioner
{
    typename dense_matrix_type_selector<Scalar, N, 1>::type m_D; //hessian scaling
    typename dense_matrix_type_selector<Scalar, N, 1>::type m_D
}
*/

namespace polympc {


/** empty preconditioner */
class IdentityPreconditioner
{
public:
    IdentityPreconditioner() = default;
    ~IdentityPreconditioner() = default;

    template<typename T>
    using MB = Eigen::Ref<T>;


    template<typename Hessian, typename Gradient, typename Jacobian, typename Constraint>
    EIGEN_STRONG_INLINE void compute(const Hessian& H, const Gradient&h, const Jacobian& A, const Constraint& Al,
                                     const Constraint& Au, const Gradient& l, const Gradient& u) const noexcept
    {
        polympc::ignore_unused_var(H);
        polympc::ignore_unused_var(h);
        polympc::ignore_unused_var(A);
        polympc::ignore_unused_var(Al);
        polympc::ignore_unused_var(Au);
        polympc::ignore_unused_var(l);
        polympc::ignore_unused_var(u);
    }

    template<typename Hessian, typename Gradient, typename Jacobian, typename Constraint>
    EIGEN_STRONG_INLINE void scale(const Hessian& H, const Gradient&h, const Jacobian& A, const Constraint& Al,
                                   const Constraint& Au, const Gradient& l, const Gradient& u) const noexcept
    {
        polympc::ignore_unused_var(H);
        polympc::ignore_unused_var(h);
        polympc::ignore_unused_var(A);
        polympc::ignore_unused_var(Al);
        polympc::ignore_unused_var(Au);
        polympc::ignore_unused_var(l);
        polympc::ignore_unused_var(u);
    }

    template<typename Hessian, typename Gradient, typename Jacobian, typename Constraint>
    EIGEN_STRONG_INLINE void unscale(const Hessian& H, const Gradient&h, const Jacobian& A, const Constraint& Al,
                                     const Constraint& Au, const Gradient& l, const Gradient& u) const noexcept
    {
        polympc::ignore_unused_var(H);
        polympc::ignore_unused_var(h);
        polympc::ignore_unused_var(A);
        polympc::ignore_unused_var(Al);
        polympc::ignore_unused_var(Au);
        polympc::ignore_unused_var(l);
        polympc::ignore_unused_var(u);
    }

    template<typename Primal, typename Dual>
    EIGEN_STRONG_INLINE void scale(const Primal& x, const Dual&y) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(y);
    }

    template<typename Primal, typename Dual>
    EIGEN_STRONG_INLINE void unscale(const Primal& x, const Dual&y) const noexcept
    {
        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(y);
    }

    template<typename Hessian>
    EIGEN_STRONG_INLINE void unscale_hessian(const Hessian& x) const noexcept
    {
        polympc::ignore_unused_var(x);
    }

    template<typename Hessian, typename Gradient, typename Jacobian, typename Constraint>
    EIGEN_STRONG_INLINE void update( const Hessian& H, const Gradient&h, const Jacobian& A, const Constraint& Al,
                                     const Constraint& Au, const Gradient& l, const Gradient& u) const noexcept
    {
        polympc::ignore_unused_var(H);
        polympc::ignore_unused_var(h);
        polympc::ignore_unused_var(A);
        polympc::ignore_unused_var(Al);
        polympc::ignore_unused_var(Au);
        polympc::ignore_unused_var(l);
        polympc::ignore_unused_var(u);
    }

};

/** Ruiz equilibration algorithm */
template<typename Scalar, int N, int M, int MatrixType = DENSE>
class RuizEquilibration
{
public:
    using scalar_t = Scalar;
    using vector_d_t = typename dense_matrix_type_selector<scalar_t, N, 1>::type;
    using vector_e_t = typename dense_matrix_type_selector<scalar_t, M, 1>::type;
private:
    vector_d_t m_D;  //hessian scaling
    vector_e_t m_E;  // Jacoabian scaling
    scalar_t m_c{1}; // cost scaling


    using hessiant_t   = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                         typename dense_matrix_type_selector<scalar_t, N, N>::type>::type;
    using jacobian_t   = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                         typename dense_matrix_type_selector<scalar_t, M, N>::type>::type;
    using gradient_t   = typename dense_matrix_type_selector<scalar_t, N, 1>::type;
    using constraint_t = typename dense_matrix_type_selector<scalar_t, M, 1>::type;
    using dual_t       = typename dense_matrix_type_selector<scalar_t, M + N, 1>::type;

public:
    RuizEquilibration()
    {
        m_D = vector_d_t::Ones(N);
        m_E = vector_e_t::Ones(M);

        D = vector_d_t::Ones(N);
        E = vector_e_t::Ones(M);
    }
    ~RuizEquilibration() = default;

    vector_d_t D, Dx;
    vector_e_t E;
    const scalar_t& c() const noexcept {return m_c;}
    scalar_t& c() noexcept {return m_c;}

    template<int T = MatrixType>
    typename std::enable_if<T == DENSE>::type
    compute(Eigen::Ref<hessiant_t> H, Eigen::Ref<gradient_t> h, Eigen::Ref<jacobian_t> A,
            Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
            Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) noexcept
    {
        constexpr int max_iter = 4;
        m_c = scalar_t(1);
        scalar_t gamma_scaling = scalar_t(1);

        m_D = vector_d_t::Ones(N, 1);
        m_E = vector_e_t::Ones(M, 1);
        D = m_D;
        E = m_E;

        const scalar_t _approx_zero = std::numeric_limits<scalar_t>::epsilon();
        const scalar_t _tolerance   = scalar_t(1e-3);
        scalar_t scaling_norm = 10 * _tolerance;
        Eigen::Index iter = 0;

        while ( (iter < max_iter) && ((scalar_t(1) - scaling_norm) >= _tolerance) )
        {
            // compute the Infinity norm of i-th column
            m_E = A.rowwise().template lpNorm<Eigen::Infinity>();
            m_D = H.colwise().template lpNorm<Eigen::Infinity>();
            gradient_t x = A.colwise().template lpNorm<Eigen::Infinity>();
            m_D = m_D.cwiseMax(x);

            scaling_norm = std::max(m_D.maxCoeff(), m_E.maxCoeff());

            // avoid singularities
            if (m_D.minCoeff() < _approx_zero)
                for (int k = 0; k < N; ++k) {if (m_D(k) < _approx_zero) m_D(k) = scalar_t(1.0);}

            if (m_E.minCoeff() < _approx_zero)
                for (int k = 0; k < M; ++k) {if (m_E(k) < _approx_zero) m_E(k) = scalar_t(1.0);}

            // compute diagonal scaling factors
            m_D = m_D.cwiseSqrt().cwiseInverse();
            m_E = m_E.cwiseSqrt().cwiseInverse();

            // scale the matrices
            H.noalias() = m_D.asDiagonal() * H * m_D.asDiagonal();
            A.noalias() = m_E.asDiagonal() * A * m_D.asDiagonal();
            h.noalias() = h.cwiseProduct(m_D);

            // update scaling matrices
            D = D.cwiseProduct(m_D);
            E = E.cwiseProduct(m_E);

            // update gamma factor
            m_D = H.colwise().template lpNorm<Eigen::Infinity>();
            scalar_t h_inf = h.template lpNorm<Eigen::Infinity>();
            h_inf = h_inf > _approx_zero ? h_inf : scalar_t(1.0);
            gamma_scaling = scalar_t(1) / std::max(m_D.mean(), h_inf);
            H *= gamma_scaling;
            h *= gamma_scaling;
            m_c *= gamma_scaling;

            ++iter;
        }

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }

    /** Max coefficient of a sparse expression */
    template<typename Derived>
    typename Derived::Scalar maxxCoeff(const Eigen::SparseMatrixBase<Derived>& mat) const noexcept
    {
        typename Derived::Scalar max_coeff = std::numeric_limits<typename Derived::Scalar>::lowest();
        Eigen::internal::evaluator<Derived> mat_eval(mat.derived());
        for(Eigen::Index j = 0; j < mat.outerSize(); ++j)
            for (typename Eigen::internal::evaluator<Derived>::InnerIterator iter(mat_eval,j); iter; ++iter)
                max_coeff = max_coeff > iter.value() ? max_coeff : iter.value();

        return max_coeff;
    }

    template<int T = MatrixType>
    typename std::enable_if<T == SPARSE>::type
    compute(hessiant_t& H, Eigen::Ref<gradient_t> h, jacobian_t& A,
            Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
            Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) noexcept
    {
        constexpr int max_iter = 4;
        m_c = scalar_t(1);
        scalar_t gamma_scaling = scalar_t(1);

        m_D = vector_d_t::Ones(N, 1);
        m_E = vector_e_t::Ones(M, 1);
        D = m_D;
        E = m_E;

        const scalar_t _approx_zero = scalar_t(1e-4); //std::numeric_limits<scalar_t>::epsilon();
        const scalar_t _tolerance   = scalar_t(1e-3);
        scalar_t scaling_norm = 10 * _tolerance;
        Eigen::Index iter = 0;

        while ( (iter < max_iter) && ((scalar_t(1) - scaling_norm) >= _tolerance) )
        {
            // compute the Infinity norm of i-th column
            for(int i = 0; i < M; ++i) m_E(i) = maxxCoeff(A.transpose().col(i).cwiseAbs());
            for(int i = 0; i < N; ++i) m_D(i) = maxxCoeff(H.col(i).cwiseAbs());

            gradient_t x;
            for(int i = 0; i < N; ++i) x(i) = maxxCoeff(A.col(i).cwiseAbs());

            m_D = m_D.cwiseMax(x);

            scaling_norm = std::max(m_D.maxCoeff(), m_E.maxCoeff());

            // avoid singularities
            if (m_D.minCoeff() < _approx_zero)
                for (int k = 0; k < N; ++k) {if (m_D(k) < _approx_zero) m_D(k) = scalar_t(1.0);}

            if (m_E.minCoeff() < _approx_zero)
                for (int k = 0; k < M; ++k) {if (m_E(k) < _approx_zero) m_E(k) = scalar_t(1.0);}

            // compute diagonal scaling factors
            m_D = m_D.cwiseSqrt().cwiseInverse();
            m_E = m_E.cwiseSqrt().cwiseInverse();

            D = D.cwiseProduct(m_D);
            E = E.cwiseProduct(m_E);

            // scale the matrices
            H = m_D.asDiagonal() * H * m_D.asDiagonal();
            A = m_E.asDiagonal() * (A * m_D.asDiagonal());
            h = h.cwiseProduct(m_D);

            // update gamma factor
            for(int i = 0; i < N; ++i) m_D(i) = maxxCoeff(H.col(i).cwiseAbs());
            gamma_scaling = scalar_t(1) / std::max(m_D.mean(), h.template lpNorm<Eigen::Infinity>());
            H *= gamma_scaling;
            h *= gamma_scaling;
            m_c *= gamma_scaling;

            ++iter;
        }

        H.uncompress();
        A.uncompress();

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }


    /** apply existing scaling to new matrices */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    scale(Eigen::Ref<hessiant_t> H, Eigen::Ref<gradient_t> h, Eigen::Ref<jacobian_t> A,
          Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
          Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) const noexcept
    {
        H.noalias() = m_c * D.asDiagonal() * H * D.asDiagonal();
        A.noalias() = E.asDiagonal() * A * D.asDiagonal();
        h.noalias() = m_c * h.cwiseProduct(D);

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    scale(hessiant_t& H, Eigen::Ref<gradient_t> h, jacobian_t& A,
          Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
          Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) const noexcept
    {
        H = m_c * D.asDiagonal() * H * D.asDiagonal();
        A = E.asDiagonal() * (A * D.asDiagonal());
        H.uncompress();
        A.uncompress();

        h.noalias() = m_c * h.cwiseProduct(D);

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }

    /** scale / unscale primal and dual solutions */
    EIGEN_STRONG_INLINE void scale(Eigen::Ref<gradient_t> x, Eigen::Ref<dual_t> y) const noexcept
    {
        x = x.cwiseProduct(D.cwiseInverse());
        y.template head<M>() = m_c * y.template head<M>().cwiseProduct(E.cwiseInverse());
        y.template tail<N>() = m_c * y.template tail<N>().cwiseProduct(D);
    }

    EIGEN_STRONG_INLINE void unscale(Eigen::Ref<gradient_t> x, Eigen::Ref<dual_t> y) const noexcept
    {
        x = x.cwiseProduct(D);
        y.template head<M>().noalias() = (1 / m_c) * y.template head<M>().cwiseProduct(E);
        y.template tail<N>().noalias() = (1 / m_c) * y.template tail<N>().cwiseProduct(D.cwiseInverse());
    }

    /** apply unscaling */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    unscale(Eigen::Ref<hessiant_t> H, Eigen::Ref<gradient_t> h, Eigen::Ref<jacobian_t> A,
            Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
            Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) const noexcept
    {
        H.noalias() = scalar_t(1 / m_c) * (D.cwiseInverse()).asDiagonal() * H * (D.cwiseInverse()).asDiagonal();
        A.noalias() = (E.cwiseInverse()).asDiagonal() * A * (D.cwiseInverse()).asDiagonal();
        h.noalias() = scalar_t(1 / m_c) * h.cwiseProduct(D.cwiseInverse());

        // apply to the constraints cone
        Au = Au.cwiseProduct(E.cwiseInverse());
        Al = Al.cwiseProduct(E.cwiseInverse());

        // box constraints
        l = l.cwiseProduct(D);
        u = u.cwiseProduct(D);
    }

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    unscale(hessiant_t& H, Eigen::Ref<gradient_t> h, jacobian_t& A,
            Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
            Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) const noexcept
    {
        H = scalar_t(1 / m_c) * (D.cwiseInverse()).asDiagonal() * H * (D.cwiseInverse()).asDiagonal();
        A = (E.cwiseInverse()).asDiagonal() * A * (D.cwiseInverse()).asDiagonal();
        H.uncompress();
        A.uncompress();
        h.noalias() = scalar_t(1 / m_c) * h.cwiseProduct(D.cwiseInverse());

        // apply to the constraints cone
        Au = Au.cwiseProduct(E.cwiseInverse());
        Al = Al.cwiseProduct(E.cwiseInverse());

        // box constraints
        l = l.cwiseProduct(D);
        u = u.cwiseProduct(D);
    }

    /** unscale hessian */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    unscale_hessian(hessiant_t& H) const noexcept
    {
        H = scalar_t(1 / m_c) * (D.cwiseInverse()).asDiagonal() * H * (D.cwiseInverse()).asDiagonal();
        H.uncompress();
    }

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    unscale_hessian(Eigen::Ref<hessiant_t> H) const noexcept
    {
        H.noalias() = scalar_t(1 / m_c) * (D.cwiseInverse()).asDiagonal() * H * (D.cwiseInverse()).asDiagonal();
    }

    template<int T = MatrixType>
    typename std::enable_if<T == DENSE>::type
    update(Eigen::Ref<hessiant_t> H, Eigen::Ref<gradient_t> h, Eigen::Ref<jacobian_t> A,
                Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
                Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) noexcept
    {
        constexpr int max_iter = 4;
        scalar_t gamma_scaling = scalar_t(1);

        const scalar_t _approx_zero = std::numeric_limits<scalar_t>::epsilon();
        const scalar_t _tolerance   = scalar_t(1e-3);
        scalar_t scaling_norm = std::max(m_D.maxCoeff(), m_E.maxCoeff());
        Eigen::Index iter = 0;

        while ( (iter < max_iter) && ((scalar_t(1) - scaling_norm) <= _tolerance) )
        {
            // compute the Infinity norm of i-th column
            m_E = A.rowwise().template lpNorm<Eigen::Infinity>();
            m_D = H.colwise().template lpNorm<Eigen::Infinity>();
            m_D = m_D.cwiseMax(A.colwise().template lpNorm<Eigen::Infinity>());

            scaling_norm = std::max(m_D.maxCoeff(), m_E.maxCoeff());

            // avoid singularities
            if (m_D.minCoeff() < _approx_zero)
                for (int k = 0; k < N; ++k) {if (m_D(k) < _approx_zero) m_D(k) = scalar_t(1.0);}

            if (m_E.minCoeff() < _approx_zero)
                for (int k = 0; k < M; ++k) {if (m_E(k) < _approx_zero) m_E(k) = scalar_t(1.0);}

            // compute diagonal scaling factors
            m_D = m_D.cwiseSqrt().cwiseInverse();
            m_E = m_E.cwiseSqrt().cwiseInverse();

            D = D.cwiseProduct(m_D);
            E = E.cwiseProduct(m_E);

            // scale the matrices
            H.noalias() = m_c * m_D.asDiagonal() * H * m_D.asDiagonal();
            A.noalias() = m_E.asDiagonal() * A * m_D.asDiagonal();
            h.noalias() = m_c * h.cwiseProduct(m_D);

            // update gamma factor
            m_D = H.colwise().template lpNorm<Eigen::Infinity>();
            gamma_scaling = scalar_t(1) / std::max(m_D.mean(), h.template lpNorm<Eigen::Infinity>());
            H.noalias() *= gamma_scaling;
            h.noalias() *= gamma_scaling;
            m_c *= gamma_scaling;

            ++iter;
        }

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }

    template<int T = MatrixType>
    typename std::enable_if<T == SPARSE>::type
    update(hessiant_t& H, Eigen::Ref<gradient_t> h, jacobian_t& A,
           Eigen::Ref<constraint_t> Al, Eigen::Ref<constraint_t> Au,
           Eigen::Ref<gradient_t> l, Eigen::Ref<gradient_t> u) noexcept
    {
        constexpr int max_iter = 4;
        scalar_t gamma_scaling = scalar_t(1);

        const scalar_t _approx_zero = std::numeric_limits<scalar_t>::epsilon();
        const scalar_t _tolerance   = scalar_t(1e-3);
        scalar_t scaling_norm = 10 * _tolerance;
        Eigen::Index iter = 0;

        while ( (iter < max_iter) && ((scalar_t(1) - scaling_norm) >= _tolerance) )
        {
            // compute the Infinity norm of i-th column
            for(int i = 0; i < M; ++i) m_E(i) = maxxCoeff(A.transpose().col(i).cwiseAbs());
            for(int i = 0; i < N; ++i) m_D(i) = maxxCoeff(H.col(i).cwiseAbs());

            gradient_t x;
            for(int i = 0; i < N; ++i) x(i) = maxxCoeff(A.col(i).cwiseAbs());
            m_D = m_D.cwiseMax(x);

            scaling_norm = std::max(m_D.maxCoeff(), m_E.maxCoeff());

            // avoid singularities
            if (m_D.minCoeff() < _approx_zero)
                for (int k = 0; k < N; ++k) {if (m_D(k) < _approx_zero) m_D(k) = scalar_t(1.0);}

            if (m_E.minCoeff() < _approx_zero)
                for (int k = 0; k < M; ++k) {if (m_E(k) < _approx_zero) m_E(k) = scalar_t(1.0);}

            // compute diagonal scaling factors
            m_D = m_D.cwiseSqrt().cwiseInverse();
            m_E = m_E.cwiseSqrt().cwiseInverse();

            D = D.cwiseProduct(m_D);
            E = E.cwiseProduct(m_E);

            // scale the matrices
            H = m_c * m_D.asDiagonal() * H * m_D.asDiagonal();
            A = m_E.asDiagonal() * (A * m_D.asDiagonal());
            h = m_c * h.cwiseProduct(m_D);

            // update gamma factor
            for(int i = 0; i < N; ++i) m_D(i) = maxxCoeff(H.col(i).cwiseAbs());
            gamma_scaling = scalar_t(1) / std::max(m_D.mean(), h.template lpNorm<Eigen::Infinity>());
            H *= gamma_scaling;
            h *= gamma_scaling;
            m_c *= gamma_scaling;

            ++iter;
        }

        H.uncompress();
        A.uncompress();

        // apply to the constraints cone
        Au = Au.cwiseProduct(E);
        Al = Al.cwiseProduct(E);

        // box constraints
        l = l.cwiseProduct(D.cwiseInverse());
        u = u.cwiseProduct(D.cwiseInverse());
    }

};


} // end of polympc namespace



#endif // QP_PRECONDITIONERS_HPP
