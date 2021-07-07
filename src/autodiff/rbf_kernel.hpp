#ifndef RBF_KERNEL_HPP
#define RBF_KERNEL_HPP

#include "autodiff/AutoDiffScalar.h"
#include "utils/helpers.hpp"
#include <iostream>


template<int Size, typename Scalar = double>
struct rbf_kernel
{
    rbf_kernel() = default;
    ~rbf_kernel() = default;

    template<typename T>
    using vector_t = Eigen::Matrix<T, Size, 1>;
    using real_vector_t = Eigen::Matrix<Scalar, Size, 1>;
    using adscalar_t = Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, Size, 1>>;
    using outer_deriv_t = Eigen::Matrix<adscalar_t, Size, 1>;
    using outer_adscalar_t = Eigen::AutoDiffScalar<outer_deriv_t>;


    template<typename scalar_t>
    EIGEN_STRONG_INLINE scalar_t eval(Eigen::Ref<vector_t<scalar_t>> x, const Eigen::Ref<const real_vector_t>& y)
    {
        return exp(-(x - y.template cast<scalar_t>()).squaredNorm());
    }

    // provide partial specialisations
    // double
    template<>
    EIGEN_STRONG_INLINE double eval<double>(Eigen::Ref<vector_t<double>> x, const Eigen::Ref<const real_vector_t>& y)
    {
        std::cout << " double \n";
        return exp(-(x - y).squaredNorm());
    }

    // float
    template<>
    EIGEN_STRONG_INLINE float eval<float>(Eigen::Ref<vector_t<float>> x, const Eigen::Ref<const real_vector_t>& y)
    {
        std::cout << " float \n";
        return exp(-(x - y).squaredNorm());
    }

    // autodiff
    template<>
    adscalar_t eval<adscalar_t>(Eigen::Ref<vector_t<adscalar_t>> x, const Eigen::Ref<const real_vector_t>& y)
    {
        std::cout << " adscalar \n";
        real_vector_t xt;
        for (Eigen::Index i  = 0; i < Size; ++i)
            xt(i) = x(i).value();

        Scalar val = exp(-(xt - y).squaredNorm());
        return adscalar_t(val, -2 * val * (xt - y));
    }

    // autodiff 2
    template<>
    outer_adscalar_t eval<outer_adscalar_t>(Eigen::Ref<vector_t<outer_adscalar_t>> x, const Eigen::Ref<const real_vector_t>& y)
    {
        std::cout << " adscalar2 \n";
        real_vector_t xt;
        for (Eigen::Index i  = 0; i < Size; ++i)
            xt(i) = x(i).value().value();

        Scalar val = exp(-(xt - y).squaredNorm());
        Eigen::Matrix<Scalar, Size, Size> hes;
        hes.noalias() = val * (4 * (xt - y) * (xt - y).transpose() - Eigen::Matrix<Scalar, Size, Size>::Identity());

        outer_adscalar_t result;
        // value
        result.value().value() = val;
        result.value().derivatives().noalias() = -2 * val * (xt - y);

        for (Eigen::Index i  = 0; i < Size; ++i)
            result.derivatives()(i).derivatives() = hes.col(i);

        return result;
    }

};

#endif // RBF_KERNEL_HPP

