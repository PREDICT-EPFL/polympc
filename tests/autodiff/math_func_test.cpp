#include "Eigen/Core"
//#include "autodiff/AutoDiffScalar.h"
#include "polynomials/splines.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <type_traits>

int main(void)
{
    Eigen::Matrix<double,3,1> x0;
    x0 << 1.0, 1.0, M_PI_4;
    Eigen::Matrix<double,2,1>u;
    u << 1.0, M_PI_4;
    Eigen::Matrix<double,3,3> jacobian;
    Eigen::Matrix<double,3,1> output;

    using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<double, 5, 1>>;
    using outer_deriv_type = Eigen::Matrix<ADScalar, 5, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outer_deriv_type>;
    Eigen::Matrix<outerADScalar, 3, 1> Ax0;
    Eigen::Matrix<outerADScalar, 2, 1> Au;

    /** initialize values */
    for(int i = 0; i < Ax0.SizeAtCompileTime; ++i) Ax0(i).value().value() = x0[i];
    for(int i = 0; i < Au.SizeAtCompileTime; ++i) Au(i).value().value() = u[i];

    /** initialize derivatives */
    int div_size = Ax0.size() + Au.size();
    int derivative_idx = 0;
    for(int i = 0; i < Ax0.size(); ++i)
    {
        Ax0(i).value().derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        Ax0(i).derivatives() =  Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        // initialize hessian matrix to zero
        for(int idx=0; idx<div_size; idx++)
        {
            Ax0(i).derivatives()(idx).derivatives()  = Eigen::Matrix<double, 5, 1>::Zero();
        }
        derivative_idx++;
    }

    for(int i = 0; i < Au.size(); ++i)
    {
        Au(i).value().derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        Au(i).derivatives() = Eigen::Matrix<double, 5, 1>::Unit(div_size, derivative_idx);
        for(int idx=0; idx<div_size; idx++)
        {
            Au(i).derivatives()(idx).derivatives()  = Eigen::Matrix<double, 5, 1>::Zero();
        }
        derivative_idx++;
    }

    ADScalar ads(0.2);
    outerADScalar ads2; ads2.value().value() = ads.value();

    ADScalar floor_ads = floor(ads);
    outerADScalar floor_ads2 = floor(ads2);

    std::cout << "floor( " << ads.value() << " ) = " << floor_ads.value() << "\n";
    std::cout << "floor( " << ads2.value() << " ) = " << floor_ads2.value().value() << "\n";

    Eigen::MatrixXd coeffs(2,4); coeffs << 0, 1, 2, 3, 4, 5, 6, 7;
    using namespace polympc;
    EquidistantCubicSpline<double> cubic_spline(coeffs, 10);
    double d_spline = cubic_spline.eval(0.2);
    ADScalar ads_spline = cubic_spline.eval(ads);
    outerADScalar ads2_spline = cubic_spline.eval(ads2);

    std::cout << "Spline eval: " << d_spline << "\n";
    std::cout << "Spline eval: " << ads_spline.value() << "\n";
    std::cout << "Spline eval: " << ads2_spline.value().value() << "\n";

    return EXIT_SUCCESS;
}
