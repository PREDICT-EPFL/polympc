#include <cmath>
#include <iostream>
#include <string>

#include "polynomials/splines.hpp"

template<typename _scalar, int Size>
using Vec = typename Eigen::Matrix<_scalar, Size, 1>

// compute the lateral forces in the wheels
template<typename _scalar> 
void LateralForces(const Eigen::Ref<const Vec<_scalar, 4>>& Vel,
		      const Eigen::Ref<const Vec<_scalar, 2>>& FzForce, const _scalar& Steering,
                      Eigen::Ref<Vec<_scalar, 2>> FYLateral) 
{
    //-------MODEL------
    // Assume all parameters (i.e., L_r, L_f, Cxx, tire parameters are defined somewhere)

    //slip angles 
    _scalar a_r = atan2(Vel(3), Vel(2) + 0.01);
    _scalar a_f = atan2(Vel(1), Vel(0) + 0.01) - Steering;
    
    // Front lateral slip force
    FYLateral(0) = -FzForce(0) * D_f * sin(C_f * atan2(B_f * a_f - E_f * (B_f * a_f - atan2(B_f * a_f,(_scalar)(1))),(_scalar)(1)));

     // Rear lateral slip force
     FYLateral(1) = -FzForce(1) * D_r * sin(C_r * atan2(B_r * a_r - E_r * (B_r * a_r - atan2(B_r * a_r,(_scalar)(1))),(_scalar)(1)));
}

// For the centreline parametrisation a cubic spline from the PolyMPC polynomial library is used in this example
// the user must provide a segment length (assumed equal) and a [4 x N] matrix containing poly. coefficients for each of [N] segments
// it is then possible to evaluate and differentiate the resulting spline using the standard autodiff interface 
polympc::EquidistantCubicSpline<double> kappa_spline;
void setSplineData(Eigen::Ref<Eigen::MatrixXd>& _data_coeff, double segment_length) 
{
    // set the number of spline segments and polynomial coefficients for each segment
    kappa_spline.segment_length() = segment_length;
    kappa_spline.coefficients()   = Eigen::Map<Eigen::Matrix<Scalar, 4, Eigen::Dynamic>>(Data.col(4).data(), 4, Data.rows() / 4);
}


// Finally, evaluate the dynamics
template<typename _scalar>
void Dynamics(const Eigen::Ref<const Vec<_scalar, NX_>>& x, const Eigen::Ref<const Vec<_scalar, NU_>>& u,
		 const Eigen::Ref<const Vec<_scalar, NP_>>& param, Eigen::Ref<Vec<_scalar, NX_>> xdot)
{
    //Look-up table to get kappa as a function of s = x[3]
    // Assume all parameters (i.e., L_r, L_f, Cxx, tire parameters are defined somewhere)
    // ================ MODEL EQUATIONS ==================

    // Rear Axis
    _scalar vx_r = x(0);
    _scalar vy_r = x(1) - x(2) * L_r;
    _scalar FZr  = (_scalar)(m * g * L_f / (L_r + L_f));

    //Front Axis
    _scalar vx_f = x(0);
    _scalar vy_f = x(1) + x(2) * L_f;
    _scalar FZf = (_scalar)(m * g * L_r / (L_f + L_r));

    // Drag Force
    _scalar Fdrag = (rollResist + Cxx * x(0)*x(0));

    Vec<_scalar, 4> velocities;
    Vec<_scalar, 2> FZForce;
    Vec<_scalar, 2> FYLateral;

    velocities << vx_f, vy_f, vx_r, vy_r;
    FZForce << FZf, FZr;

    //Get Lateral Forces
    getLateralForces<_scalar, T>(velocities, FZForce, u(0), FYLateral);
    FYr = FYLateral(1);
    FYf = FYLateral(0);

    xdot(0) = x(2) * x(1) + (u(2) + u(1) * cos(u(0)) -  FYf * sin(u(0)) - sgn(u(0)) * Fdrag) / m; //vx_dot
    xdot(1) = -x(2) * x(0) + (FYf * cos(u(0)) + u(1) * sin(u(0)) + FYr) / m;  //vy_dot
    xdot(2) = (-FYr * L_r + (FYf * cos(u(0)) + u(1) * sin(u(0))) * L_f) / Iz; //omega_dot = theta_dot_dot

    kappa_c = kappa_spline.eval(x(3));
    xdot(3) = (1.0 / (1 - kappa_c * x(4))) * (x(0) * cos(x(5)) - x(1) * sin(x(5)));	//s_dot
    xdot(4) = x(0)*sin(state(5)) + x(1)*cos(x(5));					//w_dot
    xdot(5) = x(2) - kappa_c * xdot(3); 						//theta_dot

};