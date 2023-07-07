
#include "solvers/sqp_base.hpp"
#include "solvers/box_admm.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "control/mpc_wrapper.hpp"
#include "polynomials/splines.hpp"

#include "models/car_model.hpp"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2

using namespace Eigen;


/** choose the trajectory parametrisation and numerical quadratures */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ CarOCP, /*NX*/ 10, /*NU*/ 3, /*NP*/ 0, /*ND*/ 0, /*NG*/0, /*TYPE*/ double)
class RacingOCP : public ContinuousOCP<RacingOCP, Approximation, SPARSE>
{
public:

    RacingOCP()
    {	
	// initialise curvature somehow
        m_data_coeff.resize(20, 5);
	for (int i  = 0; i < 5; i++) //Straight line
	{
            m_data_coeff.row(4 * i)     << m_spline_length * i, 0, 0, 0, 0;
	    m_data_coeff.row(4 * i + 1) << m_spline_length * i, 1, 0, 0, 0;
	    m_data_coeff.row(4 * i + 2) << m_spline_length * i, 0, 0, 0, 0;
	    m_data_coeff.row(4 * i + 3) << m_spline_length * i, 0, 0, 0, 0;
	}

	//Define the Cost Function Matrices
	ScaleU << 1.0,  3000.0, 3000.0;
	Q  << 7.0, 0.0, 0.0, 0.0, 10.0, 100.0, 5.0, 0.0, 0.5, 0.5;
	R  << 1.0, 0.25, 0.25;
	QN << 7.0, 0.0, 0.0, 0.0, 10.0, 100.0, 5.0, 0.0, 0.5, 0.5;

	//For tracking a velocity profile
        m_velocity_coeff.resize(20, 2);
	m_velocity_coeff.setZero();
	for (int i  = 0; i < 5; i++) // Constant reference speed at 1m/s
	{
		m_velocity_coeff.row(4 * i)     << m_spline_length * i, 1; // default = 1m/s
		m_velocity_coeff.row(4 * i + 1) << m_spline_length * i, 0;
		m_velocity_coeff.row(4 * i + 2) << m_spline_length * i, 0;
		m_velocity_coeff.row(4 * i + 3) << m_spline_length * i, 0;
	}
	velocity_spline.segment_length() = m_spline_length;
	velocity_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(m_velocity_coeff.col(1).data(), 4, m_velocity_coeff.rows() / 4);

    }
    ~CarOCP() = default;

    scalar_t m_spline_length{ 12.5 };
    //The Data below are is follows [Xc,Yc,Psic,Kappac]
    MatrixXd m_data_coeff;

    Eigen::Matrix<scalar_t, 3, 1> ScaleU;
    Eigen::Matrix<scalar_t, 10, 1> Q_, QN_;
    Eigen::Matrix<scalar_t, 3, 1> R_;

    //For tracking a velocity profile
    polympc::EquidistantCubicSpline<scalar_t> velocity_spline; //For tracking the velocity
    MatrixXd m_velocity_coeff;

    //inverse of steering delay 
    scalar_t m_delay_inverse{20.0};

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const control_t<T>>& u,
			      const Eigen::Ref<const parameter_t<T>>& p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        //x = [vx,vy,r,s,w,theta, steering, Fxf, Fxr]
        //u = [steering_rate, Fxf_rate, Fxr_rate]
        Dynamics<T>(x.template head<6>(), x.template tail<3>().cwiseProduct(ScaleU), p, xdot.template head<6>());
        
        //Steering delay
        xdot.template tail<2>() = u.template tail<2>(); //Fxf rate, Fxr rate
        xdot(6) = u(0); // steering
        xdot(7) = (T)(m_delay_inverse) * (-x(7) + x(6));      // delayed steering

    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const control_t<T>>& u,
				   const Eigen::Ref<const parameter_t<T>>& p, const Eigen::Ref<const static_parameter_t>& d,
				   const scalar_t &t, T &lagrange) noexcept
    {
         //x = [v-v_ref, vy, phi_dot, s, w, theta]
	 //u = [steering, Fxf, Fxr]
	 state_t<T> x_aug{ x };

         T v_ref;
         v_ref = (T)(velocity_spline.eval(x(3))); //For tracking reference velocity profile
	 x_aug(0) = x(0) - v_ref; // (vx - v_ref)

         lagrange = x_aug.dot(x_aug.cwiseProduct(Q_)) + u.dot(u.cwiseProduct(R_))
	            + (x(8) - x(9)) * (x(8) - x(9)) * (T)(100.0); //Last two: differences between front and rear throttle

    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const control_t<T>>& u,
				const Eigen::Ref<const parameter_t<T>>& p, const Eigen::Ref<const static_parameter_t>& d,
				const scalar_t &t, T &mayer) noexcept
    {
        state_t<T> x_aug{ x };

        T v_ref{ 13.2 };
	v_ref = (T)(velocity_spline.eval(x(3))); //For tracking reference velocity profile
	x_aug(0) = x(0) - v_ref; // (vx - v_ref)

	mayer = x_aug.dot(x_aug.cwiseProduct(Q_)) + (x(8) - x(9)) * (x(8) - x(9)) * (T)(100.0);
    }

    void set_data(const Eigen::Ref<const data_t>& _data_new, const scalar_t& _spline_length) noexcept
    {
        m_data_coeff = _data_new;
	m_spline_length = _spline_length;
    }

    void set_vel_data(const Eigen::Ref<const data_t>& data, const double& spline_length) noexcept
    {
	velocity_spline.segment_length() = spline_length;
	velocity_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(data.col(1).data(), 4, data.rows() / 4);
    }
};