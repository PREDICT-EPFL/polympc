#include "solvers/sqp_base.hpp"
#include "polynomials/splines.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/nlproblem.hpp"


// Minimise Euclidean distance to a spline curve
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ ClosestPoint, /*NX*/ 1, /*NE*/0, /*NI*/0, /*NP*/2, /*Type*/double);
class ClosestPoint : public ProblemBase<ClosestPoint>
{
public:

    Eigen::MatrixXd m_data_coeff;
    double m_spline_length{ 0.5 }; // each segment is 0.5 [m] long

    polympc::EquidistantCubicSpline<double> m_xc_spline;
    polympc::EquidistantCubicSpline<double> m_yc_spline;

    // constructor
    ClosestPoint()
    {
        // initialise the track as a straight line (for instance)
	m_data_coeff.resize(20, 5);

	for (int i  = 0; i < 5; i++) 
	{
		m_data_coeff.row(4 * i)     << m_spline_length * i, 0, 0, 0, 0;
		m_data_coeff.row(4 * i + 1) << m_spline_length * i, 1, 0, 0, 0;
		m_data_coeff.row(4 * i + 2) << m_spline_length * i, 0, 0, 0, 0;
		m_data_coeff.row(4 * i + 3) << m_spline_length * i, 0, 0, 0, 0;
	}

        //Xc
        m_xc_spline.segment_length() = m_spline_length;
        m_xc_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(m_data_coeff.col(1).data(), 4, m_data_coeff.rows() / 4);

        //Yc
        m_yc_spline.segment_length() = m_spline_length;
        m_yc_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(m_data_coeff.col(2).data(), 4, m_data_coeff.rows() / 4);
    }

    ~ClosestPoint() = default;

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        T xc_ref = m_xc_spline.eval(x(0));
        T yc_ref = m_yc_spline.eval(x(0));

        cost = (p(0) - xc_ref)*(p(0) - xc_ref) + (p(1) - yc_ref)*(p(1) - yc_ref);
    }

    // set the data (if new coefficients are available)
    void set_data(const Eigen::Ref<const Eigen::MatrixXd>& _data_new, double _spline_length) 
    {
        //Xc
        m_xc_spline.segment_length() = _spline_length;
        m_xc_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(m_data_coeff.col(1).data(), 4, m_data_coeff.rows() / 4);

        //Yc
        m_yc_spline.segment_length() = _spline_length;
        m_yc_spline.coefficients() = Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic>>(m_data_coeff.col(2).data(), 4, m_data_coeff.rows() / 4);
    }


};