#ifndef POLYMATH_H
#define POLYMATH_H

#include "casadi/casadi.hpp"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "unsupported/Eigen/Polynomials"
#include "unsupported/Eigen/MatrixFunctions"

enum IntType {RK4, CVODES, CHEBYCHEV};

namespace polymath
{
    /** quaternion arithmetic */
    casadi::SX T1quat(const casadi::SX &rotAng);
    casadi::SX T2quat(const casadi::SX &rotAng);
    casadi::SX T3quat(const casadi::SX &rotAng);

    casadi::SX quat_multiply(const casadi::SX &q1, const casadi::SX &q2);
    casadi::SX quat_inverse(const casadi::SX &q);
    casadi::SX quat_transform(const casadi::SX &q_ba, const casadi::SX &a_vect);

    /** collection of custom functions */
    casadi::SX heaviside(const casadi::SX &x, const double K);
    inline double deg2rad(const double &deg){return (M_PI / 180) * deg;}

    /** integration */
    casadi::SX rk4_symbolic(const casadi::SX &X,
                            const casadi::SX &U,
                            casadi::Function &func,
                            const casadi::SX &h);

    /** @brief: compute Chebyshev collocation points for a given interval */
    void cheb(casadi::DM &CollocPoints, casadi::DM &DiffMatrix,
              const unsigned &N, const std::pair<double, double> interval);

    casadi::SX mat_func(const casadi::SX &matrix_in, casadi::Function &func);
    casadi::SX mat_dynamics(const casadi::SX &arg_x, const casadi::SX &arg_u, casadi::Function &func);

    /** range of numbers between first and the last */
    template<typename T>
    std::vector<T> range(const T &first, const T &last, const T &step = 1)
    {
        std::vector<T> _range;
        for (T value = first; value <= last; value += step)
            _range.push_back(value);

        return _range;
    }

    /** flip matrix along "X" axis */
    casadi::DM flip(const casadi::DM &matrix, const unsigned &axis);

    /** factorial computation 'n!' */
    uint factorial(const uint &n);

    /** Chebyshev exapnsion of a function u(x) = Sum_0^K uk * Tk(x) */
    template<typename Scalar>
    Scalar chebyshev_expansion(const std::vector<Scalar> &FuncValues, const Scalar &Value)
    {
        if(FuncValues.empty())
            return std::numeric_limits<Scalar>::infinity();

        /** initialize polynomial basis*/
        std::vector<Scalar> T = {1, Value};
        Scalar result = 0;

        if(FuncValues.size() > 2)
        {
            for (int k = 2; k < FuncValues.size(); ++k)
            {
                Scalar Tk = 2 * Value * T[k-1] - T[k-2];
                T.push_back(Tk);
            }
        }

        for (int i = 0; i < FuncValues.size(); ++i)
        {
            result += T[i] * FuncValues[i];
        }

        return result;
    }

    /** Chebyshev exapnsion of a function u(x) = Sum_0^K uk * Tk(x) using series expansion on [-1,1]*/
    template<typename Scalar>
    Scalar chebyshev_expansion2(const std::vector<Scalar> &FuncValues, const Scalar &Value)
    {
        if(FuncValues.empty())
            return std::numeric_limits<Scalar>::infinity();

        Scalar result = 0;
        for (int k = 0; k < FuncValues.size(); ++k)
        {
            Scalar Tk = cos(k * acos(Value));

            /** compute the Chebyshev polynomial of order k */
            result  += Tk * FuncValues[k];
        }
        return result;
    }

    template<typename BaseClass>
    BaseClass spheric2cart(const BaseClass &azimuth, const BaseClass &elevation, const BaseClass &radius)
    {
        BaseClass cart_coord = BaseClass::zeros(3,1);
        cart_coord[0] = radius * cos(elevation) * cos(azimuth);
        cart_coord[1] = radius * cos(elevation) * sin(azimuth);
        cart_coord[2] = radius * sin(elevation);

        return cart_coord;
    }

    /** @brief : compute Lagrange polynomial of degree "n_degree"
     * Lagrange polynomials
     * returns: Function({point, nodes})
     * */
    casadi::Function lagrange_poly(const uint &n_degree);

    /** @brief: compute lagrangian interpolator
     * returns: Function({point}) : y_point = polyval(P, point)
     * P - coeeficients of the interpolation polynomial
     * */
    casadi::Function lagrange_interpolant(const casadi::DM &X, const casadi::DM &Y);
    /** @brief: compute lagrangian interpolant
     * returns: polynomial coefficients : [C0,..., Cn-1, Cn]
     * P - coefficients of the interpolation polynomial
     * */
    Eigen::VectorXd lagrange_interpolant(const Eigen::VectorXd &X, const Eigen::VectorXd &Y);

    /** @brief: compute the matrix of basis polynomial coefficients
     * nodes: basis nodes
     * return: NxN matrix of lagrange polynomials
     */
    Eigen::MatrixXd lagrange_poly_basis(const Eigen::VectorXd &nodes);

    /** Chebyshev-Gauss_Lobatto collocation nodes for the interval [-1, 1]*/
    casadi::SX cgl_nodes( const int &n_points);

    /** class to keep the lagrange interpolation data */
    class  LagrangeInterpolator
    {
    public:
        LagrangeInterpolator(const Eigen::VectorXd &nodes, const Eigen::VectorXd &values);
        LagrangeInterpolator(const casadi::DM &nodes, const casadi::DM &values);
        LagrangeInterpolator() = default;
        ~LagrangeInterpolator() = default;

        double eval(const double &arg);
        double eval(const double &arg, const Eigen::VectorXd &values);
        double eval(const double &arg, const casadi::DM &values);

        void update_basis(const Eigen::VectorXd &nodes);
        void init(const Eigen::VectorXd &nodes, const Eigen::VectorXd &values);
        void init(const casadi::DM &nodes, const casadi::DM &values);

    private:
        Eigen::MatrixXd m_poly_basis;
        Eigen::VectorXd m_interpolant;
    };


    /** Linear Systems Control and Analysis routines */
    struct LinearSystem
    {
        Eigen::MatrixXd F; // dynamics
        Eigen::MatrixXd G; // control mapping matrix
        Eigen::MatrixXd H; // output mapping

        LinearSystem(){}
        LinearSystem(const Eigen::MatrixXd &_F, const Eigen::MatrixXd &_G, const Eigen::MatrixXd &_H) :
            F(_F), H(_H), G(_G) {}
        virtual ~LinearSystem(){}

        bool is_controllable();
        bool is_observable();
        /** involves stable/unstable modes decomposition */
        bool is_stabilizable();
    };

    /** fast optimal control methods */
    namespace oc
    {
        /** Solve Lyapunov equation: AX + XA' = Q */
        Eigen::MatrixXd lyapunov(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Q);

        /** solve Continuous Riccati Equation using Newton iteration with line search */
        /** C + XA + A'X - XBX = 0 */
        Eigen::MatrixXd newton_ls_care(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                                       const Eigen::MatrixXd &C, const Eigen::MatrixXd &X0);

        /** Compute a stabilizing intial approximation for X */
        Eigen::MatrixXd init_newton_care(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

        /** Moore-Penrose pseudo-inverse */
        Eigen::MatrixXd pinv(const Eigen::MatrixXd &mat);

        /** solve CARE : C + XA + A'X - XBX*/
        Eigen::MatrixXd care(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C);

        /** Line search for CARE Newton iteration */
        double line_search_care(const double &a, const double &b, const double &c);

        /** Linear Quadratic Regulator:
         * J(x) = INT { xQx + xMu + uRu }dt
         * xdot = Fx + Gu
         */
        Eigen::MatrixXd lqr(const LinearSystem &sys, const Eigen::MatrixXd Q,
                            const Eigen::MatrixXd R, const Eigen::MatrixXd M, const bool &check = false);
    }

}

#endif // POLYMATH_H
