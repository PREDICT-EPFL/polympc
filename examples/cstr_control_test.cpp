#include "generic_ocp.hpp"

static constexpr int NX = 4; // number of system states
static constexpr int NU = 2; // number of input signals
static constexpr int NP = 0; // number of unknown parameters (can be optimised)
static constexpr int ND = 0; // number of user specified parameters (changed excusively by the user)

static constexpr int POLY_ORDER = 2;
static constexpr int NUM_SEGMENTS = 1;

using Approximation  = Chebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>;     // standard collocation
using Approximation2 = MSChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>;   // ZOH controls in segments
using Approximation3 = SoftChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>; // relaxed collocation constraints

class cstr_ocp : public GenericOCP<cstr_ocp, Approximation>
//class MyOCP : public GenericOCP<MyOCP, Approximation>
{
public:
    /** constructor inheritance */
    using GenericOCP::GenericOCP;
    //MyOCP() = default;
    ~cstr_ocp()  = default;

    casadi::Dict solver_options;

    static constexpr double t_start = 0.0;
    static constexpr double t_final = 30.0;

    /**
     * x - state
     * u - control
     * p - optimised parameters
     * d - static parameters
     */

    casadi::SX mayer_term_impl(const casadi::SX &x, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX p1 = casadi::SX::horzcat({1.464677837458437, 0.667688951672119,  0.35446715117028615, 0.10324422005086348});
        casadi::SX p2 = casadi::SX::horzcat({0.6676889516721198, 1.407812935783267, 0.17788030743777067, 0.050059833257226405});
        casadi::SX p3 = casadi::SX::horzcat({0.3544671511702861, 0.1778803074377706, 0.6336052592712396, 0.01110329497282364});
        casadi::SX p4 = casadi::SX::horzcat({.10324422005086348, 0.0500598332572264, 0.0111032949728236, 0.229412393739723});

        casadi::SX P = casadi::SX::vertcat({p1, p2, p3, p4});
        casadi::SX xs = casadi::SX::vertcat({2.1402105301746182e00, 1.0903043613077321e00, 1.1419108442079495e02, 1.1290659291045561e02});
        casadi::SX delta_x = x - xs;

        return casadi::SX::dot(delta_x, casadi::SX::mtimes(P, delta_x));
    }

    casadi::SX lagrange_term_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX xs = casadi::SX::vertcat({2.1402105301746182e00, 1.0903043613077321e00, 1.1419108442079495e02, 1.1290659291045561e02});
        casadi::SX us = casadi::SX::vertcat({14.19, -1113.50});

        casadi::SX Q = casadi::SX::diag(casadi::SX::vertcat({0.2, 1.0, 0.5, 0.2}));
        casadi::SX R = casadi::SX::diag(casadi::SX::vertcat({0.5, 5.0 * 1.0e-7}));
        casadi::SX delta_x = x - xs;
        casadi::SX delta_u = u - us;
        return casadi::SX::dot(delta_x, casadi::SX::mtimes(Q, delta_x)) + casadi::SX::dot(delta_u, casadi::SX::mtimes(R, delta_u));
    }

    casadi::SX system_dynamics_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        double c_AO = 5.1;
        double v_0 = 104.9;
        double k_w = 4032.0;
        double A_R = 0.215;
        double rho = 0.9342;
        double C_P = 3.01;
        double V_R = 10.0;
        double H_1 = 4.2;
        double H_2 = -11.0;
        double H_3 = -41.85;
        double m_K = 5.0;
        double C_PK = 2.0;
        double k10 =  1.287e12;
        double k20 =  1.287e12;
        double k30 =  9.043e09;
        double E1  =  -9758.3;
        double E2  =  -9758.3;
        double E3  =  -8560.0;
        casadi::SX k_1 = k10 * exp(E1 / (273.15 + x(2)));
        casadi::SX k_2 = k20 * exp(E2 / (273.15 + x(2)));
        casadi::SX k_3 = k30 * exp(E3 / (273.15 + x(2)));
        double TIMEUNITS_PER_HOUR = 3600.0;

        casadi::SX dx1 = (1 / TIMEUNITS_PER_HOUR) * (u(0) * (c_AO - x(0)) - k_1 * x(0) - k_3 * x(0) * x(0));
        casadi::SX dx2 = (1 / TIMEUNITS_PER_HOUR) * (-u(0) * x(1) + k_1 * x(0) - k_2*x(1));
        casadi::SX dx3 = (1 / TIMEUNITS_PER_HOUR) * (u(0) * (v_0-x(2)) + (k_w * A_R / (rho * C_P * V_R)) *
                                                    (x(3) - x(2)) - (1 /( rho*C_P)) * (k_1*x(0) * H_1 + k_2 * x(1) * H_2 + k_3 * x(0) * x(1) * H_3));
        casadi::SX dx4 = (1 / TIMEUNITS_PER_HOUR) * ((1 / (m_K * C_PK)) * (u(1) + k_w * A_R * (x(2)-x(3))));

        return casadi::SX::vertcat({dx1, dx2, dx3, dx4});
    }

    casadi::SX inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX();
    }

    casadi::SX final_inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX();
    }
};

int main(void)
{
    casadi::Dict user_options;
    user_options["ipopt.print_level"] = 5;

    casadi::Dict mpc_options;
    mpc_options["mpc.scaling"] = false;
    mpc_options["mpc.scale_x"] = std::vector<double>{1,1,1,1};
    mpc_options["mpc.scale_u"] = std::vector<double>{1,1};

    cstr_ocp cstr(user_options, mpc_options);
    //casadi::DM lbx = -casadi::DM::inf(4);
    //casadi::DM ubx =  casadi::DM::inf(4);

    casadi::DM lbu = casadi::DM::vertcat({3.0, -9000.0});
    casadi::DM ubu = casadi::DM::vertcat({35.0, 0.0});

    //cstr.set_state_box_constraints(lbx, ubx);
    cstr.set_control_box_constraints(lbu,ubu);

    //cstr.set_parameter_mapping("lox1", std::pair<int, int>(0,2));
    //cstr.set_parameter_mapping("lox2", std::pair<int, int>(2,3));

    //cstr.set_parameters("lox1", casadi::DM::vertcat({1,2}));
    //cstr.set_parameters("lox2", 3);

    casadi::DM x0 = casadi::DM::vertcat({1.0, 0.5, 100.0, 100.0});
    cstr.solve(x0, x0);

    x0 = casadi::DM::vertcat({1.1, 0.508, 100.5, 100.1});
    cstr.solve(x0, x0);

    casadi::DM solution = cstr.get_optimal_control();
    std::cout << "Optimal Control: " << solution << "\n";

    casadi::DM trajectory = cstr.get_optimal_trajectory();
    std::cout << " \n Optimal Trajectory " << trajectory << "\n";

    return EXIT_SUCCESS;
}
