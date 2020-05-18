#include "generic_ocp.hpp"

static constexpr int NX = 2; // number of system states
static constexpr int NU = 2; // number of input signals
static constexpr int NP = 0; // number of unknown parameters (can be optimised)
static constexpr int ND = 3; // number of user specified parameters (changed excusively by the user)

static constexpr int POLY_ORDER = 4;
static constexpr int NUM_SEGMENTS = 3;

using Approximation  = Chebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>;     // standard collocation
using Approximation2 = MSChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>;   // ZOH controls in segments
using Approximation3 = SoftChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP, ND>; // relaxed collocation constraints

class MyOCP : public GenericOCP<MyOCP, Approximation3>
//class MyOCP : public GenericOCP<MyOCP, Approximation>
{
public:
    /** constructor inheritance */
    using GenericOCP::GenericOCP;
    //MyOCP() = default;
    ~MyOCP() {}

    casadi::Dict solver_options;

    static constexpr double t_start = 0.0;
    static constexpr double t_final = 1.0;

    /**
     * x - state
     * u - control
     * p - optimised parameters
     * d - static parameters
     */

    casadi::SX mayer_term_impl(const casadi::SX &x, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX::dot(x,x);
    }

    casadi::SX lagrange_term_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::DM Weight = casadi::DM::eye(NU);
        return casadi::SX::dot(x,x) + casadi::SX::dot(u,u) + norm_diff(u, Weight); //+ norm_ddiff(u, Weight);
    }

    casadi::SX system_dynamics_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return x + u;
    }

    casadi::SX inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        casadi::SX ineq1 = pow(u,2) - casadi::SX::vertcat({10, 10});
        casadi::SX ineq2 = diff(x, 5 * casadi::DM::ones(NX));
        return casadi::SX::vertcat({ineq1, ineq2});
    }

    casadi::SX final_inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return casadi::SX::vertcat({x(0),-x(0)});
    }
};


int main(void)
{
    casadi::Dict user_options;
    user_options["ipopt.print_level"] = 5;

    casadi::Dict mpc_options;
    mpc_options["mpc.scaling"] = true;
    mpc_options["mpc.scale_x"] = std::vector<double>{1,2};
    mpc_options["mpc.scale_u"] = std::vector<double>{3,4};

    MyOCP lox(user_options, mpc_options);
    casadi::DM lbx = casadi::DM::vertcat({-10,-10});
    casadi::DM ubx = casadi::DM::vertcat({10, 10});

    casadi::DM lbu = casadi::DM::vertcat({-5,-5});
    casadi::DM ubu = casadi::DM::vertcat({5, 5});

    lox.set_state_box_constraints(lbx, ubx);
    lox.set_control_box_constraints(lbu,ubu);

    lox.set_parameter_mapping("lox1", std::pair<int, int>(0,2));
    lox.set_parameter_mapping("lox2", std::pair<int, int>(2,3));

    lox.set_parameters("lox1", casadi::DM::vertcat({1,2}));
    lox.set_parameters("lox2", 3);

    casadi::DM x0 = casadi::DM::vertcat({1.0,1.0});
    lox.solve(x0, x0);

    casadi::DM solution = lox.get_optimal_control();
    std::cout << "Optimal Control: " << solution << "\n";

    casadi::DM trajectory = lox.get_optimal_trajectory();
    std::cout << " \n Optimal Trajectory " << trajectory << "\n";

    return 0;
}
