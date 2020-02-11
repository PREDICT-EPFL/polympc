#include "generic_ocp.hpp"

static constexpr int NX = 2;
static constexpr int NU = 2;
static constexpr int NP = 3;

static constexpr int POLY_ORDER = 4;
static constexpr int NUM_SEGMENTS = 3;

using Approximation  = Chebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP>;
using Approximation2 = MSChebyshev<casadi::SX, POLY_ORDER, NUM_SEGMENTS, NX, NU, NP>;

//class MyOCP : public GenericOCP<MyOCP, Approximation>
class MyOCP : public GenericOCP<MyOCP, Approximation2>
{
public:
    /** constructor inheritance */
    using GenericOCP::GenericOCP;
    //MyOCP() = default;
    ~MyOCP() {}

    casadi::Dict solver_options;

    static constexpr double t_start = 0.0;
    static constexpr double t_final = 1.0;

    casadi::SX mayer_term_impl(const casadi::SX &x, const casadi::SX &p)
    {
        return casadi::SX::dot(x,x);
    }

    casadi::SX lagrange_term_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return casadi::SX::dot(x,x) + casadi::SX::dot(u,u);
    }

    casadi::SX system_dynamics_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return x + u;
    }

    casadi::SX inequality_constraints_impl(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return pow(u,2) - casadi::SX::vertcat({10, 10});
        //return casadi::SX(); //!!! deal with it!!!
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

    return 0;
}
