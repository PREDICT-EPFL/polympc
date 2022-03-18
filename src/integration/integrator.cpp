#include "integrator.h"

using namespace casadi;

namespace polympc {

/** ODESolver class implementation */
ODESolver::ODESolver(const Function &rhs, const Dict &params)
{
    /** define default values for parameters */
    Tolerance           = 1e-8;
    MaxIter             = 300;
    dT                  = 1;
    Method              = CVODES; //{RK4}

    Parameters["method"]        = Method;
    Parameters["tf"]            = dT;
    Parameters["max_iter"]      = MaxIter;
    Parameters["tol"]           = Tolerance;

    /** set user defined parameters */
    if(params.empty())
        std::cout << "Running ODE solver with default parameters: \n" << Parameters << "\n";
    else
    {
        updateParams(params);
        std::cout << "Running ODE solver with specified parameters: \n" << Parameters << "\n";
    }

    /** @todo: revise parameters */
    dT = Parameters["tf"];
    /** Define integration scheme here */
    /** space dimensionality */
    nx = static_cast<int>(rhs.numel_in(0));
    nu = static_cast<int>(rhs.numel_in(1));
    if(rhs.n_in() > 2)
        np = static_cast<int>(rhs.numel_in(2));

    SX x = SX::sym("x", nx);
    SX u = SX::sym("u", nu);
    SX p = SX::sym("p", np);
    SX dt = SX::sym("dt");
    SXVector sym_ode;
    if(np > 0)
        sym_ode = rhs(SXVector{x, u, p});
    else
        sym_ode = rhs(SXVector{x, u});

    RHS = casadi::Function("RHS", {x, u, p},{sym_ode[0]});

    SXDict ode = {{"x", x}, {"p", SX::vertcat({u, dt, p})}, {"ode", dt * sym_ode[0]}};
    Dict opts = {{"tf", 1.0}, {"abstol", Tolerance}, {"max_num_steps" , MaxIter}};

    /** initialization of integration methods */
    Method = Parameters["method"];
    switch (Method) {
    case RK4:
        std::cout << "Creating RK4 solver... \n";
        break;
    case CVODES:
        std::cout << "Creating CVODES solver... \n";
        cvodes_integrator = casadi::integrator("CVODES_INT", "cvodes", ode, opts);
        break;
    default:
        std::cerr << "Unknown method: " << Method << "\n";
        break;
    }
}

DM ODESolver::rk4_solve(const DM &x0, const DM &u, const DM &p, const DM &dt)
{
    DMVector res = RHS(DMVector{x0, u, p});
    DM k1 = res[0];
    res = RHS(DMVector{x0 + 0.5 * dt * k1, u, p});
    DM k2 = res[0];
    res = RHS(DMVector{x0 + 0.5 * dt * k2, u, p});
    DM k3 = res[0];
    res = RHS(DMVector{x0 + dt * k3, u, p});
    DM k4 = res[0];

    return x0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
}

void ODESolver::updateParams(const Dict &params)
{
    for (Dict::const_iterator it = params.begin(); it != params.end(); ++it)
    {
        if(Parameters.count(it->first) > 0)
            Parameters[it->first] = it->second;
        else
            std::cout << "Unknown parameter: " << it->first << "\n";
    }
}

DM ODESolver::cvodes_solve(const DM &x0, const DM &u, const DM &p, const DM &dt)
{
    DMDict out;
    try
    {
        DMDict args = {{"x0", x0}, {"p", DM::vertcat({u, dt, p})}};
        out = cvodes_integrator(args);
    }
    catch(std::exception &e)
    {
        std::cout << "At state x0 : " << x0 << "\n";
        std::cout << "At control u:"  << u  << "\n";
        std::cout << "At control p:"  << p  << "\n";
        std::cout << "At control dt:" << dt << "\n";
        std::cout << "CVODES exception " << e.what() << "\n";
    }

    return out["xf"];
}

DM ODESolver::solve(const DM &x0, const DM &u, const DM &p, const double &dt)
{
    DM solution;
    Method = Parameters["method"];
    dT     = Parameters["tf"];
    switch (Method) {
    case RK4:
        solution = rk4_solve(x0, u, p, dt);
        break;
    case CVODES:
        solution = cvodes_solve(x0, u, p , dt);
        break;
    default:
        break;
    }

    return solution;
}

DM ODESolver::solve(const DM &x0, const DM &u, const double &dt)
{
    casadi::DM p = casadi::DM::zeros(np); // dummy input
    return solve(x0, u, p, dt);
}

} // polympc namespace