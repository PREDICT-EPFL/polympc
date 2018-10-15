#include "integrator.h"

using namespace casadi;

/** ODESolver class implementation */
ODESolver::ODESolver(const Function &rhs, const Dict &params)
{
    RHS = rhs;

    /** define default values for parameters */
    NumCollocationPoints = 10;
    Tolerance           = 1e-8;
    MaxIter             = 300;
    Restart             = false;
    UseWarmStart        = false;
    dT                  = 1;
    Method              = CVODES; //{CHEBYCHEV, RK4}

    Parameters["method"]        = Method;
    Parameters["tf"]            = dT;
    Parameters["restart"]       = Restart;
    Parameters["max_iter"]      = MaxIter;
    Parameters["tol"]           = Tolerance;
    Parameters["poly_order"]    = NumCollocationPoints;

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
    nx = RHS.nnz_out();
    nu = RHS.nnz_in() - nx;
    std::pair<double, double> time_interval;

    SX x = SX::sym("x", nx);
    SX u = SX::sym("u", nu);
    SXVector sym_ode = RHS(SXVector{x, u});

    SXDict ode = {{"x", x}, {"p", u}, {"ode", sym_ode[0]}};
    Dict opts = {{"tf", dT}, {"abstol", Tolerance}, {"max_num_steps" , MaxIter}};

    /** initialization of integration methods */
    Method = Parameters["method"];
    switch (Method) {
    case RK4:
        std::cout << "Creating RK4 solver... \n";
        break;
    case CVODES:
        std::cout << "Creating CVODES solver... \n";
        cvodes_initialized = false;
        cvodes_integrator = integrator("CVODES_INT", "cvodes", ode, opts);
        break;
    case CHEBYCHEV:
        std::cout << "Creating CHEB solver... \n";
        // generate grid and differentiation matrix
        //redo this with chebyshev class
        time_interval = std::make_pair<double, double>(0, double(dT));
        polymath::cheb(Xch, D, NumCollocationPoints, time_interval);
        D(D.size1() - 1, Slice(0, D.size2())) = DM::zeros(NumCollocationPoints + 1);
        D(D.size1() - 1, D.size2() - 1) = 1;
        Dn = DM::kron(D, DM::eye(nx));

        //set up equations
        z = SX::sym("z", nx, NumCollocationPoints + 1);
        z_u = SX::sym("z_u", nu, NumCollocationPoints);

        F = polymath::mat_dynamics( z,  z_u, RHS);
        G = SX::mtimes(Dn, SX::vec(z)) - F;
        G = G(Slice(0, NumCollocationPoints * nx), 0);

        break;
    default:
        std::cerr << "Unknown method: " << Method << "\n";
        break;
    }
}

DM ODESolver::rk4_solve(const DM &x0, const DM &u, const DM &dt)
{
    DMVector res = RHS(DMVector{x0, u});
    DM k1 = res[0];
    res = RHS(DMVector{x0 + 0.5 * dt * k1, u});
    DM k2 = res[0];
    res = RHS(DMVector{x0 + 0.5 * dt * k2, u});
    DM k3 = res[0];
    res = RHS(DMVector{x0 + dt * k3, u});
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

DM ODESolver::cvodes_solve(const DM &X0, const DM &U)
{
    DMDict out;
    try
    {
        DMDict args = {{"x0", X0}, {"p", U}};
        out = cvodes_integrator(args);
    }
    catch(std::exception &e)
    {
        std::cout << "At state x0 : " << X0 << "\n";
        std::cout << "At control U:" << U << "\n";
        std::cout << "CVODES exception " << e.what() << "\n";
    }

    return out["xf"];
}

DM ODESolver::pseudospectral_solve(const DM &X0, const DM &U)
{
    /** extend nonlinear equalities with initial condition **/
    G = SX::vertcat(SXVector{G, z(Slice(0, z.size1()), z.size2()-1) - SX(X0)});

    /** nature merit function */
    SX sym_V = 0.5 * SX::norm_inf(G);
    Function V = Function("MeritFun", {SX::vec(z), SX::vec(z_u)}, {sym_V});

    //std::cout << "Lyapunov full: " << V.n_nodes() << "\n";
    /** prepare auxiliary functions */
    SX G_jacobian       = SX::jacobian(G, SX::vec(z));
    Function eval_jac_G = Function("Gcobian", {SX::vec(z), SX::vec(z_u)}, {G_jacobian});
    Function eval_G     = Function("Gfunc", {SX::vec(z), SX::vec(z_u)}, {G});

    /** initialization */
    DM xk = DM::repmat(X0, NumCollocationPoints + 1, 1);
    DM uk = DM::repmat(U, NumCollocationPoints, 1);
    double alpha_max = 1;
    double alpha_min = 0;
    double alpha = (alpha_max + alpha_min) / 2;

    double err    = std::numeric_limits<double>::infinity();
    uint counter  = 0;
    /** count also backtracking iterations */
    uint k        = 0;
    bool accepted = false;

    /** initialize lyapunov/merit function */
    DMVector Vk_res = V(DMVector{xk, uk});
    DM Vk = Vk_res[0];

    Tolerance = Parameters["tol"];
    MaxIter   = Parameters["max_iter"];

    Eigen::MatrixXd JacG;
    Eigen::VectorXd GEig;
    /** damped Newton iterations */
    while (err >= Tolerance)
    {
        counter++;

        DMVector dG_dx_res = eval_jac_G(DMVector{xk, uk});
        DM dG_dx = dG_dx_res[0];
        DMVector G_res = eval_G(DMVector{xk, uk});
        DM G_ = G_res[0];

        //DM dx = -DM::solve(dG_dx, G_);

        JacG = Eigen::MatrixXd::Map(DM::densify(dG_dx).nonzeros().data(), G.size1(), SX::vec(z).size1());
        GEig = Eigen::VectorXd::Map(DM::densify(G_).nonzeros().data(), SX::vec(z).size1());

        Eigen::VectorXd sol = JacG.partialPivLu().solve(-GEig);
        std::vector<double> sold;
        sold.resize(static_cast<size_t>(sol.size()));
        Eigen::Map<Eigen::VectorXd>(&sold[0], sol.size()) = sol;
        DM dx = DM(sold);

        /** backtracking (not exactly) */
        DMVector Vtrial_res;
        DM Vtrial;
        while(!accepted)
        {
            Vtrial_res = V(DMVector{xk +  alpha * dx, uk});
            Vtrial     = Vtrial_res[0];
            k++;
            if(Vtrial.nonzeros()[0] > Vk.nonzeros()[0])
            {
                alpha_max = alpha;
                alpha = (alpha_max + alpha_min) / 2;
                accepted = false;
            }
            else
            {
                alpha_max = (err < 1) ? 2 : 1;
                accepted = true;
                k = 0;
            }
        }

        /** update solution and merit function */
        xk     = xk + alpha * dx;
        Vk_res = V(DMVector{xk, uk});
        Vk     = Vk_res[0];

        if (alpha < 1e-10)
        {
            std::cerr << "ODE cannot be solved to specified precision: linesearch infeasible \n";
            break;
        }
        /** increase step */
        alpha = (alpha_max + alpha_min) / 2;
        accepted = false;

        DM err_inf = DM::norm_inf(G_);
        err = err_inf.nonzeros()[0];
        std::cout << "iteration: " << counter << " " << "error: " << err << "\n";
        if(counter >= MaxIter)
        {
            std::cerr << "ODE cannot be solved to specified precision \n";
            break;
        }
    }

    DM xt = xk(Slice(0, X0.size1()), 0);
    XT = xk;

    DMVector G_res = eval_G(DMVector{xk, uk});
    DM G_ = G_res[0];
    DM err_inf = DM::norm_inf(G_);
    std::cout << "Chebyshev solver: error : " << err_inf << "\n";
    //int nx = X0.size1();
    //G = G(Slice(0, NumCollocationPoints * nx), 0);
    return xt;
}

DM ODESolver::solve(const DM &x0, const DM &u, const double &dt)
{
    DM solution;
    Method = Parameters["method"];
    dT     = Parameters["tf"];
    switch (Method) {
    case RK4:
        "Solving with RK4 ... \n";
        solution = rk4_solve(x0, u, dt);
        break;
    case CVODES:
        "Solving with CVODES ... \n";
        /** @todo: consider time scaling */
        if (fabs(dT - dt) > 1e-5)
            std::cerr << "Inconsistent integration time: CVODES solver should be reinitialized \n";
        solution = cvodes_solve(x0, u);
        break;
    case CHEBYCHEV:
        "Solving with CHEB ... \n";
        if (fabs(dT - dt) > 1e-5)
            std::cerr << "Inconsistent integration time: CHEBYCHEV solver should be reinitialized \n";
        solution = pseudospectral_solve(x0, u);
        break;
    default:
        break;
    }

    return solution;
}
