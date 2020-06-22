#ifndef CHEBYSHEV_INTEGRATOR_HPP
#define CHEBYSHEV_INTEGRATOR_HPP

//#include "chebyshev.hpp"
#include "../chebyshev.hpp"

template<typename ODE, int PolyOrder, int NumSegments, int NX, int NU, int NP>
class eig_chebyshev_solver{

private:
    /** Chebychev parameters */
    /*
    casadi::DM       Xch, D, Dn, XT;
    casadi::SX       F, G;
    casadi::SX       z, z_u;
    casadi::DM       pseudospectral_solve(const casadi::DM &X0, const casadi::DM &U);
    */

    eig_chebyshev_solver()
    {
        /**
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
        */
    }
    ~eig_chebyshev_solver(){}

    void solve(const casadi::DM &X0, const casadi::DM &U)
    {
        /** extend nonlinear equalities with initial condition */
        //G = SX::vertcat(SXVector{G, z(Slice(0, z.size1()), z.size2()-1) - SX(X0)});

        /** nature merit function */
        //SX sym_V = 0.5 * SX::norm_inf(G);
        //Function V = Function("MeritFun", {SX::vec(z), SX::vec(z_u)}, {sym_V});

        //std::cout << "Lyapunov full: " << V.n_nodes() << "\n";
        /** prepare auxiliary functions */
        //SX G_jacobian       = SX::jacobian(G, SX::vec(z));
        //Function eval_jac_G = Function("Gcobian", {SX::vec(z), SX::vec(z_u)}, {G_jacobian});
        //Function eval_G     = Function("Gfunc", {SX::vec(z), SX::vec(z_u)}, {G});

        /** initialization
        DM xk = DM::repmat(X0, NumCollocationPoints + 1, 1);
        DM uk = DM::repmat(U,  NumCollocationPoints, 1);
        double alpha_max = 1;
        double alpha_min = 0;
        double alpha = (alpha_max + alpha_min) / 2;

        double err    = std::numeric_limits<double>::infinity();
        int counter  = 0;
        // count also backtracking iterations
        uint k        = 0;
        bool accepted = false;
        */

        /** initialize lyapunov/merit function
        DMVector Vk_res = V(DMVector{xk, uk});
        DM Vk = Vk_res[0];

        Tolerance = Parameters["tol"];
        MaxIter   = Parameters["max_iter"];

        Eigen::MatrixXd JacG;
        Eigen::VectorXd GEig;
        */
        /** damped Newton iterations
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

            // backtracking (not exactly)
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

            // update solution and merit function
            xk     = xk + alpha * dx;
            Vk_res = V(DMVector{xk, uk});
            Vk     = Vk_res[0];

            if (alpha < 1e-10)
            {
                std::cerr << "ODE cannot be solved to specified precision: linesearch infeasible \n";
                break;
            }
            // increase step
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
        */
    }

};



/** Pseudospectral solver based on CasADi framework and IPOPT */
template<int PolyOrder, int NumSegments, int NX, int NU, int NP>
class PSODESolver{
public:
    PSODESolver(casadi::Function ODE, const float &dt, const casadi::DMDict &props, const casadi::Dict &solver_opts = casadi::Dict());
    virtual ~PSODESolver(){}
    casadi::DM solve(const casadi::DM &X0, const casadi::DM &U,const casadi::DM &P = casadi::DM(), const bool full = false);
    casadi::DMDict solve_trajectory(const casadi::DM &X0, const casadi::DM &U, const casadi::DM &P = casadi::DM(), const bool full = false);

    /** scaling matrices */
    casadi::DM P, R;
    int scale;
    void set_dt(const double &_dt);

private:
    casadi::SX       G;
    casadi::SX       opt_var;
    casadi::SXDict   NLP;
    casadi::Dict     OPTS;
    casadi::DMDict   ARG;
    casadi::Function NLP_Solver;

    casadi::Function Jacobian;
};

template<int PolyOrder, int NumSegments, int NX, int NU, int NP>
PSODESolver<PolyOrder, NumSegments, NX, NU, NP>::PSODESolver(casadi::Function ODE, const float &dt,
                                                             const casadi::DMDict &props, const casadi::Dict &solver_opts)
{
    scale = 0;
    P = casadi::DM::eye(NX);
    R = casadi::DM::eye(NU);

    if(props.find("scale") != props.end())
        scale = static_cast<int>(props.find("scale")->second.nonzeros()[0]);
    if(props.find("P") != props.end())
        P = props.find("P")->second;
    if(props.find("R") != props.end())
        R = props.find("R")->second;

    Chebyshev<casadi::SX, PolyOrder, NumSegments, NX, NU, NP>spectral;
    casadi::SX p = casadi::SX::sym("p", NP);

    if(scale)
    {
        casadi::SX z = casadi::SX::sym("z", NX);
        casadi::SX r = casadi::SX::sym("r", NU);
        casadi::SX invP = casadi::SX::inv(P);
        casadi::SX invR = casadi::SX::inv(R);

        casadi::SX SODE;
        casadi::Function FunSODE;

        if(NP > 0)
        {
            SODE = ODE(casadi::SXVector{casadi::SX::mtimes(invP,z), casadi::SX::mtimes(invR, r), p})[0];
            SODE = casadi::SX::mtimes(P, SODE);
            FunSODE = casadi::Function("scaled_ode", {z, r, p}, {SODE});
        }
        else
        {
            /** deal with the old concept */
            SODE = ODE(casadi::SXVector{casadi::SX::mtimes(invP,z), casadi::SX::mtimes(invR, r)})[0];
            SODE = casadi::SX::mtimes(P, SODE);
            FunSODE = casadi::Function("scaled_ode", {z, r}, {SODE});
        }

        G = spectral.CollocateDynamics(FunSODE, 0, dt);
    }
    else
    {
        G = spectral.CollocateDynamics(ODE, 0, dt);
    }

    G = G(casadi::Slice(0, G.size1() - NX), 0);

    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();
    casadi::SX varp = spectral.VarP();

    opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    casadi::SX lbg = casadi::SX::zeros(G.size());
    casadi::SX ubg = casadi::SX::zeros(G.size());

    /** set inequality (box) constraints */
    /** state */
    casadi::SX LBX = casadi::SX::repmat(-casadi::SX::inf(), NX, 1);
    casadi::SX UBX = casadi::SX::repmat(casadi::SX::inf(), NX, 1);

    casadi::SX lbx = casadi::SX::repmat(LBX, (NumSegments * PolyOrder + 1), 1);
    casadi::SX ubx = casadi::SX::repmat(UBX, (NumSegments * PolyOrder + 1), 1);

    /** control */
    casadi::SX LBU = casadi::SX::repmat(-casadi::SX::inf(), NU, 1);
    casadi::SX UBU = casadi::SX::repmat(casadi::SX::inf(), NU, 1);
    lbx = casadi::SX::vertcat( {lbx, casadi::SX::repmat(LBU, (NumSegments * PolyOrder + 1), 1)} );
    ubx = casadi::SX::vertcat( {ubx, casadi::SX::repmat(UBU, (NumSegments * PolyOrder + 1), 1)} );

    /** formulate NLP */
    NLP["x"] = opt_var;
    NLP["f"] = 1e-3 * casadi::SX::dot(G,G);
    NLP["g"] = G;
    NLP["p"] = varp;

    OPTS["ipopt.linear_solver"]  = "mumps"; // mumps
    OPTS["ipopt.print_level"]    = 5;
    OPTS["ipopt.tol"]            = 1e-4;
    OPTS["ipopt.acceptable_tol"] = 1e-4;
    OPTS["ipopt.max_iter"]       = 3000;
    OPTS["ipopt.hessian_approximation"] = "limited-memory";

    /** set uder-defined options */
    if(solver_opts.find("ipopt.linear_solver") != solver_opts.end())
        OPTS["ipopt.linear_solver"] = solver_opts.find("ipopt.linear_solver")->second;

    NLP_Solver = nlpsol("solver", "ipopt", NLP, OPTS);

    std::cout << "problem set \n";

    /** set default args */
    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;
    ARG["p"]   = casadi::DM::zeros(varp.size1());

    casadi::DM feasible_state = (UBX + LBX) / 2;
    casadi::DM feasible_control = (UBU + LBU) / 2;

    ARG["x0"] = casadi::DM::vertcat(casadi::DMVector{casadi::DM::repmat(feasible_state, (NumSegments * PolyOrder + 1), 1),
                                    casadi::DM::repmat(feasible_control, (NumSegments * PolyOrder + 1), 1)});
}

template<int PolyOrder, int NumSegments, int NX, int NU, int NP>
casadi::DM PSODESolver<PolyOrder, NumSegments, NX, NU, NP>::solve(const casadi::DM &X0, const casadi::DM &U, const casadi::DM &P, const bool full)
{
    casadi::Slice x_var = casadi::Slice(0, (NumSegments * PolyOrder + 1) * NX);
    casadi::Slice u_var = casadi::Slice((NumSegments * PolyOrder + 1) * NX, (NumSegments * PolyOrder + 1) * (NX + NU));
    int idx_in = NumSegments * PolyOrder * NX;
    int idx_out = idx_in + NX;

    if(U.size1() != NU)
    {
        std::cout << "Control vector should be " << NU << " provided " << U.size1() << "\n";
        return X0;
    }

    std::cout << "x_var indices: " << x_var << " idx_in: " << idx_in << "\n";

    if (scale)
    {
        ARG["x0"](x_var, 0) = casadi::DM::repmat(casadi::DM::mtimes(P, X0), (NumSegments * PolyOrder + 1), 1);
        ARG["x0"](u_var, 0) = casadi::DM::repmat(casadi::DM::mtimes(R, U), (NumSegments * PolyOrder + 1), 1);

        /** initial point */
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = casadi::DM::mtimes(P, X0);
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = casadi::DM::mtimes(P, X0);

        /** control */
        ARG["lbx"](u_var, 0) = casadi::DM::repmat(casadi::DM::mtimes(R, U), (NumSegments * PolyOrder + 1), 1);
        ARG["ubx"](u_var, 0) = casadi::DM::repmat(casadi::DM::mtimes(R, U), (NumSegments * PolyOrder + 1), 1);

        /** parameters */
        ARG["p"] = P;
    }
    else
    {
        ARG["x0"](x_var, 0) = casadi::DM::repmat(X0, (NumSegments * PolyOrder + 1), 1);
        ARG["x0"](u_var, 0) = casadi::DM::repmat(U, (NumSegments * PolyOrder + 1), 1);

        /** initial point */
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = X0;
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = X0;

        /** control */
        ARG["lbx"](u_var, 0) = casadi::DM::repmat(U, (NumSegments * PolyOrder + 1), 1);
        ARG["ubx"](u_var, 0) = casadi::DM::repmat(U, (NumSegments * PolyOrder + 1), 1);

        /** parameters */
        ARG["p"] = P;

    }
    /** solve */
    casadi::DMDict res = NLP_Solver(ARG);
    casadi::DM NLP_X   = res.at("x");
    casadi::DM xt;

    if(full)
    {
        xt = NLP_X(casadi::Slice(0, (NumSegments * PolyOrder + 1) * NX ));
        if(scale)
        {
            casadi::DM invP = casadi::DM::inv(P);
            casadi::DM solution = casadi::DM::reshape(NLP_X, NX, (NumSegments * PolyOrder + 1));
            solution = casadi::DM::mtimes(invP, solution);
            xt = casadi::DM::vec(solution);
        }
    }
    else
    {
        xt = NLP_X(casadi::Slice(0, NX ));
        if(scale)
        {
            casadi::DM invP = casadi::DM::inv(P);
            xt = casadi::DM::mtimes(invP, xt);
        }
    }

    std::cout << NLP_Solver.stats() << "\n";
    return xt;
}

/** provide the whole control trajectory */
template<int PolyOrder, int NumSegments, int NX, int NU, int NP>
casadi::DMDict PSODESolver<PolyOrder, NumSegments, NX, NU, NP>::solve_trajectory(const casadi::DM &X0, const casadi::DM &U,
                                                                                 const casadi::DM &P,  const bool full)
{
    casadi::Slice x_var = casadi::Slice(0, (NumSegments * PolyOrder + 1) * NX);
    casadi::Slice u_var = casadi::Slice((NumSegments * PolyOrder + 1) * NX, (NumSegments * PolyOrder + 1) * (NX + NU));
    int idx_in = NumSegments * PolyOrder * NX;
    int idx_out = idx_in + NX;

    if(U.size1() != (NumSegments * PolyOrder + 1) * NU)
    {
        std::cout << "Control vector size should be " << (NumSegments * PolyOrder + 1) * NU << " provided: " << U.size1() << "\n";
        casadi::DMDict fault;
        fault["x"] = X0;
        fault["lam_x"] = casadi::DM::zeros(X0.size1());
        fault["lam_g"] = casadi::DM::zeros(X0.size1());
        return fault;
    }

    if (scale)
    {
        casadi::DM U_ = casadi::DM::reshape(U, NU, (NumSegments * PolyOrder + 1));
        U_ = casadi::DM::mtimes(R, U_);

        ARG["x0"](x_var, 0) = casadi::DM::repmat(casadi::DM::mtimes(P, X0), (NumSegments * PolyOrder + 1), 1);
        ARG["x0"](u_var, 0) = casadi::DM::vec(U_);

        /** initial point */
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = casadi::DM::mtimes(P, X0);
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = casadi::DM::mtimes(P, X0);

        /** control */
        ARG["lbx"](u_var, 0) = casadi::DM::vec(U_);
        ARG["ubx"](u_var, 0) = casadi::DM::vec(U_);

        /** parameters */
        ARG["p"] = P;
    }
    else
    {
        ARG["x0"](x_var, 0) = casadi::DM::repmat(X0, (NumSegments * PolyOrder + 1), 1);
        ARG["x0"](u_var, 0) = U;

        /** initial point */
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = X0;
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = X0;

        /** control */
        ARG["lbx"](u_var, 0) = U;
        ARG["ubx"](u_var, 0) = U;

        /** parameters */
        ARG["p"] = P;
    }
    /** solve */
    casadi::DMDict res = NLP_Solver(ARG);
    casadi::DM NLP_X     = res.at("x");
    casadi::DM xt;

    if(full)
    {
        xt = NLP_X(casadi::Slice(0, (NumSegments * PolyOrder + 1) * NX ));
        if(scale)
        {
            casadi::DM invP = casadi::DM::inv(P);
            casadi::DM solution = casadi::DM::reshape(xt, NX, (NumSegments * PolyOrder + 1));
            solution = casadi::DM::mtimes(invP, solution);
            xt = casadi::DM::vec(solution);
        }
    }
    else
    {
        xt = NLP_X(casadi::Slice(0, NX ));
        if(scale)
        {
            casadi::DM invP = casadi::DM::inv(P);
            xt = casadi::DM::mtimes(invP, xt);
        }
    }
    res.at("x") = xt;

    std::cout << NLP_Solver.stats() << "\n";
    return res;
}





#endif // CHEBYSHEV_INTEGRATOR_HPP
