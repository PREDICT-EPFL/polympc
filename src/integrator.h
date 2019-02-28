#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "casadi/casadi.hpp"
#include "Eigen/Dense"
#include "chebyshev.hpp"

/** Solve ODE of the form : xdot = f(x, u) */
class ODESolver
{
public:
    ODESolver(const casadi::Function &rhs, const casadi::Dict &params = casadi::Dict());
    virtual ~ODESolver(){}

    casadi::DM solve(const casadi::DM &x0, const casadi::DM &u, const double &dt);

    void updateParams(const casadi::Dict &params);
    casadi::Dict getParams(){return Parameters;}
    int dim_x(){return nx;}
    int dim_u(){return nu;}

private:
    /** right hand side of the ODE : f(x, u)*/
    casadi::Function RHS;
    casadi::Dict     Parameters;
    int              nx, nu;

    casadi::DM       InitCond;
    int              NumCollocationPoints;
    double           dT;
    int              Method;
    /** optional */
    casadi::DM       XScaling;
    casadi::DM       UScaling;
    int              MaxIter;
    double           Tolerance;
    /** reset line search parameter after each iteration */
    bool             Restart;
    bool             UseWarmStart;

    /** stats */
    double           accuracy;
    int              num_iterations;

    /** Chebychev parameters */
    casadi::DM       Xch, D, Dn, XT;
    casadi::SX       F, G;
    casadi::SX       z, z_u;
    casadi::DM       pseudospectral_solve(const casadi::DM &X0, const casadi::DM &U);

    /** RK4 */
    casadi::DM       rk4_solve(const casadi::DM &X0, const casadi::DM &U, const casadi::DM &dT);

    /** CVODES */
    casadi::DM       cvodes_solve(const casadi::DM &X0, const casadi::DM &U);
    casadi::Function cvodes_integrator;
    casadi::Function create_cvodes_integrator(const casadi::SX &x, const casadi::SX &u, const casadi::Dict);
    bool             cvodes_initialized;
};

/** Pseudospectral solver */
template<int PolyOrder, int NumSegments, int NX, int NU>
class PSODESolver{
public:
    PSODESolver(casadi::Function ODE, const float &dt, const casadi::DMDict &props);
    virtual ~PSODESolver(){}
    casadi::DM solve(const casadi::DM &X0, const casadi::DM &U, const bool full = false);
    casadi::DMDict solve_trajectory(const casadi::DM &X0, const casadi::DM &U, const bool full = false);

    /** scaling matrices */
    casadi::DM P, R;
    int scale;

private:
    casadi::SX G;
    casadi::SX opt_var;
    casadi::SXDict   NLP;
    casadi::Dict     OPTS;
    casadi::DMDict   ARG;
    casadi::Function NLP_Solver;

    casadi::Function Jacobian;
};

template<int PolyOrder, int NumSegments, int NX, int NU>
PSODESolver<PolyOrder, NumSegments, NX, NU>::PSODESolver(casadi::Function ODE, const float &dt, const casadi::DMDict &props)
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

    Chebyshev<casadi::SX, PolyOrder, NumSegments, NX, NU, 0>spectral;

    if(scale)
    {
        casadi::SX z = casadi::SX::sym("z", NX);
        casadi::SX r = casadi::SX::sym("r", NU);
        casadi::SX invP = casadi::SX::inv(P);
        casadi::SX invR = casadi::SX::inv(R);

        casadi::SX SODE = ODE(casadi::SXVector{casadi::SX::mtimes(invP,z), casadi::SX::mtimes(invR, r)})[0];
        SODE = casadi::SX::mtimes(P, SODE);
        casadi::Function FunSODE = casadi::Function("scaled_ode", {z, r}, {SODE});

        G = spectral.CollocateDynamics(FunSODE, 0, dt);
    }
    else
    {
        G = spectral.CollocateDynamics(ODE, 0, dt);
    }

    G = G(casadi::Slice(0, G.size1() - NX), 0);

    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();

    opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    casadi::SX lbg = casadi::SX::zeros(G.size());
    casadi::SX ubg = casadi::SX::zeros(G.size());

    /** set inequality (box) constraints */
    /** state */
    casadi::SX LBX = casadi::SX::repmat(-casadi::SX::inf(), NX, 1);
    casadi::SX UBX = casadi::SX::repmat(casadi::SX::inf(), NX, 1);
    /** dirty hack */
    //LBX[0] = P(1,1) * 2.0;
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

    OPTS["ipopt.linear_solver"]  = "ma97";
    OPTS["ipopt.print_level"]    = 5;
    OPTS["ipopt.tol"]            = 1e-4;
    OPTS["ipopt.acceptable_tol"] = 1e-4;
    OPTS["ipopt.max_iter"]       = 3000;
    OPTS["ipopt.hessian_approximation"] = "limited-memory";
    NLP_Solver = nlpsol("solver", "ipopt", NLP, OPTS);

    std::cout << "problem set \n";

    /** set default args */
    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;

    casadi::DM feasible_state = (UBX + LBX) / 2;
    casadi::DM feasible_control = (UBU + LBU) / 2;

    ARG["x0"] = casadi::DM::vertcat(casadi::DMVector{casadi::DM::repmat(feasible_state, (NumSegments * PolyOrder + 1), 1),
                                    casadi::DM::repmat(feasible_control, (NumSegments * PolyOrder + 1), 1)});
}

template<int PolyOrder, int NumSegments, int NX, int NU>
casadi::DM PSODESolver<PolyOrder, NumSegments, NX, NU>::solve(const casadi::DM &X0, const casadi::DM &U, const bool full)
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

    }
    /** solve */
    casadi::DMDict res = NLP_Solver(ARG);
    casadi::DM NLP_X = res.at("x");
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
template<int PolyOrder, int NumSegments, int NX, int NU>
casadi::DMDict PSODESolver<PolyOrder, NumSegments, NX, NU>::solve_trajectory(const casadi::DM &X0,
                                                                         const casadi::DM &U, const bool full)
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
            casadi::DM solution = casadi::DM::reshape(NLP_X, NX, (NumSegments * PolyOrder + 1));
            solution = casadi::DM::mtimes(invP, solution);
            xt = casadi::DM::vec(solution);
            res.at("x") = xt;
        }
    }
    else
    {
        xt = NLP_X(casadi::Slice(0, NX ));
        if(scale)
        {
            casadi::DM invP = casadi::DM::inv(P);
            xt = casadi::DM::mtimes(invP, xt);
            res.at("x") = xt;
        }
    }

    std::cout << NLP_Solver.stats() << "\n";
    return res;
}


#endif // INTEGRATOR_H
