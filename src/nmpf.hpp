#ifndef NMPF_HPP
#define NMPF_HPP

#include <memory>
#include "polymath.h"
#include "chebyshev.hpp"

namespace polympc {

/**
std::set<std::string> available_options = {"spectral.number_segments", "spectral.poly_order", "spectral.tf"};

template<typename ParamType>
ParamType get_param(const std::string &key, const casadi::Dict dict, const ParamType &default_val)
{
    if((available_options.find(key) != available_options.end()) && (dict.find(key) != dict.end()))
        return dict.find(key)->second;
    else if((available_options.find(key) == available_options.end()) && (dict.find(key) != dict.end()))
    {
        std::cout << "Unknown  parameter: " << key << "\n"; // << "Available parameters: " << available_options << "\n";
        return default_val;
    }
    else
        return default_val;
}
*/

template <typename System, typename Path, int NX, int NU, int NumSegments = 2, int PolyOrder = 5>
class nmpf
{
public:
    nmpf(const double &tf = 1.0, const casadi::DMDict &mpc_options = casadi::DMDict(), const casadi::Dict &solver_options = casadi::Dict());
    ~nmpf(){}

    /** contsraints setters */
    void setLBX(const casadi::DM &_lbx)
    {
        ARG["lbx"](casadi::Slice(0, (NX + 2) * (PolyOrder * NumSegments + 1 ))) =
                   casadi::SX::repmat(casadi::SX::mtimes(Scale_X, _lbx), PolyOrder * NumSegments + 1, 1);
    }

    void setUBX(const casadi::DM &_ubx)
    {
        ARG["ubx"](casadi::Slice(0, (NX + 2) * (PolyOrder * NumSegments + 1 ))) =
                   casadi::SX::repmat(casadi::SX::mtimes(Scale_X, _ubx), PolyOrder * NumSegments + 1, 1);
    }


    void setLBU(const casadi::DM &_lbu)
    {
        int start = (NX + 2) * (PolyOrder * NumSegments + 1 );
        int finish = start + (NU + 1) * (PolyOrder * NumSegments + 1 );
        ARG["lbx"](casadi::Slice(start, finish)) = casadi::SX::repmat(casadi::SX::mtimes(Scale_U, _lbu), PolyOrder * NumSegments + 1, 1);
    }
    void setUBU(const casadi::DM &_ubu)
    {
        int start = (NX + 2) * (PolyOrder * NumSegments + 1 );
        int finish = start + (NU + 1) * (PolyOrder * NumSegments + 1 );
        ARG["ubx"](casadi::Slice(start, finish)) = casadi::SX::repmat(casadi::SX::mtimes(Scale_U, _ubu), PolyOrder * NumSegments + 1, 1);
    }

    void setStateScaling(const casadi::DM &Scaling){Scale_X = Scaling;
                                                      invSX = casadi::DM::solve(Scale_X, casadi::DM::eye(Scale_X.size1()));}
    void setControlScaling(const casadi::DM &Scaling){Scale_U = Scaling;
                                                      invSU = casadi::DM::solve(Scale_U, casadi::DM::eye(Scale_U.size1()));}

    void setReferenceVelocity(const casadi::DM &vel_ref){ARG["p"](0) = Scale_X(nx + 1,nx + 1) * vel_ref;}
    void setPathParameters(const casadi::DM &path_params){ARG["p"](casadi::Slice(1, num_path_parameters)) = path_params;}

    void setPath(const casadi::SX &_path);
    void createNLP(const casadi::Dict &solver_options);
    void updateParams(const casadi::Dict &params);

    void enableWarmStart(){WARM_START = true;}
    void disableWarmStart(){WARM_START = false;}
    void computeControl(const casadi::DM &_X0);
    casadi::DM findClosestPointOnPath(const casadi::DM &position, const casadi::DM &init_guess = casadi::DM(0));

    casadi::DM getOptimalControl(){return OptimalControl;}
    casadi::DM getOptimalTrajetory(){return OptimalTrajectory;}
    casadi::Function getPathFunction(){return PathFunc;}
    casadi::Function getAugDynamics(){return AugDynamics;}
    casadi::Dict getStats(){return stats;}
    bool initialized(){return _initialized;}

    double getPathError();
    double getVirtState();
    double getVelocityError();

private:
    System system;
    Path   PathFunc;
    uint   nx, nu, ny, np;
    double Tf;

    casadi::SX       Contraints;
    casadi::Function ContraintsFunc;

    casadi::SX reference_velocity;
    casadi::SX path_parameters;
    casadi::DM default_path_parameters;

    /** state box constraints */
    casadi::DM LBX, UBX;

    /** nonlinear inequality constraints */
    casadi::DM LBG, UBG;

    /** control box constraints */
    casadi::DM LBU, UBU;

    /** state and control scaling matrixces */
    casadi::DM Scale_X, invSX;
    casadi::DM Scale_U, invSU;

    /** cost function weight matrices */
    casadi::SX Q, R, W, Wq;

    casadi::DM NLP_X, NLP_LAM_G, NLP_LAM_X;
    casadi::Function NLP_Solver;
    casadi::SXDict NLP;
    casadi::Dict OPTS;
    casadi::DMDict ARG;
    casadi::Dict stats;

    casadi::DM OptimalControl;
    casadi::DM OptimalTrajectory;

    unsigned NUM_COLLOCATION_POINTS;
    bool WARM_START;
    bool _initialized;
    bool scale;
    double reset_path_after;
    int num_path_parameters;

    /** TRACE FUNCTIONS */
    casadi::Function DynamicsFunc;
    casadi::Function DynamicConstraints;
    casadi::Function PerformanceIndex;
    casadi::Function CostFunction;
    casadi::Function PathError;
    casadi::Function VelError;

    casadi::Function AugJacobian;
    casadi::Function AugDynamics;
};

template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::nmpf(const double &tf, const casadi::DMDict &mpc_options, const casadi::Dict &solver_options)
{
    /** set up default */
    casadi::Function dynamics = system.getDynamics();
    nx = dynamics.nnz_out();
    nu = dynamics.nnz_in() - nx;
    Tf = tf;

    assert(NX == nx);
    assert(NU == nu);

    casadi::Function output   = system.getOutputMapping();
    ny = output.nnz_out();

    Q  = casadi::SX::eye(ny);
    R  = casadi::SX::eye(nu + 1);
    W  = 1.0;


    Scale_X = casadi::DM::eye(nx + 2);  invSX = casadi::DM::eye(nx + 2);
    Scale_U = casadi::DM::eye(nu + 1);  invSU = casadi::DM::eye(nu + 1);

    /** problem scaling */
    scale = false;
    if(mpc_options.find("mpc.scaling") != mpc_options.end())
        scale = static_cast<bool>(mpc_options.find("mpc.scaling")->second.nonzeros()[0]);

    if(mpc_options.find("mpc.scale_x") != mpc_options.end() && scale)
    {
        Scale_X = mpc_options.find("mpc.scale_x")->second;
        assert((nx+2) == Scale_X.size1());
        assert((nx+2) == Scale_X.size2());
        invSX = casadi::DM::solve(Scale_X, casadi::DM::eye(Scale_X.size1()));
    }

    if(mpc_options.find("mpc.scale_u") != mpc_options.end() && scale)
    {
        Scale_U = mpc_options.find("mpc.scale_u")->second;
        assert((nu+1) == Scale_U.size1());
        assert((nu+1) == Scale_U.size2());
        invSU = casadi::DM::solve(Scale_U, casadi::DM::eye(Scale_U.size1()));
    }

    /** cost function weights */
    if(mpc_options.find("mpc.Q") != mpc_options.end())
    {
        Q = mpc_options.find("mpc.Q")->second;
        assert(ny == Q.size1());
        assert(ny == Q.size2());
    }

    if(mpc_options.find("mpc.R") != mpc_options.end())
    {
        R = mpc_options.find("mpc.R")->second;
        assert((nu + 1) == R.size1());
        assert((ny + 1) == R.size2());
    }

    if(mpc_options.find("mpc.W") != mpc_options.end())
    {
        W = mpc_options.find("mpc.W")->second;
        assert(1 == W.size1());
        assert(1 == W.size2());
    }

    reset_path_after = 2 * M_PI;
    if(mpc_options.find("mpc.reset_path_after") != mpc_options.end())
    {
        casadi::DM tmp = mpc_options.find("mpc.reset_path_after")->second;
        reset_path_after  = tmp.nonzeros()[0];
    }

    num_path_parameters = 0;
    if(mpc_options.find("mpc.num_path_params") != mpc_options.end())
    {
        casadi::DM tmp = mpc_options.find("mpc.num_path_params")->second;
        num_path_parameters  = tmp.nonzeros()[0] >= 0 ? tmp.nonzeros()[0] : 0;
    }

    if(mpc_options.find("mpc.default_path_params") != mpc_options.end())
    {
        default_path_parameters = mpc_options.find("mpc.default_path_params")->second;
        assert(default_path_parameters.size1() == num_path_parameters);
    }

    /** assume unconstrained problem */
    LBX = -casadi::DM::inf(nx + 2);
    UBX = casadi::DM::inf(nx + 2);
    LBU = -casadi::DM::inf(nu + 1);
    UBU = casadi::DM::inf(nu + 1);

    WARM_START  = false;
    _initialized = false;

    /** create NLP */
    createNLP(solver_options);
}

/** update solver paramters */
template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
void nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::updateParams(const casadi::Dict &params)
{
    for (casadi::Dict::const_iterator it = params.begin(); it != params.end(); ++it)
    {
        OPTS[it->first] = it->second;
    }
}

template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
void nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::createNLP(const casadi::Dict &solver_options)
{
    /** get dynamics function and state Jacobian */
    casadi::Function dynamics = system.getDynamics();
    casadi::Function output   = system.getOutputMapping();
    casadi::SX x = casadi::SX::sym("x", nx);
    casadi::SX u = casadi::SX::sym("u", nu);

    /** define augmented dynamics of path parameter */
    casadi::SX v = casadi::SX::sym("v", 2);
    casadi::SX uv = casadi::SX::sym("uv");
    casadi::SX Av = casadi::SX::zeros(2,2); Av(0,1) = 1;
    casadi::SX Bv = casadi::SX::zeros(2,1); Bv(1,0) = 1;

    /** parameter dynamics */
    casadi::SX p_dynamics = casadi::SX::mtimes(Av, v) + casadi::SX::mtimes(Bv, uv);

    /** augmented system */
    casadi::SX aug_state = casadi::SX::vertcat({x, v});
    casadi::SX aug_control = casadi::SX::vertcat({u, uv});

    /** evaluate dynamics */
    casadi::SX sym_dynamics = dynamics(casadi::SXVector({x,u}))[0];
    casadi::SX aug_dynamics = casadi::SX::vertcat({sym_dynamics, p_dynamics});

    /** evaluate augmented dynamics */
    casadi::Function aug_dynamo = casadi::Function("AUG_DYNAMO", {aug_state, aug_control}, {aug_dynamics});
    DynamicsFunc = aug_dynamo;

    /** create problem parameters: velocity and path */
    reference_velocity = casadi::SX::sym("reference_velocity");
    if(num_path_parameters > 0)
        path_parameters = casadi::SX::sym("path_parameters", num_path_parameters);

    /** set default properties of approximation */
    const int num_segments = NumSegments;
    const int poly_order   = PolyOrder;
    const int dimx         = NX + 2;
    const int dimu         = NU + 1;
    const int dimp         = 0;
    const double tf        = Tf;

    NUM_COLLOCATION_POINTS = num_segments * poly_order;
    /** Order of polynomial interpolation */

    Chebyshev<casadi::SX, poly_order, num_segments, dimx, dimu, dimp> spectral;
    casadi::SX diff_constr;

    if(scale)
    {
        casadi::SX SODE = aug_dynamo(casadi::SXVector{casadi::SX::mtimes(invSX, aug_state), casadi::SX::mtimes(invSU, aug_control)})[0];
        SODE = casadi::SX::mtimes(Scale_X, SODE);
        casadi::Function FunSODE = casadi::Function("scaled_ode", {aug_state, aug_control}, {SODE});

        diff_constr = spectral.CollocateDynamics(FunSODE, 0, tf);
    }
    else
    {
        diff_constr = spectral.CollocateDynamics(DynamicsFunc, 0, tf);
    }

    //diff_constr = diff_constr(casadi::Slice(0, diff_constr.size1() - dimx));

    /** define an integral cost */
    casadi::SX lagrange, residual;
    if(scale)
    {
        /** @bug dimensions bug here: _invSX */
        casadi::SX sym_path;
        if(num_path_parameters > 0)
            sym_path = PathFunc(casadi::SXVector{casadi::SX::mtimes(invSX(nx, nx), v(0)), path_parameters})[0];
        else
            sym_path = PathFunc(casadi::SXVector{casadi::SX::mtimes(invSX(nx, nx), v(0))})[0];

        casadi::SX _invSX = invSX(casadi::Slice(0, NX), casadi::Slice(0, NX));
        residual  = sym_path - output({casadi::SX::mtimes(_invSX, x)})[0];
        lagrange  = casadi::SX::sum1( casadi::SX::mtimes(Q, pow(residual, 2)) ) +
                    casadi::SX::sum1( casadi::SX::mtimes(W, pow(reference_velocity - v(1), 2)) );
        lagrange = lagrange + casadi::SX::sum1( casadi::SX::mtimes(R, pow(aug_control, 2)) );
    }
    else
    {
        casadi::SX sym_path;
        if(num_path_parameters > 0)
            sym_path = PathFunc(casadi::SXVector{v(0), path_parameters})[0];
        else
            sym_path = PathFunc(casadi::SXVector{v(0)})[0];

        residual  = sym_path - output({x})[0];
        lagrange  = casadi::SX::sum1( casadi::SX::mtimes(Q, pow(residual, 2)) ) +
                    casadi::SX::sum1( casadi::SX::mtimes(W, pow(reference_velocity - v(1), 2)) );

        lagrange = lagrange + casadi::SX::sum1( casadi::SX::mtimes(R, pow(aug_control, 2)) );
    }

    casadi::Function LagrangeTerm = casadi::Function("Lagrange", {aug_state, aug_control}, {lagrange});

    /** trace functions */
    PathError = casadi::Function("PathError", {aug_state}, {residual});
    VelError  = casadi::Function("VelError", {aug_state, reference_velocity}, {reference_velocity - v(1)});

    casadi::SX mayer           =  casadi::SX::sum1( casadi::SX::mtimes(Q, pow(residual, 2)) );
    casadi::Function MayerTerm = casadi::Function("Mayer",{aug_state}, {mayer});
    casadi::SX performance_idx = spectral.CollocateCost(MayerTerm, LagrangeTerm, 0.0, tf);

    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();

    casadi::SX opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu});

    /** debugging output */
    DynamicConstraints = casadi::Function("constraint_func", {opt_var}, {diff_constr});
    PerformanceIndex   = casadi::Function("performance_idx", {opt_var}, {performance_idx});

    casadi::SX lbg = casadi::SX::zeros(diff_constr.size());
    casadi::SX ubg = casadi::SX::zeros(diff_constr.size());

    /** set inequality (box) constraints */
    /** state */
    casadi::SX lbx = casadi::SX::repmat(casadi::SX::mtimes(Scale_X, LBX), poly_order * num_segments + 1, 1);
    casadi::SX ubx = casadi::SX::repmat(casadi::SX::mtimes(Scale_X, UBX), poly_order * num_segments + 1, 1);

    /** control */
    lbx = casadi::SX::vertcat( {lbx, casadi::SX::repmat(casadi::SX::mtimes(Scale_U, LBU), poly_order * num_segments + 1, 1)} );
    ubx = casadi::SX::vertcat( {ubx, casadi::SX::repmat(casadi::SX::mtimes(Scale_U, UBU), poly_order * num_segments + 1, 1)} );

    casadi::SX diff_constr_jacobian = casadi::SX::jacobian(diff_constr, opt_var);
    /** Augmented Jacobian */
    AugJacobian = casadi::Function("aug_jacobian",{opt_var}, {diff_constr_jacobian});

    /** formulate NLP */
    NLP["x"] = opt_var;
    NLP["f"] = performance_idx;
    NLP["g"] = diff_constr;
    if(num_path_parameters > 0)
        NLP["p"] = casadi::SX::vertcat({reference_velocity, path_parameters});
    else
        NLP["p"] = reference_velocity;

    /** default solver options */
    OPTS["ipopt.linear_solver"]         = "ma97";
    OPTS["ipopt.print_level"]           = 0;
    OPTS["ipopt.tol"]                   = 1e-4;
    OPTS["ipopt.acceptable_tol"]        = 1e-4;
    OPTS["ipopt.max_iter"]              = 40;
    OPTS["ipopt.warm_start_init_point"] = "yes";

    /** set user defined options */
    if(!solver_options.empty())
        updateParams(solver_options);

    NLP_Solver = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    /** set default args */
    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;

    /** set default parameters */
    setReferenceVelocity(1.0); // ARG["p"] = 1.0
    setPathParameters(default_path_parameters);

    casadi::DM feasible_state = casadi::DM::zeros(UBX.size());
    casadi::DM feasible_control = casadi::DM::zeros(UBU.size());

    ARG["x0"] = casadi::DM::vertcat(casadi::DMVector{casadi::DM::repmat(feasible_state, poly_order * num_segments + 1, 1),
                                     casadi::DM::repmat(feasible_control, poly_order * num_segments + 1, 1)});
}

template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
void nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::computeControl(const casadi::DM &_X0)
{
    int N = NUM_COLLOCATION_POINTS;

    /** rectify / reset virtual state */
    casadi::DM X0 = _X0;
    bool rectify = false;
    if(X0(nx).nonzeros()[0] > reset_path_after)
    {
        X0(nx) -= reset_path_after;
        rectify = true;
    }
    else if (X0(nx).nonzeros()[0] < -reset_path_after)
    {
        X0(nx) += reset_path_after;
        rectify = true;
    }

    /** scale input */
    X0 = casadi::DM::mtimes(Scale_X, X0);
    double critical_val = Scale_X(nx, nx).nonzeros()[0] * reset_path_after;
    double flexibility  = Scale_X(nx, nx).nonzeros()[0] * 0.78;     // ~ pi / 4

    int idx_theta;

    if(WARM_START)
    {
        int idx_in = N * (NX + 2);
        int idx_out = idx_in + (NX + 2);
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = X0;
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = X0;

        /** relax virtual state constraint */
        idx_theta = idx_out - 2;
        ARG["lbx"](idx_theta) = X0(nx) - flexibility;
        ARG["ubx"](idx_theta) = X0(nx) + flexibility;

        ARG["lbx"](idx_theta + 1) = X0(nx + 1) - flexibility;
        ARG["ubx"](idx_theta + 1) = X0(nx + 1) + flexibility;

        /** rectify initial guess */
        if(rectify)
        {
            for(int i = 0; i < (N + 1) * nx; i += nx)
            {
                int idx = i + nx;
                if(NLP_X(idx).nonzeros()[0] > critical_val)
                    NLP_X(idx) -= critical_val;
                else if (NLP_X(idx).nonzeros()[0] < -critical_val)
                    NLP_X(idx) += critical_val;
            }
        }
        ARG["x0"]     = NLP_X;
        ARG["lam_g0"] = NLP_LAM_G;
        ARG["lam_x0"] = NLP_LAM_X;
    }
    else
    {
        ARG["x0"](casadi::Slice(0, (N + 1) * (NX + 2)), 0) = casadi::DM::repmat(X0, (N + 1), 1);
        int idx_in = N * (NX + 2);
        int idx_out = idx_in + (NX + 2);
        ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = X0;
        ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = X0;

        /** relax virtual state constraint */
        idx_theta = idx_out - 2;
        ARG["lbx"](idx_theta) = X0(nx) - flexibility;
        ARG["ubx"](idx_theta) = X0(nx) + flexibility;

        ARG["lbx"](idx_theta + 1) = X0(nx + 1) - flexibility;
        ARG["ubx"](idx_theta + 1) = X0(nx + 1) + flexibility;
    }

    //DMVector dbg_prf    = PerformanceIndex({ARG["x0"]});
    //DMVector dbg_constr = DynamicConstraints({ARG["x0"]});

    //std::cout << "Cost Function : " << dbg_prf[0] << " Constraint Function : " << DM::norm_inf(dbg_constr[0]) << "\n";
    //DM state = ARG["x0"](Slice(N * nx, N * nx + nx));
    //std::cout << "State: " << DM::mtimes(invSX, state) << "\n";

    /** store optimal solution */
    casadi::DMDict res = NLP_Solver(ARG);
    NLP_X     = res.at("x");
    NLP_LAM_X = res.at("lam_x");
    NLP_LAM_G = res.at("lam_g");

    casadi::DM opt_x = NLP_X(casadi::Slice(0, (N + 1) * (NX + 2) ));
    OptimalTrajectory = casadi::DM::mtimes(invSX, casadi::DM::reshape(opt_x, (NX + 2), N + 1));
    casadi::DM opt_u = NLP_X( casadi::Slice((N + 1) * (NX + 2), NLP_X.size1()) );
    OptimalControl = casadi::DM::mtimes(invSU, casadi::DM::reshape(opt_u, (NU + 1), N + 1));

    stats = NLP_Solver.stats();
    //std::cout << stats << "\n";

    std::string solve_status = static_cast<std::string>(stats["return_status"]);
    if(solve_status.compare("Invalid_Number_Detected") == 0)
    {
        std::cout << "X0 : " << ARG["x0"] << "\n";
    }
    if(solve_status.compare("Infeasible_Problem_Detected") == 0)
    {
        std::cout << "X0 : " << ARG["x0"] << "\n";
    }

    enableWarmStart();
}

/** get path error */
template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
double nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::getPathError()
{
    double error = 0;
    if(!OptimalTrajectory.is_empty())
    {
        casadi::DM state = OptimalTrajectory(casadi::Slice(0, OptimalTrajectory.size1()), OptimalTrajectory.size2() - 1);
        state = casadi::DM::mtimes(Scale_X, state);
        casadi::DMVector tmp = PathError(casadi::DMVector{state});
        error = casadi::DM::norm_2( tmp[0] ).nonzeros()[0];
    }
    return error;
}

/** get velocity error */
template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
double nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::getVelocityError()
{
    double error = 0;
    if(!OptimalTrajectory.is_empty())
    {
        casadi::DM state = OptimalTrajectory(casadi::Slice(0, OptimalTrajectory.size1()), OptimalTrajectory.size2() - 1);
        state = casadi::DM::mtimes(Scale_X, state);
        casadi::DMVector tmp = VelError(casadi::DMVector{state, ARG["p"](0)});
        error = casadi::DM::norm_2( tmp[0] ).nonzeros()[0];
    }
    return error;
}

/** get virtual state */
template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
double nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::getVirtState()
{
    double virt_state = 0;
    if(!OptimalTrajectory.is_empty())
    {
        virt_state = OptimalTrajectory(13, OptimalTrajectory.size2() - 1).nonzeros()[0];
    }
    return virt_state;
}

/** compute intial guess for virtual state */
template<typename System, typename Path, int NX, int NU, int NumSegments, int PolyOrder>
casadi::DM nmpf<System, Path, NX, NU, NumSegments, PolyOrder>::findClosestPointOnPath(const casadi::DM &position,
                                                                                      const casadi::DM &init_guess)
{
    casadi::SX theta = casadi::SX::sym("theta");
    casadi::SX sym_residual = 0.5 * casadi::SX::norm_2(PathFunc(casadi::SXVector{theta})[0] - casadi::SX(position));
    casadi::SX sym_gradient = casadi::SX::gradient(sym_residual,theta);

    casadi::Function grad = casadi::Function("gradient", {theta}, {sym_gradient});

    double tol = 1e-2;
    int counter = 0;
    casadi::DM theta_i = init_guess;
    casadi::DMVector result  = grad(casadi::DMVector{theta_i});
    double step = 0.25;

    /** check for local minimum */
    if(fabs(result[0].nonzeros()[0]) < tol)
    {
        theta_i = {M_PI_2 + 0.1};
        result  = grad(casadi::DMVector{theta_i});
    }

    /** @badcode : shitty code */
    while (fabs(result[0].nonzeros()[0]) >= tol)
    {
        counter++;
        theta_i -= step * grad(theta_i);
        /** gradient update */
        result  = grad(casadi::DMVector{theta_i});
        if(counter > 10)
            break;
    }
    //std::cout << theta_i << "\n";
    return theta_i;
}

}

#endif // NMPF_HPP
