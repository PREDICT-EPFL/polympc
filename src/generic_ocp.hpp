#ifndef GENERIC_OCP_HPP
#define GENERIC_OCP_HPP

#include "chebyshev.hpp"
#include "chebyshev_ms.hpp"

template<typename OCP, typename Approximation>
class GenericOCP
{
public:
    GenericOCP(){ setup(); }
    GenericOCP(const casadi::Dict &nlp_options)
    {
        USER_OPTS = nlp_options;
        setup();
    }
    GenericOCP(const casadi::Dict &nlp_options, const casadi::Dict &mpc_options)
    {
        USER_OPTS = nlp_options;
        MPC_OPTS  = mpc_options;
        setup();
    }
    virtual ~GenericOCP() = default;

protected:
    /** generic Lagrange cost term */
    casadi::SX lagrange_term(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return static_cast<OCP*>(this)->lagrange_term_impl(x,u,p);
    }
    /** generic Mayer term */
    casadi::SX mayer_term(const casadi::SX &x, const casadi::SX &p)
    {
        return static_cast<OCP*>(this)->mayer_term_impl(x,p);
    }
    /** evaluation of the ODEs RHS */
    casadi::SX system_dynamics(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return static_cast<OCP*>(this)->system_dynamics_impl(x,u,p);
    }
    /** system output mapping  */
    casadi::SX system_output(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return static_cast<OCP*>(this)->system_output_impl(x,u,p);
    }
    /** generic inequality constraints */
    casadi::SX inequality_constraints(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p)
    {
        return static_cast<OCP*>(this)->inequality_constraints_impl(x,u,p);
    }

public:
    /** state box constraints */
    void set_state_box_constraints(const casadi::DM &lower_bound, const casadi::DM &upper_bound)
    {
        assert((lower_bound.size1() == NX) && (upper_bound.size1() == NX));
        ARG["lbx"](casadi::Slice(X_START, X_END)) = casadi::DM::repmat(lower_bound, Approximation::_NUM_COLLOC_PTS_X);
        ARG["ubx"](casadi::Slice(X_START, X_END)) = casadi::DM::repmat(upper_bound, Approximation::_NUM_COLLOC_PTS_X);
    }
    /** constrol box constraints */
    void set_control_box_constraints(const casadi::DM &lower_bound, const casadi::DM &upper_bound)
    {
        assert((lower_bound.size1() == NU) && (upper_bound.size1() == NU));
        ARG["lbx"](casadi::Slice(U_START, U_END)) = casadi::DM::repmat(lower_bound, Approximation::_NUM_COLLOC_PTS_U);
        ARG["ubx"](casadi::Slice(U_START, U_END)) = casadi::DM::repmat(upper_bound, Approximation::_NUM_COLLOC_PTS_U);
    }

    void set_parameters(const casadi::DM &param_vector)
    {
        assert(param_vector.size1() == ARG["p"].size1());
        ARG["p"] = param_vector;
    }

    void set_parameters(const std::string &name, const casadi::DM &value)
    {
        const bool UNKNOWN_PARAMETER_NAME = false;
        if(parameter_map.find(name) != parameter_map.end())
        {
            std::pair<int, int> indices = parameter_map[name];
            ARG["p"](casadi::Slice(indices.first, indices.second)) = value;
        }
        else
        {
            assert(UNKNOWN_PARAMETER_NAME);
        }
    }

    void set_parameter_mapping(const std::map<std::string, std::pair<int, int>> &map)
    {
        parameter_map = map;
    }

    void set_parameter_mapping(const std::string &name, const std::pair<int, int> &indices)
    {
        parameter_map[name] = indices;
    }

    void update_solver_params()
    {
        for (casadi::Dict::const_iterator it = USER_OPTS.begin(); it != USER_OPTS.end(); ++it)
        {
            OPTS[it->first] = it->second;
        }
    }

    void setup();
    void solve(const casadi::DM &lbx0, const casadi::DM &ubx0,
               const casadi::DM &X0 = casadi::DM(), const casadi::DM &LAM_X0 = casadi::DM(), const casadi::DM &LAM_G0 = casadi::DM());

    //casadi::DM solution_primal;
    //casadi::DM solution_dual;
    casadi::DM NLP_X, NLP_LAM_G, NLP_LAM_X;
    casadi::Dict solve_status;

    static constexpr int NX = Approximation::_NX;
    static constexpr int NU = Approximation::_NU;
    static constexpr int NP = Approximation::_NP;

    static constexpr int POLY_ORDER = Approximation::_POLY_ORDER;
    static constexpr int NUM_SEGMENTS = Approximation::_NUM_SEGMENTS;

    const double t_start = static_cast<OCP*>(this)->t_start;
    const double t_final = static_cast<OCP*>(this)->t_final;

    /** store the approximation class */
    Approximation spectral;
    static constexpr int X_START = Approximation::_X_START_IDX;
    static constexpr int X_END   = Approximation::_X_END_IDX;
    static constexpr int U_START = Approximation::_U_START_IDX;
    static constexpr int U_END   = Approximation::_U_END_IDX;

private:
    bool scaling{false};
    casadi::SX SX, SU, invSX, invSU;
    casadi::SXDict NLP;
    casadi::Function NLP_Solver;
    casadi::Dict OPTS, USER_OPTS, MPC_OPTS;
    casadi::DMDict ARG;
    std::map<std::string, std::pair<int, int>> parameter_map;
};

template<typename OCP, typename Approximation>
void GenericOCP<OCP, Approximation>::setup()
{
    casadi::SX x = casadi::SX::sym("x", NX);
    casadi::SX u = casadi::SX::sym("u", NU);
    casadi::SX p = casadi::SX::sym("p", NP);

    /** set scaling of the problem */
    if(MPC_OPTS.find("mpc.scaling") != MPC_OPTS.end())
        scaling = MPC_OPTS["mpc.scaling"];

    if(scaling)
        std::cout << "I AM USING SCALING \n";


    /** collocate dynamic equations*/
    casadi::SX diff_constr;
    casadi::SX dynamics = system_dynamics(x,u,p);
    casadi::Function dyn_func;
    /** @badcode : have to deal with the old Chebyshev interface */
    if(NP == 0)
        dyn_func = casadi::Function("dyn_func",{x,u},{dynamics});
    else
        dyn_func = casadi::Function("dyn_func",{x,u,p},{dynamics});

    diff_constr = spectral.CollocateDynamics(dyn_func, t_start, t_final);
    casadi::SX lbg_diff = casadi::SX::zeros(diff_constr.size1(), 1);
    casadi::SX ubg_diff = lbg_diff;
    //std::cout << "Differential constraints: " << diff_constr << "\n";

    /** collocate performance index */
    casadi::SX cost;
    casadi::SX mayer_cost = mayer_term(x,p);
    casadi::Function mayer_cost_func = casadi::Function("mayer_cost",{x,p},{mayer_cost});

    casadi::SX lagrange_cost = lagrange_term(x,u,p);
    casadi::Function lagrange_cost_func = casadi::Function("lagrange_cost",{x,u,p},{lagrange_cost});

    cost = spectral.CollocateParametricCost(mayer_cost_func, lagrange_cost_func, t_start, t_final);
    //std::cout << "Cost: " << cost << "\n";

    /** collocate generic inequality constraints */
    casadi::SX ineq_constraints;
    casadi::SX lbg_ic;
    casadi::SX ubg_ic;
    casadi::SX ic = inequality_constraints(x,u,p);

    if (!ic->empty())
    {
        casadi::Function ic_func = casadi::Function("ic_func",{x,u,p},{ic});
        ineq_constraints = spectral.CollocateFunction(ic_func);
        lbg_ic = -casadi::SX::inf(ineq_constraints.size1());
        ubg_ic =  casadi::SX::zeros(ineq_constraints.size1());
    }
    //std::cout << "Inequality constraints: " << ineq_constraints << "\n";

    /** initialise NLP interface*/
    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();
    casadi::SX varp = spectral.VarP();

    casadi::SX opt_var = casadi::SX::vertcat({varx, varu});
    casadi::SX constraints = casadi::SX::vertcat({diff_constr, ineq_constraints});
    casadi::SX lbg = casadi::SX::vertcat({lbg_diff, lbg_ic});
    casadi::SX ubg = casadi::SX::vertcat({ubg_diff, ubg_ic});

    NLP["x"] = opt_var;
    NLP["f"] = cost;
    NLP["g"] = constraints;
    NLP["p"] = varp;

    /** default solver options */
    OPTS["ipopt.linear_solver"]         = "mumps";
    OPTS["ipopt.print_level"]           = 0;
    OPTS["ipopt.tol"]                   = 1e-4;
    OPTS["ipopt.acceptable_tol"]        = 1e-4;
    OPTS["ipopt.warm_start_init_point"] = "yes";

    update_solver_params();

    NLP_Solver = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;
    ARG["lbx"] = -casadi::DM::inf(opt_var.size1());
    ARG["ubx"] =  casadi::DM::inf(opt_var.size1());
    ARG["p"]   = casadi::DM::zeros(varp.size1());

    /** default initial guess */
    ARG["x0"] = casadi::DM ::zeros(opt_var.size1(), 1);
}

template<typename OCP, typename Approximation>
void GenericOCP<OCP, Approximation>::solve(const casadi::DM &lbx0, const casadi::DM &ubx0,
                                           const casadi::DM &X0, const casadi::DM &LAM_X0, const casadi::DM &LAM_G0)
{
    ARG["lbx"](casadi::Slice(X_END - NX, X_END), 0) = lbx0;
    ARG["ubx"](casadi::Slice(X_END - NX, X_END), 0) = ubx0;

    /** apply warmstarting if possible */
    if(!X0.is_empty())
        ARG["x0"] = X0;
    else
    {
        casadi::DM mid_point = 0.5 * ( lbx0 + ubx0 );
        ARG["x0"](casadi::Slice(0, X_END)) = casadi::DM::repmat(mid_point, NUM_SEGMENTS * POLY_ORDER + 1, 1);
    }

    if(!LAM_X0.is_empty())
        ARG["lam_x0"] = LAM_X0;

    if(!LAM_G0.is_empty())
        ARG["lam_g0"] = LAM_G0;


    /** store optimal solution */
    casadi::DMDict res = NLP_Solver(ARG);
    NLP_X     = res.at("x");
    NLP_LAM_X = res.at("lam_x");
    NLP_LAM_G = res.at("lam_g");
    solve_status = NLP_Solver.stats();
}


#endif // GENERIC_OCP_HPP
