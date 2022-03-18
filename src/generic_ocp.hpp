// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef GENERIC_OCP_HPP
#define GENERIC_OCP_HPP

#include "chebyshev.hpp"
#include "chebyshev_ms.hpp"
#include "chebyshev_soft.hpp"

namespace polympc {

enum PENALTY { QUADRATIC, EXACT };
enum OPERATOR_OUTPUT {NORM_DIFF_VALUE = 5};

static casadi::SX _mtimes(const casadi::SX &m1, const casadi::SX &m2){ return casadi::SX::mtimes(m1,m2); }
static casadi::DM _mtimes(const casadi::DM &m1, const casadi::DM &m2){ return casadi::DM::mtimes(m1,m2); }

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
    casadi::SX lagrange_term(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return static_cast<OCP*>(this)->lagrange_term_impl(x,u,p,d);
    }
    /** generic Mayer term */
    casadi::SX mayer_term(const casadi::SX &x, const casadi::SX &p, const casadi::SX &d)
    {
        return static_cast<OCP*>(this)->mayer_term_impl(x,p,d);
    }
    /** evaluation of the ODEs RHS */
    casadi::SX system_dynamics(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return static_cast<OCP*>(this)->system_dynamics_impl(x,u,p,d);
    }
    /** generic path inequality constraints */
    casadi::SX inequality_constraints(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return static_cast<OCP*>(this)->inequality_constraints_impl(x,u,p,d);
    }

    /** generic final time inequality constraints */
    casadi::SX final_inequality_constraints(const casadi::SX &x, const casadi::SX &u, const casadi::SX &p, const casadi::SX &d)
    {
        return static_cast<OCP*>(this)->final_inequality_constraints_impl(x,u,p,d);
    }

    /** regularization of an expression derivative*/
    /**  || expr_dot ||_{WeightMat} */
    casadi::SX norm_diff(const casadi::SX &expr, const casadi::SX &WeightMat);

    /** regularization of an expression second derivative*/
    /**  || expr_dot_dot ||_{WeightMat} */
    casadi::SX norm_ddiff(const casadi::SX &expr, const casadi::SX &WeightMat);

    /** expression spectral derivative */
    /** [g(x,u)]' <= c */
    casadi::SX diff(const casadi::SX &expr, const casadi::DM &c);

    /** expression spectral second derivative */
    /** [g(x,u)]'' <= c */
    casadi::SX ddiff(const casadi::SX &expr, const casadi::DM &c);

    casadi::SXVector m_norm_diff, m_norm_ddiff;
    casadi::SXVector m_diff, m_ddiff;
    casadi::DMVector m_diff_bound, m_ddiff_bound;

public:
    /** state box constraints */
    void set_state_box_constraints(const casadi::DM &lower_bound, const casadi::DM &upper_bound)
    {
        casadi_assert((lower_bound.size1() == NX) && (upper_bound.size1() == NX), "set_state_constraints: incorrect supplied bounds sizes");
        ARG["lbx"](casadi::Slice(X_START, X_END)) = casadi::DM::repmat(_mtimes(casadi::DM(ScX), lower_bound), Approximation::_NUM_COLLOC_PTS_X);
        ARG["ubx"](casadi::Slice(X_START, X_END)) = casadi::DM::repmat(_mtimes(casadi::DM(ScX), upper_bound), Approximation::_NUM_COLLOC_PTS_X);
    }
    /** constrol box constraints */
    void set_control_box_constraints(const casadi::DM &lower_bound, const casadi::DM &upper_bound)
    {
        casadi_assert((lower_bound.size1() == NU) && (upper_bound.size1() == NU), "set_control_constraints: incorrect supplied bounds sizes");
        ARG["lbx"](casadi::Slice(U_START, U_END)) = casadi::DM::repmat(_mtimes(casadi::DM(ScU), lower_bound), Approximation::_NUM_COLLOC_PTS_U);
        ARG["ubx"](casadi::Slice(U_START, U_END)) = casadi::DM::repmat(_mtimes(casadi::DM(ScU), upper_bound), Approximation::_NUM_COLLOC_PTS_U);
    }
    /** parameters box constraints */
    void set_parameters_box_constraints(const casadi::DM &lower_bound, const casadi::DM &upper_bound)
    {
        casadi_assert((lower_bound.size1() == NP) && (upper_bound.size1() == NP), "set_parameters_constraints: incorrect supplied bounds sizes");
        ARG["lbx"](casadi::Slice(P_START, P_END)) = _mtimes(casadi::DM(ScP), lower_bound);
        ARG["ubx"](casadi::Slice(P_START, P_END)) = _mtimes(casadi::DM(ScP), upper_bound);
    }


    /** get optimal state trajectory */
    casadi::DM get_optimal_trajectory() const
    {
        casadi::DM opt_x = NLP_X(casadi::Slice(X_START, X_END));
        casadi::DM reshaped = casadi::DM::reshape(opt_x, NX, Approximation::_NUM_COLLOC_PTS_X);
        return _mtimes(casadi::DM(invSX), reshaped);
    }

    /** get optimal control trajectory */
    casadi::DM get_optimal_control() const
    {
        casadi::DM opt_u = NLP_X(casadi::Slice(U_START, U_END));
        casadi::DM reshaped = casadi::DM::reshape(opt_u, NU, Approximation::_NUM_COLLOC_PTS_U);
        return _mtimes(casadi::DM(invSU), reshaped);
    }

    /** get optimal parameters */
    casadi::DM get_optimal_parameters() const
    {
        casadi::DM opt_p = NLP_X(casadi::Slice(P_START, P_END));
        return _mtimes(casadi::DM(invSP), opt_p);
    }

    /** get slacks if used */
    casadi::DM get_slacks() const
    {
        return NLP_X(casadi::Slice(P_END, NLP_X.size1()));
    }
    casadi_int get_slacks_size() const
    {
        return slacks.size1();
    }

    void set_parameters(const casadi::DM &param_vector)
    {
        casadi_assert(param_vector.size1() == ARG["p"].size1(), "set_parameters: given parameter vector has a wrong dimension");
        ARG["p"] = param_vector;
    }

    void set_parameters(const std::string &name, const casadi::DM &value)
    {
        if(parameter_map.find(name) != parameter_map.end())
        {
            std::pair<int, int> indices = parameter_map[name];
            ARG["p"](casadi::Slice(indices.first, indices.second)) = value;
        }
        else
            casadi_assert(false, "set_parameters: unknown parameter name: " + name);
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

    void solve(const casadi::DM &lbx0, const casadi::DM &ubx0,
               const casadi::DM &X0 = casadi::DM(), const casadi::DM &LAM_X0 = casadi::DM(), const casadi::DM &LAM_G0 = casadi::DM());

    casadi::DM NLP_X, NLP_LAM_G, NLP_LAM_X;
    std::vector<double> SCALER_X, SCALER_U, SCALER_P;
    casadi::Dict solve_status;

    static constexpr int NX = Approximation::_NX;
    static constexpr int NU = Approximation::_NU;
    static constexpr int NP = Approximation::_NP;
    static constexpr int ND = Approximation::_ND;

    static constexpr int POLY_ORDER       = Approximation::_POLY_ORDER;
    static constexpr int NUM_SEGMENTS     = Approximation::_NUM_SEGMENTS;
    static constexpr int NUM_COLLOC_PTS_X = Approximation::_NUM_COLLOC_PTS_X;

    const double t_start = static_cast<OCP*>(this)->t_start;
    const double t_final = static_cast<OCP*>(this)->t_final;

    /** store the approximation class */
    static constexpr int X_START = Approximation::_X_START_IDX;
    static constexpr int X_END   = Approximation::_X_END_IDX;
    static constexpr int U_START = Approximation::_U_START_IDX;
    static constexpr int U_END   = Approximation::_U_END_IDX;
    static constexpr int P_START = Approximation::_P_START_IDX;
    static constexpr int P_END   = Approximation::_P_END_IDX;

private:
    bool scaling{false};
    casadi::SX ScX, ScU, ScP, invSX, invSU, invSP;

    bool use_slacks{false};
    int  slack_penalty_type{PENALTY::QUADRATIC};
    casadi::SX slacks;

    double soft_rho{1e6};

    casadi::SXDict NLP;
    casadi::Function NLP_Solver;
    casadi::Dict OPTS, USER_OPTS, MPC_OPTS;
    casadi::DMDict ARG;
    std::map<std::string, std::pair<int, int>> parameter_map;

    void setup();
    Approximation spectral;
};

/** add derivative regularisation to the cost : || g_dot(x,u,p,d) || _ {W} */
template<typename OCP, typename Approximation>
casadi::SX GenericOCP<OCP, Approximation>::norm_diff(const casadi::SX &expr, const casadi::SX &WeightMat)
{
    casadi_assert(expr.size1() == WeightMat.size1(), "norm_diff: weight matrix has wrong dimension");
    m_norm_diff.push_back(casadi::SX::mtimes(sqrt(WeightMat), expr));

    return casadi::SX(NORM_DIFF_VALUE);
}

/** add second derivative regularisation to the cost : || g_ddot(x,u,p,d) || _ {W} */
template<typename OCP, typename Approximation>
casadi::SX GenericOCP<OCP, Approximation>::norm_ddiff(const casadi::SX &expr, const casadi::SX &WeightMat)
{
    casadi_assert(expr.size1() == WeightMat.size1(), "norm_ddiff: weight matrix has wrong dimension");
    m_norm_ddiff.push_back(casadi::SX::mtimes(sqrt(WeightMat), expr));

    return casadi::SX(NORM_DIFF_VALUE);
}

/** derivative of the expression to the path constraints */
template<typename OCP, typename Approximation>
casadi::SX GenericOCP<OCP, Approximation>::diff(const casadi::SX &expr, const casadi::DM &c)
{
    casadi_assert(expr.size1() == c.size1(), "g(x,u,p,d) and c should have the same dimensions");
    m_diff.push_back(expr);
    m_diff_bound.push_back(c);
    return casadi::SX();
}

template<typename OCP, typename Approximation>
casadi::SX GenericOCP<OCP, Approximation>::ddiff(const casadi::SX &expr, const casadi::DM &c)
{
    casadi_assert(expr.size1() == c.size1(), "g(x,u,p,d) and c should have the same dimensions");
    m_ddiff.push_back(expr);
    m_ddiff_bound.push_back(c);
    return casadi::SX();
}

/** ------------------------------------------- */

template<typename OCP, typename Approximation>
void GenericOCP<OCP, Approximation>::setup()
{
    casadi::SX x = casadi::SX::sym("x", NX);
    casadi::SX u = casadi::SX::sym("u", NU);
    casadi::SX p = casadi::SX::sym("p", NP);
    casadi::SX d = casadi::SX::sym("d", ND);

    /** set scaling of the problem */
    /** default scalers values */
    ScX   = casadi::SX::eye(NX);
    invSX = casadi::SX::eye(NX);
    ScU   = casadi::SX::eye(NU);
    invSU = casadi::SX::eye(NU);
    ScP   = casadi::SX::eye(NP);
    invSP = casadi::SX::eye(NP);

    SCALER_X = std::vector<double>(NX, 1.0);
    SCALER_U = std::vector<double>(NU, 1.0);
    SCALER_P = std::vector<double>(NU, 1.0);

    /** scaling options */
    if(MPC_OPTS.find("mpc.scaling") != MPC_OPTS.end())
        scaling = MPC_OPTS["mpc.scaling"];

    if(scaling && (MPC_OPTS.find("mpc.scale_x") != MPC_OPTS.end()))
    {
        std::vector<double> scale_x = MPC_OPTS["mpc.scale_x"];
        casadi_assert(NX == scale_x.size(), "setup: provided state scaling matrix has wrong dimension");
        ScX   = casadi::SX::diag({scale_x});
        invSX = casadi::DM::solve(ScX, casadi::DM::eye(NX));
        SCALER_X = scale_x;
    }

    if(scaling && (MPC_OPTS.find("mpc.scale_u") != MPC_OPTS.end()))
    {
        std::vector<double> scale_u = MPC_OPTS["mpc.scale_u"];
        casadi_assert(NU == scale_u.size(), "setup: provided control scaling matrix has wrong dimension");
        ScU   = casadi::SX::diag({scale_u});
        invSU = casadi::DM::solve(ScU, casadi::DM::eye(NU));
        SCALER_U = scale_u;
    }

    if(scaling && (MPC_OPTS.find("mpc.scale_p") != MPC_OPTS.end()))
    {
        std::vector<double> scale_p = MPC_OPTS["mpc.scale_p"];
        casadi_assert(NP == scale_p.size(), "setup: provided parameters scaling matrix has wrong dimension");
        ScP   = casadi::SX::diag({scale_p});
        invSP = casadi::DM::solve(ScP, casadi::DM::eye(NU));
        SCALER_P = scale_p;
    }

    /** slack reformulation options */
    if(MPC_OPTS.find("mpc.use_slacks") != MPC_OPTS.end())
        use_slacks = MPC_OPTS["mpc.use_slacks"];

    if(scaling && (MPC_OPTS.find("mpc.slack_penaly_type") != MPC_OPTS.end()))
        slack_penalty_type = MPC_OPTS["mpc.slack_penalty_type"];

    /** soft formulation */
    if(MPC_OPTS.find("mpc.soft_rho") != MPC_OPTS.end())
            soft_rho = MPC_OPTS["mpc.soft_rho"];

    /** collocate dynamic equations*/
    casadi::SX diff_constr;
    casadi::SX dynamics = system_dynamics(_mtimes(invSX, x), _mtimes(invSU, u), p, d);
    dynamics = _mtimes(ScX, dynamics);
    casadi::Function dyn_func;
    /** @badcode : have to deal with the old Chebyshev interface */
    if((NP == 0) && (ND == 0))
        dyn_func = casadi::Function("dyn_func",{x,u},{dynamics});
    else
        dyn_func = casadi::Function("dyn_func",{x,u,p,d},{dynamics});

    diff_constr = spectral.CollocateDynamics(dyn_func, t_start, t_final);
    casadi::SX lbg_diff = casadi::SX::zeros(diff_constr.size1(), 1);
    casadi::SX ubg_diff = lbg_diff;

    /** collocate performance index */
    casadi::SX cost;
    casadi::SX mayer_cost = mayer_term(_mtimes(invSX, x), _mtimes(invSP, p), d);
    casadi::Function mayer_cost_func = casadi::Function("mayer_cost",{x,p,d},{mayer_cost});

    casadi::SX lagrange_cost = lagrange_term(_mtimes(invSX, x), _mtimes(invSU, u), _mtimes(invSP, p), d);
    casadi::Function lagrange_cost_func = casadi::Function("lagrange_cost",{x,u,p,d},{lagrange_cost});

    cost = spectral.CollocateParametricCost(mayer_cost_func, lagrange_cost_func, t_start, t_final);

    /** handle the special case of relaxed collocation: soft collocation */
    if(diff_constr.size1() == 1)
    {
        cost += soft_rho * diff_constr;
        lbg_diff = casadi::SX();
        ubg_diff = casadi::SX();
        diff_constr = casadi::SX();
    }

    /** process differential operators */
    if(!m_norm_diff.empty() && !cost.is_zero())
    {
        casadi::SXVector::const_iterator it;
        for(it = m_norm_diff.begin(); it != m_norm_diff.end(); ++it)
        {
            casadi::Function expr_fun = casadi::Function("expr", {x, u, p, d}, {*it});
            casadi::SX tmp = spectral.DifferentiateFunction(expr_fun, 1);
            /** @bug: should integrate u_dot^2 here? */
            cost += casadi::SX::dot(tmp,tmp);
        }
    }

    if(!m_norm_ddiff.empty() && !cost.is_zero())
    {
        casadi::SXVector::const_iterator it;
        for(it = m_norm_ddiff.begin(); it != m_norm_ddiff.end(); ++it)
        {
            casadi::Function expr_fun = casadi::Function("expr", {x, u, p, d}, {*it});
            casadi::SX tmp = spectral.DifferentiateFunction(expr_fun, 2);
            /** @bug: should integrate u_dotdot^2 here? */
            cost += casadi::SX::dot(tmp,tmp);
        }
    }
    /** clear memory */
    m_norm_diff.clear();
    m_norm_ddiff.clear();

    /** collocate generic inequality constraints */
    casadi::SX ineq_constraints;
    casadi::SX lbg_ic;
    casadi::SX ubg_ic;
    casadi::SX ic = inequality_constraints(_mtimes(invSX, x), _mtimes(invSU, u), _mtimes(invSP, p), d);

    if (!ic->empty())
    {
        casadi::Function ic_func = casadi::Function("ic_func",{x,u,p,d},{ic});
        ineq_constraints = spectral.CollocateFunction(ic_func);

        /** process differential operators */
        if(!m_diff.empty())
        {
            for (unsigned i = 0; i < m_diff.size(); ++i)
            {
                casadi::Function expr_fun = casadi::Function("expr", {x, u, p, d}, {m_diff[i]});
                casadi::SX tmp = spectral.DifferentiateFunction(expr_fun, 1);

                unsigned dim = tmp.size1() / m_diff[i].size1();
                tmp = tmp - casadi::SX::repmat(m_diff_bound[i], dim, 1);
                ineq_constraints = casadi::SX::vertcat({ineq_constraints, tmp});
            }
            /** clear data */
            m_diff.clear();
            m_diff_bound.clear();
        }

        if(!m_ddiff.empty())
        {
            for (unsigned i = 0; i < m_ddiff.size(); ++i)
            {
                casadi::Function expr_fun = casadi::Function("expr", {x, u, p, d}, {m_ddiff[i]});
                casadi::SX tmp = spectral.DifferentiateFunction(expr_fun, 2);

                unsigned dim = tmp.size1() / m_ddiff[i].size1();
                tmp = tmp - casadi::SX::repmat(m_ddiff_bound[i], dim, 1);
                ineq_constraints = casadi::SX::vertcat({ineq_constraints, tmp});
            }
            /** clear data */
            m_ddiff.clear();
            m_ddiff_bound.clear();
        }

        if(use_slacks)
        {
            casadi::SX slack_ic = casadi::SX::sym("slack_ic", ineq_constraints.size1());
            ineq_constraints = ineq_constraints + slack_ic;

            lbg_ic =  casadi::SX::zeros(ineq_constraints.size1());
            ubg_ic =  casadi::SX::zeros(ineq_constraints.size1());

            slacks = casadi::SX::vertcat({slacks, slack_ic});
        }
        else
        {
            lbg_ic = -casadi::SX::inf(ineq_constraints.size1());
            ubg_ic =  casadi::SX::zeros(ineq_constraints.size1());
        }
    }


    /** collocate generic final time inequality constraints */
    casadi::SX final_ineq_constraints;
    casadi::SX final_lbg_ic;
    casadi::SX final_ubg_ic;
    casadi::SX final_ic = final_inequality_constraints(_mtimes(invSX, x), _mtimes(invSU, u), _mtimes(invSP, p), d);

    if (!final_ic->empty())
    {
        casadi::Function ic_func = casadi::Function("final_ic_func",{x,u,p,d},{final_ic});
        final_ineq_constraints = spectral.CollocateFunction(ic_func);
        final_ineq_constraints = final_ineq_constraints(casadi::Slice(0, ic_func.n_out())); // take only the final state

        if(use_slacks)
        {
            casadi::SX slack_fic = casadi::SX::sym("slack_fic", final_ineq_constraints.size1());
            final_ineq_constraints = final_ineq_constraints + slack_fic;

            final_lbg_ic = casadi::SX::zeros(final_ineq_constraints.size1());
            final_ubg_ic = casadi::SX::zeros(final_ineq_constraints.size1());

            slacks = casadi::SX::vertcat({slacks, slack_fic});
        }
        else
        {
            final_lbg_ic = -casadi::SX::inf(final_ineq_constraints.size1());
            final_ubg_ic =  casadi::SX::zeros(final_ineq_constraints.size1());
        }
    }

    /** initialise NLP interface*/
    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();
    casadi::SX varp = spectral.VarP();
    casadi::SX vard = spectral.VarD();

    casadi::SX opt_var = casadi::SX::vertcat({varx, varu, varp});
    casadi::SX constraints = casadi::SX::vertcat({diff_constr, ineq_constraints, final_ineq_constraints});
    casadi::SX lbg = casadi::SX::vertcat({lbg_diff, lbg_ic, final_lbg_ic});
    casadi::SX ubg = casadi::SX::vertcat({ubg_diff, ubg_ic, final_ubg_ic});

    if(use_slacks && !(slacks->empty()))
    {
        /** append optvar and cost */
        opt_var = casadi::SX::vertcat({opt_var, slacks});
        if(slack_penalty_type == PENALTY::QUADRATIC)
            cost += 10 * casadi::SX::dot(slacks, slacks);
        else if (slack_penalty_type == PENALTY::EXACT)
        {
            cost += casadi::SX::dot(slacks, slacks) + 10 * slacks;
        }
        else
            casadi_assert(false, "Unknown penalty function type, available: EXACT and QUADRATIC");
    }

    NLP["x"] = opt_var;
    NLP["f"] = cost;
    NLP["g"] = constraints;
    NLP["p"] = vard;

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
    ARG["p"]   = casadi::DM::zeros(vard.size1());

    if(use_slacks && !slacks->empty())
    {
        casadi::DM lbs = casadi::SX::zeros(slacks.size1());
        casadi::DM ubs = casadi::SX::inf(slacks.size1());

        casadi::DM ubx =  casadi::DM::inf(varx.size1() + varu.size1() + varp.size1());
        casadi::DM lbx = -ubx;

        ARG["lbx"] = casadi::DM::vertcat({lbx, lbs});
        ARG["ubx"] = casadi::DM::vertcat({ubx, ubs});
    }
    else
    {
        ARG["lbx"] = -casadi::DM::inf(opt_var.size1());
        ARG["ubx"] =  casadi::DM::inf(opt_var.size1());
    }

    /** default initial guess */
    ARG["x0"] = casadi::DM ::zeros(opt_var.size1(), 1);
}

template<typename OCP, typename Approximation>
void GenericOCP<OCP, Approximation>::solve(const casadi::DM &lbx0, const casadi::DM &ubx0,
                                           const casadi::DM &X0, const casadi::DM &LAM_X0, const casadi::DM &LAM_G0)
{
    ARG["lbx"](casadi::Slice(X_END - NX, X_END), 0) = _mtimes(casadi::DM(ScX), lbx0);
    ARG["ubx"](casadi::Slice(X_END - NX, X_END), 0) = _mtimes(casadi::DM(ScX), ubx0);

    /** apply warmstarting if possible */
    if(!X0.is_empty())
        ARG["x0"] = X0;
    else
    {
        casadi::DM mid_point = _mtimes(casadi::DM(ScX), 0.5 * ( lbx0 + ubx0 ));
        ARG["x0"](casadi::Slice(X_START, X_END)) = casadi::DM::repmat(mid_point, NUM_COLLOC_PTS_X, 1);
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

} // polympc namespace

#endif // GENERIC_OCP_HPP
