// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "integrator.h"
#include <fstream>
#include "pseudospectral/chebyshev.hpp"
#include "kiteNMPF.h"


int main(void)
{
    /** Load control signal */
    std::ifstream id_control_file("id_data_control.txt", std::ios::in);
    const int DATA_POINTS = 501;
    const int state_size   = 13;
    const int control_size = 3;

    casadi::DM id_control = casadi::DM::zeros(control_size, DATA_POINTS);

    /** load control data */
    if(!id_control_file.fail())
    {
    for(uint i = 0; i < DATA_POINTS; ++i){
        for(uint j = 0; j < control_size; ++j){
            double entry;
            id_control_file >> entry;
            /** put in reverse order to comply with Chebyshev method */
            id_control(j,DATA_POINTS - 1 - i) = entry;
        }
    }
    }
    else
    {
        std::cout << "Could not open : id control data file \n";
        id_control_file.clear();
    }

    /** create the kite model */
    std::string kite_params_file = "umx_radian.yaml";
    KiteProperties kite_props = kite_utils::LoadProperties(kite_params_file);
    AlgorithmProperties algo_props;
    algo_props.Integrator = CVODES;
    algo_props.sampling_time = 0.02;

    /** kite model */
    KiteDynamics kite(kite_props, algo_props);

    /** no tether */
    kite_props.Tether.Ks = 0.0;
    kite_props.Tether.Kd = 0.0;

    KiteDynamics aircraft(kite_props, algo_props); // "simple" model

    /** get dynamics function and state Jacobian */
    casadi::Function DynamicsFunc = aircraft.getNumericDynamics();

    /** state bounds */
    casadi::DM INF = casadi::DM::inf(1);
    casadi::DM LBX = casadi::DM::vertcat({2.0, -INF, -INF, -4 * M_PI, -4 * M_PI, -4 * M_PI, -INF, -INF, -INF,
                                          -1.05, -1.05, -1.05, -1.05});
    casadi::DM UBX = casadi::DM::vertcat({INF, INF, INF, 4 * M_PI, 4 * M_PI, 4 * M_PI, INF, INF, INF,
                                          1.05, 1.05, 1.05, 1.05});
    /** control bounds */
    casadi::DM LBU = casadi::DM::vec(id_control);
    casadi::DM UBU = casadi::DM::vec(id_control);

    /** ----------------------------------------------------------------------------------*/
    const int num_segments = 100;
    const int poly_order   = 5;
    const int dimx         = 13;
    const int dimu         = 3;
    const int dimp         = 0;
    const double tf        = 20.0;

    Chebyshev<casadi::SX, poly_order, num_segments, dimx, dimu, dimp> spectral;
    casadi::SX diff_constr = spectral.CollocateDynamics(DynamicsFunc, 0, tf);
    diff_constr = diff_constr(casadi::Slice(0, num_segments * poly_order * dimx));

    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();
    casadi::SX varp = spectral.VarP();

    casadi::SX opt_var = casadi::SX::vertcat(casadi::SXVector{varx, varu, varp});

    casadi::SX lbg = casadi::SX::zeros(diff_constr.size());
    casadi::SX ubg = casadi::SX::zeros(diff_constr.size());

    /** set inequality (box) constraints */
    /** state */
    casadi::SX lbx = casadi::SX::repmat(LBX, num_segments * poly_order + 1, 1);
    casadi::SX ubx = casadi::SX::repmat(UBX, num_segments * poly_order + 1, 1);

    /** control */
    lbx = casadi::SX::vertcat({lbx, LBU});
    ubx = casadi::SX::vertcat({ubx, UBU});

    /** formulate NLP */
    casadi::SXDict NLP;
    casadi::Dict OPTS;
    casadi::DMDict ARG;
    NLP["x"] = opt_var;
    NLP["f"] = 0;
    NLP["g"] = diff_constr;

    OPTS["ipopt.linear_solver"]  = "ma97";
    OPTS["ipopt.print_level"]    = 5;
    OPTS["ipopt.tol"]            = 1e-4;
    OPTS["ipopt.acceptable_tol"] = 1e-4;
    OPTS["ipopt.warm_start_init_point"] = "yes";

    casadi::Function NLP_Solver = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    /** set default args */
    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;

    casadi::DM feasible_control = (UBU + LBU) / 2;
    casadi::DM init_state = casadi::DM::vertcat({5.3104597e+00,  -2.0782698e-01,  1.3057420e+00, 1.6608125e-01,  -1.7443906e+00,  -8.5414779e-01,
                                                 2.5189837e+00,  -7.3947045e-01,  -2.1751468e-01,   3.6023340e-01,   5.4059382e-01,  -1.8748153e-01,  -7.3681384e-01});
    casadi::DM feasible_state = casadi::DM::repmat(init_state, (num_segments * poly_order + 1), 1);

    /** if the solutions available load them from file */
    if(kite_utils::file_exists("solution_x0.txt"))
    {
        ARG["x0"] = kite_utils::read_from_file("solution_x0.txt");
    }
    else
    {
        ARG["x0"] = casadi::DM::vertcat(casadi::DMVector{feasible_state, feasible_control});
    }

    if(kite_utils::file_exists("solution_lam_g.txt"))
    {
        ARG["lam_g0"] = kite_utils::read_from_file("solution_lam_g.txt");
    }

    if(kite_utils::file_exists("solution_lam_x.txt"))
    {
        ARG["lam_x0"] = kite_utils::read_from_file("solution_lam_x.txt");
    }

    int idx_in = num_segments * poly_order * dimx;
    int idx_out = idx_in + dimx;
    ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = init_state;
    ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = init_state;


    /** get a initial solution for the "simple" model */
    casadi::DMDict solution = NLP_Solver(ARG);

    casadi::DM x0     = solution.at("x");
    casadi::DM lam_g0 = solution.at("lam_g");
    casadi::DM lam_x0 = solution.at("lam_x");

    /** save the initial guess if optimal solution is found */
    casadi::Dict stats = NLP_Solver.stats();
    std::string solve_status = static_cast<std::string>(stats["return_status"]);
    std::cout << stats << "\n";
    if(solve_status.compare("Solve_Succeeded") == 0)
    {
        kite_utils::write_to_file("solution_x0.txt", x0);
        kite_utils::write_to_file("solution_lam_g.txt", lam_g0);
        kite_utils::write_to_file("solution_lam_x.txt", lam_x0);
    }


    /** ---------------------------------------------- */
    /** appproximate the stiff model */
    casadi::Function KiteDynamicsFunc = kite.getNumericDynamics();
    casadi::SX kite_constr = spectral.CollocateDynamics(KiteDynamicsFunc, 0, tf);
    kite_constr = kite_constr(casadi::Slice(0, num_segments * poly_order * dimx));

    /** create the homotopy equations */
    casadi::SX lambda = casadi::SX::sym("lambda"); // homotopy paramter
    casadi::SX homotopy = (lambda) * kite_constr + (1 - lambda) * diff_constr;

    /** adjoin the homotopy parameter to the optimization variable and create new NLP */
    opt_var = casadi::SX::vertcat(casadi::SXVector{opt_var, lambda});
    lbx = casadi::SX::vertcat({lbx, 0});
    ubx = casadi::SX::vertcat({ubx, 0});

    NLP["x"] = opt_var;
    NLP["f"] = 0;
    NLP["g"] = homotopy;

    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;

    ARG["lbx"](casadi::Slice(idx_in, idx_out), 0) = init_state;
    ARG["ubx"](casadi::Slice(idx_in, idx_out), 0) = init_state;

    ARG["x0"]     = casadi::DM::vertcat({x0, 0.05});
    ARG["lam_g0"] = lam_g0;
    ARG["lam_x0"] = casadi::DM::vertcat({lam_x0, 0});

    OPTS["ipopt.tol"]            = 1e-4;
    OPTS["ipopt.acceptable_tol"] = 1e-4;
    OPTS["ipopt.print_level"]    = 1;

    casadi::Function HomotopyCorrector = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    /** gradually increase homotopy parameter and solve nonlinear problem */
    double lambda_val = 0.0;
    double d_lambda_val = 0.0001;
    while(lambda_val < 1.0)
    {
        lambda_val += d_lambda_val;

        /** round-off last lambda */
        if(std::fabs(lambda_val - 1.0) < 0.006)
            lambda_val = 1.0;

        /** update homotopy parameter */
        ARG["lbx"](lbx.size1() - 1, 0) = lambda_val;
        ARG["ubx"](ubx.size1() - 1, 0) = lambda_val;

        std::cout << "------------------------------------------------ \n";
        std::cout << "LAMBDA: " << lambda_val << "\n";
        std::cout << "------------------------------------------------ \n";

        /** solve homotopy equations */
        solution = HomotopyCorrector(ARG);
        stats = HomotopyCorrector.stats();

        /** update warm starting parameters if solve succeeded*/
        std::string it_solve_status = static_cast<std::string>(stats["return_status"]);
        if(it_solve_status.compare("Solve_Succeeded") == 0)
        {
            ARG["x0"]     = solution.at("x");
            ARG["lam_g0"] = solution.at("lam_g");
            ARG["lam_x0"] = solution.at("lam_x");
        }

        /** adjust step: heuristic */
        std::cout << stats << "\n";
        int num_iter = static_cast<int>(stats["iter_count"]);
        if((num_iter < 50) && (d_lambda_val < 0.01))
            d_lambda_val *= 1.0;
    }

    /** save solution to the files */
    casadi::DM res_x     = solution.at("x");
    casadi::DM res_lam_g = solution.at("lam_g");
    casadi::DM res_lam_x = solution.at("lam_x");
    kite_utils::write_to_file("id_x0.txt", res_x(casadi::Slice(0, res_x.size1() - 1))); // exclude homotopy parameter from the solution
    kite_utils::write_to_file("id_lam_g.txt", res_lam_g);
    kite_utils::write_to_file("id_lam_x.txt", res_lam_x(casadi::Slice(0, res_lam_x.size1() - 1)));

    casadi::DM trajectory = res_x(casadi::Slice(0, varx.size1()));
    std::ofstream est_trajectory_file("estimated_trajectory.txt", std::ios::out);

    if(!est_trajectory_file.fail())
    {
        for (int i = 0; i < trajectory.size1(); i = i + dimx)
        {
            std::vector<double> tmp = trajectory(casadi::Slice(i, i + dimx),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                est_trajectory_file << tmp[j] << " ";
            }
            est_trajectory_file << "\n";
        }
    }
    est_trajectory_file.close();

    return 0;
}
