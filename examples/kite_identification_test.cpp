// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "kiteNMPF.h"
#include "integrator.h"
#include <fstream>
#include "pseudospectral/chebyshev.hpp"

using namespace casadi;


int main(void)
{
    /** Load identification data */
    std::ifstream id_data_file("id_data_state.txt", std::ios::in);
    std::ifstream id_control_file("id_data_control.txt", std::ios::in);
    const int DATA_POINTS = 251;
    const int state_size   = 13;
    const int control_size = 3;

    DM id_data    = DM::zeros(state_size, DATA_POINTS);
    DM id_control = DM::zeros(control_size, DATA_POINTS);

    /** load state trajectory */
    if(!id_data_file.fail())
    {
    for(uint i = 0; i < DATA_POINTS; ++i) {
        for(uint j = 0; j < state_size; ++j){
            double entry;
            id_data_file >> entry;
            id_data(j,i) = entry;
        }
    }
    }
    else
    {
        std::cout << "Could not open : id state data file \n";
        id_data_file.clear();
    }

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

    /** define kite dynamics */
    std::string kite_params_file = "umx_radian.yaml";
    KiteProperties kite_props = kite_utils::LoadProperties(kite_params_file);

    AlgorithmProperties algo_props;
    algo_props.Integrator = CVODES;
    algo_props.sampling_time = 0.02;
    KiteDynamics kite(kite_props, algo_props, true);
    KiteDynamics kite_int(kite_props, algo_props); //integration model
    Function ode = kite_int.getNumericDynamics();

    /** get dynamics function and state Jacobian */
    Function DynamicsFunc = kite.getNumericDynamics();

    /** state bounds */
    DM LBX = DM::vertcat({2.0, -DM::inf(1), -DM::inf(1), -4 * M_PI, -4 * M_PI, -4 * M_PI, -DM::inf(1), -DM::inf(1), -DM::inf(1),
                          -1.05, -1.05, -1.05, -1.05});
    DM UBX = DM::vertcat({DM::inf(1), DM::inf(1), DM::inf(1), 4 * M_PI, 4 * M_PI, 4 * M_PI, DM::inf(1), DM::inf(1), DM::inf(1),
                          1.05, 1.05, 1.05, 1.05});
    /** control bounds */
    DM LBU = DM::vec(id_control);
    DM UBU = DM::vec(id_control);

    /** parameter bounds */
    YAML::Node config = YAML::LoadFile("umx_radian.yaml");
    double CL0 = config["aerodynamic"]["CL0"].as<double>();
    double CLa_tot = config["aerodynamic"]["CLa_total"].as<double>();

    double CD0_tot = config["aerodynamic"]["CD0_total"].as<double>();
    double CYb = config["aerodynamic"]["CYb"].as<double>();
    double Cm0 = config["aerodynamic"]["Cm0"].as<double>();
    double Cma = config["aerodynamic"]["Cma"].as<double>();
    double Cnb = config["aerodynamic"]["Cnb"].as<double>();
    double Clb = config["aerodynamic"]["Clb"].as<double>();

    double CLq = config["aerodynamic"]["CLq"].as<double>();
    double Cmq = config["aerodynamic"]["Cmq"].as<double>();
    double CYr = config["aerodynamic"]["CYr"].as<double>();
    double Cnr = config["aerodynamic"]["Cnr"].as<double>();
    double Clr = config["aerodynamic"]["Clr"].as<double>();
    double CYp = config["aerodynamic"]["CYp"].as<double>();
    double Clp = config["aerodynamic"]["Clp"].as<double>();
    double Cnp = config["aerodynamic"]["Cnp"].as<double>();

    double CLde = config["aerodynamic"]["CLde"].as<double>();
    double CYdr = config["aerodynamic"]["CYdr"].as<double>();
    double Cmde = config["aerodynamic"]["Cmde"].as<double>();
    double Cndr = config["aerodynamic"]["Cndr"].as<double>();
    double Cldr = config["aerodynamic"]["Cldr"].as<double>();

    double Lt = config["tether"]["length"].as<double>();
    double Ks = config["tether"]["Ks"].as<double>();
    double Kd = config["tether"]["Kd"].as<double>();
    double rx = config["tether"]["rx"].as<double>();
    double rz = config["tether"]["rz"].as<double>();

    DM REF_P = DM::vertcat({CL0, CLa_tot, CD0_tot, CYb, Cm0, Cma, Cnb, Clb, CLq, Cmq,
                            CYr, Cnr, Clr, CYp, Clp, Cnp, CLde, CYdr, Cmde, Cndr, Cldr,
                            Lt, Ks, Kd, rx, rz});
    DM LBP = REF_P; DM UBP = REF_P;
    LBP = -DM::inf(26);
    UBP = DM::inf(26);


    LBP[0] = REF_P[0] -  0.1 * fabs(REF_P[0]); UBP[0] = REF_P[0] +  0.2 * fabs(REF_P[0]); // CL0
    LBP[1] = REF_P[1] - 0.1 * fabs(REF_P[1]); UBP[1] = REF_P[1] +  0.2 * fabs(REF_P[1]); // CLa
    LBP[2] = REF_P[2] -  0.2 * fabs(REF_P[2]); UBP[2] = REF_P[2] + 0.25 * fabs(REF_P[2]); // CD0
    LBP[3] = REF_P[3] -  0.5 * fabs(REF_P[3]); UBP[3] = REF_P[3] +  0.5 * fabs(REF_P[3]); // CYb
    LBP[4] = REF_P[4] -  0.5 * fabs(REF_P[4]); UBP[4] = REF_P[4] +  0.5 * fabs(REF_P[4]); // Cm0
    LBP[5] = REF_P[5] -  0.1 * fabs(REF_P[5]); UBP[5] = REF_P[5] + 0.30 * fabs(REF_P[5]); // Cma
    LBP[6] = REF_P[6] -  0.5 * fabs(REF_P[6]); UBP[6] = REF_P[6] +  0.5 * fabs(REF_P[6]); // Cnb
    LBP[7] = REF_P[7] -  0.5 * fabs(REF_P[7]); UBP[7] = REF_P[7] +  0.5 * fabs(REF_P[7]); // Clb
    LBP[8] = REF_P[8] -  0.2 * fabs(REF_P[8]); UBP[8] = REF_P[8] +  0.2 * fabs(REF_P[8]); // CLq
    LBP[9] = REF_P[9] -  0.3 * fabs(REF_P[9]); UBP[9] = REF_P[9] +  0.3 * fabs(REF_P[9]); // Cmq

    LBP[10] = REF_P[10] -  0.3 * fabs(REF_P[10]); UBP[10] = REF_P[10] +  0.3 * fabs(REF_P[10]); // CYr
    LBP[11] = REF_P[11] -  0.5 * fabs(REF_P[11]); UBP[11] = REF_P[11] +  0.5 * fabs(REF_P[11]); // Cnr
    LBP[12] = REF_P[12] -  0.5 * fabs(REF_P[12]); UBP[12] = REF_P[12] +  0.5 * fabs(REF_P[12]); // Clr
    LBP[13] = REF_P[13] -  0.5 * fabs(REF_P[13]); UBP[13] = REF_P[13] +  0.5 * fabs(REF_P[13]); // CYp
    LBP[14] = REF_P[14] -  0.5 * fabs(REF_P[14]); UBP[14] = REF_P[14] +  0.5 * fabs(REF_P[14]); // Clp
    LBP[15] = REF_P[15] -  0.3 * fabs(REF_P[15]); UBP[15] = REF_P[15] +  1.0 * fabs(REF_P[15]); // Cnp
    LBP[16] = REF_P[16] -  0.5 * fabs(REF_P[16]); UBP[16] = REF_P[16] +  0.5 * fabs(REF_P[16]); // CLde
    LBP[17] = REF_P[17] -  0.5 * fabs(REF_P[17]); UBP[17] = REF_P[17] +  0.5 * fabs(REF_P[17]); // CYdr
    LBP[18] = REF_P[18] -  0.5 * fabs(REF_P[18]); UBP[18] = REF_P[18] +  0.5 * fabs(REF_P[18]); // Cmde
    LBP[19] = REF_P[19] -  0.5 * fabs(REF_P[19]); UBP[19] = REF_P[19] +  0.5 * fabs(REF_P[19]); // Cndr
    LBP[20] = REF_P[20] -  0.5 * fabs(REF_P[20]); UBP[20] = REF_P[20] +  0.5 * fabs(REF_P[20]); // Cldr

    LBP[21] = 2.60;    UBP[21] = 2.70;   // tether length
    LBP[22] = 50.0;    UBP[22] = 100.0;  // Ks
    LBP[23] = 5.0;     UBP[23] = 15;     // Kd
    LBP[24] = -0.01;   UBP[24] = 0.01;   // rx
    LBP[25] = -0.01;   UBP[25] = 0.01;   // rz

    std::cout << "OK so far \n";

    /** ----------------------------------------------------------------------------------*/
    const int num_segments = 50;
    const int poly_order   = 5;
    const int dimx         = 13;
    const int dimu         = 3;
    const int dimp         = 26;
    const double tf        = 10.0;

    Chebyshev<SX, poly_order, num_segments, dimx, dimu, dimp> spectral;
    SX diff_constr = spectral.CollocateDynamics(DynamicsFunc, 0, tf);
    diff_constr = diff_constr(casadi::Slice(0, num_segments * poly_order * dimx));

    SX varx = spectral.VarX();
    SX varu = spectral.VarU();
    SX varp = spectral.VarP();

    SX opt_var = SX::vertcat(SXVector{varx, varu, varp});

    SX lbg = SX::zeros(diff_constr.size());
    SX ubg = SX::zeros(diff_constr.size());

    /** set inequality (box) constraints */
    /** state */
    SX lbx = SX::repmat(LBX, num_segments * poly_order + 1, 1);
    SX ubx = SX::repmat(UBX, num_segments * poly_order + 1, 1);

    /** control */
    lbx = SX::vertcat({lbx, LBU});
    ubx = SX::vertcat({ubx, UBU});

    /** parameters */
    lbx = SX::vertcat({lbx, LBP});
    ubx = SX::vertcat({ubx, UBP});


    DM Q  = SX::diag(SX({1e3, 1e2, 1e2,  1e2, 1e2, 1e2,  1e1, 1e1, 1e2,  1e2, 1e2, 1e2, 1e2})); //good one as well
    //DM Q = 1e1 * DM::eye(13);
    double alpha = 100.0;


    SX fitting_error = 0;
    SX varx_ = SX::reshape(varx, state_size, DATA_POINTS);
    for (uint j = 0; j < DATA_POINTS; ++j)
    {
        SX measurement = id_data(Slice(0, id_data.size1()), j);
        SX error = measurement - varx_(Slice(0, varx_.size1()), varx_.size2() - j - 1);
        fitting_error += static_cast<double>(1.0 / DATA_POINTS) * SX::sumRows( SX::mtimes(Q, pow(error, 2)) );
    }

    /** add regularisation */
    // fitting_error = fitting_error + alpha * SX::dot(varp - SX({REF_P}), varp - SX({REF_P}));

    /** alternative approximation */
    /**
    SX x = SX::sym("x", state_size);
    SX y = SX::sym("y", state_size);
    SX cost_function = SX::sumRows( SX::mtimes(Q, pow(x - y, 2)) );
    Function IdCost = Function("IdCost",{x,y}, {cost_function});
    SX fitting_error2 = spectral.CollocateIdCost(IdCost, id_data, 0, tf);
    fitting_error2 = fitting_error2 + alpha * SX::dot(varp - SX({REF_P}), varp - SX({REF_P}));
    */

    /** formulate NLP */
    SXDict NLP;
    Dict OPTS;
    DMDict ARG;
    NLP["x"] = opt_var;
    NLP["f"] = fitting_error;
    NLP["g"] = diff_constr;

    OPTS["ipopt.linear_solver"]  = "ma97";
    OPTS["ipopt.print_level"]    = 5;
    OPTS["ipopt.tol"]            = 1e-4;
    OPTS["ipopt.acceptable_tol"] = 1e-4;
    OPTS["ipopt.warm_start_init_point"] = "yes";
    //OPTS["ipopt.max_iter"]       = 20;

    Function NLP_Solver = nlpsol("solver", "ipopt", NLP, OPTS);

    std::cout << "Ok here as well \n";

    /** set default args */
    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;

    DMDict solution;
    DM feasible_state;
    DM init_state = id_data(Slice(0, id_data.size1()), 0);

    /** if the solutions available load them from file */
    if(kite_utils::file_exists("id_x0.txt"))
    {
        DM sol_x  = kite_utils::read_from_file("id_x0.txt");
        ARG["x0"] = DM::vertcat(DMVector{sol_x, REF_P});
        feasible_state = sol_x;

        std::cout << "Initial guess loaded from a file \n";
    }
    else
    {
        /** otherwise, provide initial guess from integrator */
        casadi::DMDict props;
        props["scale"] = 0;
        props["P"] = casadi::DM::diag(casadi::DM({0.1, 1/3.0, 1/3.0, 1/2.0, 1/5.0, 1/2.0, 1/3.0, 1/3.0, 1/3.0, 1.0, 1.0, 1.0, 1.0}));
        props["R"] = casadi::DM::diag(casadi::DM({1/0.15, 1/0.2618, 1/0.2618}));
        PSODESolver<poly_order,num_segments,dimx,dimu>ps_solver(ode, tf, props);

        DM init_control = DM({0.1, 0.0, 0.0});
        init_control = casadi::DM::repmat(init_control, (num_segments * poly_order + 1), 1);
        solution = ps_solver.solve_trajectory(init_state, init_control, true);
        feasible_state = solution.at("x");
        ARG["x0"] = casadi::DM::vertcat(casadi::DMVector{feasible_state, REF_P});
    }

    if(kite_utils::file_exists("id_lam_g.txt"))
    {
        ARG["lam_g0"] = kite_utils::read_from_file("id_lam_g.txt");
    }
    else
    {
        ARG["lam_g0"] = solution.at("lam_g");
    }

    if(kite_utils::file_exists("id_lam_x.txt"))
    {
        DM sol_lam_x = kite_utils::read_from_file("id_lam_x.txt");
        ARG["lam_x0"] = DM::vertcat({sol_lam_x, DM::zeros(REF_P.size1())});
    }
    else
    {
        ARG["lam_x0"] = DM::vertcat({solution.at("lam_x"), DM::zeros(REF_P.size1())});
    }


    /** write to initial trajectory to a file */
    std::ofstream trajectory_file("integrated_trajectory.txt", std::ios::out);
    if(!trajectory_file.fail())
    {
        for (int i = 0; i < varx.size1(); i = i + 13)
        {
            std::vector<double> tmp = feasible_state(Slice(i, i + 13),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                trajectory_file << tmp[j] << " ";
            }
            trajectory_file << "\n";
        }
    }
    trajectory_file.close();


    int idx_in = num_segments * poly_order * dimx;
    int idx_out = idx_in + dimx;
    ARG["lbx"](Slice(idx_in, idx_out), 0) = init_state;
    ARG["ubx"](Slice(idx_in, idx_out), 0) = init_state;

    /** solve the identification problem */
    DMDict res = NLP_Solver(ARG);
    DM result = res.at("x");
    DM lam_x  = res.at("lam_x");

    DM new_params = result(Slice(result.size1() - varp.size1(), result.size1()));
    DM param_sens = lam_x(Slice(lam_x.size1() - varp.size1(), lam_x.size1()));

    std::cout << "PARAMETER SENSITIVITIES: " << param_sens << "\n";

    std::vector<double> new_params_vec = new_params.nonzeros();

    DM trajectory = result(Slice(0, varx.size1()));
    //DM trajectory = DM::reshape(traj, DATA_POINTS, dimx );
    std::ofstream est_trajectory_file("estimated_trajectory.txt", std::ios::out);

    if(!est_trajectory_file.fail())
    {
        for (int i = 0; i < trajectory.size1(); i = i + dimx)
        {
            std::vector<double> tmp = trajectory(Slice(i, i + dimx),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                est_trajectory_file << tmp[j] << " ";
            }
            est_trajectory_file << "\n";
        }
    }
    est_trajectory_file.close();

    /** update parameter file */
    config["aerodynamic"]["CL0"] = new_params_vec[0];
    config["aerodynamic"]["CLa_total"] = new_params_vec[1];
    config["aerodynamic"]["CD0_total"] = new_params_vec[2];
    config["aerodynamic"]["CYb"] = new_params_vec[3];
    config["aerodynamic"]["Cm0"] = new_params_vec[4];
    config["aerodynamic"]["Cma"] = new_params_vec[5];
    config["aerodynamic"]["Cnb"] = new_params_vec[6];
    config["aerodynamic"]["Clb"] = new_params_vec[7];

    config["aerodynamic"]["CLq"] = new_params_vec[8];
    config["aerodynamic"]["Cmq"] = new_params_vec[9];
    config["aerodynamic"]["CYr"] = new_params_vec[10];
    config["aerodynamic"]["Cnr"] = new_params_vec[11];
    config["aerodynamic"]["Clr"] = new_params_vec[12];
    config["aerodynamic"]["CYp"] = new_params_vec[13];
    config["aerodynamic"]["Clp"] = new_params_vec[14];
    config["aerodynamic"]["Cnp"] = new_params_vec[15];

    config["aerodynamic"]["CLde"] = new_params_vec[16];
    config["aerodynamic"]["CYdr"] = new_params_vec[17];
    config["aerodynamic"]["Cmde"] = new_params_vec[18];
    config["aerodynamic"]["Cndr"] = new_params_vec[19];
    config["aerodynamic"]["Cldr"] = new_params_vec[20];

    config["tether"]["length"] = new_params_vec[21];
    config["tether"]["Ks"] = new_params_vec[22];
    config["tether"]["Kd"] = new_params_vec[23];
    config["tether"]["rx"] = new_params_vec[24];
    config["tether"]["rz"] = new_params_vec[25];

    std::ofstream fout("umx_radian_id.yaml");
    fout << config;
}
