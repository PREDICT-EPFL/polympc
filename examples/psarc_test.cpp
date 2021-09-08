// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "homotopy/psarc.hpp"
#include "pseudospectral/chebyshev.hpp"
#include "kite.h"

struct FX
{
    FX();
    ~FX(){}

    using num = casadi::DM;
    using sym = casadi::SX;

    /** symbolic variable */
    sym var;
    sym sym_func;
    sym sym_jac;

    casadi::Function func;
    casadi::Function jac;

    num eval(const num &arg)
    {
        return func({arg})[0];
    }

    num jacobian(const num &arg)
    {
        return jac({arg})[0];
    }

    sym operator()()
    {
        return sym_func;
    }
};

FX::FX()
{
    /** create a kite model */
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

    kite_props.Tether.Ks = 0.0;
    kite_props.Tether.Kd = 0.0;

    /** kite model */
    KiteDynamics kite(kite_props, algo_props);
    casadi::Function KiteDynamicsFunc = kite.getNumericDynamics();

    /** collocate equations */
    const int num_segments = 100;
    const int poly_order   = 5;
    const int dimx         = 13;
    const int dimu         = 3;
    const int dimp         = 0;
    const double tf        = 20.0;

    Chebyshev<casadi::SX, poly_order, num_segments, dimx, dimu, dimp> spectral;
    casadi::SX diff_constr = spectral.CollocateDynamics(KiteDynamicsFunc, 0, tf);
    diff_constr = diff_constr(casadi::Slice(0, num_segments * poly_order * dimx));

    casadi::SX varx = spectral.VarX();
    casadi::SX varu = spectral.VarU();
    casadi::SX opt_var = casadi::SX::vertcat({varx, varu});

    casadi::SX init_state = casadi::SX::vertcat({5.3104597e+00,  -2.0782698e-01,  1.3057420e+00, 1.6608125e-01,  -1.7443906e+00,  -8.5414779e-01,
                                                 2.5189837e+00,  -7.3947045e-01,  -2.1751468e-01,   3.6023340e-01,   5.4059382e-01,  -1.8748153e-01,  -7.3681384e-01});
    casadi::SX init_constr = varx(casadi::Slice(varx.size1() - dimx, varx.size1())) - init_state;

    /** control constraints */
    casadi::SX u = casadi::SX::vec(id_control);
    casadi::SX ctl_constr = varu - u;

    /** define system of nonlinear inequalities */
    var = opt_var;
    sym_func = casadi::SX::vertcat({diff_constr, ctl_constr, init_constr});
    sym_jac = casadi::SX::jacobian(sym_func, var);

    func = casadi::Function("lox",{var},{sym_func});
    jac = casadi::Function("pidor",{var},{sym_jac});
}

int main(void)
{	
    /** create an initial guess */
    casadi::DM init_state = casadi::DM::vertcat({5.3104597e+00,  -2.0782698e-01,  1.3057420e+00, 1.6608125e-01,  -1.7443906e+00,  -8.5414779e-01,
                                                 2.5189837e+00,  -7.3947045e-01,  -2.1751468e-01,   3.6023340e-01,   5.4059382e-01,  -1.8748153e-01,  -7.3681384e-01});
    casadi::DM init_guess = casadi::DM::repmat(init_state, 501, 1);
    casadi::DM init_control = casadi::DM::vertcat({0.1, 0, 0});
    init_control = casadi::DM::repmat(init_control, 501, 1);
    init_guess = casadi::DM::vertcat({init_guess, init_control});

    symbolic_psarc<FX, casadi::Dict>psarc(init_guess);
    return 0;
}
