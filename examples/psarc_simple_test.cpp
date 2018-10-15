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
    casadi::SX x = casadi::SX::sym("x");
    casadi::SX y = casadi::SX::sym("y");
    casadi::SX fx = casadi::SX::vertcat({x*x + 4*y*y - 9, 18*y - 14*x*x + 45});

    /** define system of nonlinear inequalities */
    var = casadi::SX::vertcat({x,y});
    sym_func = fx;
    sym_jac = casadi::SX::jacobian(sym_func, var);

    func = casadi::Function("lox",{var},{sym_func});
    jac = casadi::Function("pidor",{var},{sym_jac});
}

int main(void)
{
    /** create an initial guess */
    casadi::DM init_guess = casadi::DM::vertcat({0,0});

    symbolic_psarc<FX, casadi::Dict>psarc(init_guess);
    return 0;
}
