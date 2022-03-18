#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "casadi/casadi.hpp"
#include "Eigen/Dense"

namespace polympc {

enum IntType {RK4, CVODES};

/** Solve ODE of the form : xdot = f(x, u, p) */
class ODESolver
{
public:
    ODESolver(const casadi::Function &rhs, const casadi::Dict &params = casadi::Dict());
    virtual ~ODESolver(){}

    casadi::DM solve(const casadi::DM &x0, const casadi::DM &u, const double &dt);
    casadi::DM solve(const casadi::DM &x0, const casadi::DM &u, const casadi::DM &p, const double &dt);

    void updateParams(const casadi::Dict &params);
    casadi::Dict getParams(){return Parameters;}
    int dim_x(){return nx;}
    int dim_u(){return nu;}
    int dim_p(){return np;}

private:
    /** right hand side of the ODE : f(x, u, p)*/
    casadi::Function RHS;
    casadi::Dict     Parameters;
    int              nx{0}, nu{0}, np{0};

    /** numerical options */
    double           dT;
    double           Tolerance;
    int              Method;
    int              MaxIter;

    /** RK4 */
    casadi::DM       rk4_solve(const casadi::DM &x0, const casadi::DM &u, const casadi::DM &p, const casadi::DM &dt);

    /** CVODES */
    casadi::DM       cvodes_solve(const casadi::DM &x0, const casadi::DM &u, const casadi::DM &p, const casadi::DM &dt);
    casadi::Function cvodes_integrator;
};

} // polympc namespace

#endif // INTEGRATOR_H
