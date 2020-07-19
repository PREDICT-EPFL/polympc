//#include "legendre.hpp"
#include "polynomials/legendre.hpp"
#include "polynomials/projection.hpp"
#include "integration/chebyshev_integrator.hpp"

/** uniform random variable */
struct Unif
{
    double operator ()(const double &x) const
    {
        return  x;
    }
};


int main(void)
{
    using namespace polympc;
    using Legendre = Legendre<5>;
    Legendre leg;
    Unif distr;

    /** project lambda random variable */
    Projection<Legendre> proj(distr, 4, 6);
    std::cout << "Projection: " << proj.coeff.transpose() << "\n";

    /** solve x_dot = v * f_kite(x,u), where v ~ Unif(1,3) */
    casadi::SX x1 = casadi::SX::sym("x1"); casadi::SX x2 = casadi::SX::sym("x2"); casadi::SX x3 = casadi::SX::sym("x3");
    casadi::SX state = casadi::SX::vertcat({x1, x2, x3});
    casadi::SX x = casadi::SX::sym("x", 3 * 6);
    casadi::SX u = casadi::SX::sym("u");
    const double a = 1;
    const double b = 1;
    casadi::SX f1 = a * cos(x1) * cos(x2) * cos(x3);
    casadi::SX f2 = a * cos(x2) * sin(x3);
    casadi::SX f3 = b * cos(x1) * cos(x2) * u;
    casadi::Function f1_fun = casadi::Function("f1", {state}, {f1});
    casadi::Function f2_fun = casadi::Function("f2", {state}, {f2});
    casadi::Function f3_fun = casadi::Function("f3", {state}, {f3});

    Legendre::q_weights_t weights = leg.QWeights();
    Legendre::nodes_t nodes = leg.CPoints();
    Legendre::q_weights_t nfactors = leg.NFactors();
    int N = nodes.SizeAtCompileTime;

    /** construct K = N differential equations */
    casadi::SX sode = casadi::SX::zeros(3 * 6, 1);
    for(int k = 0; k < N; ++k)
    {
        casadi::SX integral1 = 0, integral2 = 0, integral3 = 0;
        for(int n = 0; n < N; ++n)
        {
            casadi::SX arg1 = 0, arg2 = 0, arg3 = 0;
            double omega = 0;
            for(int j = 0; j < N; ++j)
            {
                arg1 += x(j) * leg.eval(nodes[n], j);
                arg2 += x(j + N) * leg.eval(nodes[n], j);
                arg3 += x(j + 2 * N) * leg.eval(nodes[n], j);
                omega += proj.coeff[j] * leg.eval(nodes[n], j);
            }
            double fi_k = leg.eval(nodes[n], k);
            casadi::SX f1_n = f1_fun(casadi::SX::vertcat({arg1, arg2, arg3}))[0];
            casadi::SX f2_n = f2_fun(casadi::SX::vertcat({arg1, arg2, arg3}))[0];
            casadi::SX f3_n = f3_fun(casadi::SX::vertcat({arg1, arg2, arg3}))[0];
            integral1 += (weights[n] * omega * fi_k) * f1_n;
            integral2 += (weights[n] * omega * fi_k) * f2_n;
            integral3 += (weights[n] * omega * fi_k) * f3_n;
        }
        sode(k) = nfactors[k] * integral1;
        sode(k + N) = nfactors[k] * integral2;
        sode(k + 2 * N) = nfactors[k] * integral3;
    }

    casadi::Function sode_fun = casadi::Function("sode_fun",{x, u},{sode});
    casadi::DM x0  = casadi::DM({M_PI_4,0,0,0,0,0,  M_PI_4,0,0,0,0,0,  0,0,0,0,0,0,});
    casadi::DM ctl = 0.05;

    double tf = 1.0;
    casadi::DMDict props;
    casadi::DM ps_sol;
    PSODESolver<10, 1, 3*6, 1, 0>ps_solver(sode_fun, tf, props);
    ps_sol = ps_solver.solve(x0, ctl, true);

    /** save trajectory */
    std::ofstream solution_sode("solution_sode.txt", std::ios::out);
    int dimx = 3 * 6;

    if(!solution_sode.fail())
    {
        for (int i = 0; i < ps_sol.size1(); i = i + dimx)
        {
            std::vector<double> tmp = ps_sol(casadi::Slice(i, i + dimx),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                solution_sode << tmp[j] << " ";
            }
            solution_sode << "\n";
        }
    }
    solution_sode.close();

    casadi::DMVector res = sode_fun(casadi::DMVector{x0, ctl});
    std::cout << res[0] << "\n";

    return 0;
}
