#include <iostream>
#include <fstream>
#include <vector>

#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"

template <typename Solver>
void callback(void *solver_p)
{
    Solver& s = *static_cast<Solver*>(solver_p);
    std::cout << s._x.transpose() << std::endl;
}

using Problem = polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>>;
using Approximation = Chebyshev<3>; // POLY_ORDER = 3

using controller_t = polympc::nmpc<Problem, Approximation, int>;
using var_t = controller_t::var_t;
using dual_t = controller_t::dual_t;
using State = controller_t::State;
using Control = controller_t::Control;
using Parameters = controller_t::Parameters;

enum
{
    NX = State::RowsAtCompileTime,
    NU = Control::RowsAtCompileTime,
    NP = Parameters::RowsAtCompileTime,
    VARX_SIZE = controller_t::VARX_SIZE,
    VARU_SIZE = controller_t::VARU_SIZE,
    VARP_SIZE = controller_t::VARP_SIZE,
    VAR_SIZE = var_t::RowsAtCompileTime,
};

void print_info(void)
{
    printf("NX = %d\n", NX);
    printf("NU = %d\n", NU);
    printf("NP = %d\n", NP);
    printf("VARX_SIZE = %d\n", VARX_SIZE);
    printf("VARU_SIZE = %d\n", VARU_SIZE);
    printf("VARP_SIZE = %d\n", VARP_SIZE);
    printf("VAR_SIZE = %d\n", VAR_SIZE);

    printf("controller_t size %lu\n", sizeof(controller_t));
    printf("controller_t::cost_colloc_t size %lu\n", sizeof(controller_t::cost_colloc_t));
    printf("controller_t::ode_colloc_t size %lu\n", sizeof(controller_t::ode_colloc_t));
    printf("controller_t::sqp_t size %lu\n", sizeof(controller_t::sqp_t));
    printf("controller_t::sqp_t::qp_t size %lu\n", sizeof(controller_t::sqp_t::qp_t));
    printf("controller_t::sqp_t::qp_solver_t size %lu\n", sizeof(controller_t::sqp_t::qp_solver_t));
}

void print_duals(const dual_t& y)
{
    Eigen::IOFormat fmt(3, 0, ", ", ",", "[", "]");
    std::cout << "duals" << std::endl;
    std::cout << "ode   " << y.template segment<VARX_SIZE>(0).transpose().format(fmt) << std::endl;
    std::cout << "x     " << y.template segment<VARX_SIZE-NX>(VARX_SIZE).transpose().format(fmt) << std::endl;
    std::cout << "x0    " << y.template segment<NX>(2*VARX_SIZE-NX).transpose().format(fmt) << std::endl;
    std::cout << "u     " << y.template segment<VARU_SIZE>(2*VARX_SIZE).transpose().format(fmt) << std::endl;
}

void print_sol(const var_t& sol)
{
    Eigen::IOFormat fmt(4, 0, ", ", ",", "[", "]");
    std::cout << "xyt" << std::endl;
    for (int i = 0; i < VARX_SIZE/NX; i++) {
       std::cout << sol.segment<NX>(NX*i).transpose().format(fmt) << ",\n";
    }
    std::cout << "u" << std::endl;
    for (int i = 0; i < VARU_SIZE/NU; i++) {
       std::cout << sol.segment<NU>(VARX_SIZE+NU*i).transpose().format(fmt) << ",\n";
    }
}

template <typename Var>
void save_csv(const char *name, const std::vector<Var>& vec)
{
    std::ofstream out(name);
    Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "", "");
    for (const Var& x : vec) {
        out << x.transpose().format(fmt) << "\n";
    }
}

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
    return !((x.array() == x.array())).all();
}

int main(int argc, char **argv)
{
    controller_t robot_controller;
    // robot_controller.solver.settings().iteration_callback = callback<controller_t::sqp_t>;

    State x = {-1, -1, 1.5};
    State dx;
    Control u;
    Parameters p(1.0);

    if (argc == 4) {
        double x0, y0, t0;
        x0 = atof(argv[1]);
        y0 = atof(argv[2]);
        t0 = atof(argv[3]);
        x << x0, y0, t0;
    }

    // bounds
    controller_t::State xu, xl;
    xu << 10, 10, 1e20;
    xl << -xu;
    controller_t::Control uu, ul;
    uu << 10, 1;
    ul << -10, -1;

    robot_controller.set_constraints(xl, xu, ul, uu);

    print_info();
    std::cout << "x0 " << x.transpose() << std::endl;

    std::vector<var_t> var_log;
    var_log.reserve(100);

    std::vector<State> traj_log;
    traj_log.reserve(100+1);
    traj_log.push_back(x);

    for (int i = 0; i < 100; i++) {

        var_t sol;
        if (i == 0) {
            sol = robot_controller.solve(x, p);
        } else {
            sol = robot_controller.solve(x, p);
            // sol = robot_controller.solve_warm_start(x, p);
        }

        u = sol.segment<2>(VARX_SIZE+VARU_SIZE-NU);

        // crude integrator
        const double dt = 0.001;
        for (int j = 0; j < 1000; j++) {
            robot_controller.ps_ode.m_f(x, u, p, dx);
            x = x + dt * dx;
        }

        // print_sol(sol);
        // print_duals(robot_controller.solver.dual_solution());
        std::cout << "iter " << robot_controller.solver.info().iter << "    ";
        std::cout << "x " << x.transpose() << "    ";
        std::cout << "u " << u.transpose() << std::endl;

        if (is_nan<var_t>(sol)) {
            std::cout << "ERROR: NAN" << std::endl;
            break;
        }

        traj_log.push_back(x);
        var_log.push_back(sol);
    }

    save_csv<State>("traj_log.csv", traj_log);
    save_csv<var_t>("var_log.csv", var_log);
}
