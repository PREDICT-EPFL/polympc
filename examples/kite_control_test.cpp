#include "nmpf.hpp"
#include "kite.h"

using namespace casadi;

struct Path
{
    SXVector operator()(const SXVector &arg)
    {
        SX x = SX::sym("x");
        double h = M_PI / 6.0;
        double a = 0.2;
        double L = 5;
        SX theta = h + a * sin(2 * x);
        SX phi   = 4 * a * cos(x);
        SX Path  = SX::vertcat({theta, phi});
        Function path = Function("path", {x}, {Path});
        return path(arg);
    }
};


int main(int argc, char **argv)
{
    const int dimx = 3;
    const int dimu = 1;
    double tf = 2.0;
    polympc::nmpf<SimpleKinematicKite, Path, dimx, dimu> controller(tf);

    /** set state and control constraints */
    DM lbu = DM(std::vector<double>{-5, -10});
    DM ubu = DM(std::vector<double>{5, 10});
    controller.setLBU(lbu);
    controller.setUBU(ubu);

    DM lbx = DM::vertcat({0, -M_PI_2, -M_PI, -100, -100});
    DM ubx = DM::vertcat({M_PI_2, M_PI_2, M_PI, 100, 100});
    controller.setLBX(lbx);
    controller.setUBX(ubx);

    controller.setReferenceVelocity(0.05);

    DM state = DM::vertcat({M_PI_4, 0, 0, 0, 0});
    controller.computeControl(state);
    DM opt_ctl  = controller.getOptimalControl();
    DM opt_traj = controller.getOptimalTrajetory();

    std::cout << opt_ctl << "\n";
    std::cout << opt_traj << "\n";

    return 0;
}
