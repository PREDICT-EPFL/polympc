#include "mobile_robot.hpp"
#include "nmpc.hpp"

using namespace casadi;

int main(void)
{
    using nmpc_controller = polympc::nmpc<MobileRobot, 3, 2, 2, 4>;

    DMDict options;
    options["mpc.scaling"] = 0;
    options["mpc.scale_x"] = DM::diag(DM({1,1,1}));
    options["mpc.scale_u"] = DM::diag(DM(std::vector<double>{1,1}));

    DM Q  = casadi::SX::diag(casadi::SX({10,150,50}));
    DM R  = casadi::SX::diag(casadi::SX(std::vector<double>{1,1}));
    DM P  = 1e1 * Q;

    options["mpc.Q"] = Q;
    options["mpc.R"] = R;
    options["mpc.P"] = P;

    DM target  = DM({0, 0, 0});
    double tf = 2.0;
    nmpc_controller robot_controller(target, tf, options);

    /** set state and control constraints */
    DM lbu = DM::vertcat({-1, -0.6});
    DM ubu = DM::vertcat({ 1,  0.6});
    robot_controller.setLBU(lbu);
    robot_controller.setUBU(ubu);

    /** set state/virtual state constraints */
    DM lbx = DM::vertcat({-DM::inf(), -DM::inf(), -DM::inf()});
    DM ubx = DM::vertcat({ DM::inf(),  DM::inf(),  DM::inf()});
    robot_controller.setLBX(lbx);
    robot_controller.setUBX(ubx);

    DM state = DM({ -1, -1, 0});
    robot_controller.computeControl(state);

    return 0;
}


