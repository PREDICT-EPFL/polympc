#include "solvers/qp_base.hpp"
#include "solvers/box_admm.hpp"

class MyQP : public QPBase<MyQP, 55, 40>
{

};




int main(void)
{
    MyQP qp;

    MyQP::qp_dual_t lb;
    MyQP::qp_dual_t ub;

    lb.setZero();
    ub.setOnes();

    qp.parse_constraints_bounds(lb, ub);


    /** box ADMM test */
    using Scalar = double;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 1,2> A;
    Eigen::Matrix<Scalar, 1,1> Al;
    Eigen::Matrix<Scalar, 1,1> Au;
    Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1;
    Al << 1;
    Au << 1;
    xl << 0, 0;
    xu << 0.7, 0.7;
    solution << 0.3, 0.7;

    boxADMM<2, 1, Scalar> prob;
    prob.settings().max_iter = 150;
    //prob.settings().check_termination = 150;

    prob.solve_box(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2d sol = prob.primal_solution();

    //EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    //EXPECT_LT(prob.iter, prob.settings().max_iter);
    //EXPECT_EQ(prob.info().status, status_t::SOLVED);

    return EXIT_SUCCESS;
}
