#include "gtest/gtest.h"
#include "solvers/box_admm.hpp"
#include <Eigen/IterativeLinearSolvers>

TEST(ADMMSolverTest, box_admmSimpleQP)
{
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

    prob.solve(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}


TEST(ADMMSolverTest, box_admmSinglePrecisionFloat)
{
    using Scalar = float;

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
    xu << Scalar(0.7), Scalar(0.7);
    solution << Scalar(0.3), Scalar(0.7);

    boxADMM<2, 1, Scalar> prob;
    prob.settings().max_iter = 150;

    prob.solve(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2f sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, Scalar(1e-2)));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}

TEST(ADMMSolverTest, box_admmConstraintViolation)
{
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

    prob.settings().eps_rel = 1e-4;
    prob.settings().eps_abs = 1e-4;

    prob.solve(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2d sol = prob.primal_solution();

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower;
    Eigen::Vector3d upper;
    lower.segment<1>(0) = A * sol - Al;
    upper.segment<1>(0) = A * sol - Au;
    lower.segment<2>(1) = sol - xl;
    upper.segment<2>(1) = sol - xu;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(ADMMSolverTest, box_admmAdaptiveRho)
{
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

    prob.settings().adaptive_rho = false;
    prob.settings().adaptive_rho_interval = 10;

    prob.solve(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_EQ(prob.info().status, SOLVED);
}


TEST(ADMMSolverTest, box_admmLox)
{
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

    prob.settings().warm_start = false;
    prob.settings().max_iter = 1000;
    prob.settings().rho = 0.1;

    // solve whithout adaptive rho
    prob.settings().adaptive_rho = false;
    prob.solve(H,h,A,Al,Au,xl,xu);
    int prev_iter = prob.info().iter;

    // solve with adaptive rho
    prob.settings().adaptive_rho = true;
    prob.settings().adaptive_rho_interval = 10;
    prob.solve(H,h,A,Al,Au,xl,xu);

    auto info = prob.info();
    EXPECT_LT(info.iter, prob.settings().max_iter);
    EXPECT_LT(info.iter, prev_iter); // adaptive rho should improve :)
    EXPECT_EQ(info.status, SOLVED);
}

#ifdef EIGEN_NO_DEBUG
TEST(ADMMSolverTest, box_admmConjugateGradientLinearSolver)
{
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

    boxADMM<2,1,double, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> prob;
    prob.solve(H,h,A,Al,Au,xl,xu);
    Eigen::Vector2d sol = prob.primal_solution();

    auto info = prob.info();
    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_EQ(info.status, SOLVED);
    EXPECT_LT(info.iter, prob.settings().max_iter); // convergence test
}
#endif

TEST(ADMMSolverTest, box_admmSimpleLP)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 1,1> H;
    Eigen::Matrix<Scalar, 1,1> h;
    Eigen::Matrix<Scalar, 0,1> A;
    Eigen::Matrix<Scalar, 0,1> al;
    Eigen::Matrix<Scalar, 0,1> au;
    Eigen::Matrix<Scalar, 1,1> xl,xu;
    Eigen::Matrix<Scalar, 1,1> solution;

    H << 0;
    h << 1;
    xl << -1e6;
    xu <<  1e6;
    solution << -1e6;

    boxADMM<1, 0, Scalar> prob;

    prob.settings().max_iter = 200;
    prob.settings().alpha = 1.0;
    prob.settings().adaptive_rho = true;
    prob.settings().check_termination = 10;

    prob.solve(H,h,A,al,au,xl,xu);
    Eigen::VectorXd sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}

TEST(ADMMSolverTest, box_admmNonConvex)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 1,1> H;
    Eigen::Matrix<Scalar, 1,1> h;
    Eigen::Matrix<Scalar, 0,1> A;
    Eigen::Matrix<Scalar, 0,1> al;
    Eigen::Matrix<Scalar, 0,1> au;
    Eigen::Matrix<Scalar, 1,1> xl,xu;
    Eigen::Matrix<Scalar, 1,1> solution, guess, dual_guess;

    H << -1;
    h << 0;
    xl << -1;
    xu <<  2;
    solution << 2;
    guess << 0.1;
    dual_guess << 0.1;


    boxADMM<1, 0, Scalar> prob;

    prob.settings().max_iter = 200;
    prob.settings().alpha = 1.0;
    prob.settings().adaptive_rho = true;
    prob.settings().rho = 2;
    prob.settings().check_termination = 10;

    prob.solve(H,h,A,al,au,xl,xu, guess, dual_guess);
    Eigen::VectorXd sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}


