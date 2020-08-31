#include "gtest/gtest.h"
#include "solvers/admm.hpp"
#include <Eigen/IterativeLinearSolvers>

TEST(QPSolverTest, admmSimpleQP)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 3,2> A;
    Eigen::Matrix<Scalar, 3,1> l;
    Eigen::Matrix<Scalar, 3,1> u;
    Eigen::Matrix<Scalar, 2,1> solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, 0.7, 0.7;
    solution << 0.3, 0.7;

    ADMM<2, 3, Scalar> prob;
    prob.settings().max_iter = 1000;

    prob.solve(H,h,A,l,u);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}

TEST(QPSolverTest, admmSinglePrecisionFloat)
{
    using Scalar = float;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 3,2> A;
    Eigen::Matrix<Scalar, 3,1> l;
    Eigen::Matrix<Scalar, 3,1> u;
    Eigen::Matrix<Scalar, 2,1> solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, Scalar(0.7), Scalar(0.7);
    solution << Scalar(0.3), Scalar(0.7);

    ADMM<2, 3, Scalar> prob;
    prob.settings().max_iter = 1000;

    prob.solve(H,h,A,l,u);
    Eigen::Vector2f sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(solution, Scalar(1e-2)));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, status_t::SOLVED);
}

TEST(QPSolverTest, admmConstraintViolation)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 3,2> A;
    Eigen::Matrix<Scalar, 3,1> l;
    Eigen::Matrix<Scalar, 3,1> u;
    Eigen::Matrix<Scalar, 2,1> solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, 0.7, 0.7;
    solution << 0.3, 0.7;

    ADMM<2, 3, Scalar> prob;

    prob.settings().eps_rel = 1e-4;
    prob.settings().eps_abs = 1e-4;

    prob.solve(H,h,A,l,u);
    Eigen::Vector2d sol = prob.primal_solution();

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower = A * sol - l;
    Eigen::Vector3d upper = A * sol - u;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(QPSolverTest, admmAdaptiveRho)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 3,2> A;
    Eigen::Matrix<Scalar, 3,1> l;
    Eigen::Matrix<Scalar, 3,1> u;
    Eigen::Matrix<Scalar, 2,1> solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, 0.7, 0.7;
    solution << 0.3, 0.7;

    ADMM<2, 3, Scalar> prob;

    prob.settings().adaptive_rho = false;
    prob.settings().adaptive_rho_interval = 10;

    prob.solve(H,h,A,l,u);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_EQ(prob.info().status, SOLVED);
}


TEST(QPSolverTest, admmLox)
{
    using Scalar = double;

    Eigen::Matrix<Scalar, 2,2> H;
    Eigen::Matrix<Scalar, 2,1> h;
    Eigen::Matrix<Scalar, 3,2> A;
    Eigen::Matrix<Scalar, 3,1> l;
    Eigen::Matrix<Scalar, 3,1> u;
    Eigen::Matrix<Scalar, 2,1> solution;

    H << 4, 1,
         1, 2;
    h << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, 0.7, 0.7;
    solution << 0.3, 0.7;

    ADMM<2, 3, Scalar> prob;

    prob.settings().warm_start = false;
    prob.settings().max_iter = 1000;
    prob.settings().rho = 0.1;

    // solve whithout adaptive rho
    prob.settings().adaptive_rho = false;
    prob.solve(H,h,A,l,u);
    int prev_iter = prob.info().iter;

    // solve with adaptive rho
    prob.settings().adaptive_rho = true;
    prob.settings().adaptive_rho_interval = 10;
    prob.solve(H,h,A,l,u);

    auto info = prob.info();
    EXPECT_LT(info.iter, prob.settings().max_iter);
    EXPECT_LT(info.iter, prev_iter); // adaptive rho should improve :)
    EXPECT_EQ(info.status, SOLVED);
}

