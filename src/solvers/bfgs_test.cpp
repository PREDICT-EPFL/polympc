#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "gtest/gtest.h"
#include "bfgs.hpp"

bool is_posdef(Eigen::MatrixXd H)
{
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(H);
    for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
        double v = eigensolver.eigenvalues()(i).real();
        if (v <= 0) {
            return false;
        }
    }
    return true;
}

TEST(BFGSTestCase, Test2D_posdef) {
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;

    Vec step, delta_grad;
    Mat H; // true constant hessian;
    H << 2, 0,
         0, 1;
    Mat B = Mat::Identity();

    for (int i = 0; i < 10; i++) {
        // do some random steps
        step = {sin(i), cos(i)};

        delta_grad = H*step;
        BFGS_update(B, step, delta_grad);

        EXPECT_TRUE(is_posdef(B));
    }

    std::cout << "B\n" << B << std::endl;
    EXPECT_TRUE(B.isApprox(H, 1e-3));
}

TEST(BFGSTestCase, Test2D_indefinite) {
    using Scalar = double;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;

    Vec step, delta_grad;
    Mat H; // true constant hessian;
    H << 2, 0,
         0, -1;
    Mat B = Mat::Identity();

    for (int i = 0; i < 10; i++) {
        // do some random steps
        step = {sin(i), cos(i)};

        delta_grad = H*step;
        BFGS_update(B, step, delta_grad);

        EXPECT_TRUE(is_posdef(B));
    }
    std::cout << "B\n" << B << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
