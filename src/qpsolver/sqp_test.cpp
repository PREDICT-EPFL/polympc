#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

TEST(SQPTestCase, TestSimpleNLP) {
    SQP prob;
    SQP::Settings settings;
    settings.max_iter = 100;

    prob.solve(settings);

    std::cout << "Solution: x = \n" << prob._x.transpose() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
