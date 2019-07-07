#include "gtest/gtest.h"
#include <Eigen/Dense>


/*
 * minimize     0.5 * x'.H.x + h'.x
 * subject to   l_i <= x_i <= u_i
 */

class GradProj {
public:
    using Scalar = double;
    using x_t = Eigen::Vector2d;
    Eigen::Matrix2d H;
    Eigen::Vector2d h;
    Eigen::Vector2d l, u;
    GradProj() {
        H << 10, 0,
             0, 0.1;
        h << -1, -2;

        l << -1, -1;
        u << 1, 1;
    }

    Scalar f(const Eigen::Vector2d& x)
    {
        return 0.5 * x.dot(H * x) + h.dot(x);
    }

    Eigen::Vector2d grad_f(const Eigen::Vector2d& x)
    {
        Eigen::Vector2d grad;
        grad << H * x + h;
        return grad;
    }

    Eigen::Vector2d solve(const Eigen::Vector2d& x0)
    {
        Eigen::Vector2d x;
        x = x0;

        std::cout << "x " << x.transpose() << std::endl;

        for (int iter = 1; iter <= 100; iter++) {
            Scalar fx;
            x_t grad;

            fx = f(x);
            grad = grad_f(x);

            std::cout << "grad " << grad.transpose() << std::endl;

            // backtracking line search
            // Scalar alpha = 1.0;
            Scalar alpha = 0.9;
            const Scalar beta = 0.3; // 0 < beta < 1
            const Scalar c = 1e-5; // 0 < c < 1
            int i;
            for (i = 1;; i++) {
                x_t x_step;
                x_t p;

                p = alpha*grad;
                // gradient projection
                box_projection(p, l, u);
                x_step = x - p;

                if (f(x_step) <= fx - alpha * c * grad.dot(x_step - x)) {
                    // std::cout << f(x_step) << " f " << fx << " d " << alpha * c * grad.dot(x_step - x) << " alpha " << alpha << std::endl;
                    std::cout << "alpha " << alpha << "  step " << p.transpose() << std::endl;
                    x = x_step;
                    break;
                } else {
                    alpha = beta*alpha;
                }
            }
            std::cout << "x " << x.transpose() << std::endl;
        }

        return x;
    }

    void box_projection(x_t& x, const x_t& l, const x_t& u) const
    {
        x = x.cwiseMax(l).cwiseMin(u);
    }
};

TEST(GradProjTestCase, TestSimple) {
    GradProj test;
    Eigen::Vector2d x0(0, 0);
    Eigen::Vector2d sol;

    sol = test.solve(x0);

    std::cout << "sol " << sol.transpose() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
