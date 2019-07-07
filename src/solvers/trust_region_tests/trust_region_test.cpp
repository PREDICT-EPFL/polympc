#include "gtest/gtest.h"
#include <Eigen/Dense>


#include <casadi/casadi.hpp>
#include <vector>

class Rosenbrock {
public:
    using var_t = Eigen::Vector2d;
    using gradient_t = Eigen::Vector2d;
    using hessian_t = Eigen::Matrix2d;

    casadi::Function m_f;
    casadi::Function m_grad;
    casadi::Function m_hess;

    Rosenbrock()
    {
        casadi::SX x = casadi::SX::sym("x", 2);
        const double a = 1;
        const double b = 100;

        casadi::SX z = 0;
        for (int i = 0; i < x.rows() - 1; i++) {
            z += pow(a - x(i), 2) + b * pow(x(i + 1) - pow(x(i), 2), 2);
        }

        casadi::SX f_sym = z;
        casadi::SX grad_sym = casadi::SX::gradient(f_sym, x);
        casadi::SX hess_sym = casadi::SX::jacobian(grad_sym, x);

        m_f = casadi::Function("f", {x}, {f_sym});
        m_grad = casadi::Function("g", {x}, {grad_sym});
        m_hess = casadi::Function("h", {x}, {hess_sym});
    }

    void operator()(const var_t& x, double& value)
    {
        std::vector<double> xv(x.size());
        var_t::Map(&xv[0]) = x;
        casadi::DM res;
        res = m_f({casadi::DM(xv)})[0];
        value = *casadi::DM::densify(res).nonzeros().data();
    }

    void gradient(const var_t& x, gradient_t& grad, double& value)
    {
        this->operator()(x, value);

        casadi::DM res;
        std::vector<double> xv(x.size());
        var_t::Map(&xv[0]) = x;

        res = m_grad({casadi::DM(xv)})[0];
        grad = gradient_t::Map(casadi::DM::densify(res).nonzeros().data());
    }

    void hessian(const var_t& x, hessian_t& hess, gradient_t& grad, double& value)
    {
        gradient(x, grad, value);

        casadi::DM res;
        std::vector<double> xv(x.size());
        var_t::Map(&xv[0]) = x;

        res = m_hess({casadi::DM(xv)})[0];
        hess = hessian_t::Map(casadi::DM::densify(res).nonzeros().data());
    }
};

class SimpleQP {
public:
    Eigen::Matrix2d H;
    Eigen::Vector2d h;
    Eigen::Vector2d l, u;

    SimpleQP()
    {
        H << 10, 0,
            0, 0.1;
        h << -1, -2;

        l << -1, -1;
        u << 1, 1;
    }

    void operator()(const Eigen::Vector2d& x, double& value)
    {
        value = 0.5 * x.dot(H * x) + h.dot(x);
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& grad, double& value)
    {
        this->operator()(x, value);
        grad << H * x + h;
    }

    void hessian(const Eigen::Vector2d& x,
                 Eigen::Matrix2d& hess,
                 Eigen::Vector2d& grad,
                 double& value)
    {
        gradient(x, grad, value);
        hess << H;
    }
};

TEST(TrustRegionTestCase, TestRosenbrock) {
    SimpleQP prob;
    // Rosenbrock prob;

    using Scalar = double;
    using var_t = Eigen::Vector2d;
    using gradient_t = Eigen::Vector2d;
    using hessian_t = Eigen::Matrix2d;

    var_t x0(0, 0);

    Scalar cost, trust_region;
    var_t x;
    hessian_t B;
    gradient_t g;
    Eigen::LLT<hessian_t> cholesky;

    const Scalar epsilon = 1e-4;
    const Scalar eta = 0; // 0 < eta < 10e-3
    const Scalar trust_region_max_radius = 1e3;
    x = x0;
    trust_region = 0.1;

    // Algorithm 6.2 Trust-Region Method
    int iter;
    for (iter = 0; iter < 100; iter++) {
        std::cout << "x " << x.transpose() << std::endl;

        prob.hessian(x, B, g, cost);
        var_t p, q, x_step;

        Scalar lambda = 0.1; // TODO: initial value?
        for (int i = 0; i < 3; i++) {
            // Algorithm 4.3 Trust Region Subproblem
            cholesky.compute(B + lambda * B.Identity());

            // TODO: what if B is indefinite?
            if (cholesky.info() != Eigen::Success) {
                lambda *= 2;
                continue;
            }

            p = cholesky.solve(-g);
            q = cholesky.matrixL().solve(p);
            // std::cout << "p " << p.transpose() << std::endl;
            // std::cout << "q " << p.transpose() << std::endl;
            lambda = lambda + p.dot(p) / q.dot(q) * (p.norm() - trust_region) / trust_region;

            // std::cout << "B\n" << B << std::endl;
            // std::cout << "B+\n" << (B + lambda * B.Identity()) << std::endl;

        }
        cholesky.compute(B + lambda * B.Identity());
        p = cholesky.solve(-g);

        std::cout << "g " << g.transpose() << std::endl;
        std::cout << "p " << p.transpose() << std::endl;

        x_step = x + p;

        Scalar ared, pred, rho;
        Scalar cost_step;
        gradient_t g_step;

        prob.gradient(x_step, g_step, cost_step);
        pred = -(g.dot(p) + 0.5 * p.dot(B * p));
        ared = cost - cost_step;
        rho = ared / pred;

        printf("rho %f\n", rho);

        if (rho > eta) {
            x = x + p;
        } else {
            printf("reject step\n");
            // reject step: x = x;
        }

        // trust region update
        if (rho < 0.1) {
            printf("shrink trust region, %f\n", trust_region);
            trust_region = 0.5 * trust_region;
        } else if (rho > 0.75) {
            if (p.norm() < 0.8 * trust_region) {
                // keep trust region
                printf("keep trust region\n");
            } else {
                trust_region = fmin(2.0 * trust_region, trust_region_max_radius);
                printf("increase trust region, %f\n", trust_region);
            }
        } else {
            // keep trust region
            printf("keep trust region\n");
        }

        // do SR1 or BFGS update
        // if (SR1 condition 6.26) {
        //     y = grad - grad_prev;
        //     SR1_update(B, p, y);
        // }

        if (g.lpNorm<Eigen::Infinity>() < epsilon) {
            break;
        }
    }
    std::cout << "x " << x.transpose() << std::endl;
    std::cout << "iter " << iter << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
