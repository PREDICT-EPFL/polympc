#ifndef OSQP_SOLVER_H
#define OSQP_SOLVER_H

#include <Eigen/Dense>
#include <cmath>

namespace osqp_solver {

/*
 *  minimize        0.5 x' P x + q' x
 *  subject to      l <= A x <= u
*/
class OSQPSolver {
public:
    enum {
        n = 2,
        m = 3
    };

    using Scalar = double;
    using Vn = Eigen::Matrix<Scalar, n, 1>;
    using Vm = Eigen::Matrix<Scalar, m, 1>;
    using Vnm = Eigen::Matrix<Scalar, n+m, 1>;
    using Mn = Eigen::Matrix<Scalar, n, n>;
    using Mmn = Eigen::Matrix<Scalar, m, n>;
    using KKT = Eigen::Matrix<Scalar, n+m, n+m>;

    // Problem parameters
    Mn P;
    Vn q;
    Mmn A;
    Vm l, u;

    // Solver settings
    Scalar sigma;
    Scalar rho;
    Scalar rho_inv;
    Scalar alpha;
    int max_iter;

    // Solver state variables
    int iter;
    Vn x;
    Vm z;
    Vm y;
    Vn x_tilde;
    Vm z_tilde;
    Vm z_prev;
    // TODO: Inplace matrix decompositions
    // LDLT<Ref<KKT>>, with matrix passed to constructor!
    // https://eigen.tuxfamily.org/dox/group__InplaceDecomposition.html
    KKT kkt_mat;
    Eigen::LDLT<KKT> kkt_ldlt;

    void setup(const Mn& _P, const Vn& _q, const Mmn& _A, const Vm& _l, const Vm& _u)
    {
        P = _P;
        q = _q;
        A = _A;
        l = _l;
        u = _u;

        // TODO: parameter
        rho = 1.0f;
        rho_inv = 1.0f/rho;
        sigma = 1e-6f;
        alpha = 1.0f;
        max_iter = 50;
    }

    void update()
    {
        // TOOD: update problem
    }

    void solve()
    {
        // TODO: warm-start
        x.setZero();
        z.setZero();
        y.setZero();

        form_KKT_mat(kkt_mat);
        kkt_ldlt.compute(kkt_mat);

        for (iter = 1; iter <= max_iter; iter++) {
            z_prev = z;

            // update x_tilde z_tilde
            Vnm rhs, x_tilde_nu;
            form_KKT_rhs(rhs);
            x_tilde_nu = kkt_ldlt.solve(rhs);

            x_tilde = x_tilde_nu.head<n>();
            z_tilde = z_prev + rho_inv*(x_tilde_nu.tail<m>() - y);

            // update x
            x = alpha * x_tilde + (1 - alpha) * x;

            // update z
            z = alpha * z_tilde + (1 - alpha) * z_prev + rho_inv * y;
            clip(z, l, u); // euclidean projection

            // update y
            y = y + rho * (alpha * z_tilde + (1 - alpha) * z_prev - z);

            if (termination_criteria()) {
                break;
            }
        }
        // print summary
    }

private:
    void form_KKT_mat(KKT& kkt)
    {
        kkt.topLeftCorner<n,n>() = P + sigma*P.Identity();
        kkt.topRightCorner<n,m>() = A.transpose();
        kkt.bottomLeftCorner<m,n>() = A;
        kkt.bottomRightCorner<m,m>().setIdentity();
        kkt.bottomRightCorner<m,m>() *= -rho_inv;
    }

    // void form_KKT_rhs(Vnm& rhs, const Vn& x, const Vm& z, const Vm& y)
    void form_KKT_rhs(Vnm& rhs)
    {
        rhs.head<n>() = sigma*x - q;
        rhs.tail<m>() = z - rho_inv*y;
    }

    void clip(Vm& z, const Vm& l, const Vm& u)
    {
        for (int i = 0; i < z.RowsAtCompileTime; i++) {
            z(i) = fmax(l(i), fmin(z(i), u(i)));
        }
    }

    bool termination_criteria()
    {
        // TODO
        return false;
    }
};

} // namespace osqp_solver

#endif // OSQP_SOLVER_H
