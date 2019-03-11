#ifndef OSQP_SOLVER_H
#define OSQP_SOLVER_H

#include <Eigen/Dense>
#include <cmath>

namespace osqp_solver {

template <int _n, int _m, typename _Scalar = double>
struct QP {
    using Scalar = _Scalar;
    enum {
        n=_n,
        m=_m
    };
    Eigen::Matrix<Scalar, n, n> P;
    Eigen::Matrix<Scalar, n, 1> q;
    Eigen::Matrix<Scalar, m, n> A;
    Eigen::Matrix<Scalar, m, 1> l, u;
};

/** Direct linear system solver
 *  solve A x = b
 */
template<typename AType>
class DirectLinSysSolver {
public:
    enum {
        n=AType::RowsAtCompileTime
    };
    using Scalar = typename AType::Scalar;
    using Vn = Eigen::Matrix<Scalar, n, 1>;
    Eigen::LDLT<AType> ldlt;

    void setup(AType &A)
    {
        // TODO: Inplace matrix decompositions
        // LDLT<Ref<KKT>>, with matrix passed to constructor!
        // https://eigen.tuxfamily.org/dox/group__InplaceDecomposition.html
        ldlt.compute(A);
    }

    Vn solve(Vn &rhs)
    {
        return ldlt.solve(rhs);
    }
};

template <typename Scalar>
struct OSQPSettings {
    Scalar sigma = 1e-6;
    Scalar rho = 1e-1;
    Scalar alpha = 1.0;
    Scalar eps_rel = 1e-3;
    Scalar eps_abs = 1e-3;
    int max_iter = 1000;
};

/**
 *  minimize        0.5 x' P x + q' x
 *  subject to      l <= A x <= u
 *
 *  with:
 *    x element of R^n
 *    Ax element of R^m
 */
template <typename _QPType>
class OSQPSolver {
public:
    enum {
        n=_QPType::n,
        m=_QPType::m
    };

    using QPType = _QPType;
    using Scalar = typename _QPType::Scalar;
    using Vn = Eigen::Matrix<Scalar, n, 1>;
    using Vm = Eigen::Matrix<Scalar, m, 1>;
    using Vnm = Eigen::Matrix<Scalar, n + m, 1>;
    using Mn = Eigen::Matrix<Scalar, n, n>;
    using Mmn = Eigen::Matrix<Scalar, m, n>;
    using KKT = Eigen::Matrix<Scalar, n + m, n + m>;
    using Settings = OSQPSettings<Scalar>;

    // Solver state variables
    int iter;
    Vn x;
    Vm z;
    Vm y;
    Vn x_tilde;
    Vm z_tilde;
    Vm z_prev;

    KKT kkt_mat;
    // TODO: choose between direct and indirect method
    DirectLinSysSolver<KKT> lin_sys_solver;

    void solve(const QPType &qp, const Settings &settings)
    {
        Vnm rhs, x_tilde_nu;

        // TODO: warm-start
        x.setZero();
        z.setZero();
        y.setZero();

        form_KKT_mat(qp, settings, kkt_mat);
        lin_sys_solver.setup(kkt_mat);

        for (iter = 1; iter <= settings.max_iter; iter++) {
            z_prev = z;

            // update x_tilde z_tilde
            form_KKT_rhs(qp, settings, rhs);
            x_tilde_nu = lin_sys_solver.solve(rhs);

            x_tilde = x_tilde_nu.template head<n>();
            z_tilde = z_prev + 1.0 / settings.rho * (x_tilde_nu.template tail<m>() - y);

            // update x
            x = settings.alpha * x_tilde + (1 - settings.alpha) * x;

            // update z
            z = settings.alpha * z_tilde + (1 - settings.alpha) * z_prev + 1.0 / settings.rho * y;
            clip(z, qp.l, qp.u); // euclidean projection

            // update y
            y = y + settings.rho * (settings.alpha * z_tilde + (1 - settings.alpha) * z_prev - z);

            if (termination_criteria()) {
                break;
            }
        }
        // print summary
    }

private:
    void form_KKT_mat(const QPType &qp, const Settings &settings, KKT& kkt)
    {
        kkt.template topLeftCorner<n, n>() = qp.P + settings.sigma * qp.P.Identity();
        kkt.template topRightCorner<n, m>() = qp.A.transpose();
        kkt.template bottomLeftCorner<m, n>() = qp.A;
        kkt.template bottomRightCorner<m, m>().setIdentity();
        kkt.template bottomRightCorner<m, m>() *= -1.0 / settings.rho;
    }

    void form_KKT_rhs(const QPType &qp, const Settings &settings, Vnm& rhs)
    {
        rhs.template head<n>() = settings.sigma * x - qp.q;
        rhs.template tail<m>() = z - 1.0 / settings.rho * y;
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
