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
    // TODO: add check_termination param for minimal number of iterations before termination criteria is checked
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
    Vm rho;
    Vm rho_inv;

    enum {
        INEQUALITY_CONSTRAINT,
        EQUALITY_CONSTRAINT,
        LOOSE_BOUNDS
    } constr_type[m]; /**< constraint type classification */

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

        constr_type_init(qp);
        rho_init(settings);

        form_KKT_mat(qp, settings, kkt_mat);
        lin_sys_solver.setup(kkt_mat);

        for (iter = 1; iter <= settings.max_iter; iter++) {
            z_prev = z;

            // update x_tilde z_tilde
            form_KKT_rhs(qp, settings, rhs);
            x_tilde_nu = lin_sys_solver.solve(rhs);

            x_tilde = x_tilde_nu.template head<n>();
            z_tilde = z_prev + rho_inv.cwiseProduct(x_tilde_nu.template tail<m>() - y);

            // update x
            x = settings.alpha * x_tilde + (1 - settings.alpha) * x;

            // update z
            z = settings.alpha * z_tilde + (1 - settings.alpha) * z_prev + rho_inv.cwiseProduct(y);
            clip(z, qp.l, qp.u); // euclidean projection

            // update y
            y = y + rho.cwiseProduct(settings.alpha * z_tilde + (1 - settings.alpha) * z_prev - z);

            if (termination_criteria(qp, settings)) {
                break;
            }

            // TODO: adaptive rho
        }

        // TODO: return summary
    }

private:
    void form_KKT_mat(const QPType &qp, const Settings &settings, KKT& kkt)
    {
        kkt.template topLeftCorner<n, n>() = qp.P + settings.sigma * qp.P.Identity();
        kkt.template topRightCorner<n, m>() = qp.A.transpose();
        kkt.template bottomLeftCorner<m, n>() = qp.A;
        kkt.template bottomRightCorner<m, m>() = -1.0 * rho_inv.asDiagonal();
    }

    void form_KKT_rhs(const QPType &qp, const Settings &settings, Vnm& rhs)
    {
        rhs.template head<n>() = settings.sigma * x - qp.q;
        rhs.template tail<m>() = z - rho_inv.cwiseProduct(y);
    }

    void clip(Vm& z, const Vm& l, const Vm& u)
    {
        for (int i = 0; i < z.RowsAtCompileTime; i++) {
            z(i) = fmax(l(i), fmin(z(i), u(i)));
        }
    }

    void constr_type_init(const QPType &qp)
    {
        for (int i = 0; i < qp.l.RowsAtCompileTime; i++) {
            if (qp.l[i] < -1e-16 && qp.u[i] > 1e-16) {
                constr_type[i] = LOOSE_BOUNDS;
            } else if (qp.u[i] - qp.l[i] < 1e-4) {
                constr_type[i] = EQUALITY_CONSTRAINT;
            } else {
                constr_type[i] = INEQUALITY_CONSTRAINT;
            }
        }
    }

    void rho_init(const Settings &settings)
    {
        for (int i = 0; i < rho.RowsAtCompileTime; i++) {
            if (constr_type[i] == LOOSE_BOUNDS) {
                rho[i] = 1e-6; // TODO: constant RHO_MIN
            } else if (constr_type[i] == EQUALITY_CONSTRAINT) {
                rho[i] = 1e3*settings.rho; // TODO: constant
            } else { // INEQUALITY_CONSTRAINT
                rho[i] = settings.rho;
            }
        }
        rho_inv = rho.cwiseInverse();
    }

    Scalar eps_prim(const QPType &qp, const Settings &settings) const
    {
        Scalar norm_Ax, norm_z;
        norm_Ax = (qp.A*x).template lpNorm<Eigen::Infinity>();
        norm_z = z.template lpNorm<Eigen::Infinity>();
        return settings.eps_abs + settings.eps_rel * fmax(norm_Ax, norm_z);
    }

    Scalar eps_dual(const QPType &qp, const Settings &settings) const
    {
        Scalar norm_Px, norm_ATy, norm_q;
        norm_Px = (qp.P*x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
        norm_q = qp.q.template lpNorm<Eigen::Infinity>();
        return settings.eps_abs + settings.eps_rel * fmax(norm_Px, fmax(norm_ATy, norm_q));
    }

    Vm residual_prim(const QPType &qp) const
    {
        return qp.A*x - z;
    }

    Vn residual_dual(const QPType &qp) const
    {
        return qp.P*x + qp.q + qp.A.transpose()*y;
    }

    bool termination_criteria(const QPType &qp, const Settings &settings)
    {
        Scalar rp_norm, rd_norm;
        rp_norm = residual_prim(qp).template lpNorm<Eigen::Infinity>();
        rd_norm = residual_dual(qp).template lpNorm<Eigen::Infinity>();

        // check residual norms to detect optimality
        if (rp_norm <= eps_prim(qp, settings) && rd_norm <= eps_dual(qp, settings)) {
            return true;
        }

        return false;
    }
};

} // namespace osqp_solver

#endif // OSQP_SOLVER_H
