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

template <typename Scalar>
struct OSQPSettings {
    Scalar rho = 1e-1;          /**< ADMM rho step, 0 < rho */
    Scalar sigma = 1e-6;        /**< ADMM sigma step, 0 < sigma, (small) */
    Scalar alpha = 1.0;         /**< ADMM overrelaxation parameter, 0 < alpha < 2,
                                     values in [1.5, 1.8] give good results (empirically) */
    Scalar eps_rel = 1e-3;      /**< Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs = 1e-3;      /**< Absolute tolerance for termination, 0 < eps_abs */
    int max_iter = 1000;        /**< Maximal number of iteration, 0 < max_iter */
    int check_termination = 25; /**< Check termination after every Nth iteration, 0 (disabled) or 0 < check_termination */
    bool warm_start = false;    /**< Warm start solver, reuses previous x,z,y */
    bool adaptive_rho = false;  /**< Adapt rho to optimal estimate */
    Scalar adaptive_rho_tolerance = 5;  /**< Minimal for rho update factor, 1 < adaptive_rho_tolerance */
    int adaptive_rho_interval = 25; /**< change rho every Nth iteration, 0 < adaptive_rho_interval,
                                         set equal to check_termination to save computation  */
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

    using qp_t = _QPType;
    using Scalar = typename _QPType::Scalar;
    using primal_t = Eigen::Matrix<Scalar, n, 1>;
    using constraint_t = Eigen::Matrix<Scalar, m, 1>;
    using dual_t = Eigen::Matrix<Scalar, m, 1>;
    using kkt_vec_t = Eigen::Matrix<Scalar, n + m, 1>;
    using kkt_mat_t = Eigen::Matrix<Scalar, n + m, n + m>;
    using Settings = OSQPSettings<Scalar>;

    static constexpr Scalar RHO_MIN = 1e-6;
    static constexpr Scalar RHO_MAX = 1e+6;
    static constexpr Scalar RHO_TOL = 1e-4;
    static constexpr Scalar RHO_EQ_FACTOR = 1e+3;
    static constexpr Scalar LOOSE_BOUNDS_THRESH = 1e+16;
    static constexpr Scalar DIV_BY_ZERO_REGUL = 1e-10;

    // Solver state variables
    int iter;
    primal_t x;
    constraint_t z;
    dual_t y;
    primal_t x_tilde;
    constraint_t z_tilde;
    constraint_t z_prev;
    dual_t rho_vec;
    dual_t rho_inv_vec;
    Scalar rho;

    // State
    Scalar res_prim;
    Scalar res_dual;
    Scalar _max_Ax_z_norm;
    Scalar _max_Px_ATy_q_norm;

    enum {
        INEQUALITY_CONSTRAINT,
        EQUALITY_CONSTRAINT,
        LOOSE_BOUNDS
    } constr_type[m]; /**< constraint type classification */

    Settings settings;

    kkt_mat_t kkt_mat;

    // TODO: choose between direct and indirect method
    // TODO: Inplace matrix decompositions
    // LDLT<Ref<KKT>>, with matrix passed to constructor!
    // https://eigen.tuxfamily.org/dox/group__InplaceDecomposition.html
    Eigen::LDLT<kkt_mat_t> lin_sys_solver;

    OSQPSolver()
    {
        x.setZero();
        z.setZero();
        y.setZero();
    }

    void solve(const qp_t &qp)
    {
        kkt_vec_t rhs, x_tilde_nu;
        bool check_termination = false;

#ifdef OSQP_PRINTING
        print_settings(settings);
#endif
        if (!settings.warm_start) {
            x.setZero();
            z.setZero();
            y.setZero();
        }

        constr_type_init(qp);
        rho_update(settings.rho);

        KKT_mat_update(qp, kkt_mat);
        lin_sys_solver.compute(kkt_mat);

        for (iter = 1; iter <= settings.max_iter; iter++) {
            z_prev = z;

            // update x_tilde z_tilde
            form_KKT_rhs(qp, rhs);
            x_tilde_nu = lin_sys_solver.solve(rhs);

            x_tilde = x_tilde_nu.template head<n>();
            z_tilde = z_prev + rho_inv_vec.cwiseProduct(x_tilde_nu.template tail<m>() - y);

            // update x
            x = settings.alpha * x_tilde + (1 - settings.alpha) * x;

            // update z
            z = settings.alpha * z_tilde + (1 - settings.alpha) * z_prev + rho_inv_vec.cwiseProduct(y);
            clip_z(z, qp.l, qp.u); // euclidean projection

            // update y
            y = y + rho_vec.cwiseProduct(settings.alpha * z_tilde + (1 - settings.alpha) * z_prev - z);

            if (settings.check_termination != 0 && iter % settings.check_termination == 0) {
                check_termination = true;
            } else {
                check_termination = false;
            }

            if (check_termination) {
                update_state(qp);

#ifdef OSQP_PRINTING
                print_status(qp);
#endif
                if (termination_criteria(qp)) {
                    break;
                }
            }

            if (settings.adaptive_rho && iter % settings.adaptive_rho_interval == 0) {
                if (!check_termination) {
                    // state was not yet updated
                    update_state(qp);
                }
                Scalar new_rho = rho_estimate(rho, qp);
                new_rho = fmax(RHO_MIN, fmin(new_rho, RHO_MAX));

                if (new_rho < rho / settings.adaptive_rho_tolerance ||
                    new_rho > rho * settings.adaptive_rho_tolerance) {
                    rho_update(new_rho);
                    KKT_mat_update(qp, kkt_mat);
                    lin_sys_solver.compute(kkt_mat);
                }
            }
        }

        // TODO: return summary
    }

private:
    void KKT_mat_update(const qp_t &qp, kkt_mat_t& kkt)
    {
        kkt.template topLeftCorner<n, n>() = qp.P + settings.sigma * qp.P.Identity();
        kkt.template topRightCorner<n, m>() = qp.A.transpose();
        kkt.template bottomLeftCorner<m, n>() = qp.A;
        kkt.template bottomRightCorner<m, m>() = -1.0 * rho_inv_vec.asDiagonal();
    }

    void form_KKT_rhs(const qp_t &qp, kkt_vec_t& rhs)
    {
        rhs.template head<n>() = settings.sigma * x - qp.q;
        rhs.template tail<m>() = z - rho_inv_vec.cwiseProduct(y);
    }

    void clip_z(constraint_t& z, const constraint_t& l, const constraint_t& u)
    {
        for (int i = 0; i < z.RowsAtCompileTime; i++) {
            z(i) = fmax(l(i), fmin(z(i), u(i)));
        }
    }

    void constr_type_init(const qp_t &qp)
    {
        for (int i = 0; i < qp.l.RowsAtCompileTime; i++) {
            if (qp.l[i] < -LOOSE_BOUNDS_THRESH && qp.u[i] > LOOSE_BOUNDS_THRESH) {
                constr_type[i] = LOOSE_BOUNDS;
            } else if (qp.u[i] - qp.l[i] < RHO_TOL) {
                constr_type[i] = EQUALITY_CONSTRAINT;
            } else {
                constr_type[i] = INEQUALITY_CONSTRAINT;
            }
        }
    }

    void rho_update(Scalar rho0)
    {
        for (int i = 0; i < rho_vec.RowsAtCompileTime; i++) {
            switch (constr_type[i]) {
            case LOOSE_BOUNDS:
                rho_vec[i] = RHO_MIN;
                break;
            case EQUALITY_CONSTRAINT:
                rho_vec[i] = RHO_EQ_FACTOR*rho0;
                break;
            case INEQUALITY_CONSTRAINT: /* fall through */
            default:
                rho_vec[i] = rho0;
            };
        }
        rho_inv_vec = rho_vec.cwiseInverse();
        rho = rho0;
    }

    void update_state(const qp_t& qp)
    {
        Scalar norm_Ax, norm_z;
        norm_Ax = (qp.A*x).template lpNorm<Eigen::Infinity>();
        norm_z = z.template lpNorm<Eigen::Infinity>();
        _max_Ax_z_norm = fmax(norm_Ax, norm_z);

        Scalar norm_Px, norm_ATy, norm_q;
        norm_Px = (qp.P*x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
        norm_q = qp.q.template lpNorm<Eigen::Infinity>();
        _max_Px_ATy_q_norm = fmax(norm_Px, fmax(norm_ATy, norm_q));

        res_prim = residual_prim(qp);
        res_dual = residual_dual(qp);
    }

    Scalar rho_estimate(const Scalar rho0, const qp_t &qp) const
    {
        Scalar rp_norm, rd_norm;
        rp_norm = res_prim / (_max_Ax_z_norm + DIV_BY_ZERO_REGUL);
        rd_norm = res_dual / (_max_Px_ATy_q_norm + DIV_BY_ZERO_REGUL);

        Scalar rho_new = rho0 * sqrt(rp_norm/(rd_norm + DIV_BY_ZERO_REGUL));
        return rho_new;
    }

    Scalar eps_prim(const qp_t &qp) const
    {
        Scalar norm_Ax, norm_z;
        norm_Ax = (qp.A*x).template lpNorm<Eigen::Infinity>();
        norm_z = z.template lpNorm<Eigen::Infinity>();
        return settings.eps_abs + settings.eps_rel * fmax(norm_Ax, norm_z);
    }

    Scalar eps_dual(const qp_t &qp) const
    {
        Scalar norm_Px, norm_ATy, norm_q;
        norm_Px = (qp.P*x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
        norm_q = qp.q.template lpNorm<Eigen::Infinity>();
        return settings.eps_abs + settings.eps_rel * fmax(norm_Px, fmax(norm_ATy, norm_q));
    }

    Scalar residual_prim(const qp_t &qp) const
    {
        return (qp.A*x - z).template lpNorm<Eigen::Infinity>();
    }

    Scalar residual_dual(const qp_t &qp) const
    {
        return (qp.P*x + qp.q + qp.A.transpose()*y).template lpNorm<Eigen::Infinity>();
    }

    bool termination_criteria(const qp_t &qp)
    {
        // check residual norms to detect optimality
        if (res_prim <= eps_prim(qp) && res_dual <= eps_dual(qp)) {
            return true;
        }

        return false;
    }

#ifdef OSQP_PRINTING
    void print_status(const qp_t &qp) const
    {
        Scalar obj = 0.5 * x.dot(qp.P*x) + qp.q.dot(x);

        if (iter == 1) {
            printf("iter   obj       rp        rd\n");
        }
        printf("%4d  %.2e  %.2e  %.2e\n", iter, obj, res_prim, res_dual);
    }

    void print_settings(const Settings &settings) const
    {
        printf("ADMM settings:\n");
        printf("  sigma %.2e\n", settings.sigma);
        printf("  rho %.2e\n", settings.rho);
        printf("  alpha %.2f\n", settings.alpha);
        printf("  eps_rel %.1e\n", settings.eps_rel);
        printf("  eps_abs %.1e\n", settings.eps_abs);
        printf("  max_iter %d\n", settings.max_iter);
        printf("  adaptive_rho %d\n", settings.adaptive_rho);
        printf("  warm_start %d\n", settings.warm_start);
    }
#endif
};

} // namespace osqp_solver

#endif // OSQP_SOLVER_H
