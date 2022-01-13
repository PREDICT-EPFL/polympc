#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "control/mpc_wrapper.hpp"
#include "polynomials/splines.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#include "control/simple_robot_model.hpp"
#include "solvers/box_admm.hpp"

#include "unsupported/Eigen/SparseExtra"

#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ ParkingOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 1, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class ParkingOCP : public ContinuousOCP<ParkingOCP, Approximation, DENSE>
{
public:
    ~ParkingOCP() = default;

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = p(0) * u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = p(0) * u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = p(0) * u(0) * sin(u(1)) / d(0);

        polympc::ignore_unused_var(t);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        mayer = p(0);

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
    }
};

POLYMPC_FORWARD_DECLARATION(/*Name*/ ParkingOCPSparse, /*NX*/ 3, /*NU*/ 2, /*NP*/ 1, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)
class ParkingOCPSparse : public ContinuousOCP<ParkingOCPSparse, Approximation, SPARSE>
{
public:
    ~ParkingOCPSparse() = default;

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = p(0) * u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = p(0) * u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = p(0) * u(0) * sin(u(1)) / d(0);

        polympc::ignore_unused_var(t);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        mayer = p(0);

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(u);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
    }
};

/** create solver */
template<typename Problem, typename QPSolver> class Solver;
template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ,
                                             typename Problem::scalar_t, Problem::MATRIXFMT, linear_solver_traits<ParkingOCP::MATRIXFMT>::default_solver>>
class Solver : public SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;


    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }

    EIGEN_STRONG_INLINE void hessian_regularisation_dense_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept
    {
        const int n = this->m_H.rows();
        /**Regularize by the estimation of the minimum negative eigen value--does not work with inexact Hessian update(matrix is already PSD)*/
        scalar_t aii, ri;
        for (int i = 0; i < n; i++)
        {
            aii = lag_hessian(i,i);
            ri  = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

            if (aii - ri <= 0) {lag_hessian(i,i) += (ri - aii) + scalar_t(0.01);} //All Greshgorin discs are in the positive half

        }
    }

    EIGEN_STRONG_INLINE void hessian_regularisation_sparse_impl(nlp_hessian_t& lag_hessian) noexcept
    {
        const int n = this->m_H.rows(); //132=m_H.toDense().rows()
        /**Regularize by the estimation of the minimum negative eigen value*/
        scalar_t aii, ri;
        for (int i = 0; i < n; i++)
        {
            aii = lag_hessian.coeffRef(i, i);
            ri = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

            if (aii - ri <= 0)
                lag_hessian.coeffRef(i, i) += (ri - aii) + 0.001;//All Gershgorin discs are in the positive half
        }
    }
};

using dense_admm_t = boxADMM<ParkingOCP::VAR_SIZE, ParkingOCP::NUM_EQ + ParkingOCP::NUM_INEQ, typename ParkingOCP::scalar_t,
                     ParkingOCP::MATRIXFMT, linear_solver_traits<ParkingOCP::MATRIXFMT>::default_solver>;

using dense_solver_t = dense_admm_t::linear_solver_t;

using sparse_admm_t = boxADMM<ParkingOCPSparse::VAR_SIZE, ParkingOCPSparse::NUM_EQ + ParkingOCPSparse::NUM_INEQ, typename ParkingOCPSparse::scalar_t,
                      ParkingOCPSparse::MATRIXFMT, linear_solver_traits<ParkingOCPSparse::MATRIXFMT>::default_solver>;

using sparse_solver_t = sparse_admm_t::linear_solver_t;


int main(void)
{
    using namespace polympc;

    ParkingOCP robot_nlp;
    ParkingOCP::nlp_variable_t var = ParkingOCP::nlp_variable_t::Zero();
    ParkingOCP::nlp_eq_constraints_t eq_constr;
    ParkingOCP::nlp_ineq_constraints_t ineq_constr;
    ParkingOCP::nlp_constraints_t constr;
    ParkingOCP::nlp_eq_jacobian_t eq_jac(static_cast<int>(ParkingOCP::VARX_SIZE), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCP::nlp_ineq_jacobian_t ineq_jac(static_cast<int>(ParkingOCP::NUM_INEQ), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCP::nlp_jacobian_t jac(static_cast<int>(ParkingOCP::NUM_INEQ + ParkingOCP::NUM_EQ), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCP::nlp_cost_t cost = 0;
    ParkingOCP::nlp_cost_t lagrangian = 0;                     /** suppres warn */  //polympc::ignore_unused_var(lagrangian);
    ParkingOCP::nlp_dual_t lam = ParkingOCP::nlp_dual_t::Zero(); /** suppres warn */  // polympc::ignore_unused_var(lam);
    ParkingOCP::static_parameter_t p; p(0) = 2.0;
    ParkingOCP::nlp_variable_t cost_gradient, lag_gradient;
    ParkingOCP::nlp_hessian_t cost_hessian(static_cast<int>(ParkingOCP::VAR_SIZE),  static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCP::nlp_hessian_t lag_hessian(static_cast<int>(ParkingOCP::VAR_SIZE), static_cast<int>(ParkingOCP::VAR_SIZE));

    var << 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5,
           0.5, 0.5, 1.5, 0.5, 0.5, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0.5;

    robot_nlp.lagrangian_gradient_hessian(var, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, constr, eq_jac);

    // same for sparse problem
    ParkingOCPSparse sparse_nlp;
    ParkingOCPSparse::nlp_variable_t svar; svar = var;
    ParkingOCPSparse::nlp_eq_constraints_t seq_constr;
    ParkingOCPSparse::nlp_ineq_constraints_t sineq_constr;
    ParkingOCPSparse::nlp_constraints_t sconstr;
    ParkingOCPSparse::nlp_eq_jacobian_t seq_jac(static_cast<int>(ParkingOCP::VARX_SIZE), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCPSparse::nlp_ineq_jacobian_t sineq_jac(static_cast<int>(ParkingOCP::NUM_INEQ), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCPSparse::nlp_jacobian_t sjac(static_cast<int>(ParkingOCP::NUM_INEQ + ParkingOCP::NUM_EQ), static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCPSparse::nlp_cost_t scost = 0;
    ParkingOCPSparse::nlp_cost_t slagrangian = 0;                     /** suppres warn */  //polympc::ignore_unused_var(lagrangian);
    ParkingOCPSparse::nlp_dual_t slam = lam;                          /** suppres warn */   //polympc::ignore_unused_var(lam);
    ParkingOCPSparse::static_parameter_t sp; sp(0) = 2.0;
    ParkingOCPSparse::nlp_variable_t scost_gradient, slag_gradient;
    ParkingOCPSparse::nlp_hessian_t scost_hessian(static_cast<int>(ParkingOCP::VAR_SIZE),  static_cast<int>(ParkingOCP::VAR_SIZE));
    ParkingOCPSparse::nlp_hessian_t slag_hessian(static_cast<int>(ParkingOCP::VAR_SIZE), static_cast<int>(ParkingOCP::VAR_SIZE));
    sparse_nlp.lagrangian_gradient_hessian(svar, sp, slam, slagrangian, slag_gradient, slag_hessian, scost_gradient, sconstr, seq_jac);

    // compare Hessians
    Eigen::MatrixXd H = slag_hessian.toDense();
    Eigen::MatrixXd dH = lag_hessian - H;
    std::cout << "dH: \n" << dH << "\n";

    //compare Jacobians
    Eigen::MatrixXd J = seq_jac.toDense();
    Eigen::MatrixXd dJ = eq_jac - J;
    std::cout << "dJ: \n" << dJ<< "\n";

    //compare cost Gradients
    Eigen::MatrixXd dG = scost_gradient - cost_gradient;
    std::cout << "dG: \n" << dG.transpose() << "\n";

    //compare constraints
    Eigen::MatrixXd dC = constr - sconstr;
    std::cout << "dC: \n" << dC.transpose() << "\n";

    //compare KKT systems
    dense_admm_t  dense_qp;
    sparse_admm_t sparse_qp;

    dense_qp.settings().warm_start = false;
    dense_qp.settings().check_termination = 10;
    dense_qp.settings().eps_abs = 1e-4;
    dense_qp.settings().eps_rel = 1e-4;
    dense_qp.settings().max_iter = 100;
    dense_qp.settings().adaptive_rho = true;
    dense_qp.settings().adaptive_rho_interval = 50;
    dense_qp.settings().alpha = 1.0;

    sparse_qp.settings().warm_start = false;
    sparse_qp.settings().check_termination = 10;
    sparse_qp.settings().eps_abs = 1e-4;
    sparse_qp.settings().eps_rel = 1e-4;
    sparse_qp.settings().max_iter = 100;
    sparse_qp.settings().adaptive_rho = true;
    sparse_qp.settings().adaptive_rho_interval = 50;
    sparse_qp.settings().alpha = 1.0;

    // regularise hessian
    lag_hessian.diagonal()  += dense_admm_t::qp_var_t::Constant(0.01);
    slag_hessian.diagonal() += dense_admm_t::qp_var_t::Constant(0.01);

    dense_qp.construct_kkt_matrix(lag_hessian, eq_jac);
    sparse_qp.construct_kkt_matrix(slag_hessian, seq_jac);

    Eigen::MatrixXd sK = sparse_qp.m_K.toDense();
    std::cout << "dH: \n" << sK - dense_qp.m_K << "\n";

    dense_solver_t  dense_ldlt;
    sparse_solver_t sparse_ldlt;

    dense_ldlt.compute(dense_qp.m_K);
    sparse_ldlt.analyzePattern(sparse_qp.m_K);
    sparse_ldlt.factorize(sparse_qp.m_K);

    Eigen::MatrixXd dL = dense_ldlt.matrixL();
    Eigen::MatrixXd sL = sparse_ldlt.matrixL().toDense();
    std::cout << "diffL: \n" << (dL - sL).norm() << "\n";
    std::cout << "diffD: \n" << (dense_ldlt.vectorD() - sparse_ldlt.vectorD()).transpose() << "\n";

    std::cout << "pdm: " << dense_ldlt.isPositive() << "\n";
    Eigen::EigenSolver<dense_admm_t::kkt_mat_t> eig_solver;
    eig_solver.compute(dense_qp.m_K);
    std::cout << "K eigen values: " << eig_solver.eigenvalues().real().transpose() << "\n";

    // solve test
    dense_admm_t::kkt_vec_t rhs, xd, xs;
    dense_qp.compute_kkt_rhs(cost_gradient, rhs);

    xd = dense_ldlt.solve(rhs);
    xs = sparse_ldlt.solve(rhs);

    std::cout << "X diff: \n" << (xd - xs).transpose() << "\n";

    std::cout << "x: " << xs.transpose() << "\n";
    std::cout << "rhs: " << rhs.transpose() << "\n";


    return EXIT_SUCCESS;
}
