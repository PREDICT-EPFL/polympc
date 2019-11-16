#define QP_SOLVER_PRINTING
#include "gtest/gtest.h"
#include "solvers/qp_solver.hpp"
#include "load_matrix_from_csv.hpp"
#include <Eigen/IterativeLinearSolvers>

using namespace qp_solver; 

template <typename _Scalar=double>
class _QP0018 : public QP<50, 51, _Scalar>
{
public:
	_QP0018()
	{
        Eigen::MatrixXd P = load_csv<Eigen::MatrixXd>("./qplib/QP0018/P.csv");
        Eigen::MatrixXd q = load_csv<Eigen::MatrixXd>("./qplib/QP0018/q.csv");
        Eigen::MatrixXd A = load_csv<Eigen::MatrixXd>("./qplib/QP0018/A.csv");
        Eigen::MatrixXd l = load_csv<Eigen::MatrixXd>("./qplib/QP0018/l.csv");
        Eigen::MatrixXd u = load_csv<Eigen::MatrixXd>("./qplib/QP0018/u.csv");

        this->P = P;
        this->q = q;
        this->A = A;
        this->l = l;
        this->u = u;
	}
};

using QP0018 = _QP0018<double>;

TEST(QPProblemSets, testQPQP0018default) {
	QP0018 qp;
	QPSolver<QP0018> prob;

    	prob.settings().max_iter = 1000;
	prob.settings().verbose = true;
	prob.settings().alpha = 1.6;
    	prob.settings().adaptive_rho = true;
	prob.settings().check_termination = 25;

	prob.setup(qp);
	prob.solve(qp);
	Eigen::VectorXd sol = prob.primal_solution();

	EXPECT_LT(prob.iter, prob.settings().max_iter);
	EXPECT_EQ(prob.info().status, SOLVED);
	// check feasibility (with some epsilon margin)
	Eigen::VectorXd lower = qp.A*sol - qp.l;
	Eigen::VectorXd upper = qp.A*sol - qp.u;
	EXPECT_GE(lower.minCoeff(), -1e-3);
	EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(QPProblemSets, testQPQP0018adaptive) {
	QP0018 qp;
	QPSolver<QP0018> prob;

	prob.settings().max_iter = 4000;
	prob.settings().verbose = true;
	prob.settings().alpha = 1.6;
	prob.settings().adaptive_rho = true;
	prob.settings().adaptive_rho_interval = 25;
	prob.settings().check_termination = 25;

	prob.setup(qp);
	prob.solve(qp);
	Eigen::VectorXd sol = prob.primal_solution();

	EXPECT_LT(prob.iter, prob.settings().max_iter);
	EXPECT_EQ(prob.info().status, SOLVED);
	// check feasibility (with some epsilon margin)
	Eigen::VectorXd lower = qp.A*sol - qp.l;
	Eigen::VectorXd upper = qp.A*sol - qp.u;
	EXPECT_GE(lower.minCoeff(), -1e-3);
	EXPECT_LE(upper.maxCoeff(), 1e-3);
} 
