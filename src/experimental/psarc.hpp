#ifndef PSARC_HPP
#define PSARC_HPP

#include "casadi/casadi.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Eigenvalues"
#include "kite.h"

namespace psarc_math
{

enum MAT : int {CASADI, EIGEN_DENSE, EIGEN_SPARSE};

template<typename Scalar>
Eigen::SparseMatrix<Scalar> C2ESparse(const casadi::DM &matrix)
{
    casadi::Sparsity SpA = matrix.get_sparsity();
    std::vector<casadi_int> output_row, output_col;
    SpA.get_triplet(output_row, output_col);
    std::vector<double> values = matrix.get_nonzeros();

    using T = Eigen::Triplet<double>;
    std::vector<T> TripletList;
    TripletList.resize(values.size());
    for(int k = 0; k < values.size(); ++k)
        TripletList[k] = T(output_row[k], output_col[k], values[k]);

    Eigen::SparseMatrix<double> SpMatrx(matrix.size1(), matrix.size2());
    SpMatrx.setFromTriplets(TripletList.begin(), TripletList.end());

    return SpMatrx;
}

static casadi::DM solve(const casadi::DM &A, const casadi::DM &b, MAT mat_type = MAT::EIGEN_DENSE)
{
    casadi::DM x;
    /** if A dimansion is small use native Casadi solver */
    switch(mat_type){
    case CASADI : {
        x = casadi::DM::solve(A, b);
        return x;
    }
    case EIGEN_DENSE : {
        Eigen::MatrixXd _A;
        Eigen::VectorXd _b;
        _A = Eigen::MatrixXd::Map(casadi::DM::densify(A).nonzeros().data(), A.size1(), A.size2());
        _b = Eigen::VectorXd::Map(casadi::DM::densify(b).nonzeros().data(), b.size1());

        /** solve the linear system and cast back to Casadi types */
        Eigen::VectorXd x = _A.partialPivLu().solve(_b);
        std::vector<double> sold;
        sold.resize(static_cast<size_t>(x.size()));
        Eigen::Map<Eigen::VectorXd>(&sold[0], x.size()) = x;
        return casadi::DM(sold);
    }
    case EIGEN_SPARSE : {
        Eigen::SparseMatrix<double> _A = psarc_math::C2ESparse<double>(A);
        Eigen::VectorXd _b = Eigen::VectorXd::Map(casadi::DM::densify(b).nonzeros().data(), b.size1());

        /** try direct solvers */
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(_A);
        if(solver.info() != Eigen::Success)
        {
            // decomposition failed
            std::cout << "LU decomposition of the Matrix failed: " << solver.lastErrorMessage() << "\n";
            //return casadi::DM();
        }
        Eigen::VectorXd x =solver.solve(_b);
        if(solver.info() != Eigen::Success)
        {
            // solving failed
            std::cout << "Solving failed! \n";
            return casadi::DM();
        }

        std::cout << "DET: " << solver.logAbsDeterminant() << " NORM: " << _b.norm() << "\n";

        /** cast back and return the value */
        std::vector<double> sold;
        sold.resize(static_cast<size_t>(x.size()));
        Eigen::Map<Eigen::VectorXd>(&sold[0], x.size()) = x;
        return casadi::DM(sold);
    }
    }

}

double determinant(const casadi::DM &A, MAT mat_type = MAT::EIGEN_DENSE)
{
    switch(mat_type){
    case CASADI : {
        return casadi::DM::det(A).nonzeros()[0];
    }
    case EIGEN_DENSE : {
        Eigen::MatrixXd _A;
        _A = Eigen::MatrixXd::Map(casadi::DM::densify(A).nonzeros().data(), A.size1(), A.size2());

        std::cout << "MATRIX WAS DENSIFIED \n";
        Eigen::BDCSVD<Eigen::MatrixXd> svd;
        svd.compute(_A);
        std::cout << "Singular values: \n" << svd.singularValues() << "\n";
        //Eigen::PartialPivLU<Eigen::MatrixXd>lu;
        //lu.compute(_A);
        return 0.0;
    }
    case EIGEN_SPARSE : {
        Eigen::SparseMatrix<double> _A = psarc_math::C2ESparse<double>(A);

        /** try direct solvers */
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(_A);
        if(solver.info() != Eigen::Success)
        {
            // decomposition failed
            std::cout << "LU decomposition of the Matrix failed: " << solver.lastErrorMessage() << "\n";
            //return casadi::DM();
        }

        return solver.logAbsDeterminant();
    }
    }
}


// end of namespace
}


template<typename FX, typename CorrectorProps>
typename FX::x psarc(const typename FX::x &init_guess, const CorrectorProps &props = CorrectorProps())
{
    FX system;

    typename FX::x res = system.eval(init_guess);
    std::cout << res << "\n";

    res = system.jac(init_guess);
    std::cout << res << "\n";
    return res;
}

template<typename Equalities, typename CorrectorProps>
class symbolic_psarc
{
public:
    symbolic_psarc(const typename Equalities::num &init_guess, const CorrectorProps &props = CorrectorProps());
    ~symbolic_psarc(){}

    casadi::DMDict operator()();
};

template<typename Equalities, typename CorrectorProps>
symbolic_psarc<Equalities, CorrectorProps>::symbolic_psarc(const typename Equalities::num &init_guess, const CorrectorProps &props)
{
    /** create an instance of the system */
    Equalities FX;

    /** generate convex homotopy equation */
    typename Equalities::sym x = FX.var;
    typename Equalities::sym lambda = Equalities::sym::sym("lambda");
    typename Equalities::sym x0 = {init_guess};
    typename Equalities::sym homotopy = (lambda) * (x - x0) + (1 - lambda) * FX();

    /** create symbolic expressions to evaluate jacoabians */
    typename Equalities::sym jac_x   = Equalities::sym::jacobian(homotopy, x);
    typename Equalities::sym jac_lam = Equalities::sym::jacobian(homotopy, lambda);

    /** create a corrector */
    x = Equalities::sym::vertcat({x, lambda});
    /** full jacobian */
    typename Equalities::sym jac_full = Equalities::sym::jacobian(homotopy, x);
    typename Equalities::sym w = Equalities::sym::sym("w", x.size1(), 1);
    /** Homotopy derivative wrt x */
    casadi::Function Hx = casadi::Function("jac_x", {x}, {jac_x});
    casadi::Function Hl = casadi::Function("jac_lam", {x}, {jac_lam});
    casadi::Function H = casadi::Function("jac_full", {x}, {jac_full});
    casadi::Function homo_eval = casadi::Function("homo", {x}, {homotopy});

    casadi::SXDict NLP;
    casadi::Dict   OPTS;
    casadi::DMDict ARG;
    NLP["x"] = x;
    NLP["f"] = 0.5 * Equalities::sym::dot((x - w), (x - w));
    NLP["g"] = homotopy;
    NLP["p"] = w;

    OPTS["ipopt.linear_solver"]  = "ma97";
    OPTS["ipopt.print_level"]    = 1;
    OPTS["ipopt.tol"]            = 1e-6;
    OPTS["ipopt.acceptable_tol"] = 1e-6;
    OPTS["ipopt.warm_start_init_point"] = "yes";

    casadi::Function Corrector = casadi::nlpsol("solver", "ipopt", NLP, OPTS);

    casadi::DM LBX = -casadi::DM::inf(13);
    LBX(0) = 1.0;

    int re_size = x.size1() - 501 * 13;
    casadi::DM ctl_lbx = -casadi::DM::inf(re_size);

    // trick here
    typename Equalities::num lbx = Equalities::num::repmat(LBX, 501, 1); // FIX for quaternions as well!
    lbx = casadi::DM::vertcat({lbx, ctl_lbx});
    //typename Equalities::num lbx = -Equalities::num::inf(x.size1());
    typename Equalities::num ubx = Equalities::num::inf(x.size1());
    typename Equalities::num lbg = Equalities::num::zeros(homotopy.size1());
    typename Equalities::num ubg = lbg;

    ARG["lbx"] = lbx;
    ARG["ubx"] = ubx;
    ARG["lbg"] = lbg;
    ARG["ubg"] = ubg;
    ARG["p"] = Equalities::num::vertcat({init_guess,1});

    ARG["lbx"](lbx.size1() - 1, 0) = 1;
    ARG["ubx"](ubx.size1() - 1, 0) = 1;
    ARG["x0"] = Equalities::num::vertcat({init_guess,1});

    //casadi::DM init_g = casadi::DM::vertcat({init_guess, 1});
    //casadi::DM init_g = casadi::DM::repmat(0, 8017, 1);
    //init_g[init_g.size1() - 1] = 1;

    //std::cout << "Initial guess: " << " size : " << init_g.size() << "\n" << init_g << "\n";
    //std::cout << "Jacoabian at initial guess: " << homo_eval({init_g}) << "\n";

    /** solve initial homotopy equations */
    casadi::DMDict solution = Corrector(ARG);
    casadi::Dict stats = Corrector.stats();

    //std::cout << solution.at("x") << "\n";

    /** implement predict-corrector scheme */
    double lambda_val = 1.0;
    double prop_lambda;
    casadi::DM lambda_log = 1.0;
    double h = 1.0;
    typename Equalities::num x_next;
    typename Equalities::num x0_num = solution.at("x");

    ARG["lbx"](lbx.size1() - 1, 0) = -Equalities::num::inf(1);
    ARG["ubx"](ubx.size1() - 1, 0) = Equalities::num::inf(1);

    uint iter_count = 0;
    casadi::DM t, tau;
    double r_norm_inf = 0.0;

    while(lambda_val > 0.0)
    {
        /** estimate tangent direction*/
        /** solve the system Hx x_dot = Hl lam_dot */
        casadi::DM b = -Hl({x0_num})[0];
        casadi::DM A = Hx({x0_num})[0];
        casadi::DM r = psarc_math::solve(A, b, psarc_math::EIGEN_SPARSE);

        r_norm_inf = casadi::DM::norm_inf(r).nonzeros()[0];
        if(r_norm_inf > 20)
        {
            /** try gradient rescaling */
            double factor = 20 / r_norm_inf;
            r = factor * r;
        }

        casadi::DM l_dot = 1.0 / std::sqrtf(1 + casadi::DM::dot(r,r).nonzeros()[0]);

        if(iter_count == 0)
        {
            // choose lambda-decreasing direction
            l_dot = -l_dot;
            tau = l_dot * r;
            tau = casadi::DM::vertcat({tau,l_dot});
            t = tau;
        }
        else
        {
            // try positive direction first
            tau = l_dot * r;
            tau = casadi::DM::vertcat({tau,l_dot});
            double proj_t = casadi::DM::dot(t,tau).nonzeros()[0];
            if(proj_t >= 0)
            {
                //choose this direction
                t = tau;
            }
            else
            {
                // choose the opposte direction
                std::cout << "CHANGING DIRECTION \n"; //??????
                tau = -tau;
                t = tau;
            }
        }

        /** make a prediction step */
        x_next = x0_num + h * t;
        prop_lambda = x_next(x_next.size1() - 1).nonzeros()[0];

        /** apply corrector  + warm starting*/
        ARG["p"] = x_next;
        ARG["x0"]     = solution.at("x");
        ARG["lam_g0"] = solution.at("lam_g");
        ARG["lam_x0"] = solution.at("lam_x");
        //ARG["lbx"](lbx.size1() - 1, 0) = x_next(1);
        //ARG["ubx"](ubx.size1() - 1, 0) = x_next(1);
        solution = Corrector(ARG);

        /** prepare for the next step */
        x0_num = solution.at("x");

        if(x0_num(x0_num.size1() - 1).nonzeros()[0] < 0.0)
        {   /** refine solution if it crosses 0 */
            x0_num(x0_num.size1() - 1) = 0.0;
            ARG["lbx"](lbx.size1() - 1, 0) = x0_num(x0_num.size1() - 1);
            ARG["ubx"](ubx.size1() - 1, 0) = x0_num(x0_num.size1() - 1);
            solution = Corrector(ARG);
            x0_num = solution.at("x");
        }
        //std::cout << "Corrector: " << x0_num << "\n";
        std::cout << Corrector.stats() << "\n";

        lambda_val = x0_num(x0_num.size1() - 1).nonzeros()[0];
        lambda_log = casadi::DM::vertcat({lambda_log, lambda_val});

        if(casadi::DM::dot(r,r).nonzeros()[0] > 1e6)
        {
            kite_utils::write_to_file("lox.txt", r);
            std::cout << r << "\n";
            break;
        }

        ++iter_count;

//        if(iter_count >= 80)
//        {
//            /** check bifurcation point */
//            x0_num(lbx.size1() - 1, 0) = 0.5;
//            b = -Hl({x0_num})[0];
//            A = Hx({x0_num})[0];

//            double det = psarc_math::determinant(A, psarc_math::EIGEN_DENSE);
//            double b_norm = casadi::DM::norm_2(b).nonzeros()[0];

//            std::cout << "\n" << "Gx determinant: " << psarc_math::determinant(A, psarc_math::EIGEN_SPARSE) << "\n";
//            std::cout << "\n" << "Gl norm: " << b_norm << "\n";

//            break;

//            //kite_utils::write_to_file("b.txt", b);
//            //break;
//            h = h >= 10 ? 10 : 1.5 * h;
//            std::cout << "H updated: " << h << "\n";
//        }

        std::cout << "LAMBDA: " << lambda_val << " t_norm_inf: " << casadi::DM::norm_inf(r)
                  << " l_dot " << l_dot << " iter: " << iter_count << "\n";
        std::cout << "Proposed lambda: " << prop_lambda << " accepted : " << lambda_val << "\n";
    }
    std::cout << "Total number of iterations: " << iter_count << "\n";
    //std::cout << solution.at("x") << "\n";
}

#endif // PSARC_HPP
