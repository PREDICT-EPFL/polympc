#ifndef LQR_HPP
#define LQR_HPP

#include "utils/helpers.hpp"
#include "unsupported/Eigen/Polynomials"
#include "iostream"

namespace polympc {

/** Moore-Penrose pseudo-inverse */
Eigen::MatrixXd pinv(const Eigen::Ref<const Eigen::MatrixXd> mat) noexcept
{
    /** compute SVD */
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const double pinvtol = 1e-6;
    Eigen::VectorXd singular_values = svd.singularValues();
    /** make a copy */
    Eigen::VectorXd singular_values_inv = singular_values;
    for ( int i = 0; i < mat.cols(); ++i)
    {
        if ( singular_values(i) > pinvtol )
            singular_values_inv(i) = 1.0 / singular_values(i);
        else singular_values_inv(i) = 0;
    }
    return (svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose());
}


/** Lyapunov equation */
Eigen::MatrixXd lyapunov(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> Q) noexcept
{
    const Eigen::Index m = Q.cols();
    /** compute Schur decomposition of A */
    Eigen::RealSchur<Eigen::MatrixXd> schur(A);
    Eigen::MatrixXd T = schur.matrixT();
    Eigen::MatrixXd U = schur.matrixU();

    Eigen::MatrixXd Q1;
    Q1.noalias() = (U.transpose() * Q) * U;
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(m, m);
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(m,m);

    X.col(m-1).noalias() = (T + T(m-1,m-1) * E).partialPivLu().solve(Q1.col(m-1));
    Eigen::VectorXd v;

    for(Eigen::Index i = m-2; i >= 0; --i)
    {
        v.noalias() = Q1.col(i) - X.block(0, i+1, m, m-(i+1)) * T.block(i, i+1, 1, m-(i+1)).transpose();
        X.col(i) = (T + T(i,i) * E).partialPivLu().solve(v);
    }

    X.noalias() = (U * X) * U.transpose();
    return X;
}

double line_search_care(const double &a, const double &b, const double &c) noexcept
{
    Eigen::Matrix<double, 5, 1> poly;
    Eigen::Matrix<double, 4, 1> poly_derivative;
    poly_derivative << -2*a, 2*(a-2*b), 6*b, 4*c;
    poly << a, -2*a, a-2*b, 2*b, c;
    poly_derivative = (1.0/(4*c)) * poly_derivative;
    poly = (1.0/c) * poly;

    /** find extremums */
    Eigen::PolynomialSolver<double, 3> root_finder;
    root_finder.compute(poly_derivative);

    /** compute values on the bounds */
    double lb_value = Eigen::poly_eval(poly, 1e-5);
    double ub_value = Eigen::poly_eval(poly, 2);

    double argmin = lb_value < ub_value ? 1e-5 : 2;

    /** check critical points : redo with visitor! */
    double minimum   = Eigen::poly_eval(poly, argmin);
    for (int i = 0; i < root_finder.roots().size(); ++i)
    {
        double root = root_finder.roots()(i).real();
        if((root >= 1e-5) && (root <= 2))
        {
            double candidate = Eigen::poly_eval(poly, root);
            if(candidate < minimum)
            {
                argmin = root;
                minimum   = Eigen::poly_eval(poly, argmin);
            }
        }
    }
    return argmin;
}

/** CARE Newton iteration */
Eigen::MatrixXd newton_ls_care(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> B,
                               const Eigen::Ref<const Eigen::MatrixXd> C, const Eigen::Ref<const Eigen::MatrixXd> X0) noexcept
{
    /** initial guess */
    //Eigen::EigenSolver<Eigen::MatrixXd> eig(A - B * X0);
    //std::cout << "INIT X0: \n" << eig.eigenvalues() << "\n";
    const double tol = 1e-5;
    const int kmax   = 2;
    Eigen::MatrixXd X = X0;
    double err = std::numeric_limits<double>::max();
    int k = 0;
    Eigen::MatrixXd RX, H, V;

    std::cout << "C: " << C << "\n";

    /** temporary */
    double tk = 1;

    while( (err > tol) && (k < kmax) )
    {
       RX = C;
       RX.noalias() += X * A + A.transpose() * X - (X * B) * X;
       /** newton update */
       H = lyapunov((A - B * X).transpose(), -RX);

       std::cout << "A': " << (A - B * X).transpose() << "\n";
       std::cout << "RX: " << RX << "\n";
       std::cout << "H: " << H << "\n";
       /** exact line search */
       V.noalias() = H * B * H;
       double a = (RX * RX).trace();
       double b = (RX * V).trace();
       double c = (V * V).trace();
       tk = line_search_care(a,b,c);
       /** inner loop to accept step */
       X.noalias() += tk * H;
       //err = tk * (H.lpNorm<1>() / X.lpNorm<1>());
       err = RX.norm();
       std::cout << "iter: " << k << " err: " << err << " step: " << tk << "\n";
       k++;
    }

    /** may be defect correction algorithm? */

    //std::cout << "CARE solve took " << k << " iterations. \n";
    if(k == kmax)
        std::cerr << "CARE cannot be solved to specified precision :" << err << " max number of iteration exceeded! \n ";

    return X;
}

Eigen::MatrixXd init_newton_care(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> B) noexcept
{
    const Eigen::Index n = A.cols();
    const double tolerance = 1e-12;
    /** compute Schur decomposition of A */
    Eigen::RealSchur<Eigen::MatrixXd> schur(A);
    Eigen::MatrixXd TA = schur.matrixT();
    Eigen::MatrixXd U = schur.matrixU();

    Eigen::MatrixXd TD = U.transpose() * B;
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    es.compute(TA, false);

    Eigen::VectorXd eig_r = es.eigenvalues().real();
    double b = -eig_r.minCoeff();
    b = std::fmax(b, 0.0) + 0.5;
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Z = lyapunov(TA + 100 * E, 2 * TD * TD.transpose());

    std::cout << "1 arg: \n" << TA + 100 * E << "\n";
    std::cout << "2 arg: \n" << 2 * TD * TD.transpose();
    std::cout << "Z: \n" << Z << "\n";

    Eigen::MatrixXd X = (TD.transpose() * pinv(Z)) * U.transpose();

    if( (X - X.transpose()).norm() > tolerance)
    {
        Eigen::MatrixXd M = (X.transpose() * B) * X + 0.5 * Eigen::MatrixXd::Identity(n ,n);
        X = lyapunov((A - B*X).transpose(), -M);
    }
    return X;
}

Eigen::MatrixXd care(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> B,
                     const Eigen::Ref<const Eigen::MatrixXd> C) noexcept
{
    std::cout << "A: \n" << A << "\n";
    std::cout << "B: \n" << B << "\n";

    Eigen::MatrixXd X0 = init_newton_care(A, B);
    return X0;
    //return newton_ls_care(A, B, C, X0);
}


/** Linear Quadratic Regulator:
 * J(x) = INT { xQx + xMu + uRu }dt
 * xdot = Fx + Gu
 */
void lqr(const Eigen::Ref<const Eigen::MatrixXd>& F, const Eigen::Ref<const Eigen::MatrixXd>& G, const Eigen::Ref<const Eigen::MatrixXd>& Q,
         const Eigen::Ref<const Eigen::MatrixXd>& R, const Eigen::Ref<const Eigen::MatrixXd>& M,
         Eigen::Ref<Eigen::MatrixXd> S, Eigen::Ref<Eigen::MatrixXd> K, const bool &check) noexcept
{
    /** check preliminary conditions */
    //assume F,G to be stabilizable
    if(check)
    {
        Eigen::MatrixXd QR = Q;
        QR.noalias() -= M * pinv(R) * M.transpose();
        Eigen::EigenSolver<Eigen::MatrixXd> solver(QR);
        Eigen::VectorXd values = solver.eigenvalues().real();
        if( (values.array() < 0).any() )
        {
            std::cerr << "Weight matrices did not pass positivity check! \n";
            S = Eigen::MatrixXd();
            K = Eigen::MatrixXd();
        }
    }

    /** formulate Ricatti equations */
    Eigen::MatrixXd invR = pinv(R);
    Eigen::MatrixXd A = F - M * invR * (G).transpose();
    Eigen::MatrixXd B = G * invR * (G).transpose();
    Eigen::MatrixXd C = M * invR * M.transpose() + Q;

    //std::cout << "A: \n" << A << "\n";
    //std::cout << "B: \n" << B << "\n";
    //std::cout << "C: \n" << C << "\n";
    /** solve Ricatti equation */
    S = care(A, B, C);
    //std::cout << "CARE solution: \n" << (S*A) + (A.transpose() * S) - (S * B) * S.transpose() + C << "\n";

    //std::cout << "S: \n" << S << "\n";
    /** compute gain matrix */
    K = invR * ( (G).transpose() * S + M.transpose());
}

} // polympc namespace

#endif // LQR_HPP
