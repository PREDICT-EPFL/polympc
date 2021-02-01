#include "polymath.h"

using namespace casadi;

/** collection of auxillary mathematical routines */

namespace polymath
{

/** quaternion arithmetics */
SX T1quat(const casadi::SX &rotAng) { return SX::vertcat({cos(-rotAng / 2.0), sin(-rotAng / 2.0), 0, 0}); }
SX T2quat(const casadi::SX &rotAng) { return SX::vertcat({cos(-rotAng / 2.0), 0, sin(-rotAng / 2.0), 0}); }
SX T3quat(const casadi::SX &rotAng) { return SX::vertcat({cos(-rotAng / 2.0), 0, 0, sin(-rotAng / 2.0)}); }

SX quat_multiply(const SX &q1, const SX &q2)
{
    SX s1 = q1(0);
    SX v1 = q1(Slice(1,4),0);

    SX s2 = q2(0);
    SX v2 = q2(Slice(1,4),0);

    SX s = (s1 * s2) - SX::dot(v1, v2);
    SX v = SX::cross(v1, v2) + (s1 * v2) + (s2 * v1);

    return SX::vertcat({s,v});
}

SX quat_inverse(const SX &q)
{
    SXVector tmp{q(0), -q(1), -q(2), -q(3)};
    return SX::vertcat(tmp);
}

SX quat_transform(const casadi::SX &q_ba, const casadi::SX &a_vect)
{
    SX tmp = quat_multiply( q_ba, quat_multiply(SX::vertcat({0, a_vect}), quat_inverse(q_ba)) );
    return tmp(Slice(1,4), 0);
}


/** ------------------------------------ */
SX heaviside(const SX &x, const double K)
{
    return K / (1 + exp(-4 * x));
}

SX rk4_symbolic(const SX &x,
                const SX &u,
                Function &func,
                const SX &h)
{
    SXVector res = func(SXVector{x, u});
    SX k1 = res[0];
    res = func(SXVector{x + 0.5 * h * k1, u});
    SX k2 = res[0];
    res = func(SXVector{x + 0.5 * h * k2, u});
    SX k3 = res[0];
    res = func(SXVector{x + h * k3, u});
    SX k4 = res[0];

    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
}

void cheb(DM &CollocPoints, DM &DiffMatrix, const unsigned &N,
          const std::pair<double, double> interval = std::make_pair(0,1))
{
    /** Chebyshev collocation points for the interval [-1, 1]*/
    auto grid_int = casadi::range(0, N+1);
    /** cast grid to Casadi type */
    DMVector DMgrid(grid_int.begin(), grid_int.end());
    DM grid = DM::vertcat(DMgrid);
    DM X = cos(grid * (M_PI / N));

    /** shift and scale points */
    CollocPoints = (X + interval.first + 1) * ((interval.second - interval.first) / 2);

    /** Compute Differentiation matrix */
    DM c = DM::vertcat({2, DM::ones(N-1,1), 2});
    c = SX::mtimes(SX::diag( pow(-1, grid)), c);

    DM XM = DM::repmat(CollocPoints, 1, N+1);
    DM dX = XM - XM.T();
    DM Dn  = DM::mtimes(c, (1 / c).T() ) / (dX + (DM::eye(N+1)));      /** off-diagonal entries */

    DiffMatrix  = Dn - DM::diag( DM::sum1(Dn.T() ));               /**  diagonal entries */
}

SX mat_func(const SX &matrix_in, Function &func)
{
    SXVector columns = SX::horzsplit(matrix_in, 1);
    /** evaluate fnuction for each column  and stack in a single vector*/
    SX output_vec;
    for(auto it = columns.begin(); it != columns.end(); ++it)
    {
        SX res = SX::vertcat(func(*it));
        /** append to resulting vector*/
        output_vec = SX::vertcat(SXVector{output_vec, res});
    }
    return output_vec;
}

SX mat_dynamics(const SX &arg_x, const SX &arg_u, Function &func)
{
    SXVector xdot;
    SXVector x = SX::horzsplit(arg_x, 1);
    SXVector u = SX::horzsplit(arg_u, 1);

    for(unsigned i = 0; i < u.size(); ++i)
    {
        SXVector eval = func(SXVector{x[i], u[i]});
        xdot.push_back(eval[0]);
    }
    /** discard the initial state */
    xdot.push_back(x.back());
    return SX::vertcat(xdot);
}

casadi::Function lagrange_poly(const unsigned &n_degree)
{
    casadi::SX t = casadi::SX::sym("t");
    casadi::SX nodes = casadi::SX::sym("nodes", n_degree + 1);
    casadi::SX phi_sym = casadi::SX::ones(n_degree + 1, 1); //polynomial coefficients

    for(unsigned k = 0; k <= n_degree; ++k)
    {
        for(unsigned j = 0; j <= n_degree; ++j)
        {
            if(j != k)
                phi_sym(k) *= (t - nodes(j)) / (nodes(k) - nodes(j));
        }
    }

    casadi::Function phi_fun = Function("phi_fun", SXVector{t, nodes}, {phi_sym});
    return phi_fun;
}

casadi::Function lagrange_interpolant(const casadi::DM &X, const casadi::DM &Y)
{
    casadi::SX t = casadi::SX::sym("t");
    Eigen::VectorXd Xe(X.size1()); Xe = Eigen::VectorXd::Map(X.nonzeros().data(), X.size1());
    Eigen::VectorXd Ye(Y.size1()); Ye = Eigen::VectorXd::Map(Y.nonzeros().data(), Y.size1());

    Eigen::VectorXd interpolant = lagrange_interpolant(Xe, Ye);

    /** cast back to CasADi Polynomial and evaluate at "t" */
    std::vector<double> coeffs(interpolant.size());
    Eigen::Map<Eigen::VectorXd>(&coeffs[0], interpolant.size()) = interpolant;
    casadi::Polynomial poly(coeffs);

    return casadi::Function("interpolant",{t},{poly(t)});
}

casadi::SX cgl_nodes(const int &n_points)
{
    auto grid_int = casadi::range(0, n_points);
    /** cast grid to Casadi type */
    casadi::DMVector DMgrid(grid_int.begin(), grid_int.end());
    casadi::DM grid = casadi::DM::vertcat(DMgrid);
    casadi::SX X = cos(grid * (M_PI / (n_points - 1)));

    return X;
}

/**  (flip along dimension 1 means: the first row will be the last row, flip along dimension 2 means: the first column will be the last column) */
casadi::DM flip(const casadi::DM &matrix, const unsigned &dim)
{
    switch (dim) {
        case 1:
            return DM::vertcat({matrix(Slice(-1, 0, -1), Slice()), matrix(0, Slice())});
        case 2:
            return DM::horzcat({matrix(Slice(), Slice(-1, 0, -1)), matrix(Slice(), 0)});
        default:
            return matrix;
    }
}


Eigen::VectorXd lagrange_interpolant(const Eigen::VectorXd &X, const Eigen::VectorXd &Y)
{
    Eigen::VectorXd polynomial(X.rows());

    Eigen::MatrixXd P(X.rows(), polynomial.rows());
    for(unsigned i = 0; i < P.rows(); ++i)
    {
        Eigen::VectorXd roots(X.rows() - 1);
        unsigned count = 0;
        for(unsigned j = 0; j < X.rows(); ++j)
        {    if(i != j)
            {
                roots(count) = X(j);
                count++;
            }
        }
        Eigen::roots_to_monicPolynomial(roots, polynomial);
        P.row(i) = polynomial / Eigen::poly_eval(polynomial, X(i));
    }
    //polynomial = Y.transpose() * P.rowwise().reverse();
    polynomial = Y.transpose() * P;
    return polynomial;
}

Eigen::MatrixXd lagrange_poly_basis(const Eigen::VectorXd &nodes)
{
    Eigen::VectorXd polynomial(nodes.rows());
    Eigen::MatrixXd P(nodes.rows(), nodes.rows());

    for(unsigned i = 0; i < P.rows(); ++i)
    {
        Eigen::VectorXd roots(nodes.rows() - 1);
        unsigned count = 0;
        for(unsigned j = 0; j < nodes.rows(); ++j)
        {    if(i != j)
            {
                roots(count) = nodes(j);
                count++;
            }
        }
        Eigen::roots_to_monicPolynomial(roots, polynomial);
        P.row(i) = polynomial / Eigen::poly_eval(polynomial, nodes(i));
    }
    return P;
}

/** interpolator class */
LagrangeInterpolator::LagrangeInterpolator(const Eigen::VectorXd &nodes, const Eigen::VectorXd &values)
{
    assert(nodes.rows() == values.rows());
    assert((nodes.cols() == 1) && (values.cols() == 1));

    m_poly_basis = polymath::lagrange_poly_basis(nodes);
    m_interpolant =  values.transpose() * m_poly_basis;
}

LagrangeInterpolator::LagrangeInterpolator(const casadi::DM &nodes, const casadi::DM &values)
{
    assert(nodes.rows() == values.rows());
    assert((nodes.columns() == 1) && (values.columns() == 1));
    Eigen::VectorXd X(nodes.size1()); X = Eigen::VectorXd::Map(nodes.ptr(), nodes.size1());
    Eigen::VectorXd Y(values.size1()); Y = Eigen::VectorXd::Map(values.ptr(), values.size1());

    m_poly_basis = polymath::lagrange_poly_basis(X);
    m_interpolant = Y.transpose() * m_poly_basis;
}

void LagrangeInterpolator::update_basis(const Eigen::VectorXd &nodes)
{
    m_poly_basis = polymath::lagrange_poly_basis(nodes);
}

void LagrangeInterpolator::init(const Eigen::VectorXd &nodes, const Eigen::VectorXd &values)
{
    assert(nodes.rows() == values.rows());
    assert((nodes.cols() == 1) && (values.cols() == 1));

    m_poly_basis = polymath::lagrange_poly_basis(nodes);
    m_interpolant =  values.transpose() * m_poly_basis;
}

void LagrangeInterpolator::init(const casadi::DM &nodes, const casadi::DM &values)
{
    assert(nodes.rows() == values.rows());
    assert((nodes.columns() == 1) && (values.columns() == 1));
    Eigen::VectorXd X(nodes.size1()); X = Eigen::VectorXd::Map(nodes.ptr(), nodes.size1());
    Eigen::VectorXd Y(values.size1()); Y = Eigen::VectorXd::Map(values.ptr(), values.size1());

    m_poly_basis = polymath::lagrange_poly_basis(X);
    m_interpolant = Y.transpose() * m_poly_basis;
}

double LagrangeInterpolator::eval(const double &arg)
{
    return Eigen::poly_eval(m_interpolant, arg);
}

double LagrangeInterpolator::eval(const double &arg, const Eigen::VectorXd &values)
{
    assert(m_interpolant.size() == values.rows());
    m_interpolant = values.transpose() * m_poly_basis;
    return Eigen::poly_eval(m_interpolant, arg);
}

double LagrangeInterpolator::eval(const double &arg, const casadi::DM &values)
{
    Eigen::VectorXd Y(values.size1()); Y = Eigen::VectorXd::Map(values.ptr(), values.size1());
    m_interpolant = Y.transpose() * m_poly_basis;
    return Eigen::poly_eval(m_interpolant, arg);
}







/** Linear system implementation */
bool LinearSystem::is_controllable()
{
    /** compute controllability matrix */
    int n = F.rows();
    int m = G.cols();

    Eigen::MatrixXd ctrb = Eigen::MatrixXd::Zero(n, n * m);
    for(int k = 0; k < n; ++k)
    {
        ctrb.block(0, k*m, n, m) = F.pow(k) * G;
    }

    Eigen::FullPivLU<Eigen::MatrixXd> chol(ctrb);
    if (chol.rank() == n)
        return true;
    else
        return false;
}

unsigned factorial(const unsigned &n)
{
    unsigned fact = 1;
    for (int i = 1; i <= n; ++i)
        fact *= i;

    return fact;
}

namespace oc {

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

/** CARE Newton iteration */
Eigen::MatrixXd newton_ls_care(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> B,
                               const Eigen::Ref<const Eigen::MatrixXd> C, const Eigen::Ref<const Eigen::MatrixXd> X0) noexcept
{
    /** initial guess */
    //Eigen::EigenSolver<Eigen::MatrixXd> eig(A - B * X0);
    //std::cout << "INIT X0: \n" << eig.eigenvalues() << "\n";
    const double tol = 1e-5;
    const int kmax   = 20;
    Eigen::MatrixXd X = X0;
    double err = std::numeric_limits<double>::max();
    int k = 0;
    Eigen::MatrixXd RX, H, V;

    /** temporary */
    double tk = 1;

    while( (err > tol) && (k < kmax) )
    {
       RX = C;
       RX.noalias() += X * A + A.transpose() * X - (X * B) * X;
       /** newton update */
       H = lyapunov((A - B * X).transpose(), -RX);
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
       //std::cout << "err " << err << " step " << tk << "\n";
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
    Eigen::MatrixXd Z = lyapunov(TA + b * E, 2 * TD * TD.transpose());
    Eigen::MatrixXd X = (TD.transpose() * pinv(Z)) * U.transpose();

    if( (X - X.transpose()).norm() > tolerance)
    {
        Eigen::MatrixXd M = (X.transpose() * B) * X + 0.5 * Eigen::MatrixXd::Identity(n ,n);
        X = lyapunov((A - B*X).transpose(), -M);
    }
    return X;
}

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

Eigen::MatrixXd care(const Eigen::Ref<const Eigen::MatrixXd> A, const Eigen::Ref<const Eigen::MatrixXd> B,
                     const Eigen::Ref<const Eigen::MatrixXd> C) noexcept
{
    Eigen::MatrixXd X0 = init_newton_care(A, B);
    return newton_ls_care(A, B, C, X0);
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

Eigen::MatrixXd lqr(const LinearSystem &sys, const Eigen::Ref<const Eigen::MatrixXd> Q,
                    const Eigen::Ref<const Eigen::MatrixXd> R, const Eigen::Ref<const Eigen::MatrixXd> M, const bool &check) noexcept
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
            return Eigen::MatrixXd();
        }
    }

    /** formulate Ricatti equations */
    Eigen::MatrixXd invR = pinv(R);
    Eigen::MatrixXd A = sys.F - M * invR * (sys.G).transpose();
    Eigen::MatrixXd B = sys.G * invR * (sys.G).transpose();
    Eigen::MatrixXd C = M * invR * M.transpose() + Q;

    //std::cout << "A: \n" << A << "\n";
    //std::cout << "B: \n" << B << "\n";
    //std::cout << "C: \n" << C << "\n";
    /** solve Ricatti equation */
    Eigen::MatrixXd S = care(A, B, C);
    //std::cout << "CARE solution: \n" << (S*A) + (A.transpose() * S) - (S * B) * S.transpose() + C << "\n";

    //std::cout << "S: \n" << S << "\n";
    /** compute gain matrix */
    Eigen::MatrixXd K = invR * ( (sys.G).transpose() * S + M.transpose());
    return K;
}

/**oc bracket */
}

/**polymath bracket */
}
