#include "polynomials/legendre.hpp"
#include "polynomials/projection.hpp"
#include "integrator.h"

struct Integrand
{
    double operator ()(const double &x) const
    {
        double sgm = 3.0;
        double mu = 1.0;

        double c = 1.0  / std::sqrtf(2 * M_PI * sgm * sgm);
        double arg = std::powf(x - mu,2) / (2 * std::powf(sgm, 2));
        return  c * std::expf(-arg);
        //return 0.25 * (3*x*x - 1) * x;
    }
};

struct Unif
{
    double operator ()(const double &x) const
    {
        return  x;
    }
};

/** casting tensor to matrix */
template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar,int rank, typename sizeType>
MatrixType<Scalar> Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}



int main()
{
    using Legendre = Legendre<5, GAUSS_LOBATTO>;
    Legendre leg;
    Integrand f;
    Unif distrib;

    /** integrate value */
    float result = leg.integrate<Integrand>();
    Projection<Legendre> proj(f, 1, 6);
    std::cout << "Integration resuls: " << result << "\n";
    std::cout << "Projection: " << proj.coeff.transpose() << "\n";
    std::cout << "f(x): " << f(4) << " fN(x): " << proj.eval(4) << "\n";
    //std::cout << "Weights: " << leg.QWeights() << "\n";

    /** solve test equation x_dot = -lambda * x; lambda ~ Unif(1,2)*/
    /** project lambda random variable */
    proj.project(distrib, -2, 4);
    std::cout << "Projection: " << proj.coeff.transpose() << "\n";

    /** get Galerkin tensor */
    Legendre::tensor_t Galerkin = leg.getGalerkinTensor();

    /** get a chip and cast to a matrix */
    Eigen::Tensor<double, 2> kchip = Galerkin.chip(0, 2);
    Eigen::MatrixXd mat = Tensor_to_Matrix(kchip, 6, 6);
    Eigen::VectorXd A_i;
    //std::cout << "mat: \n" << mat << "\n";

    std::ofstream galerkin_file("galerkin_tensor");
    if(galerkin_file.is_open())
        std::cout << "Galerkin file is open \n";
    /** create A(\Lam) matrix */
    casadi::SX A;
    for(int i = 0; i < Galerkin.dimension(2); ++i)
    {
        kchip = Galerkin.chip(i, 2);
        mat = Tensor_to_Matrix(kchip, 6, 6);
        A_i = mat.transpose() * proj.coeff;

        galerkin_file << "i : " << i << "\n" <<  mat << "\n";

        /** cast to CasADi*/
        std::vector<double> sold;
        sold.resize(static_cast<size_t>(A_i.size()));
        Eigen::Map<Eigen::VectorXd>(&sold[0], A_i.size()) = A_i;
        A = casadi::SX::vertcat({A, casadi::DM(sold).T()});
    }

    galerkin_file.close();


    /** solve the system x_dot = -A x */
    casadi::SX Zero = casadi::SX::zeros(A.size());
    casadi::SX Iden = casadi::SX::eye(A.size1());

    /** construct big matrix */
    casadi::SX ROW1 = casadi::SX::horzcat({Zero, Iden});
    casadi::SX ROW2 = casadi::SX::horzcat({-A, Zero});
    casadi::SX BIGA = casadi::SX::vertcat({ROW1, ROW2});

    casadi::SX x   = casadi::SX::sym("x", 12);
    casadi::SX u   = casadi::SX::sym("u",1); // dummy control input
    casadi::Function rhs = casadi::Function("sode",{x, u}, { casadi::SX::mtimes(BIGA, x)});
    casadi::DM x0  = casadi::DM({5,0,0,0,0,0, 0,0,0,0,0,0});
    casadi::DM ctl = 0.0;

    double tf = 4.0;
    casadi::DMDict props;
    PSODESolver<10,1,12,1>ps_solver(rhs, tf, props);

    bool FULL = true;
    casadi::DM ps_sol = ps_solver.solve(x0, ctl, FULL);

    /** save trajectory */
    std::ofstream solution_file("solution.txt", std::ios::out);
    int dimx = 12;

    if(!solution_file.fail())
    {
        for (int i = 0; i < ps_sol.size1(); i = i + dimx)
        {
            std::vector<double> tmp = ps_sol(casadi::Slice(i, i + dimx),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                solution_file << tmp[j] << " ";
            }
            solution_file << "\n";
        }
    }
    solution_file.close();

    /** solve x_dot = a * sin(x), where a ~ Unif(1,3) */
    casadi::SX a = casadi::SX::sym("a", 6);
    casadi::SX var = casadi::SX::sym("var");
    casadi::Function ode = casadi::Function("lox", {var}, {sin(4 * var)});
    Legendre::q_weights_t weights = leg.QWeights();
    Legendre::nodes_t nodes = leg.CPoints();
    Legendre::q_weights_t nfactors = leg.NFactors();
    int N = nodes.SizeAtCompileTime;

    /** construct K = N differential equations */
    casadi::SX sode = casadi::SX::zeros(N,1);
    for(int k = 0; k < N; ++k)
    {
        casadi::SX integral = 0;
        for(int n = 0; n < N; ++n)
        {
            casadi::SX arg = 0;
            double omega = 0;
            for(int j = 0; j < N; ++j)
            {
                arg += a(j) * leg.eval(nodes[n], j);
                omega += proj.coeff[j] * leg.eval(nodes[n], j);
            }
            double fi_k = leg.eval(nodes[n], k);
            casadi::SX f = ode({arg})[0];
            integral += (weights[n] * omega * fi_k) * f;
        }
        sode(k) = nfactors[k] * integral;
    }

    casadi::Function sode_fun = casadi::Function("sode_fun",{a, u},{sode});
    x0  = casadi::DM({1,0,0,0,0,0});
    ctl = 0.0;

    tf = 1.0;
    PSODESolver<10,1,6,1>ps_solver2(sode_fun, tf, props);
    ps_sol = ps_solver2.solve(x0, ctl, FULL);

    /** save trajectory */
    std::ofstream solution_sode("solution_sode.txt", std::ios::out);
    dimx = 6;

    if(!solution_sode.fail())
    {
        for (int i = 0; i < ps_sol.size1(); i = i + dimx)
        {
            std::vector<double> tmp = ps_sol(casadi::Slice(i, i + dimx),0).nonzeros();
            for (uint j = 0; j < tmp.size(); j++)
            {
                solution_sode << tmp[j] << " ";
            }
            solution_sode << "\n";
        }
    }
    solution_sode.close();

    return 0;
}
