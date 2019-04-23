#include "casadi/casadi.hpp"
#include "ctime"
#include "chrono"

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "chebyshev.hpp"

using namespace casadi;

int main()
{
    SX sym = SX::sym("sym",4);
    SX x = SX::sym("x",3);
    std::vector<SX> y{sym, x, 0};

    SXVector vect{sym,x,0};
    SX var = SX::vertcat({sym, x, 0});

    Slice slice(1,5);
    auto sliced = var(slice,0);
    sliced(Slice(1,4),0) = x;

    SX z = SX::sym("z");
    SX z2 = SX::sym("z2");
    Function f = Function("test", {z,z2}, {z + z2});
    SXVector res = f(SXVector{z,z2});
    SX func = SX::vertcat(res);
    SX jac = SX::jacobian(func,z);

    DM Matr = DM::diag(pow(DMVector{1,2,3,4},2));

    DM H = DM::horzcat(DMVector{DM::zeros(7,6), DM::eye(7)});
    std::cout << H << "\n";

    SX J = SX::diag(x);
    SX Jx = SX::mtimes(J, x);
    std::cout << "\n" << "J " << Jx << "\n";

    std::pair<double, double> t_pair = std::pair<double, double>(1.5,2);
    std::cout << "Pair First: " << t_pair.first << " Pair Second: " << t_pair.second << "\n";

    DM vect_test = DM::vertcat({0,1});
    vect_test = cos(vect_test);

    std::cout << "Vector COS: " << vect_test << "\n";

    //////////////////////////////////////////////////////////
    auto grid = range(0,10);
    DMVector grid_DM(grid.begin(), grid.end());
    std::cout << "GRID: " << DM::vertcat(grid_DM) + 1 << "\n";

    DM c = DM::vertcat(grid_DM);
    c = c * c;
    DM _c = pow(-1, c);

    std::cout << "C: " << c << " => " << _c << "\n";
    std::cout << "INF: " << -DM::inf(10) << "\n";

    SX Z = SX::sym("Z", 3,4);
    printf("COLS: %d, ROWS: %d \n", Z.size1(), Z.size2());
    std::cout << SX::vec(Z) << "\n";

    SX q = SX::sym("q",3);
    std::cout << q(0) << "\n";
    SX r = SX::sym("r", 3);
    Function X2 = Function("X2", {q}, {pow(q,2)});
    SXVector z_vec = SX::horzsplit(Z,1);
    SXVector result = X2(*(z_vec.begin()));

    std::cout << result << "\n";

    std::cout << SX::blockcat({{SX::eye(3), SX::zeros(3,3)}, {SX::zeros(3,3), SX::zeros(3,3)}}) << "\n";

    std::cout << SX::diagcat({SX::eye(5), 5 * SX::eye(4)}) << "\n";

    std::cout << "INF DIFF: " << DM::inf(1) - DM::inf(1) << "\n";

    std::cout << pow(Matr, 2) << "\n";
    std::cout << jac << " " << "\n";
    //std::cout << "aircraft name: " << plane << " " <<"with Aspect Ratio= "<< aspect_ratio <<"\n";
    ///////////////////////////////////////////////////////////
    //// EIGEN TEST //////
    Eigen::VectorXd eig_vec = Eigen::VectorXd::Map(c.nonzeros().data(), c.nonzeros().size());
    std::cout << eig_vec << "\n";

    std::cout << DM::sum1(c) << "\n";

    DM EYE = DM::eye(7);
    Eigen::Matrix<double, 7, 7> eig_mat = Eigen::Matrix<double, 7, 7>::Map(DM::densify(EYE).nonzeros().data(), 7, 7);
    std::cout << eig_mat << "\n";

    std::cout << "\n" << "----------------------------------------------------------------------------------" << "\n";
    int32_t number = 1900;
    std::cout << "\n CAST DOUBLE" << static_cast<double>(number) << "\n";

    std::string func_name = "RK4_somethNing";
    if (func_name.find("RK4") != std::string::npos)
        std::cout << "Found RK4 \n";
    /** CAST BACK */
    Eigen::Matrix<SXElem, 2, 1> sol;
    sol(0) = SXElem::sym("lox");
    sol(1) = SXElem::sym("pidor");
    std::vector<SXElem> sold;
    sold.resize(static_cast<size_t>(sol.size()));
    Eigen::Map<Eigen::Matrix<SXElem, 2, 1>>(&sold[0], sol.size()) = sol;

    return 0;
}
