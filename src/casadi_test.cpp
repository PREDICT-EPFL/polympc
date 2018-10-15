#include "casadi/casadi.hpp"
#include "yaml-cpp/yaml.h"
#include "ctime"
#include "chrono"

#include "ros/time.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "pseudospectral/chebyshev.hpp"

using namespace casadi;

int main()
{
    //load config information
    YAML::Node config = YAML::LoadFile("umx_radian.yaml");
    const std::string plane = config["name"].as<std::string>();
    const double aspect_ratio = config["geometry"]["AR"].as<double>();

    //config["inertia"]["mass"] = 1000.0;
    //std::ofstream fout("umx_radian3.yaml");
    //fout << config;

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

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    double ros_time = static_cast<double>(seconds);
    printf("Local time is %f \n", ros_time * 1e-6);

    ros::Time::init();
    double ros_true_time = ros::Time::now().toSec();
    printf("True Local time is %f \n", ros_true_time);
    std::cout << "Local time is: " << ros_time * 1e-6 << "\n";

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

    std::cout << DM::sumRows(c) << "\n";

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

    std::cout << "|----------------------------SPARSITY EXPERIMENT--------------------------------------| \n";
    /** create a sparse matrix */
    Chebyshev<casadi::SX, 2, 2, 2, 1, 0>cheb;
    casadi::DM Matrx = cheb.CompD();
    std::cout << Matrx << "\n";

    casadi::Sparsity SpA = Matrx.get_sparsity();
    std::cout << "Nonzeros in rows: " << SpA.get_row() << "\n";
    std::cout << "Nonzeros in columns: " << SpA.get_colind() << "\n";

    std::vector<int> output_row, output_col;
    SpA.get_triplet(output_row, output_col);
    std::vector<double> values = Matrx.get_nonzeros();

    std::cout << "Output row: " << output_row << "\n";
    std::cout << "Output col: " << output_col << "\n";
    std::cout << "Nonzeros: " << Matrx.get_nonzeros() << "\n";

    using T = Eigen::Triplet<double>;
    std::vector<T> TripletList;
    TripletList.resize(values.size());
    for(int k = 0; k < values.size(); ++k)
        TripletList[k] = T(output_row[k], output_col[k], values[k]);

    for(std::vector<T>::const_iterator it = TripletList.begin(); it != TripletList.end(); ++it)
    {
        std::cout << "triplet: " << (*it).row() << " " << (*it).col() << " " << (*it).value() << "\n";
    }

    Eigen::SparseMatrix<double> SpMatrx(Matrx.size1(), Matrx.size2());
    SpMatrx.setFromTriplets(TripletList.begin(), TripletList.end());

    std::cout << "Eigen sparse matrix: \n" << SpMatrx << "\n";

    return 0;
}
