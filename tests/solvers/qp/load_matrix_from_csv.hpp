#include <Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;

/*


Source : https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix/39146048
Usage:
MatrixXd A = load_csv<MatrixXd>("C:/Users/.../A.csv");
Matrix3d B = load_csv<Matrix3d>("C:/Users/.../B.csv");
VectorXd v = load_csv<VectorXd>("C:/Users/.../v.csv");
*/

//const double inf = std::numeric_limits<double>::infinity();

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    //const char *exp[] = { "", "inf", "NaN" };
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            bool is_inf = false;
            /* for (int i=0; i < 3; i++)
            {
                if (exp[i] == cell){values.push_back(inf); is_inf=true; break;}
            }*/
            if (!is_inf) {values.push_back(std::stod(cell));}
        }
        ++rows;
    }
    //std::cout << rows << " " << values.size() << std::endl;

    return Eigen::Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);

}
