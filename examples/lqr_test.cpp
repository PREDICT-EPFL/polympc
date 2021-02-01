#include "polymath.h"

typedef std::chrono::time_point<std::chrono::system_clock> time_point;
time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}


int main(void)
{
    Eigen::MatrixXd A(6,6);
    Eigen::MatrixXd B(6,3);
    Eigen::MatrixXd C(6,6);
    Eigen::MatrixXd Q(6,6);
    Eigen::MatrixXd R(3,3);
    Eigen::MatrixXd M(6,3);

    A <<  -0.186142, 0, 0.78059, 0, -0.0688479, 0,
          0,-0.562748, 0, 0, 0, -11,
          -2.51763, 0, -8.32444, 0, 9.81496, 0,
          0, -3.64033, 0, -31.3938, 0, 4.73357,
          -1.45204, 0, -6.53416, 0, -6.91635, 0,
          0, 2.65926, 0, -2.22952, 0, -1.84696;

    B << -0.269335, 0, 0,
          0, 3.41019, 0,
         -4.6359, 0, 0,
         0, 4.97283, -155.546,
         -52.918, 0, 0,
         0, -25.9202, -5.71874;


    C.setIdentity();
    Q.setIdentity();
    R.setIdentity();
    M.setZero();

    polymath::LinearSystem sys(A,B,C);
    std::chrono::time_point<std::chrono::system_clock> start = get_time();
    Eigen::MatrixXd K = polymath::oc::lqr(sys, Q, R, M);
    std::chrono::time_point<std::chrono::system_clock> stop = get_time();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "LQR time: " << std::setprecision(9)
              << static_cast<double>(duration.count())  << " [microseconds]" << "\n";


    std::cout << "K: \n" << K << "\n";

    return EXIT_SUCCESS;
}
