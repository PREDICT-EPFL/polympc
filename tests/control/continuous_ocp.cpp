#include "control/continuous_ocp.hpp"
#include "polynomials/ebyshev.hpp"
#include "polynomials/splines.hpp"

POLYMPC_FORWARD_DECLARATION(/*Name*/ MyOCP, /*NX*/ 5, /*NU*/ 2, /*NP*/ 0, /*ND*/ 0, /*NG*/2, /*TYPE*/ double)

using Polynomial = polympc::Chebyshev<4, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, 4>;

using namespace Eigen;

class MyOCP : public ContinuousOCP<MyOCP, Approximation>
{
public:
    ~MyOCP(){}

    static constexpr double t_start = 0.0;
    static constexpr double t_stop  = 1.0;

    template<typename T>
    void dynamics_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u, const Ref<const parameter_t<T>> p,
                       const Ref<const static_parameter_t> d, const scalar_t &t, Ref<state_t<T>> xdot)
    {
        std::cout << "I took the wright implementation \n";
        xdot << x;
    }

   template<typename T>
   void inequality_constraints_impl(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                               const static_parameter_t &d, const scalar_t &t, constraint_t<T> &g)
   {
        g << Eigen::Matrix<T, 2, 1>();
   }


};


int main(void)
{
    MyOCP ocp;
    return EXIT_SUCCESS;
}
