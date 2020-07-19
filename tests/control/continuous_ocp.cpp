#include "control/continuous_ocp.hpp"

POLYMPC_FORWARD_DECLARATION(/*Name*/ MyOCP, /*NX*/ 5, /*NU*/ 2, /*NP*/ 0, /*ND*/ 0, /*NG*/2, /*TYPE*/ double)

class MyOCP : public ContinuousOCP<MyOCP>
{
public:
    ~MyOCP(){}

    template<typename T>
    void dynamics_impl(const state_t<T> &x, const control_t<T> &u, const parameter_t<T> &p,
                       const static_parameter_t &d, const scalar_t &t, state_t<T> &xdot)
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
