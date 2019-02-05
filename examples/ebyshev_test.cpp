//#include "ebyshev.hpp"
#include "polynomials/ebyshev.hpp"

struct Integrand
{
    double operator ()(const double &x)
    {
        double sgm = 3.0;
        double mu = 1.0;

        double c = 1.0  / std::sqrtf(2 * M_PI * sgm * sgm);
        double arg = std::powf(x - mu,2) / (2 * std::powf(sgm, 2));
        return  c * std::expf(-arg);
    }
};



int main()
{
    using chebyshev = Chebyshev<10,2,3,4,5>;
    chebyshev cheb;

    /** integrate value */
    Integrand f;
    float result = cheb.integrate<Integrand>(1,6);
    chebyshev::Projection proj = cheb.project<Integrand>(1,6);
    std::cout << "Integration resuls: " << result << "\n";
    std::cout << "Projection: " << proj.coeff.transpose() << "\n";

    std::cout << "f(x): " << f(4) << " fN(x): " << proj.eval(4) << "\n";
    return 0;
}
