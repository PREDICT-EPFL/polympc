//#include "ebyshev.hpp"
#include "polynomials/ebyshev.hpp"
#include "polynomials/projection.hpp"

struct Integrand
{
    Integrand(){}
    ~Integrand(){}

    double operator ()(const double &x) const
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
    using chebyshev = Chebyshev<8, GAUSS_LOBATTO>;
    chebyshev cheb;

    /** integrate value */
    Integrand f;
    float result = cheb.integrate<Integrand>();
    Projection<chebyshev> proj(f, 1, 6);

    std::cout << "Integration resuls: " << result << "\n";
    std::cout << "Projection: " << proj.coeff.transpose() << "\n";
    std::cout << "f(x): " << f(4) << " fN(x): " << proj.eval(4) << "\n";

    return 0;
}
