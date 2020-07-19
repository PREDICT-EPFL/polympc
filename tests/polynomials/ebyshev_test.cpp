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

        double c = 1.0  / std::sqrt(2 * M_PI * sgm * sgm);
        double arg = std::pow(x - mu,2) / (2 * std::pow(sgm, 2));
        return  c * std::exp(-arg);
    }
};



int main()
{
    using namespace polympc;

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
