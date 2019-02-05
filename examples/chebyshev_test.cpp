#include "chebyshev.hpp"

int main()
{
    Chebyshev<casadi::SX, 4, 2, 3, 2, 1> cheb;
    std::cout << "Nodes: " << cheb.CPoints() << "\n";
    std::cout << "Weights: " << cheb.QWeights() << "\n";
}
