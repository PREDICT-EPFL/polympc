#include "solvers/qp_base.hpp"

class MyQP : public QPBase<MyQP, 55, 40>
{

};




int main(void)
{
    MyQP qp;

    MyQP::qp_dual_t lb;
    MyQP::qp_dual_t ub;

    lb.setZero();
    ub.setOnes();

    qp.parse_constraints_bounds(lb, ub);

    return EXIT_SUCCESS;
}
