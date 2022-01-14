#ifndef QPMAD_INTERFACE_HPP
#define QPMAD_INTERFACE_HPP

#ifdef POLYMPC_FOUND_QPMAD

#include "qp_base.hpp"
#include "qpmad/solver.h"

namespace polympc {

template<int N, int M, typename Scalar = double, int MatrixType = DENSE>
class QPMAD : public QPBase<QPMAD<N, M, Scalar, MatrixType>, N, M, Scalar, MatrixType>
{
    using Base = QPBase<QPMAD<N, M, Scalar, MatrixType>, N, M, Scalar, MatrixType>;
    using qp_var_t        = typename Base::qp_var_t;
    using qp_dual_t       = typename Base::qp_dual_t;
    using qp_dual_a_t     = typename Base::qp_dual_a_t;
    using qp_constraint_t = typename Base::qp_constraint_t;
    using qp_hessian_t    = typename Base::qp_hessian_t;
    using scalar_t        = typename Base::scalar_t;

public:
    /** constructor */
    QPMAD() : Base()
    {
        EIGEN_STATIC_ASSERT(MatrixType == DENSE, "QPMAD_Interface: QPMAD does not support sparse matrices \n");
        EIGEN_STATIC_ASSERT((std::is_same<Scalar, double>::value == true), "QPMAD_Interface: QPMAD only supports 'double' precision \n");

        set_qpmad_settings();
    }
    ~QPMAD() = default;

    void set_qpmad_settings() noexcept
    {
        /** set the settings */
        qpmad_parameters.max_iter_     = this->m_settings.max_iter;
        qpmad_parameters.tolerance_    = this->m_settings.eps_abs;
        qpmad_parameters.hessian_type_ = (qpmad::SolverParameters::HessianType)this->m_settings.hessian_type;
    }

    /** solve */
    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
    {
        set_qpmad_settings();

        /** @bug : temporary fix until the new interface is released */
        qpmad::QPVector primal  = Eigen::Map<qpmad::QPVector>(this->m_x.data(), this->m_x.rows());
        qpmad::QPMatrix hessian(H); //Eigen::Map<qpmad::QPMatrix>(H.derived().data(), N, N);
        qpmad::QPVector h_(h);
        qpmad::QPMatrix A_(A);
        qpmad::QPVector alb(Alb);
        qpmad::QPVector aub(Aub);
        qpmad::QPVector xlb_(xlb);
        qpmad::QPVector xub_(xub);

        //qpmad::Solver::ReturnStatus status = qpmad_solver.solve(primal, hessian, h, xlb, xub, A, Alb, Aub, qpmad_parameters);
        typename solver_t::ReturnStatus status = qpmad_solver.solve(primal, hessian, h_, xlb_, xub_, A_, alb, aub, qpmad_parameters);

        //Eigen::VectorXd dual;
        //Eigen::Matrix<qpmad::MatrixIndex, Eigen::Dynamic, 1> indices;
        //Eigen::Matrix<bool, Eigen::Dynamic, 1> is_lower;
        //qpmad_solver.getInequalityDual(dual, indices, is_lower);

        //std::cout << "QPMAD number of inequality duals: " << qpmad_solver.getNumberOfInequalityIterations() << "\n";
        //std::cout << "QPMAD duals:" << dual.transpose() << "\n";
        //std::cout << "QPMAD indices: " << indices.transpose() << "\n";
        //std::cout << "QPMAD is_lower: " << is_lower.transpose() << "\n";

        // temporary solution
        this->m_x = primal;

        switch (status)
        {
            case solver_t::ReturnStatus::OK : {this->m_info.status = status_t::SOLVED; return status_t::SOLVED;}
            case solver_t::ReturnStatus::MAXIMAL_NUMBER_OF_ITERATIONS : {this->m_info.status = status_t::MAX_ITER_EXCEEDED; return status_t::MAX_ITER_EXCEEDED;}
            default: return status_t::UNSOLVED;
        }
    }

    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub,
                        const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        set_qpmad_settings();
        this->m_x = x_guess;

        /** @bug : temporary fix until the new interface is released */
        qpmad::QPVector primal  = Eigen::Map<qpmad::QPVector>(this->m_x.data(), this->m_x.rows());
        qpmad::QPMatrix hessian(H);
        qpmad::Solver::ReturnStatus status = qpmad_solver.solve(primal, hessian, h, xlb, xub, A, Alb, Aub, qpmad_parameters);

        this->m_x = primal;

        switch (status)
        {
            case qpmad::Solver::OK : {this->m_info.status = status_t::SOLVED; return status_t::SOLVED;}
            case qpmad::Solver::MAXIMAL_NUMBER_OF_ITERATIONS : {this->m_info.status = status_t::MAX_ITER_EXCEEDED; return status_t::MAX_ITER_EXCEEDED;}
            default: return status_t::UNSOLVED;
        }
    }


private:
    //template <typename t_Scalar, int t_primal_size, int t_has_bounds, int t_general_ctr_number>
    using solver_t = qpmad::SolverTemplate<Scalar, N, 1, M>;
    solver_t qpmad_solver;
    qpmad::SolverParameters qpmad_parameters;
};

} // polympc namespace

#endif // POLYMPC_FOUND_QPMAD

#endif // QPMAD_INTERFACE_HPP
