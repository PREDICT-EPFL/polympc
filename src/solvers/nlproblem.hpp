#ifndef NLPROBLEM_HPP
#define NLPROBLEM_HPP

#include "utils/helpers.hpp"
#include "autodiff/AutoDiffScalar.h"

#include <iostream>

template<int I, int U>
struct more
{
    static constexpr bool ret = I > U ? true : false;
};


/** define the macro for forward declarations */
#define POLYMPC_FORWARD_NLP_DECLARATION( cNAME, cNX, cNE, cNI, cNP, TYPE ) \
class cNAME;                                        \
template<>                                          \
struct nlp_traits<cNAME>                            \
{                                                   \
public:                                             \
    using Scalar = TYPE;                            \
    enum { NX = cNX, NE = cNE, NI = cNI, NP = cNP}; \
};                                                  \

/** define derived class traits */
template<typename Derived> struct nlp_traits;
template<typename T> struct nlp_traits<const T> : nlp_traits<T> {};

/** forward declare base class */
template<typename Derived, int MatrixFormat> class ProblemBase;

template<typename Derived, int MatrixFormat = DENSE>
class ProblemBase
{
public:

    ProblemBase()
    {
        seed_derivatives();
    }

    ~ProblemBase() = default;


    enum
    {
        /** problem dimensions */
        VAR_SIZE  = nlp_traits<Derived>::NX,
        NUM_EQ    = nlp_traits<Derived>::NE,
        NUM_INEQ  = nlp_traits<Derived>::NI,
        NUM_BOX   = nlp_traits<Derived>::NX,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,

        /** Various flags */
        is_sparse = (MatrixFormat == SPARSE) ? 1 : 0,
        is_dense  = is_sparse ? 0 : 1,
        MATRIXFMT = MatrixFormat
    };

    using scalar_t = typename nlp_traits<Derived>::Scalar;

    /** optimisation variable */
    template<typename T>
    using variable_t = Eigen::Matrix<T, VAR_SIZE, 1>;

    template<typename T>
    using constraint_t = Eigen::Matrix<T, NUM_EQ + NUM_INEQ, 1>;

    /** parameters */
    template<typename T>
    using parameter_t = Eigen::Matrix<T, nlp_traits<Derived>::NP, 1>;

    /** AD variables */
    using derivatives_t = Eigen::Matrix<scalar_t, VAR_SIZE, 1>;
    using ad_scalar_t = Eigen::AutoDiffScalar<derivatives_t>;
    using ad_var_t    = Eigen::Matrix<ad_scalar_t, VAR_SIZE, 1>;
    using ad_eq_t     = Eigen::Matrix<ad_scalar_t, NUM_EQ, 1>;
    ad_var_t m_ad_var, m_ad_y;
    ad_eq_t m_ad_eq;
    ad_scalar_t m_ad_cost;

    using ad2_scalar_t = Eigen::AutoDiffScalar<ad_var_t>;
    Eigen::Matrix<ad2_scalar_t, VAR_SIZE, 1> m_ad2_var;
    ad2_scalar_t m_ad2_cost;

    /** seed edrivatives */
    void seed_derivatives();

    /** NLP variables */
    using nlp_variable_t    = typename dense_matrix_type_selector<scalar_t, VAR_SIZE, 1>::type;
    using nlp_constraints_t = typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, 1>::type;
    // choose to allocate sparse or dense jacoabian and hessian
    using nlp_eq_jacobian_t = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>::type>::type;
    using nlp_hessian_t     = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, VAR_SIZE, VAR_SIZE>::type>::type;
    using nlp_cost_t        = scalar_t;
    using nlp_dual_t        = typename dense_matrix_type_selector<scalar_t, DUAL_SIZE, 1>::type;
    using static_parameter_t = typename dense_matrix_type_selector<scalar_t, nlp_traits<Derived>::NP, 1>::type;

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        static_cast<const Derived*>(this)->cost_impl(x, p, cost);
    }

    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept
    {
        this->cost<scalar_t>(var,p,cost);
    }

    // default empty implementation
    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {}

    /** @brief
     *
     */
    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                  Eigen::Ref<constraint_t<T>> constraint) const noexcept
    {
        static_cast<const Derived*>(this)->equality_constraints_impl(x, p, constraint);
    }

    // default implementation
    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<constraint_t<T>> constraint) const noexcept
    {}

    /**  NLP interface functions */
    EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                            scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept;

    EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                   scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept;

    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                         Eigen::Ref<nlp_constraints_t> equalities) const noexcept;

    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE < 1 >::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                            const Eigen::Ref<const static_parameter_t>& p,
                                                                            Eigen::Ref<nlp_constraints_t> equalities,
                                                                            Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        jacobian   = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
        equalities = nlp_constraints_t::Zero(NUM_EQ);

        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
    }

    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE >= 1 >::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                            const Eigen::Ref<const static_parameter_t>& p,
                                                                            Eigen::Ref<nlp_constraints_t> equalities,
                                                                            Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        jacobian   = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
        equalities = nlp_constraints_t::Zero(NUM_EQ);

        m_ad_var = var;
        equality_constraints<ad_scalar_t>(m_ad_var, p, m_ad_eq);

        /** compute value and first derivatives */
        for(int i = 0; i < NUM_EQ; i++)
        {
            equalities(i) = m_ad_eq(i).value();
            jacobian.row(i) = m_ad_eq(i).derivatives();
        }
    }

    EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                        const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept;

    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept;

    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                 Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_eq_jacobian_t> jac_g) noexcept;

    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var,const Eigen::Ref<const static_parameter_t> &p,
                                                         const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                         Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                         Eigen::Ref<nlp_variable_t> cost_gradient,
                                                         Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_eq_jacobian_t> jac_g) noexcept;


};

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::seed_derivatives()
{
    const int deriv_num = VAR_SIZE;
    int deriv_idx = 0;

    for(int i = 0; i < VAR_SIZE; i++)
    {
        m_ad_var(i).derivatives()  = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_var(i).derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        m_ad2_var(i).value().derivatives() = derivatives_t::Unit(deriv_num, deriv_idx);
        for(int idx = 0; idx < deriv_num; idx++)
        {
            m_ad2_var(i).derivatives()(idx).derivatives()  = derivatives_t::Zero();
        }
        deriv_idx++;
    }
}

// compute cost gradient
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                       scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
{
    _cost = scalar_t(0);
    cost_gradient = nlp_variable_t::Zero(VAR_SIZE);

    m_ad_cost = ad_scalar_t(0);
    m_ad_var = var;
    cost<ad_scalar_t>(m_ad_var, p, m_ad_cost);
    _cost += m_ad_cost.value();
    cost_gradient.noalias() += m_ad_cost.derivatives();
}

//compute cost, gradient and hessian
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var,
                                                               const Eigen::Ref<const static_parameter_t>& p,
                                                               scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient,
                                                               Eigen::Ref<nlp_hessian_t> hessian) noexcept
{
    _cost = scalar_t(0);
    _cost_gradient = nlp_variable_t::Zero(VAR_SIZE);
    hessian  = nlp_hessian_t::Zero(VAR_SIZE, VAR_SIZE);

    //Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE> hes = Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE> ::Zero();
    m_ad2_cost.value().value() = 0;

    // set variable values
    for(int i = 0; i < VAR_SIZE; i++)
        m_ad2_var(i).value().value() = var(i);

    // compute cost, gradient, hessian
    cost<ad2_scalar_t>(m_ad2_var, p, m_ad2_cost);
    _cost = m_ad2_cost.value().value();
    _cost_gradient =  m_ad2_cost.value().derivatives();

    for(int i = 0; i < VAR_SIZE; ++i)
    {
        hessian.col(i) = m_ad2_cost.derivatives()(i).derivatives();
    }
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                    const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept
{
    /** create temporary */
    nlp_constraints_t g;
    this->cost(var, p, _lagrangian);
    this->equalities(var, p, g);
    _lagrangian += g.dot(lam.template head<NUM_EQ>());
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                             const Eigen::Ref<const static_parameter_t>& p,
                                                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                             Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept
{
    nlp_constraints_t g;
    nlp_eq_jacobian_t jac_g;
    this->cost_gradient(var, p, _lagrangian, _lag_gradient);
    this->equalities_linerised(var, p, g, jac_g);
    _lagrangian += g.dot(lam.template head<NUM_EQ>());
    /** @badcode: replace with block products ???*/
    _lag_gradient.noalias() += jac_g.transpose() * lam.template head<NUM_EQ>();
    _lag_gradient += lam.template tail<VAR_SIZE>();
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                             const Eigen::Ref<const static_parameter_t>& p,
                                                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                             Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                             Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_eq_jacobian_t> jac_g) noexcept
{
    this->cost_gradient(var, p, _lagrangian, cost_gradient);
    this->equalities_linearised(var, p, g, jac_g);
    _lagrangian += g.dot(lam.template head<NUM_EQ>());
    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();
}

template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var,
                                                                     const Eigen::Ref<const static_parameter_t> &p,
                                                                     const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                                     Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                                     Eigen::Ref<nlp_variable_t> cost_gradient,
                                                                     Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_eq_jacobian_t> jac_g) noexcept
{
    this->cost_gradient_hessian(var, p, _lagrangian, cost_gradient, lag_hessian);
    this->equalities_linearised(var, p, g, jac_g);
    _lagrangian += g.dot(lam.template head<NUM_EQ>());

    /** @badcode: replace with block products ???*/
    lag_gradient.noalias() = jac_g.transpose() * lam.template head<NUM_EQ>();
    lag_gradient += cost_gradient;
    lag_gradient += lam.template tail<NUM_BOX>();

    /** hessian part */
    Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE> hes = Eigen::Matrix<scalar_t, VAR_SIZE, VAR_SIZE>::Zero();
    Eigen::Matrix<ad2_scalar_t, NUM_EQ, 1> ad2_xdot;

    for(int i = 0; i < VAR_SIZE; i++)
        m_ad2_var(i).value().value() = var(i);

    equality_constraints<ad2_scalar_t>(m_ad2_var, p, ad2_xdot);

    for(int n = 0; n < NUM_EQ; n++)
    {
        for(int i = 0; i < VAR_SIZE; ++i)
        {
            hes.col(i) = ad2_xdot(n).derivatives()(i).derivatives();
        }
        // do we really need it?
        hes.transposeInPlace();

        lag_hessian.noalias() += lam(n) * hes;
    }
}


// evaluate constraints
template<typename Derived, int MatrixFormat>
void ProblemBase<Derived, MatrixFormat>::equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                    Eigen::Ref<nlp_constraints_t> _equalities) const noexcept
{
    equality_constraints<scalar_t>(var, p, _equalities);
}



#endif // NLPROBLEM_HPP
