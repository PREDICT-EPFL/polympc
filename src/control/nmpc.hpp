#ifndef NMPC_HPP
#define NMPC_HPP

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/unsupported/Eigen/KroneckerProduct"
#include "eigen3/unsupported/Eigen/AutoDiff"

namespace polympc
{

template <typename Dynamics, typename Polynomial, int NumSegments = 1>
class ode_collocation
{
public:

    using Scalar = typename Dynamics::Scalar;
    using diff_mat_t = typename Polynomial::diff_mat_t;
    using weights_t  = typename Polynomial::q_weights_t;
    using nodes_t    = typename Polynomial::nodes_t;

    enum
    {
        NX = Dynamics::State::RowsAtCompileTime,
        NU = Dynamics::Control::RowsAtCompileTime,
        NP = Dynamics::Parameters::RowsAtCompileTime,
        POLY_ORDER = Polynomial::POLY_ORDER,
        NUM_NODES = POLY_ORDER + 1,

        VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX,
        VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU,
        VARP_SIZE = NP
    };

    /** composite differentiation matrix */
    using comp_diff_mat_t = Eigen::Matrix<Scalar, VARX_SIZE, VARX_SIZE>;

    ode_collocation(const Dynamics &_f){}
    ode_collocation();
    ~ode_collocation(){}

    /** type to store optimization variable var = [x, u, p] */
    using var_t     = Eigen::Matrix<Scalar, VARX_SIZE + VARU_SIZE + VARP_SIZE, 1>;
    using constr_t  = Eigen::Matrix<Scalar, VARX_SIZE, 1>;
    void operator() (const var_t &var, constr_t &constr_value,
                     const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1) ) const;

    /** linearized approximation */
    using jacobian_t = Eigen::Matrix<Scalar, constr_t::RowsAtCompileTime, var_t::RowsAtCompileTime>;
    using Index = typename jacobian_t::Index;
    using local_jacobian_t = Eigen::Matrix<Scalar, NX, NX + NU>;
    using localIndex = typename local_jacobian_t::Index;
    using Derivatives = Eigen::Matrix<Scalar, NX + NU, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    /** AD variables */
    using ADx_t = Eigen::Matrix<ADScalar, NX, 1>;
    using ADu_t = Eigen::Matrix<ADScalar, NU, 1>;
    ADx_t m_ADx, m_ADy;
    ADu_t m_ADu;

    void linearized(const var_t &var, jacobian_t &A, constr_t &b,
                    const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1));

    void initialize_derivatives();

public:
    Dynamics m_f;
    Polynomial m_basis_f;

    comp_diff_mat_t m_DiffMat = comp_diff_mat_t::Zero();
    Eigen::SparseMatrix<Scalar> m_SpDiffMat;

    void compute_diff_matrix();
};




template <typename Dynamics, typename Polynomial, int NumSegments>
ode_collocation<Dynamics, Polynomial, NumSegments>::ode_collocation()
{
    compute_diff_matrix();
    m_SpDiffMat = m_DiffMat.sparseView(); //Not for embedded systems

    initialize_derivatives();
}


template <typename Dynamics, typename Polynomial, int NumSegments>
void ode_collocation<Dynamics, Polynomial, NumSegments>::compute_diff_matrix()
{
    diff_mat_t D = m_basis_f.D();
    Eigen::Matrix<Scalar, NX, NX> E = Eigen::Matrix<Scalar, NX, NX>::Identity();

    if(NumSegments < 2)
    {
        m_DiffMat = Eigen::kroneckerProduct(D,E);
        return;
    }
    else
    {
        Eigen::Matrix<Scalar, NumSegments * POLY_ORDER + 1, NumSegments * POLY_ORDER + 1> DM =
                Eigen::Matrix<Scalar, NumSegments * POLY_ORDER + 1, NumSegments * POLY_ORDER + 1>::Zero();
        DM.template bottomRightCorner<NUM_NODES, NUM_NODES>() = D;
        for(int k = 0; k < (NumSegments - 1) * POLY_ORDER; k += POLY_ORDER)
            DM.template block<NUM_NODES - 1, NUM_NODES>(k, k) = D.template topLeftCorner<NUM_NODES - 1, NUM_NODES>();

        m_DiffMat = Eigen::kroneckerProduct(DM,E);

        return;
    }
}

/** Evaluate differential constraint */
template <typename Dynamics, typename Polynomial, int NumSegments>
void ode_collocation<Dynamics, Polynomial, NumSegments>::operator()(const var_t &var, constr_t &constr_value,
                                                                    const Scalar &t0, const Scalar &tf) const
{
    constr_t value;
    Eigen::Matrix<Scalar, NX, 1> f_res;
    Scalar t_scale = (tf - t0) / (2 * NumSegments);

    int n = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        m_f(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
            var.template segment<NP>(VARX_SIZE + VARU_SIZE), f_res);

        value. template segment<NX>(k) = f_res;
        n += NU;
    }

    constr_value = m_DiffMat * var.template head<VARX_SIZE>() - t_scale * value;
}

template <typename Dynamics, typename Polynomial, int NumSegments>
void ode_collocation<Dynamics, Polynomial, NumSegments>::initialize_derivatives()
{
    int deriv_num = NX + NU;
    int deriv_idx = 0;

    for(int i = 0; i < NX; i++)
    {
        m_ADx[i].derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
    for(int i = 0; i < NU; i++)
    {
        m_ADu(i).derivatives() = Derivatives::Unit(deriv_num, deriv_idx);
        deriv_idx++;
    }
}


/** compute linearization of diferential constraints */
template <typename Dynamics, typename Polynomial, int NumSegments>
void ode_collocation<Dynamics, Polynomial, NumSegments>::linearized(const var_t &var, jacobian_t &A, constr_t &b,
                                                                    const Scalar &t0, const Scalar &tf)
{
    A = jacobian_t::Zero();
    /** compute jacoabian of dynamics */
    local_jacobian_t jac;

    constr_t value;
    Scalar t_scale = (tf - t0) / (2 * NumSegments);

    /** initialize AD veriables */
    int n = 0;
    for(int k = 0; k < VARX_SIZE; k += NX)
    {
        /** @note is it an embedded cast ??*/
        for(int i = 0; i < NX; i++)
            m_ADx(i).value() = var.template segment<NX>(k)(i);

        for(int i = 0; i < NU; i++)
            m_ADu(i).value() = var.template segment<NU>(n + VARX_SIZE)(i);

        m_f(m_ADx, m_ADu,
            var.template segment<NP>(VARX_SIZE + VARU_SIZE), m_ADy);

        /** compute value and first derivatives */
        for(int i = 0; i< NX; i++)
        {
            value. template segment<NX>(k)(i) = m_ADy(i).value();
            jac.row(i) = m_ADy(i).derivatives();
        }

        /** insert block jacobian */
        A.template block<NX, NX>(k, k) = -t_scale * jac.template leftCols<NX>();
        A.template block<NX, NU>(k, n + VARX_SIZE) = -t_scale * jac.template rightCols<NU>();

        n += NU;
    }

    b = m_DiffMat * var.template head<VARX_SIZE>() - t_scale * value;

    A.template leftCols<VARX_SIZE>() = m_DiffMat + A.template leftCols<VARX_SIZE>();
}





}
#endif // NMPC_HPP
