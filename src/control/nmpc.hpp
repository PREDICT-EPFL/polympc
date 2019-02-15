#ifndef NMPC_HPP
#define NMPC_HPP

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/unsupported/Eigen/KroneckerProduct"

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





}
#endif // NMPC_HPP
