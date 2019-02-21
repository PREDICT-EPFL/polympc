#ifndef COST_COLLOCATION_HPP
#define COST_COLLOCATION_HPP

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/unsupported/Eigen/AutoDiff"
#include <type_traits>
#include <iostream>


/** ------------------------------- */
template <typename T, typename Dummy>
struct has_lagrange
{
    template <class, class>class checker;

    template <typename C> static std::true_type test(checker<C, decltype(&C::template operator()<Dummy, Dummy, Dummy, Dummy>)> *);
    template <typename C> static std::false_type test(...);

    using type = decltype(test<T>(nullptr));
    static constexpr bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

template <typename T, typename Dummy>
struct has_mayer
{
    template <class, class>class checker;

    template <typename C> static std::true_type test(checker<C, decltype(&C::template operator()<Dummy, Dummy>)> *);
    template <typename C> static std::false_type test(...);

    using type = decltype(test<T>(nullptr));
    static constexpr bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};


namespace polympc {

template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments = 1>
class cost_collocation
{
public:
    using Scalar = typename LagrangeTerm::Scalar;
    using weights_t  = typename Polynomial::q_weights_t;
    using nodes_t    = typename Polynomial::nodes_t;

    enum
    {
        NX = LagrangeTerm::State::RowsAtCompileTime,
        NU = LagrangeTerm::Control::RowsAtCompileTime,
        NP = LagrangeTerm::Parameters::RowsAtCompileTime,
        POLY_ORDER = Polynomial::POLY_ORDER,
        NUM_NODES = POLY_ORDER + 1,

        VARX_SIZE = (NumSegments * POLY_ORDER + 1) * NX,
        VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU,
        VARP_SIZE = NP,

        HAS_LAGRANGE = has_lagrange<LagrangeTerm, Scalar>::value,
        HAS_MAYER    = has_mayer<MayerTerm, Scalar>::value
    };

    /** type to store optimization variable var = [x, u, p] */
    using var_t     = Eigen::Matrix<Scalar, VARX_SIZE + VARU_SIZE + VARP_SIZE, 1>;
    void operator() (const var_t &var, Scalar &cost_value,
                     const Scalar &t0 = Scalar(-1), const Scalar &tf = Scalar(1) ) const;


    cost_collocation();
    cost_collocation(const LagrangeTerm &L, const MayerTerm &M){}
    ~cost_collocation(){}

private:
    LagrangeTerm m_Lagrange;
    MayerTerm m_Mayer;
    Polynomial m_basis_f;

    weights_t m_weights;
};

/** constructor */
template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::cost_collocation()
{
    std::cout << HAS_LAGRANGE << "\n";
    std::cout << HAS_MAYER << "\n";

    m_weights = m_basis_f.CCQWeights(); // change interface, probably integration weights? Both in Ebyshev and Legendre
}


template<typename LagrangeTerm, typename MayerTerm, typename Polynomial, int NumSegments>
void cost_collocation<LagrangeTerm, MayerTerm, Polynomial, NumSegments>::operator ()(const var_t &var, Scalar &cost_value,
                                                                                const Scalar &t0, const Scalar &tf) const
{
    cost_value = Scalar(0);

    if(HAS_MAYER)
    {
        Scalar Mayer = Scalar(0);
        m_Mayer(var. template head<NX>(), Mayer);
        cost_value += Mayer;
    }

    if(HAS_LAGRANGE)
    {
        Scalar t_scale = (tf - t0) / (2 * NumSegments);
        Scalar Lagrange = Scalar(0);
        int n = 0, it = 0;
        for(int k = 0; k < VARX_SIZE; k += NX)
        {
            m_Lagrange(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
                    var.template segment<NP>(VARX_SIZE + VARU_SIZE), Lagrange);

            cost_value += t_scale * m_weights[it % NUM_NODES] * Lagrange;
            if( ((it % NUM_NODES) == 0) && (it != 0))
                cost_value += t_scale * m_weights[it % NUM_NODES] * Lagrange; // add twice at the border points

            n += NU;
            it++;
        }
    }
}


// end of namespace
}

#endif // COST_COLLOCATION_HPP
