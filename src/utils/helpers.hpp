#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"

enum
{
    DENSE  = 0,
    SPARSE = 1
} MEMORY;

template<typename Scalar, int Rows, int Cols>
struct dense_matrix_type_selector
{
    enum { allocate_dynamic = (Rows == Eigen::Dynamic) || (Cols == Eigen::Dynamic) ? 1 : 0,
           allocate_static =  (not allocate_dynamic) && (Rows * Cols * sizeof (Scalar) < EIGEN_STACK_ALLOCATION_LIMIT) ? 1 : 0,
           cols = (Cols == 1) ? 1 : Eigen::Dynamic};
    using type = typename std::conditional<allocate_static, Eigen::Matrix<Scalar, Rows, Cols>,
                                                            Eigen::Matrix<Scalar, Eigen::Dynamic, cols>>::type;
};


/** traits to choose the default linear solver for sparse and dense implementation */
template<int type>
struct linear_solver_traits;

template<>
struct linear_solver_traits<DENSE>
{
    template<typename Type, int Flags>
    using default_solver = typename Eigen::LDLT<Type, Flags>;
};

template<>
struct linear_solver_traits<SPARSE>
{
    template<typename Type, int Flags>
    using default_solver = typename Eigen::SimplicialLDLT<Type, Flags>;
};


/** get rid of unused variables warnings*/
namespace polympc
{
    template<typename T>
    EIGEN_STRONG_INLINE void ignore_unused_var(const T& ) noexcept {}
}




#endif // HELPERS_HPP
