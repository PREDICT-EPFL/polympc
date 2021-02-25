#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <chrono>

enum
{
    DENSE  = 0,
    SPARSE = 1
} MEMORY;

template<typename Scalar, int Rows, int Cols>
struct dense_matrix_type_selector
{
    enum { allocate_dynamic = (Rows == Eigen::Dynamic) || (Cols == Eigen::Dynamic) ? 1 : 0,
           allocate_static =  (!allocate_dynamic) && (Rows * Cols * sizeof (Scalar) < EIGEN_STACK_ALLOCATION_LIMIT) ? 1 : 0,
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
    // supress 'unused variable' warnings
    template<typename T>
    EIGEN_STRONG_INLINE void ignore_unused_var(const T& ) noexcept {}

    typedef std::chrono::time_point<std::chrono::system_clock> time_point;
    static time_point get_time()
    {
        /** OS dependent */
    #ifdef __APPLE__
        return std::chrono::system_clock::now();
    #elif defined _WIN32 || defined _WIN64 || defined _MSC_VER
        return std::chrono::system_clock::now();
    #else
        return std::chrono::high_resolution_clock::now();
    #endif
    }
}




#endif // HELPERS_HPP
