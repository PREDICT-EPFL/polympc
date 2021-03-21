#ifndef MPC_WRAPPER_HPP
#define MPC_WRAPPER_HPP

#include "utils/helpers.hpp"
#include "polynomials/splines.hpp"
#include <iostream>

/** interface/aggregate class for MPC controllers */
template<typename OCP, template<typename, typename ...Args> class Solver, typename ...Args>
class MPC
{

private:
    using nlp_solver_t = Solver<OCP, Args...>;
    nlp_solver_t m_solver;

    /** dynamic system properties */
    static constexpr int nx = OCP::NX;
    static constexpr int nu = OCP::NU;
    static constexpr int np = OCP::NP;
    static constexpr int nd = OCP::ND;

    /** optimisation properties */
    static constexpr int var_size  = OCP::VAR_SIZE;
    static constexpr int varx_size = OCP::VARX_SIZE;
    static constexpr int varu_size = OCP::VARU_SIZE;
    static constexpr int dual_size = OCP::DUAL_SIZE;
    static constexpr int num_nodes = OCP::NUM_NODES;
    static constexpr int num_segms = OCP::NUM_SEGMENTS;

public:
    MPC()
    {
        // initialise time nodes and Lagrange interpolator
        time_grid = this->ocp().time_nodes.reverse();
        time_nodes = time_grid.head(time_grid.rows());
        polympc::LagrangeSpline::compute_lagrange_basis(time_nodes, m_basis);
        sgm_length = (m_solver.get_problem().t_stop - m_solver.get_problem().t_start) / num_segms;
        nodes_per_segm = std::floor(num_nodes / num_segms);
    }
    ~MPC() = default;

    using scalar_t     = typename OCP::scalar_t;
    using state_t      = typename dense_matrix_type_selector<scalar_t, nx, 1>::type;
    using control_t    = typename dense_matrix_type_selector<scalar_t, nu, 1>::type;
    using parameter_t  = typename dense_matrix_type_selector<scalar_t, np, 1>::type;
    using static_param = typename dense_matrix_type_selector<scalar_t, nd, 1>::type;

    using traj_state_t   = Eigen::Matrix<scalar_t, varx_size, 1>;
    using traj_control_t = Eigen::Matrix<scalar_t, varu_size, 1>;
    using dual_var_t     = Eigen::Matrix<scalar_t, dual_size, 1>;

    /** data for interpolation */
    typename OCP::time_t  time_grid;
    typename OCP::nodes_t time_nodes;
    using interpolation_basis_t = Eigen::Matrix<scalar_t, OCP::nodes_t::RowsAtCompileTime, OCP::nodes_t::RowsAtCompileTime>;
    interpolation_basis_t m_basis;
    scalar_t sgm_length{1.0};
    int nodes_per_segm{1};

    /** set MPC optmisation limits */
    inline void set_time_limits(const scalar_t& t0, const scalar_t& tf) noexcept {m_solver.get_problem().set_time_limits(t0, tf);}

    /** set initial conditions bounds */
    inline void initial_conditions(const Eigen::Ref<const state_t>& x0) noexcept
    {
        m_solver.upper_bound_x().template segment<nx>(varx_size - nx) = x0;
        m_solver.lower_bound_x().template segment<nx>(varx_size - nx) = x0;
    }
    inline void initial_conditions(const Eigen::Ref<const state_t>& x0_lb,
                                   const Eigen::Ref<const state_t>& x0_ub) noexcept
    {
        m_solver.upper_bound_x().template segment<nx>(varx_size - nx) = x0_ub;
        m_solver.lower_bound_x().template segment<nx>(varx_size - nx) = x0_lb;
    }

    /** set state and control bounds */
    // state
    inline void x_lower_bound(const Eigen::Ref<const state_t>& xlb) noexcept
    {
        m_solver.lower_bound_x().template head<varx_size - nx>() = xlb.replicate(num_nodes - 1, 1);
    }
    inline void x_upper_bound(const Eigen::Ref<const state_t>& xub) noexcept
    {
        m_solver.upper_bound_x().template head<varx_size - nx>() = xub.replicate(num_nodes - 1, 1);
    }
    inline void state_bounds(const Eigen::Ref<const state_t>& xlb,
                             const Eigen::Ref<const state_t>& xub) noexcept
    {
        m_solver.lower_bound_x().template head<varx_size - nx>() = xlb.replicate(num_nodes - 1, 1);
        m_solver.upper_bound_x().template head<varx_size - nx>() = xub.replicate(num_nodes - 1, 1);
    }
    inline void state_trajectory_bounds(const Eigen::Ref<const traj_state_t>& xlb,
                                        const Eigen::Ref<const traj_state_t>& xub) noexcept
    {
        m_solver.lower_bound_x().template head<varx_size>() = xlb;
        m_solver.upper_bound_x().template head<varx_size>() = xub;
    }

    // control
    inline void u_lower_bound(const Eigen::Ref<const control_t>& lb) noexcept
    {
        m_solver.lower_bound_x().template segment<varu_size>(varx_size) = lb.replicate(num_nodes, 1);
    }
    inline void u_upper_bound(const Eigen::Ref<const control_t>& ub) noexcept
    {
        m_solver.upper_bound_x().template segment<varu_size>(varx_size) = ub.replicate(num_nodes, 1);
    }
    inline void control_trajecotry_bounds(const Eigen::Ref<const traj_control_t>& lb,
                                          const Eigen::Ref<const traj_control_t>& ub) noexcept
    {
        m_solver.lower_bound_x().template segment<varu_size>(varx_size) = lb;
        m_solver.upper_bound_x().template segment<varu_size>(varx_size) = ub;
    }
    inline void control_bounds(const Eigen::Ref<const control_t>& lb,
                               const Eigen::Ref<const control_t>& ub) noexcept
    {
        m_solver.lower_bound_x().template segment<varu_size>(varx_size) = lb.replicate(num_nodes, 1);
        m_solver.upper_bound_x().template segment<varu_size>(varx_size) = ub.replicate(num_nodes, 1);
    }

    /** set parameters bounds*/
    inline void parameters_bounds(const Eigen::Ref<const parameter_t>& lbp, const Eigen::Ref<const parameter_t>& ubp) noexcept
    {
        m_solver.lower_bound_x().template tail<np>() = lbp;
        m_solver.upper_bound_x().template tail<np>() = ubp;
    }

    /** set static parameters */
    inline void set_static_parameters(const Eigen::Ref<const static_param>& param) noexcept
    {
        m_solver.parameters() = param;
    }

    /** set initial guess */
    inline void x_guess(const Eigen::Ref<const traj_state_t>& x_guess) noexcept
    {
        m_solver.m_x.template head<varx_size>() = x_guess;
    }
    inline void u_guess(const Eigen::Ref<const traj_control_t>& u_guess) noexcept
    {
        m_solver.m_x.template segment<varu_size>(varx_size) = u_guess;
    }
    inline void lam_guess(const Eigen::Ref<const dual_var_t>& lam_guess) noexcept
    {
        m_solver.m_lam = lam_guess;
    }

    /** get and set solver settings | info | problem */
    EIGEN_STRONG_INLINE const typename nlp_solver_t::nlp_settings_t& settings() const noexcept { return m_solver.m_settings; }
    EIGEN_STRONG_INLINE typename nlp_solver_t::nlp_settings_t& settings() noexcept { return m_solver.m_settings; }

    EIGEN_STRONG_INLINE const typename nlp_solver_t::qp_solver_t::settings_t& qp_settings() const noexcept { return m_solver.m_qp_solver.m_settings; }
    EIGEN_STRONG_INLINE typename nlp_solver_t::qp_solver_t::settings_t& qp_settings() noexcept { return m_solver.m_qp_solver.m_settings; }

    EIGEN_STRONG_INLINE const typename nlp_solver_t::nlp_info_t& info() const noexcept { return m_solver.m_info; }
    EIGEN_STRONG_INLINE typename nlp_solver_t::nlp_info_t& info() noexcept { return m_solver.m_info; }

    EIGEN_STRONG_INLINE const nlp_solver_t& solver() const noexcept { return m_solver; }
    EIGEN_STRONG_INLINE nlp_solver_t& solver() noexcept { return m_solver; }

    EIGEN_STRONG_INLINE const OCP& ocp() const noexcept { return m_solver.get_problem(); }
    EIGEN_STRONG_INLINE OCP& ocp() noexcept { return m_solver.get_problem(); }   

    /** get the solution */
    // state
    inline traj_state_t solution_x() const noexcept
    {
        traj_state_t tmp = m_solver.primal_solution().template head<varx_size>();
        return tmp;
    }
    inline scalar_t* solution() noexcept
    {
        return m_solver.primal_solution().template head<varx_size>().data();
    }
    inline Eigen::Matrix<scalar_t, nx, num_nodes> solution_x_reshaped() const noexcept
    {
        traj_state_t opt_x = m_solver.primal_solution().template head<varx_size>();
        return Eigen::Map<Eigen::Matrix<scalar_t, nx, num_nodes>>(opt_x.data(), nx, num_nodes).rowwise().reverse();
    }
    inline state_t solution_x_at(const int &k) const noexcept
    {
        return m_solver.primal_solution().template segment<nx>(varx_size - (k + 1) * nx);
    }
    inline state_t solution_x_at(const scalar_t& t) const noexcept
    {
        auto x_reshaped = solution_x_reshaped();
        Eigen::Index idx = (Eigen::Index)std::floor( t / sgm_length);
        idx = std::max(Eigen::Index(0), std::min(idx, (Eigen::Index)num_segms - 1)); // clip idx to stay within the spline bounds

        state_t state = polympc::LagrangeSpline::eval(t - idx * sgm_length, x_reshaped.template block<nx,
                                                      OCP::nodes_t::RowsAtCompileTime>(0, idx * nodes_per_segm), m_basis);

        return state;
    }

    // control
    inline traj_control_t solution_u() const noexcept
    {
        return m_solver.primal_solution().template segment<varu_size>(varx_size);
    }
    inline Eigen::Matrix<scalar_t, nu, num_nodes> solution_u_reshaped() const noexcept
    {
        traj_control_t opt_u = m_solver.primal_solution().template segment<varu_size>(varx_size);
        return Eigen::Map<Eigen::Matrix<scalar_t, nu, num_nodes>>(opt_u.data(), nu, num_nodes).rowwise().reverse();
    }
    inline control_t solution_u_at(const int &k) const noexcept
    {
        return m_solver.primal_solution().template segment<nu>(varx_size + varu_size - (k + 1) * nu);
    }
    inline control_t solution_u_at(const scalar_t& t) const noexcept
    {
        auto u_reshaped = solution_u_reshaped();
        Eigen::Index idx = (Eigen::Index)std::floor( t / sgm_length);
        idx = std::max(Eigen::Index(0), std::min(idx, (Eigen::Index)num_segms - 1)); // clip idx to stay within the spline bounds

        control_t state = polympc::LagrangeSpline::eval(t - idx * sgm_length, u_reshaped.template block<nu,
                                                      OCP::nodes_t::RowsAtCompileTime>(0, idx * nodes_per_segm), m_basis);

        return state;
    }


    // parameters
    inline parameter_t solution_p() const noexcept
    {
        return m_solver.primal_solution().template tail<np>();
    }

    /** solve optimal control problem */
    inline void solve() noexcept {m_solver.solve();}

};






#endif // MPC_WRAPPER_HPP
