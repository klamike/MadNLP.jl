const SparseKKTStructure = @NamedTuple{
    n::Int, m::Int, n_slack::Int, n_tot::Int,
    n_jac::Int, n_hess::Int, nlb::Int, nub::Int,
    aug_vec_length::Int, aug_mat_length::Int,
    jac_sparsity_I::Vector{Int32},
    jac_sparsity_J::Vector{Int32},
    hess_sparsity_I::Vector{Int32},
    hess_sparsity_J::Vector{Int32},
    ind_ineq::Vector{Int},
}

"""
    SparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractReducedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

function _build_sparsekkt_structure(
    cb::SparseCallback{T,VT};
    hessian_approximation=ExactHessian,
) where {T,VT}

    n = cb.nvar
    m = cb.ncon
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)

    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    ind_ineq = cb.ind_ineq

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    return (
        n=n, m=m, n_slack=n_slack, n_tot=n_tot,
        n_jac=n_jac, n_hess=n_hess, nlb=nlb, nub=nub,
        aug_vec_length=aug_vec_length,
        aug_mat_length=aug_mat_length,
        jac_sparsity_I=jac_sparsity_I,
        jac_sparsity_J=jac_sparsity_J,
        hess_sparsity_I=hess_sparsity_I,
        hess_sparsity_J=hess_sparsity_J,
        ind_ineq=ind_ineq,
    )
end

function build_aug_indices!(
    I::AbstractVector{Int32},
    J::AbstractVector{Int32},
    structure::SparseKKTStructure,
)
    (; n, n_tot, n_jac, n_hess, n_slack, m,
       jac_sparsity_I, jac_sparsity_J,
       hess_sparsity_I, hess_sparsity_J, ind_ineq) = structure

    offset = n_tot + n_jac + n_slack + n_hess + m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    return I, J
end

function _build_sparsekkt_views(
    ::Type{VT},
    I::AbstractVector{Int32},
    J::AbstractVector{Int32},
    V::VT,
    structure::SparseKKTStructure,
) where {T, VT <: AbstractVector{T}}

    (; n, m, n_tot, n_jac, n_hess, n_slack, nlb, nub,
       aug_vec_length, jac_sparsity_I, jac_sparsity_J,
       hess_sparsity_I, hess_sparsity_J, ind_ineq) = structure

    pr_diag = _madnlp_unsafe_wrap(V, n_tot)
    du_diag = _madnlp_unsafe_wrap(V, m, n_jac + n_slack + n_hess + n_tot + 1)
    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot + 1)
    jac = _madnlp_unsafe_wrap(V, n_jac + n_slack, n_hess + n_tot + 1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess + n_tot + 1)

    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    aug_raw = SparseMatrixCOO(aug_vec_length, aug_vec_length, I, J, V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )
    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )

    return (
        pr_diag=pr_diag, du_diag=du_diag,
        reg=reg, l_diag=l_diag, u_diag=u_diag, l_lower=l_lower, u_lower=u_lower,
        hess=hess, jac=jac, jac_callback=jac_callback,
        aug_raw=aug_raw, jac_raw=jac_raw, hess_raw=hess_raw,
    )
end

function build_sparse_kkt_system(
    cb::SparseCallback{T,VT},
    I::AbstractVector{Int32},
    J::AbstractVector{Int32},
    V::VT,
    structure::SparseKKTStructure,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T, VT}

    views = _build_sparsekkt_views(VT, I, J, V, structure)

    aug_com, aug_csc_map = coo_to_csc(views.aug_raw)
    jac_com, jac_csc_map = coo_to_csc(views.jac_raw)
    hess_com, hess_csc_map = coo_to_csc(views.hess_raw)

    _linear_solver = linear_solver(aug_com; opt=opt_linear_solver)
    quasi_newton = create_quasi_newton(hessian_approximation, cb, structure.n; options=qn_options)

    return SparseKKTSystem(
        views.hess, views.jac_callback, views.jac, quasi_newton,
        views.reg, views.pr_diag, views.du_diag,
        views.l_diag, views.u_diag, views.l_lower, views.u_lower,
        views.aug_raw, aug_com, aug_csc_map,
        views.hess_raw, hess_com, hess_csc_map,
        views.jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        structure.ind_ineq, cb.ind_lb, cb.ind_ub,
    )
end

function create_kkt_system(
    ::Type{SparseKKTSystem},
    cb::SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T,VT}

    structure = _build_sparsekkt_structure(cb, ind_cons, hessian_approximation)

    I = create_array(cb, Int32, structure.aug_mat_length)
    J = create_array(cb, Int32, structure.aug_mat_length)
    V = VT(undef, structure.aug_mat_length)
    fill!(V, zero(T))

    build_aug_indices!(I, J, structure)

    return build_sparse_kkt_system(
        cb, ind_cons, I, J, V, structure, linear_solver;
        opt_linear_solver, hessian_approximation, qn_options,
    )
end

num_variables(kkt::SparseKKTSystem) = length(kkt.pr_diag)

function build_kkt!(kkt::SparseKKTSystem)
    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end
