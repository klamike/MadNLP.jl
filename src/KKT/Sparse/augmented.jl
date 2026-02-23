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
    aug_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Hessian
    hess_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

# Build KKT system directly from SparseCallback
function create_kkt_system(
    ::Type{SparseKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T,VT}
    structure = _build_sparsekkt_structure(cb; hessian_approximation=hessian_approximation, qn_options=qn_options)

    I = create_array(cb, Int32, structure.aug_mat_length)
    J = create_array(cb, Int32, structure.aug_mat_length)
    V = VT(undef, structure.aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    build_aug_indices!(I, J, structure)

    return build_sparse_kkt_system(cb, I, J, V, structure, linear_solver; opt_linear_solver=opt_linear_solver)
end

function _build_sparsekkt_structure(
    cb::SparseCallback{T,VT};
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T,VT}
    n = cb.nvar
    m = cb.ncon
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
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
        n=n, m=m, nlb=nlb, nub=nub,
        n_slack=n_slack, n_jac=n_jac, n_hess=n_hess, n_tot=n_tot,
        aug_vec_length=aug_vec_length, aug_mat_length=aug_mat_length,
        jac_sparsity_I=jac_sparsity_I, jac_sparsity_J=jac_sparsity_J,
        hess_sparsity_I=hess_sparsity_I, hess_sparsity_J=hess_sparsity_J,
        ind_ineq=ind_ineq,
        quasi_newton=quasi_newton,
    )
end

function build_aug_indices!(I, J, structure)
    (; n, n_slack, n_jac, n_hess, n_tot, m, ind_ineq,
       jac_sparsity_I, jac_sparsity_J, hess_sparsity_I, hess_sparsity_J) = structure

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
    return
end

function _build_sparsekkt_views(V::VT, structure) where {T, VT<:AbstractVector{T}}
    (; n_jac, n_hess, n_slack, n_tot, m) = structure

    pr_diag = _madnlp_unsafe_wrap(V, n_tot)
    du_diag = _madnlp_unsafe_wrap(V, m, n_jac+n_slack+n_hess+n_tot+1)
    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    return (pr_diag=pr_diag, du_diag=du_diag, hess=hess, jac=jac, jac_callback=jac_callback)
end

function build_sparse_kkt_system(
    cb::SparseCallback{T,VT},
    I, J, V::VT,
    structure,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
) where {T,VT}
    (; n, m, nlb, nub, n_slack, n_tot,
       aug_vec_length, ind_ineq,
       jac_sparsity_I, jac_sparsity_J, hess_sparsity_I, hess_sparsity_J,
       quasi_newton) = structure

    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    (; jac, hess, jac_callback, pr_diag, du_diag) = _build_sparsekkt_views(V, structure)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
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

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )

    return SparseKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, cb.ind_lb, cb.ind_ub,
    )

end

num_variables(kkt::SparseKKTSystem) = length(kkt.pr_diag)

function build_kkt!(kkt::SparseKKTSystem)
    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end
