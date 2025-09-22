export sensitivity_analysis, sensitivity_analysis!
export compute_sensitivity, compute_sensitivity!
export hess_param!, jac_param!


"""
    build_sensitivity_rhs(kkt::AbstractKKTSystem, ∇xpL, ∇pg)

Build the right-hand side for the sensitivity linear system using multiple dispatch
for different KKT system types.
"""
function build_sensitivity_rhs(kkt::AbstractUnreducedKKTSystem, ∇xpL, ∇pg)
    n_tot = length(kkt.pr_diag)
    m = length(kkt.du_diag)
    nlb = length(kkt.l_diag)
    nub = length(kkt.u_diag)
    n_p = size(∇xpL, 2)

    rhs = similar(∇xpL, n_tot + m + nlb + nub, n_p)
    fill!(rhs, zero(eltype(rhs)))

    n_x = size(∇xpL, 1)
    rhs_x = view(rhs, 1:n_x, :)
    copyto!(rhs_x, -∇xpL)

    rhs_con = view(rhs, n_tot+1:n_tot+m, :)
    copyto!(rhs_con, -∇pg)

    return rhs
end

function build_sensitivity_rhs(kkt::AbstractReducedKKTSystem, ∇xpL, ∇pg)
    n_tot = length(kkt.pr_diag)
    m = length(kkt.du_diag)
    n_p = size(∇xpL, 2)

    rhs = similar(∇xpL, n_tot + m, n_p)
    fill!(rhs, zero(eltype(rhs)))

    n_x = size(∇xpL, 1)
    rhs_x = view(rhs, 1:n_x, :)
    copyto!(rhs_x, -∇xpL)

    rhs_con = view(rhs, n_tot+1:n_tot+m, :)
    copyto!(rhs_con, -∇pg)

    return rhs
end

function build_sensitivity_rhs(kkt::AbstractCondensedKKTSystem, ∇xpL, ∇pg)
    n_x = size(∇xpL, 1)
    n_p = size(∇xpL, 2)
    n_eq = kkt.n_eq

    rhs = similar(∇xpL, n_x + n_eq, n_p)
    fill!(rhs, zero(eltype(rhs)))

    rhs_x = view(rhs, 1:n_x, :)
    copyto!(rhs_x, -∇xpL)

    if kkt.n_ineq > 0
        n_ineq = kkt.n_ineq
        m_total = size(∇pg, 1)

        if n_ineq <= m_total
            ∇pg_ineq = view(∇pg, kkt.ind_ineq, :)

            n = num_variables(kkt)
            if length(kkt.pr_diag) >= n + n_ineq
                Σs = view(kkt.pr_diag, n+1:n+n_ineq)

                A_ineq = view(kkt.jac, kkt.ind_ineq, :)
                if length(Σs) >= n_ineq
                    weighted_derivs = Σs .* ∇pg_ineq
                    rhs_x .-= A_ineq' * weighted_derivs
                end
            end
        end
    end

    if n_eq > 0
        ∇pg_eq = view(∇pg, kkt.ind_eq, :)
        rhs_eq = view(rhs, n_x+1:n_x+n_eq, :)
        copyto!(rhs_eq, -∇pg_eq)
    end

    return rhs
end

"""
    solve_sensitivity!(kkt::AbstractKKTSystem, ∇xpL, ∇pg)

Solve the sensitivity system using the factorized KKT matrix.
Uses MadNLP's multi_solve! API for efficient multiple RHS solving.
"""
function solve_sensitivity!(kkt::AbstractKKTSystem, ∇xpL, ∇pg)
    rhs = build_sensitivity_rhs(kkt, ∇xpL, ∇pg)
    multi_solve!(kkt.linear_solver, rhs)
    return rhs
end

function extract_sensitivities(kkt::AbstractKKTSystem, S, solver, ∇pg)
    n_x = get_nvar(solver.nlp)
    m = get_ncon(solver.nlp)
    n_ineq = length(solver.ind_ineq)
    n_eq = m - n_ineq
    n_p = size(S, 2)
    n_tot = length(kkt.pr_diag)

    ∇x = view(S, 1:n_x, :)
    ∇y = n_eq > 0 ? view(S, n_tot+1:n_tot+n_eq, :) : similar(S, 0, n_p)
    ∇z = n_ineq > 0 ? view(S, n_tot+n_eq+1:n_tot+m, :) : similar(S, 0, n_p)

    return ∇x, ∇y, ∇z
end

function extract_sensitivities(kkt::AbstractCondensedKKTSystem, S, solver, ∇pg)
    n_x = get_nvar(solver.nlp)
    m = get_ncon(solver.nlp)
    n_ineq = length(solver.ind_ineq)
    n_eq = m - n_ineq
    n_p = size(S, 2)

    ∇x = view(S, 1:n_x, :)

    n_eq_kkt = kkt.n_eq
    ∇y = n_eq_kkt > 0 ? view(S, n_x+1:n_x+n_eq_kkt, :) : similar(S, 0, n_p)

    ∇z = similar(S, n_ineq, n_p)
    fill!(∇z, zero(eltype(∇z)))
    
    if n_ineq > 0
        Σs = get_slack_regularization(kkt)
        Σd_ineq = view(kkt.du_diag, kkt.ind_ineq)
        ∇pg_ineq = view(∇pg, kkt.ind_ineq, :)
        A_ineq = view(kkt.jac, kkt.ind_ineq, :)

        A_ineq_Δx = A_ineq * ∇x
        ∇z .= Σs .* (∇pg_ineq .+ A_ineq_Δx) ./ (1.0 .- Σd_ineq .* Σs)
    end

    return ∇x, ∇y, ∇z
end


"""
    compute_sensitivity(solver::MadNLPSolver, ∇xpL, ∇pg)

Compute sensitivity using a solved MadNLP solver.
"""
function compute_sensitivity(solver::MadNLPSolver, ∇xpL, ∇pg)
    S = solve_sensitivity!(solver.kkt, ∇xpL, ∇pg)
    ∇x, ∇y, ∇z = extract_sensitivities(solver.kkt, S, solver, ∇pg)
    return (∇x = ∇x, ∇y = ∇y, ∇z = ∇z)
end


"""
    sensitivity_analysis(solver::MadNLPSolver, p)

High-level function for sensitivity analysis.
"""
function sensitivity_analysis(solver::MadNLPSolver, p::AbstractVector)
    solver.status != SOLVE_SUCCEEDED && error("Solver must converge before sensitivity analysis")
    length(p) == 0 && error("Parameter vector cannot be empty")
    !all(isfinite.(p)) && error("Parameter vector contains NaN or Inf values")

    x = copy(primal(solver.x))
    y = copy(solver.y)

    n_x = get_nvar(solver.nlp)
    m = get_ncon(solver.nlp)
    n_p = length(p)

    ∇xpL = similar(x, n_x, n_p)
    fill!(∇xpL, zero(eltype(∇xpL)))

    ∇pg = similar(x, m, n_p)
    fill!(∇pg, zero(eltype(∇pg)))

    hess_param!(solver.nlp, ∇xpL, x, y, p)
    jac_param!(solver.nlp, ∇pg, x, p)

    return compute_sensitivity(solver, ∇xpL, ∇pg)
end

"""
    hess_param!(nlp, ∇xpL, x, y, p)

Compute ∂²L/∂x∂p, the mixed Hessian of the Lagrangian.
Must be implemented by NLP models for sensitivity analysis.
"""
function hess_param!(nlp::AbstractNLPModel, ∇xpL::AbstractMatrix, x::AbstractVector,
                    y::AbstractVector, p::AbstractVector)
    error("$(typeof(nlp)) must implement hess_param!(nlp, ∇xpL, x, y, p)")
end

"""
    jac_param!(nlp, ∇pg, x, p)

Compute ∂g/∂p, the Jacobian of constraints with respect to parameters.
Must be implemented by NLP models for sensitivity analysis.
"""
function jac_param!(nlp::AbstractNLPModel, ∇pg::AbstractMatrix, x::AbstractVector, p::AbstractVector)
    error("$(typeof(nlp)) must implement jac_param!(nlp, ∇pg, x, p)")
end