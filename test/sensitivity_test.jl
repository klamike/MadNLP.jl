using Test
using MadNLP
using MadNLP: primal
using NLPModels
using LinearAlgebra
using SparseArrays
using JuMP
using DiffOpt
using Ipopt


mutable struct TestPQP{T,VT} <: AbstractNLPModel{T,VT}
    meta::NLPModelMeta{T,VT}
    counters::Counters

    Q::Matrix{T}
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    Bp::Matrix{T}

    p::Vector{T}
end

function TestPQP(Q, c, A, b, Bp, p; name="ParametricQP")
    T = eltype(Q)
    n, m = size(A)
    np = length(p)

    lvar = zeros(T, n)
    uvar = fill(T(Inf), n)

    lcon = zeros(T, m)
    ucon = zeros(T, m)

    meta = NLPModelMeta(
        n, x0=ones(T, n), lvar=lvar, uvar=uvar,
        ncon=m, lcon=lcon, ucon=ucon,
        nnzh=n*(n+1)÷2, nnzj=m*n,
        name=name
    )

    return TestPQP(meta, Counters(), Q, c, A, b, Bp, copy(p))
end

function NLPModels.obj(nlp::TestPQP, x)
    increment!(nlp, :neval_obj)
    # Handle different dimensions: only first min(n,n_p) variables affected by parameters
    n = length(x)
    n_p = length(nlp.p)

    obj_val = 0.5 * dot(x, nlp.Q * x) + dot(nlp.c, x)

    # Add parameter effects only to first min(n,n_p) variables
    for i in 1:min(n, n_p)
        obj_val += nlp.p[i] * x[i]
    end

    return obj_val
end

function NLPModels.grad!(nlp::TestPQP, x, g)
    increment!(nlp, :neval_grad)
    # Handle different dimensions: only first min(n,n_p) variables affected by parameters
    n = length(x)
    n_p = length(nlp.p)

    mul!(g, nlp.Q, x)
    g .+= nlp.c

    # Add parameter effects only to first min(n,n_p) variables
    for i in 1:min(n, n_p)
        g[i] += nlp.p[i]
    end

    return g
end

function NLPModels.cons!(nlp::TestPQP, x::AbstractVector, c::AbstractVector)
    increment!(nlp, :neval_cons)
    # Constraint function: c(x) = A*x - b - Bp*p
    mul!(c, nlp.A, x)
    c .-= nlp.b .+ nlp.Bp * nlp.p
    return c
end

function NLPModels.jac_coord!(nlp::TestPQP, x::AbstractVector, vals::AbstractVector)
    increment!(nlp, :neval_jac)
    vals .= vec(nlp.A)
    return vals
end

function NLPModels.hess_coord!(nlp::TestPQP, x, y, vals; obj_weight=1.0)
    increment!(nlp, :neval_hess)
    n = nlp.meta.nvar
    idx = 1
    for j = 1:n
        for i = j:n
            vals[idx] = obj_weight * nlp.Q[i,j]
            idx += 1
        end
    end
    return vals
end

function NLPModels.jac_structure!(nlp::TestPQP, rows::AbstractVector{T}, cols::AbstractVector{T}) where T
    m, n = size(nlp.A)
    idx = 1
    for j = 1:n
        for i = 1:m
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function NLPModels.hess_structure!(nlp::TestPQP, rows::AbstractVector{T}, cols::AbstractVector{T}) where T
    n = nlp.meta.nvar
    idx = 1
    for j = 1:n
        for i = j:n
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function MadNLP.hess_param!(nlp::TestPQP, ∇xpL::AbstractMatrix, x::AbstractVector, y::AbstractVector, p::AbstractVector)
    n, np = size(∇xpL)
    # For the objective f(x,p) = 0.5*x'*Q*x + (c + p)'*x
    # ∂²f/∂x∂p = ∂/∂p(∂f/∂x) = ∂/∂p(Q*x + c + p)
    # Only the linear term (c + p) depends on p, so ∂²f/∂x∂p has:
    # - ∂²f/∂x_i∂p_j = 1 if i == j and i <= min(n,np)
    # - ∂²f/∂x_i∂p_j = 0 otherwise
    # The constraint term ∂²g/∂x∂p = 0 since g doesn't have x-p cross terms
    fill!(∇xpL, 0.0)
    for i = 1:min(n, np)
        ∇xpL[i, i] = 1.0
    end
    return ∇xpL
end

function MadNLP.jac_param!(nlp::TestPQP, ∇pg::AbstractMatrix, x::AbstractVector, p::AbstractVector)
    # ∂g/∂p = ∂/∂p(A*x - b - Bp*p) = -Bp
    ∇pg .= -nlp.Bp
    return ∇pg
end


function create_jump_parametric_qp(Q, c, A, b, Bp, p; name="ParametricQP")
    m, n = size(A)
    n_p = length(p)

    model = Model()
    @variable(model, x[1:n] >= 0)

    quadratic_term = 0.5 * sum(sum(Q[i,j] * x[i] * x[j] for j in 1:n) for i in 1:n)
    linear_term = sum(c[i] * x[i] for i in 1:n)
    param_term = sum(p[i] * x[i] for i in 1:min(n, n_p))
    @objective(model, Min, quadratic_term + linear_term + param_term)

    rhs = b + Bp * p
    @constraint(model, con[j=1:m], sum(A[j,i] * x[i] for i in 1:n) == rhs[j])

    return model
end


function create_diffopt_parametric_qp(Q, c, A, b, Bp, p; name="ParametricQP")
    m, n = size(A)
    n_p = length(p)

    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, x[1:n] >= 0)
    @variable(model, p_var[i=1:n_p] in Parameter(p[i]))

    @constraint(model, con[j=1:m], sum(A[j,i] * x[i] for i in 1:n) == b[j] + sum(Bp[j,i] * p_var[i] for i in 1:n_p))
    @objective(model, Min, 0.5 * x' * Q * x + sum(c[i] * x[i] for i in 1:n) + sum(p_var[i] * x[i] for i in 1:min(n, n_p)))

    return model, p_var, x
end


function compute_diffopt_sensitivities(model, p_var, x_var, n_p)
    n_x = length(x_var)
    sens_matrix = zeros(n_x, n_p)

    for i in 1:n_p
        DiffOpt.empty_input_sensitivities!(model)
        
        direction = zeros(n_p)
        direction[i] = 1.0
        DiffOpt.set_forward_parameter(model, p_var[i], direction[i])

        DiffOpt.forward_differentiate!(model)
        for j in 1:n_x
            sens_matrix[j, i] = DiffOpt.get_forward_variable(model, x_var[j])
        end
    end

    DiffOpt.empty_input_sensitivities!(model)
    return sens_matrix
end

@testset "Sensitivity Analysis Tests" begin

    @testset "Consistency Across All KKT Systems" begin
        Q = [2.0 0.0; 0.0 4.0]
        c = [1.0, 3.0]
        A = [1.0 2.0; 0.5 -0.5]
        b = [1.0, 0.0]
        Bp = [7.0 0.0; 0.0 4.0]
        p = [2.0, 1.0]

        nlp = TestPQP(Q, c, A, b, Bp, p)
        nlp.meta.lvar .= -Inf
        nlp.meta.uvar .= Inf
        nlp.meta.lcon .= -Inf
        nlp.meta.ucon .= nlp.b + nlp.Bp * nlp.p

        kkt_systems = [
            (MadNLP.SparseUnreducedKKTSystem, "SparseUnreduced"),
            (MadNLP.SparseKKTSystem, "SparseReduced"),
            (MadNLP.SparseCondensedKKTSystem, "SparseCondensed"),
            (MadNLP.DenseKKTSystem, "DenseReduced"),
            (MadNLP.DenseCondensedKKTSystem, "DenseCondensed")
        ]

        results = []
        working_systems = []

        for (KKT, name) in kkt_systems
            linear_solver = contains(name, "Dense") ? MadNLP.LapackCPUSolver : nothing
            solver_options = linear_solver === nothing ?
                (kkt_system=KKT, print_level=MadNLP.ERROR) :
                (kkt_system=KKT, linear_solver=linear_solver, print_level=MadNLP.ERROR)

            solver = MadNLP.MadNLPSolver(nlp; solver_options...)
            MadNLP.solve!(solver)

            @test solver.status == MadNLP.SOLVE_SUCCEEDED
            sens = MadNLP.sensitivity_analysis(solver, [1.0, 1.0])
            push!(results, (name, sens))
            push!(working_systems, name)

            @test size(sens.∇x) == (2, 2)
            @test all(isfinite.(sens.∇x))
            @test all(isfinite.(sens.∇y))
            @test all(isfinite.(sens.∇z))
        end

        if length(results) > 1
            base_∇x = results[1][2].∇x
            for i = 2:length(results)
                diff_∇x = norm(base_∇x - results[i][2].∇x)
                @test diff_∇x < 1e-3
                if diff_∇x >= 1e-6
                    @warn "$(results[1][1]) vs $(results[i][1]) ∇x mismatch: $diff_∇x"
                end
            end

            for (name, sens) in results
                @test size(sens.∇y, 1) == 0
            end

            base_∇z = results[1][2].∇z
            for i = 2:length(results)
                @test size(base_∇z) == size(results[i][2].∇z)
                diff_∇z = norm(base_∇z - results[i][2].∇z)
                @test diff_∇z < 1e-3
                if diff_∇z >= 1e-6
                    @warn "$(results[1][1]) vs $(results[i][1]) ∇z mismatch: $diff_∇z"
                end
            end
        end
    end

    @testset "Different Dimensions Test" begin
        Q = [2.0 0.5 0.0; 0.5 1.0 0.2; 0.0 0.2 1.5]
        c = [1.0, 2.0, 0.5]
        A = [1.0 1.0 0.5; 2.0 -1.0 1.0; 0.5 1.5 -0.5]
        b = [3.0, 1.0, 2.0]
        Bp = [1.0 0.0; 0.0 1.0; 0.5 0.5]
        p = [0.1, 0.2]

        nlp_model = TestPQP(Q, c, A, b, Bp, p)

        solver_nlp = MadNLP.MadNLPSolver(nlp_model; print_level=MadNLP.ERROR)
        MadNLP.solve!(solver_nlp)

        @test solver_nlp.status in [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]

        jump_model = create_jump_parametric_qp(Q, c, A, b, Bp, p)
        set_optimizer(jump_model, MadNLP.Optimizer)
        set_attribute(jump_model, "print_level", MadNLP.ERROR)
        optimize!(jump_model)

        @test termination_status(jump_model) == LOCALLY_SOLVED

        x_nlp = copy(primal(solver_nlp.x))
        x_jump = value.(jump_model[:x])
        @test norm(x_nlp - x_jump) < 1e-8

        obj_nlp = obj(nlp_model, x_nlp)
        obj_jump = objective_value(jump_model)

        @test abs(obj_nlp - obj_jump) < 1e-6

        cons_nlp = zeros(nlp_model.meta.ncon)
        cons!(nlp_model, x_nlp, cons_nlp)
        cons_jump = A * x_jump - (b + Bp * p)
        @test norm(cons_nlp - cons_jump) < 1e-6

        sens_madnlp = MadNLP.sensitivity_analysis(solver_nlp, p)
        @test size(sens_madnlp.∇x) == (3, 2)

        diffopt_model, p_var, x_var = create_diffopt_parametric_qp(Q, c, A, b, Bp, p)
        optimize!(diffopt_model)

        @test termination_status(diffopt_model) == LOCALLY_SOLVED

        x_diffopt = value.(x_var)
        @test norm(x_nlp - x_diffopt) < 1e-6

        obj_diffopt = objective_value(diffopt_model)
        @test abs(obj_nlp - obj_diffopt) < 1e-6

        n_p = length(p)
        diffopt_sens_x_raw = compute_diffopt_sensitivities(diffopt_model, p_var, x_var, n_p)

        if size(diffopt_sens_x_raw) == size(sens_madnlp.∇x)
            diffopt_sens_x = diffopt_sens_x_raw
        else
            @warn "DiffOpt sensitivity has the wrong orientation"
            diffopt_sens_x = diffopt_sens_x_raw'
        end

        @test size(diffopt_sens_x) == size(sens_madnlp.∇x)
        sens_diff = norm(diffopt_sens_x - sens_madnlp.∇x)
        @test sens_diff < 1e-6
    end
end