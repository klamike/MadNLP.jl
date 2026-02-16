include("MOI_parametric_utils.jl")

function ParametricNLPModels.jac_param_structure!(
    nlp::MOIModel{T},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_qp = _n_qp(model)
    k = 1

    sparsity_qp = MOI.jacobian_structure(model.param_qp_data)
    k = _fill_param_jac_structure!(rows, cols, k, sparsity_qp, 0, n_x)

    sparsity_nlp = MOI.jacobian_structure(model.param_evaluator)
    k = _fill_param_jac_structure!(rows, cols, k, sparsity_nlp, n_qp, n_x)
    return rows, cols
end

function ParametricNLPModels.jac_param_coord!(
    nlp::MOIModel{T},
    x::AbstractVector,
    vals::AbstractVector,
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    k = 1

    _fill_x_combined!(model, x)

    sparsity_qp = MOI.jacobian_structure(model.param_qp_data)
    jac_vals_qp = fill!(Vector{T}(undef, length(sparsity_qp)), zero(T))
    MOI.eval_constraint_jacobian(model.param_qp_data, jac_vals_qp, model.param_x_combined)
    k = _fill_param_jac_values!(vals, k, sparsity_qp, jac_vals_qp, n_x)

    sparsity_nlp = MOI.jacobian_structure(model.param_evaluator)
    jac_vals_nlp = fill!(Vector{T}(undef, length(sparsity_nlp)), zero(T))
    MOI.eval_constraint_jacobian(model.param_evaluator, jac_vals_nlp, model.param_x_combined)
    k = _fill_param_jac_values!(vals, k, sparsity_nlp, jac_vals_nlp, n_x)
    return vals
end

function ParametricNLPModels.hess_param_structure!(
    nlp::MOIModel{T},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    k = 1

    sparsity_qp = MOI.hessian_lagrangian_structure(model.param_qp_data)
    k = _fill_param_hess_structure!(rows, cols, k, sparsity_qp, n_x)

    sparsity_nlp = MOI.hessian_lagrangian_structure(model.param_evaluator)
    k = _fill_param_hess_structure!(rows, cols, k, sparsity_nlp, n_x)
    return rows, cols
end

function ParametricNLPModels.hess_param_coord!(
    nlp::MOIModel{T},
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real = one(T),
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)
    k = 1

    _fill_x_combined!(model, x)

    sparsity_qp = MOI.hessian_lagrangian_structure(model.param_qp_data)
    hess_vals_qp = fill!(Vector{T}(undef, length(sparsity_qp)), zero(T))
    y_qp = n_qp > 0 ? view(y, 1:n_qp) : T[]
    MOI.eval_hessian_lagrangian(
        model.param_qp_data,
        hess_vals_qp,
        model.param_x_combined,
        T(obj_weight),
        y_qp,
    )
    k = _fill_param_hess_values!(vals, k, sparsity_qp, hess_vals_qp, n_x)

    sparsity_nlp = MOI.hessian_lagrangian_structure(model.param_evaluator)
    hess_vals_nlp = fill!(Vector{T}(undef, length(sparsity_nlp)), zero(T))
    y_nlp = n_nlp_c > 0 ? view(y, n_qp+1:n_qp+n_nlp_c) : T[]
    MOI.eval_hessian_lagrangian(
        model.param_evaluator,
        hess_vals_nlp,
        model.param_x_combined,
        T(obj_weight),
        y_nlp,
    )
    k = _fill_param_hess_values!(vals, k, sparsity_nlp, hess_vals_nlp, n_x)
    return vals
end

function ParametricNLPModels.jac_param!(
    nlp::MOIModel{T}, x::AbstractVector, J::AbstractMatrix{T},
) where {T}
    rows, cols = ParametricNLPModels.jac_param_structure(nlp)
    vals = fill!(Vector{T}(undef, length(rows)), zero(T))
    ParametricNLPModels.jac_param_coord!(nlp, x, vals)

    fill!(J, zero(T))
    for k in eachindex(vals)
        J[rows[k], cols[k]] += vals[k]
    end
    return J
end

function ParametricNLPModels.hess_param!(
    nlp::MOIModel{T}, x::AbstractVector, y::AbstractVector, H::AbstractMatrix{T};
    obj_weight::Real = one(T),
) where {T}
    rows, cols = ParametricNLPModels.hess_param_structure(nlp)
    vals = fill!(Vector{T}(undef, length(rows)), zero(T))
    ParametricNLPModels.hess_param_coord!(nlp, x, y, vals; obj_weight = obj_weight)

    fill!(H, zero(T))
    for k in eachindex(vals)
        H[rows[k], cols[k]] += vals[k]
    end
    return H
end

function ParametricNLPModels.jpprod!(
    nlp::MOIModel{T}, x::AbstractVector, v::AbstractVector, Jv::AbstractVector,
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_p = nlp.pmeta.nparam
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)

    fill!(Jv, zero(T))
    _fill_x_combined!(model, x)

    ve = model.param_v_extended
    fill!(ve, zero(T))
    ve[n_x+1:n_x+n_p] .= v

    Jv_qp = view(Jv, 1:n_qp)
    MOI.eval_constraint_jacobian_product(model.param_qp_data, Jv_qp, model.param_x_combined, ve)

    Jv_nlp = view(Jv, n_qp+1:n_qp+n_nlp_c)
    MOI.eval_constraint_jacobian_product(model.param_evaluator, Jv_nlp, model.param_x_combined, ve)

    return Jv
end

function ParametricNLPModels.jptprod!(
    nlp::MOIModel{T}, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector,
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_p = nlp.pmeta.nparam
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)

    fill!(Jtv, zero(T))
    _fill_x_combined!(model, x)

    result = model.param_result

    fill!(result, zero(T))
    v_qp = view(v, 1:n_qp)
    MOI.eval_constraint_jacobian_transpose_product(model.param_qp_data, result, model.param_x_combined, v_qp)
    Jtv .+= view(result, n_x+1:n_x+n_p)

    fill!(result, zero(T))
    v_nlp = view(v, n_qp+1:n_qp+n_nlp_c)
    MOI.eval_constraint_jacobian_transpose_product(model.param_evaluator, result, model.param_x_combined, v_nlp)
    Jtv .+= view(result, n_x+1:n_x+n_p)

    return Jtv
end

function ParametricNLPModels.hpprod!(
    nlp::MOIModel{T}, x::AbstractVector, y::AbstractVector,
    v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(T),
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_p = nlp.pmeta.nparam
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)
    σ = T(obj_weight)

    fill!(Hv, zero(T))
    _fill_x_combined!(model, x)

    ve = model.param_v_extended
    fill!(ve, zero(T))
    ve[n_x+1:n_x+n_p] .= v

    result = model.param_result

    fill!(result, zero(T))
    y_qp = n_qp > 0 ? view(y, 1:n_qp) : T[]
    MOI.eval_hessian_lagrangian_product(model.param_qp_data, result, model.param_x_combined, ve, σ, y_qp)
    Hv .+= view(result, 1:n_x)

    fill!(result, zero(T))
    y_nlp = n_nlp_c > 0 ? view(y, n_qp+1:n_qp+n_nlp_c) : T[]
    MOI.eval_hessian_lagrangian_product(model.param_evaluator, result, model.param_x_combined, ve, σ, y_nlp)
    Hv .+= view(result, 1:n_x)

    return Hv
end

function ParametricNLPModels.hptprod!(
    nlp::MOIModel{T}, x::AbstractVector, y::AbstractVector,
    v::AbstractVector, Htv::AbstractVector; obj_weight::Real = one(T),
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_p = nlp.pmeta.nparam
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)
    σ = T(obj_weight)

    fill!(Htv, zero(T))
    _fill_x_combined!(model, x)

    ve = model.param_v_extended
    fill!(ve, zero(T))
    ve[1:n_x] .= v

    result = model.param_result

    fill!(result, zero(T))
    y_qp = n_qp > 0 ? view(y, 1:n_qp) : T[]
    MOI.eval_hessian_lagrangian_product(model.param_qp_data, result, model.param_x_combined, ve, σ, y_qp)
    Htv .+= view(result, n_x+1:n_x+n_p)

    fill!(result, zero(T))
    y_nlp = n_nlp_c > 0 ? view(y, n_qp+1:n_qp+n_nlp_c) : T[]
    MOI.eval_hessian_lagrangian_product(model.param_evaluator, result, model.param_x_combined, ve, σ, y_nlp)
    Htv .+= view(result, n_x+1:n_x+n_p)

    return Htv
end

function ParametricNLPModels.grad_param!(
    nlp::MOIModel{T}, x::AbstractVector, g::AbstractVector,
) where {T}
    model = nlp.model
    n_x = nlp.meta.nvar
    n_p = nlp.pmeta.nparam

    fill!(g, zero(T))
    _fill_x_combined!(model, x)

    grad_full = model.param_result

    fill!(grad_full, zero(T))
    MOI.eval_objective_gradient(model.param_qp_data, grad_full, model.param_x_combined)
    g .+= view(grad_full, n_x+1:n_x+n_p)

    if _has_nlp_objective(model)
        fill!(grad_full, zero(T))
        MOI.eval_objective_gradient(model.param_evaluator, grad_full, model.param_x_combined)
        g .+= view(grad_full, n_x+1:n_x+n_p)
    end

    return g
end

ParametricNLPModels.lcon_jac_param!(::MOIModel{T}, J::AbstractMatrix{T}) where {T} = fill!(J, zero(T))
ParametricNLPModels.ucon_jac_param!(::MOIModel{T}, J::AbstractMatrix{T}) where {T} = fill!(J, zero(T))
ParametricNLPModels.lvar_jac_param!(::MOIModel{T}, J::AbstractMatrix{T}) where {T} = fill!(J, zero(T))
ParametricNLPModels.uvar_jac_param!(::MOIModel{T}, J::AbstractMatrix{T}) where {T} = fill!(J, zero(T))
ParametricNLPModels.lcon_jac_param_structure!(::MOIModel{T}, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) where {T} = (rows, cols)
ParametricNLPModels.ucon_jac_param_structure!(::MOIModel{T}, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) where {T} = (rows, cols)
ParametricNLPModels.lvar_jac_param_structure!(::MOIModel{T}, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) where {T} = (rows, cols)
ParametricNLPModels.uvar_jac_param_structure!(::MOIModel{T}, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) where {T} = (rows, cols)
ParametricNLPModels.lcon_jpprod!(::MOIModel{T}, ::AbstractVector, Jv::AbstractVector) where {T} = fill!(Jv, zero(T))
ParametricNLPModels.ucon_jpprod!(::MOIModel{T}, ::AbstractVector, Jv::AbstractVector) where {T} = fill!(Jv, zero(T))
ParametricNLPModels.lvar_jpprod!(::MOIModel{T}, ::AbstractVector, Jv::AbstractVector) where {T} = fill!(Jv, zero(T))
ParametricNLPModels.uvar_jpprod!(::MOIModel{T}, ::AbstractVector, Jv::AbstractVector) where {T} = fill!(Jv, zero(T))
ParametricNLPModels.lcon_jptprod!(::MOIModel{T}, ::AbstractVector, Jtv::AbstractVector) where {T} = fill!(Jtv, zero(T))
ParametricNLPModels.ucon_jptprod!(::MOIModel{T}, ::AbstractVector, Jtv::AbstractVector) where {T} = fill!(Jtv, zero(T))
ParametricNLPModels.lvar_jptprod!(::MOIModel{T}, ::AbstractVector, Jtv::AbstractVector) where {T} = fill!(Jtv, zero(T))
ParametricNLPModels.uvar_jptprod!(::MOIModel{T}, ::AbstractVector, Jtv::AbstractVector) where {T} = fill!(Jtv, zero(T))
