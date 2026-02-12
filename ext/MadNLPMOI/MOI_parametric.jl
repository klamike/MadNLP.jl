function _remap_params_to_vars(expr, n_x)
    new_expr = MOI.Nonlinear.Expression()
    resize!(new_expr.nodes, length(expr.nodes))
    for (i, node) in enumerate(expr.nodes)
        if node.type == MOI.Nonlinear.NODE_PARAMETER
            new_expr.nodes[i] = MOI.Nonlinear.Node(
                MOI.Nonlinear.NODE_MOI_VARIABLE, n_x + node.index, node.parent,
            )
        else
            new_expr.nodes[i] = node
        end
    end
    append!(new_expr.values, expr.values)
    return new_expr
end

function _create_param_as_vars_model(nlp_model, n_x)
    new_model = MOI.Nonlinear.Model()
    new_model.operators = nlp_model.operators

    for expr in nlp_model.expressions
        new_expr = _remap_params_to_vars(expr, n_x)
        push!(new_model.expressions, new_expr)
    end

    if !isnothing(nlp_model.objective)
        new_model.objective = _remap_params_to_vars(nlp_model.objective, n_x)
    end

    for (ci, constraint) in nlp_model.constraints
        new_expr = _remap_params_to_vars(constraint.expression, n_x)
        new_model.constraints[ci] = MOI.Nonlinear.Constraint(new_expr, constraint.set)
    end

    return new_model
end

_get_param_idx(model::Optimizer, vi::MOI.VariableIndex) = get(model.param_vi_to_idx, vi, 0)
_get_primal_idx(model::Optimizer, vi::MOI.VariableIndex) = _is_parameter(vi) ? 0 : vi.value

function _fill_x_combined!(model::Optimizer, x)
    xc = model.param_x_combined
    n_x = length(x)
    xc[1:n_x] .= x
    for (i, pvi) in enumerate(model.param_order)
        ref = model.parameters[pvi]
        xc[n_x + i] = model.nlp_model[ref]
    end
    return xc
end

_n_qp(model::Optimizer) = length(model.qp_data.constraints)
_n_nlp(model::Optimizer) = length(model.nlp_data.constraint_bounds)

function _has_nlp_constraints(model::Optimizer)
    return model.param_evaluator !== nothing && _n_nlp(model) > 0
end

function _has_nlp_objective(model::Optimizer)
    return model.param_evaluator !== nothing &&
           model.nlp_model !== nothing &&
           model.nlp_model.objective !== nothing
end

_has_nlp_lagrangian(model::Optimizer) = _has_nlp_constraints(model) || _has_nlp_objective(model)

function _quad_term_indices(term, model)
    vi, vj = term.variable_1, term.variable_2
    return (
        coef = term.coefficient,
        pi = _get_param_idx(model, vi),
        pj = _get_param_idx(model, vj),
        xi = _get_primal_idx(model, vi),
        xj = _get_primal_idx(model, vj),
    )
end

function _get_param_value(model, pidx)
    return model.nlp_model[model.parameters[model.param_order[pidx]]]
end

abstract type DiffDirection end
struct ForwardMode <: DiffDirection end
struct ReverseMode <: DiffDirection end

function ParametricNLPModels.jpprod!(
    nlp::MOIModel{T}, x::AbstractVector, v::AbstractVector, Jv::AbstractVector,
) where {T}
    model = nlp.model
    fill!(Jv, zero(T))
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)

    for (row, constraint) in enumerate(model.qp_data.constraints)
        _jprod_qp!(ForwardMode(), Jv, constraint, model, x, v, row)
    end

    if _has_nlp_constraints(model)
        n_x = length(x)
        n_p = model.param_n_p
        _fill_x_combined!(model, x)

        ve = model.param_v_extended
        fill!(ve, zero(T))
        ve[n_x+1:n_x+n_p] .= v

        Jv_nlp = view(Jv, n_qp+1:n_qp+n_nlp_c)
        MOI.eval_constraint_jacobian_product(model.param_evaluator, Jv_nlp, model.param_x_combined, ve)
    end

    return Jv
end

function ParametricNLPModels.jptprod!(
    nlp::MOIModel{T}, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector,
) where {T}
    model = nlp.model
    fill!(Jtv, zero(T))
    n_qp = _n_qp(model)
    n_nlp_c = _n_nlp(model)

    for (row, constraint) in enumerate(model.qp_data.constraints)
        _jprod_qp!(ReverseMode(), Jtv, constraint, model, x, v, row)
    end

    if _has_nlp_constraints(model)
        n_x = length(x)
        n_p = model.param_n_p
        _fill_x_combined!(model, x)

        v_nlp = view(v, n_qp+1:n_qp+n_nlp_c)
        result = model.param_result
        MOI.eval_constraint_jacobian_transpose_product(model.param_evaluator, result, model.param_x_combined, v_nlp)
        Jtv .+= view(result, n_x+1:n_x+n_p)
    end

    return Jtv
end

function _jprod_qp!(::ForwardMode, Jv, f::MOI.ScalarAffineFunction, model, x, v, row)
    for term in f.terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (Jv[row] += term.coefficient * v[i])
    end
end

function _jprod_qp!(::ForwardMode, Jv, f::MOI.ScalarQuadraticFunction, model, x, v, row)
    for term in f.affine_terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (Jv[row] += term.coefficient * v[i])
    end
    for term in f.quadratic_terms
        (; coef, pi, pj, xi, xj) = _quad_term_indices(term, model)
        !iszero(pi) && !iszero(xj) && (Jv[row] += coef * x[xj] * v[pi])
        !iszero(pj) && !iszero(xi) && (Jv[row] += coef * x[xi] * v[pj])
        if !iszero(pi) && !iszero(pj)
            p_i, p_j = _get_param_value(model, pi), _get_param_value(model, pj)
            if pi == pj
                Jv[row] += coef * p_i * v[pi]
            else
                Jv[row] += coef * p_j * v[pi] + coef * p_i * v[pj]
            end
        end
    end
end

function _jprod_qp!(::ReverseMode, Jtv, f::MOI.ScalarAffineFunction, model, x, v, row)
    v_row = v[row]
    for term in f.terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (Jtv[i] += term.coefficient * v_row)
    end
end

function _jprod_qp!(::ReverseMode, Jtv, f::MOI.ScalarQuadraticFunction, model, x, v, row)
    v_row = v[row]
    for term in f.affine_terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (Jtv[i] += term.coefficient * v_row)
    end
    for term in f.quadratic_terms
        (; coef, pi, pj, xi, xj) = _quad_term_indices(term, model)
        !iszero(pi) && !iszero(xj) && (Jtv[pi] += coef * x[xj] * v_row)
        !iszero(pj) && !iszero(xi) && (Jtv[pj] += coef * x[xi] * v_row)
        if !iszero(pi) && !iszero(pj)
            p_i, p_j = _get_param_value(model, pi), _get_param_value(model, pj)
            if pi == pj
                Jtv[pi] += coef * p_i * v_row
            else
                Jtv[pi] += coef * p_j * v_row
                Jtv[pj] += coef * p_i * v_row
            end
        end
    end
end

function ParametricNLPModels.hpprod!(
    nlp::MOIModel{T}, x::AbstractVector, y::AbstractVector,
    v::AbstractVector, Hv::AbstractVector; obj_weight::Real = one(T),
) where {T}
    model = nlp.model
    fill!(Hv, zero(T))
    n_qp = _n_qp(model)
    σ = T(obj_weight)

    _hprod_qp!(ForwardMode(), Hv, model.qp_data.objective, model, v, σ)
    for (row, constraint) in enumerate(model.qp_data.constraints)
        _hprod_qp!(ForwardMode(), Hv, constraint, model, v, y[row])
    end

    if _has_nlp_lagrangian(model)
        n_x = length(x)
        n_p = model.param_n_p
        n_nlp_c = _n_nlp(model)
        _fill_x_combined!(model, x)

        ve = model.param_v_extended
        fill!(ve, zero(T))
        ve[n_x+1:n_x+n_p] .= v

        y_nlp = n_nlp_c > 0 ? view(y, n_qp+1:n_qp+n_nlp_c) : T[]

        result = model.param_result
        MOI.eval_hessian_lagrangian_product(model.param_evaluator, result, model.param_x_combined, ve, σ, y_nlp)
        Hv .+= view(result, 1:n_x)
    end

    return Hv
end

function ParametricNLPModels.hptprod!(
    nlp::MOIModel{T}, x::AbstractVector, y::AbstractVector,
    v::AbstractVector, Htv::AbstractVector; obj_weight::Real = one(T),
) where {T}
    model = nlp.model
    fill!(Htv, zero(T))
    n_qp = _n_qp(model)
    σ = T(obj_weight)

    _hprod_qp!(ReverseMode(), Htv, model.qp_data.objective, model, v, σ)
    for (row, constraint) in enumerate(model.qp_data.constraints)
        _hprod_qp!(ReverseMode(), Htv, constraint, model, v, y[row])
    end

    if _has_nlp_lagrangian(model)
        n_x = length(x)
        n_p = model.param_n_p
        n_nlp_c = _n_nlp(model)
        _fill_x_combined!(model, x)

        ve = model.param_v_extended
        fill!(ve, zero(T))
        ve[1:n_x] .= v

        y_nlp = n_nlp_c > 0 ? view(y, n_qp+1:n_qp+n_nlp_c) : T[]

        result = model.param_result
        MOI.eval_hessian_lagrangian_product(model.param_evaluator, result, model.param_x_combined, ve, σ, y_nlp)
        Htv .+= view(result, n_x+1:n_x+n_p)
    end

    return Htv
end

_hprod_qp!(::DiffDirection, output, ::MOI.ScalarAffineFunction, model, v, scale) = nothing

function _hprod_qp!(::ForwardMode, Hv, f::MOI.ScalarQuadraticFunction, model, v, scale)
    for term in f.quadratic_terms
        (; coef, pi, pj, xi, xj) = _quad_term_indices(term, model)
        !iszero(pi) && !iszero(xj) && (Hv[xj] += scale * coef * v[pi])
        !iszero(pj) && !iszero(xi) && (Hv[xi] += scale * coef * v[pj])
    end
end

function _hprod_qp!(::ReverseMode, Htv, f::MOI.ScalarQuadraticFunction, model, v, scale)
    for term in f.quadratic_terms
        (; coef, pi, pj, xi, xj) = _quad_term_indices(term, model)
        !iszero(pi) && !iszero(xj) && (Htv[pi] += scale * coef * v[xj])
        !iszero(pj) && !iszero(xi) && (Htv[pj] += scale * coef * v[xi])
    end
end

function ParametricNLPModels.grad_param!(
    nlp::MOIModel{T}, x::AbstractVector, g::AbstractVector,
) where {T}
    model = nlp.model
    fill!(g, zero(T))

    _grad_param_qp!(g, model.qp_data.objective, model, x)

    if _has_nlp_objective(model)
        n_x = length(x)
        n_p = model.param_n_p
        _fill_x_combined!(model, x)

        grad_full = model.param_result
        MOI.eval_objective_gradient(model.param_evaluator, grad_full, model.param_x_combined)

        g .+= view(grad_full, n_x+1:n_x+n_p)
    end

    return g
end

function _grad_param_qp!(g, f::MOI.ScalarAffineFunction, model, x)
    for term in f.terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (g[i] += term.coefficient)
    end
end

function _grad_param_qp!(g, f::MOI.ScalarQuadraticFunction, model, x)
    for term in f.affine_terms
        i = _get_param_idx(model, term.variable)
        !iszero(i) && (g[i] += term.coefficient)
    end

    for term in f.quadratic_terms
        (; coef, pi, pj, xi, xj) = _quad_term_indices(term, model)

        !iszero(pi) && !iszero(xj) && (g[pi] += coef * x[xj])
        !iszero(pj) && !iszero(xi) && (g[pj] += coef * x[xi])

        if !iszero(pi) && !iszero(pj)
            p_i, p_j = _get_param_value(model, pi), _get_param_value(model, pj)
            if pi == pj
                g[pi] += coef * p_i
            else
                g[pi] += coef * p_j
                g[pj] += coef * p_i
            end
        end
    end
end
