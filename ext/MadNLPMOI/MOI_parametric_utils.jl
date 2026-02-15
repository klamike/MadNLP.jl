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

function _remap_params_to_vars(
    vi::MOI.VariableIndex,
    model::Optimizer,
    n_x::Int,
)
    pidx = _get_param_idx(model, vi)
    iszero(pidx) && return vi
    return MOI.VariableIndex(n_x + pidx)
end

function _remap_params_to_vars(
    f::MOI.ScalarAffineFunction{T},
    model::Optimizer,
    n_x::Int,
) where {T}
    terms = Vector{MOI.ScalarAffineTerm{T}}(undef, length(f.terms))
    for i in eachindex(f.terms)
        term = f.terms[i]
        terms[i] = MOI.ScalarAffineTerm{T}(
            term.coefficient,
            _remap_params_to_vars(term.variable, model, n_x),
        )
    end
    return MOI.ScalarAffineFunction(terms, f.constant)
end

function _remap_params_to_vars(
    f::MOI.ScalarQuadraticFunction{T},
    model::Optimizer,
    n_x::Int,
) where {T}
    affine_terms = Vector{MOI.ScalarAffineTerm{T}}(undef, length(f.affine_terms))
    for i in eachindex(f.affine_terms)
        term = f.affine_terms[i]
        affine_terms[i] = MOI.ScalarAffineTerm{T}(
            term.coefficient,
            _remap_params_to_vars(term.variable, model, n_x),
        )
    end

    quadratic_terms = Vector{MOI.ScalarQuadraticTerm{T}}(undef, length(f.quadratic_terms))
    for i in eachindex(f.quadratic_terms)
        term = f.quadratic_terms[i]
        quadratic_terms[i] = MOI.ScalarQuadraticTerm{T}(
            term.coefficient,
            _remap_params_to_vars(term.variable_1, model, n_x),
            _remap_params_to_vars(term.variable_2, model, n_x),
        )
    end

    return MOI.ScalarQuadraticFunction(quadratic_terms, affine_terms, f.constant)
end

function _constraint_set_from_bounds(lower::T, upper::T) where {T}
    if lower == upper
        return MOI.EqualTo(lower)
    elseif isfinite(lower) && isfinite(upper)
        return MOI.Interval(lower, upper)
    elseif isfinite(lower)
        return MOI.GreaterThan(lower)
    end
    return MOI.LessThan(upper)
end

function _create_param_as_vars_qp_data(model::Optimizer, n_x::Int)
    src = model.qp_data
    dst = QPBlockData{Float64}()

    objective = _remap_params_to_vars(src.objective, model, n_x)
    MOI.set(dst, MOI.ObjectiveFunction{typeof(objective)}(), objective)

    for i in eachindex(src.constraints)
        constraint = _remap_params_to_vars(src.constraints[i], model, n_x)
        set = _constraint_set_from_bounds(src.g_L[i], src.g_U[i])
        ci = MOI.add_constraint(dst, constraint, set)
        MOI.set(dst, MOI.ConstraintDualStart(), ci, src.mult_g[i])
    end

    return dst
end

function _fill_x_combined!(model::Optimizer, x)
    xc = model.param_x_combined
    n_x = length(x)
    xc[1:n_x] .= x
    for (i, pvi) in enumerate(model.param_order)
        if model.nlp_model !== nothing
            ref = model.parameters[pvi]
            xc[n_x + i] = model.nlp_model[ref]
        else
            xc[n_x + i] = get(model.qp_data.parameters, pvi.value, zero(eltype(xc)))
        end
    end
    return xc
end

_n_qp(model::Optimizer) = length(model.qp_data.constraints)
_n_nlp(model::Optimizer) = length(model.nlp_data.constraint_bounds)

_has_nlp_objective(model::Optimizer) =
    model.nlp_model !== nothing && model.nlp_model.objective !== nothing

function _fill_param_jac_structure!(
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
    k::Int,
    sparsity::AbstractVector{Tuple{Int,Int}},
    row_offset::Int,
    n_x::Int,
)
    for (row, col) in sparsity
        col > n_x || continue
        rows[k] = row + row_offset
        cols[k] = col - n_x
        k += 1
    end
    return k
end

function _fill_param_jac_values!(
    vals::AbstractVector,
    k::Int,
    sparsity::AbstractVector{Tuple{Int,Int}},
    jac_vals::AbstractVector,
    n_x::Int,
)
    for (i, (_, col)) in enumerate(sparsity)
        col > n_x || continue
        vals[k] = jac_vals[i]
        k += 1
    end
    return k
end

_is_param_hess_entry(row::Int, col::Int, n_x::Int) =
    (row <= n_x && col > n_x) || (col <= n_x && row > n_x)

function _param_jac_nnz(model::Optimizer)
    n_x = length(model.param_var_order)
    nnz = 0
    for (_, col) in MOI.jacobian_structure(model.param_qp_data)
        col > n_x && (nnz += 1)
    end
    for (_, col) in MOI.jacobian_structure(model.param_evaluator)
        col > n_x && (nnz += 1)
    end
    return nnz
end

function _param_hess_nnz(model::Optimizer)
    n_x = length(model.param_var_order)
    nnz = 0
    for (row, col) in MOI.hessian_lagrangian_structure(model.param_qp_data)
        _is_param_hess_entry(row, col, n_x) && (nnz += 1)
    end
    for (row, col) in MOI.hessian_lagrangian_structure(model.param_evaluator)
        _is_param_hess_entry(row, col, n_x) && (nnz += 1)
    end
    return nnz
end

function _fill_param_hess_structure!(
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
    k::Int,
    sparsity::AbstractVector{Tuple{Int,Int}},
    n_x::Int,
)
    for (row, col) in sparsity
        _is_param_hess_entry(row, col, n_x) || continue
        if row <= n_x
            rows[k] = row
            cols[k] = col - n_x
        else
            rows[k] = col
            cols[k] = row - n_x
        end
        k += 1
    end
    return k
end

function _fill_param_hess_values!(
    vals::AbstractVector,
    k::Int,
    sparsity::AbstractVector{Tuple{Int,Int}},
    hess_vals::AbstractVector,
    n_x::Int,
)
    for (i, (row, col)) in enumerate(sparsity)
        _is_param_hess_entry(row, col, n_x) || continue
        vals[k] = hess_vals[i]
        k += 1
    end
    return k
end
