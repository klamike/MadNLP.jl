mutable struct LapackGPUSolver{T,MT} <: AbstractLinearSolver{T}
    A::MT
    fact::CuMatrix{T}
    n::Int64
    sol::CuVector{T}
    tau::CuVector{T}
    Λ::CuVector{T}
    work_gpu::CuVector{UInt8}
    lwork_gpu::Csize_t
    work_cpu::Vector{UInt8}
    lwork_cpu::Csize_t
    info::CuVector{Cint}
    ipiv::CuVector{Cint}
    ipiv64::CuVector{Int64}
    opt::LapackOptions
    logger::MadNLPLogger
    legacy::Bool
    params::CuSolverParameters

    function LapackGPUSolver(
        A::MT;
        option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
        opt = LapackOptions(),
        logger = MadNLPLogger(),
        legacy::Bool = true,
        kwargs...,
    ) where {MT<:AbstractMatrix}
        set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m,n = size(A)
        @assert m == n
        fact = CuMatrix{T}(undef, m, n)
        sol = CuVector{T}(undef, 0)
        tau = CuVector{T}(undef, 0)
        Λ = CuVector{T}(undef, 0)
        work_gpu = CuVector{UInt8}(undef, 0)
        lwork_gpu = zero(Int64)
        work_cpu = Vector{UInt8}(undef, 0)
        lwork_cpu = zero(Int64)
        info = CuVector{Cint}(undef, 1)
        ipiv = CuVector{Cint}(undef, 0)
        ipiv64 = CuVector{Int64}(undef, 0)
        params = CuSolverParameters()
        solver = new{T,MT}(A, fact, n, sol, tau, Λ, work_gpu, lwork_gpu, work_cpu, lwork_cpu,
                           info, ipiv, ipiv64, opt, logger, legacy, params)
        setup!(solver)
        return solver
    end
end

function setup!(M::LapackGPUSolver)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        setup_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        setup_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        setup_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        setup_cholesky!(M)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        setup_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function factorize!(M::LapackGPUSolver)
    gpu_transfer!(M.fact, M.A)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        tril_to_full!(M.fact)
        factorize_lu!(M)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        tril_to_full!(M.fact)
        factorize_qr!(M)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        factorize_cholesky!(M)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        factorize_evd!(M)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

for T in (:Float32, :Float64)
    @eval begin
        function solve!(M::LapackGPUSolver{$T}, x::CuVector{$T})
            if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
                solve_bunchkaufman!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.LU
                solve_lu!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.QR
                solve_qr!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
                solve_cholesky!(M, x)
            elseif M.opt.lapack_algorithm == MadNLP.EVD
                solve_evd!(M, x)
            else
                error(M.logger, "Invalid lapack_algorithm")
            end
        end

        is_supported(::Type{LapackGPUSolver}, ::Type{$T}) = true
    end
end
function solve_multirhs!(M::LapackGPUSolver, N)
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        # error(LOGGER, "Multi-RHS Bunch-Kaufman not yet implemented. Use LU or Cholesky.")
        solve_bunchkaufman_multirhs!(M, N)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        solve_lu_multirhs!(M, N)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        # error(LOGGER, "Multi-RHS QR not yet implemented. Use LU or Cholesky.")
        solve_qr_multirhs!(M, N)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        solve_cholesky_multirhs!(M, N)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        # error(LOGGER, "Multi-RHS EVD not yet implemented. Use LU or Cholesky.")
        solve_evd_multirhs!(M, N)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

function solve!(M::LapackGPUSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

function multi_solve!(M::LapackGPUSolver, X::AbstractMatrix)
    solve_multirhs!(M, X)
    return X
end

improve!(M::LapackGPUSolver) = false
is_inertia(M::LapackGPUSolver) = (M.opt.lapack_algorithm == MadNLP.CHOLESKY) || (M.opt.lapack_algorithm == MadNLP.EVD)
introduce(M::LapackGPUSolver) = "Lapack-GPU ($(M.opt.lapack_algorithm))"

for (
    sytrf,
    sytrf_buffer,
    getrf,
    getrf_buffer,
    getrs,
    geqrf,
    geqrf_buffer,
    ormqr,
    ormqr_buffer,
    trsm,
    potrf,
    potrf_buffer,
    potrs,
    sytrs_buffer,
    sytrs,
    typ,
    cutyp,
) in (
    (
        :cusolverDnDsytrf,
        :cusolverDnDsytrf_bufferSize,
        :cusolverDnDgetrf,
        :cusolverDnDgetrf_bufferSize,
        :cusolverDnDgetrs,
        :cusolverDnDgeqrf,
        :cusolverDnDgeqrf_bufferSize,
        :cusolverDnDormqr,
        :cusolverDnDormqr_bufferSize,
        :cublasDtrsm_v2,
        :cusolverDnDpotrf,
        :cusolverDnDpotrf_bufferSize,
        :cusolverDnDpotrs,
        :cusolverDnXsytrs_bufferSize,
        :cusolverDnXsytrs,
        Float64,
        CUDA.R_64F,
    ),
    (
        :cusolverDnSsytrf,
        :cusolverDnSsytrf_bufferSize,
        :cusolverDnSgetrf,
        :cusolverDnSgetrf_bufferSize,
        :cusolverDnSgetrs,
        :cusolverDnSgeqrf,
        :cusolverDnSgeqrf_bufferSize,
        :cusolverDnSormqr,
        :cusolverDnSormqr_bufferSize,
        :cublasStrsm_v2,
        :cusolverDnSpotrf,
        :cusolverDnSpotrf_bufferSize,
        :cusolverDnSpotrs,
        :cusolverDnXsytrs_bufferSize,
        :cusolverDnXsytrs,
        Float32,
        CUDA.R_32F,
    ),
)
    @eval begin
        function factorize_bunchkaufman!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef, size(M.A, 1)))
            haskey(M.etc, :ipiv64) ||
                (M.etc[:ipiv64] = CuVector{Int64}(undef, length(M.etc[:ipiv])))

            transfer!(M.fact, M.A)
            CUSOLVER.$sytrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$sytrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:ipiv],
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_bunchkaufman!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.etc[:ipiv64], M.etc[:ipiv])
            copyto!(M.rhs, x)
            CUSOLVER.$sytrs_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                size(M.fact, 1),
                1,
                $cutyp,
                M.fact,
                size(M.fact, 2),
                M.etc[:ipiv64],
                $cutyp,
                M.rhs,
                length(M.rhs),
                M.lwork,
                M.lwork_host,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            length(M.work_host) < M.lwork_host[] && resize!(work_host, Int(M.lwork_host[]))
            CUSOLVER.$sytrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                size(M.fact, 1),
                1,
                $cutyp,
                M.fact,
                size(M.fact, 2),
                M.etc[:ipiv64],
                $cutyp,
                M.rhs,
                length(M.rhs),
                M.work,
                M.lwork[],
                M.work_host,
                M.lwork_host[],
                M.info,
            )
            copyto!(x, M.rhs)

            return x
        end

        function solve_bunchkaufman_multirhs!(M::LapackGPUSolver{$typ}, N)
            copyto!(M.etc[:ipiv64], M.etc[:ipiv])
            copyto!(M.rhs, N)
            CUSOLVER.$sytrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                size(M.fact, 1),
                size(N, 2),
                $cutyp,
                M.fact,
                size(M.fact, 2),
                M.etc[:ipiv64],
                $cutyp,
                M.rhs,
                size(N, 1),
                M.work,
                M.lwork[],
                M.work_host,
                M.lwork_host[],
                M.info,
            )
            copyto!(N, M.rhs)
            return N
        end

        function factorize_lu!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef, size(M.A, 1)))
            transfer!(M.fact, M.A)
            tril_to_full!(M.fact)
            CUSOLVER.$getrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$getrf(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.work,
                M.etc[:ipiv],
                M.info,
            )
            return M
        end

        function solve_lu!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$getrs(
                dense_handle(),
                CUBLAS_OP_N,
                Int32(size(M.fact, 1)),
                Int32(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:ipiv],
                M.rhs,
                Int32(length(M.rhs)),
                M.info,
            )
            copyto!(x, M.rhs)
            return x
        end
        function solve_lu_multirhs!(M::LapackGPUSolver{$typ}, N)
            CUSOLVER.$getrs(
                dense_handle(),
                CUBLAS_OP_N,
                Int32(size(M.fact, 1)),
                Int32(size(N, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:ipiv],
                N,
                Int32(size(N, 1)),
                M.info,
            )
            return N
        end

        function factorize_qr!(M::LapackGPUSolver{$typ})
            haskey(M.etc, :tau) || (M.etc[:tau] = CuVector{$typ}(undef, size(M.A, 1)))
            transfer!(M.fact, M.A)
            tril_to_full!(M.fact)
            CUSOLVER.$geqrf_buffer(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$geqrf(
                dense_handle(),
                Int32(size(M.fact, 1)),
                Int32(size(M.fact, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_qr!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$ormqr_buffer(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(1),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.rhs,
                Int32(length(M.rhs)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$ormqr(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(1),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                M.rhs,
                Int32(length(M.rhs)),
                M.work,
                M.lwork[],
                M.info,
            )
            CUBLAS.$trsm(
                handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                Int32(size(M.fact, 1)),
                Int32(1),
                $typ(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.rhs,
                Int32(length(M.rhs)),
            )
            copyto!(x, M.rhs)
            return x
        end

        function solve_qr_multirhs!(M::LapackGPUSolver{$typ}, N)
            CUSOLVER.$ormqr_buffer(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(size(N, 2)),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                N,
                Int32(size(N, 1)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$ormqr(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Int32(size(M.fact, 1)),
                Int32(size(N, 2)),
                Int32(length(M.etc[:tau])),
                M.fact,
                Int32(size(M.fact, 2)),
                M.etc[:tau],
                N,
                Int32(size(N, 1)),
                M.work,
                M.lwork[],
                M.info,
            )
            CUBLAS.$trsm(
                handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                Int32(size(M.fact, 1)),
                Int32(size(N, 2)),
                $typ(1),
                M.fact,
                Int32(size(M.fact, 2)),
                N,
                Int32(size(N, 1)),
            )
            copyto!(N, M.rhs)
            return N
        end

        function factorize_cholesky!(M::LapackGPUSolver{$typ})
            transfer!(M.fact, M.A)
            CUSOLVER.$potrf_buffer(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.lwork,
            )
            length(M.work) < M.lwork[] && resize!(M.work, Int(M.lwork[]))
            CUSOLVER.$potrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                M.fact,
                Int32(size(M.fact, 2)),
                M.work,
                M.lwork[],
                M.info,
            )
            return M
        end

        function solve_cholesky!(M::LapackGPUSolver{$typ}, x)
            copyto!(M.rhs, x)
            CUSOLVER.$potrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                Int32(1),
                M.fact,
                Int32(size(M.fact, 2)),
                M.rhs,
                Int32(length(M.rhs)),
                M.info,
            )
            copyto!(x, M.rhs)
            return x
        end
        function solve_cholesky_multirhs!(M::LapackGPUSolver{$typ}, N)
            CUSOLVER.$potrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Int32(size(M.fact, 1)),
                Int32(size(N, 2)),
                M.fact,
                Int32(size(M.fact, 2)),
                N,
                Int32(size(N, 1)),
                M.info,
            )
            return N
        end
    end
end
function inertia(M::LapackGPUSolver)
    if M.opt.lapack_algorithm == MadNLP.CHOLESKY
        sum(M.info) == 0 ? (M.n, 0, 0) : (0, M.n, 0)
    elseif M.opt.lapack_algorithm == MadNLP.EVD
        numpos = count(λ -> λ > 0, M.Λ)
        numneg = count(λ -> λ < 0, M.Λ)
        numzero = M.n - numpos - numneg
        (numpos, numzero, numneg)
    else
        error(M.logger, "Invalid lapack_algorithm")
    end
end

input_type(::Type{LapackGPUSolver}) = :dense
MadNLP.default_options(::Type{LapackGPUSolver}) = LapackOptions()
introduce(M::LapackGPUSolver) = "cuSOLVER v$(CUSOLVER.version()) -- ($(M.opt.lapack_algorithm))"
is_supported(::Type{LapackGPUSolver}, ::Type{Float32}) = true
is_supported(::Type{LapackGPUSolver}, ::Type{Float64}) = true
