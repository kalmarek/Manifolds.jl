#
# Default implementation for the matrix type, i.e. as congruence class Stiefel matrices
#
"""
    GrassmannBasisPoint <: AbstractManifoldPoint

A point on the [`Grassmann`](@ref) manifold represented by an orthonormal basis,
i.e. a point on the corresponding [`Stiefel`](@ref) manifold. Note that a point
represented this way has several points on the Stiefel manifold it corresponds to,
that is all bases that span the same subspace as the point represented here.

!!! note
    The [`Grassmann`](@ref) manifold has this type as default type in the sense that
    using this type for points `p` is the same as using arbitrary `AbstractMatrix` types.
    This type is mainly provided for completeness.
"""
struct GrassmannBasisPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

"""
    GrassmannBasisTVector <: TVector

A tangent vector on the [`Grassmann`](@ref) manifold represented by a tangent vector from
the tangent space of a corresponding point from the [`Stiefel`](@ref) manifold,
see [`GrassmannBasisPoint`](@ref).


!!! note
    The [`Grassmann`](@ref) manifold has this type as default type in the sense that
    using this type for tangent vectors `X` is the same as using arbitrary `AbstractMatrix` types.
    This type is mainly provided for completeness.
"""
struct GrassmannBasisTVector{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

@default_manifold_fallbacks Grassmann GrassmannBasisPoint GrassmannBasisTVector value value

@doc raw"""
    inner(M::Grassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`Grassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inner(::Grassmann, p, X, Y) = dot(X, Y)

@doc raw"""
    inverse_retract(M::Grassmann, p, q, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q*(p^\mathrm{H}q)^{-1} - p,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract_polar!(::Grassmann, X, p, q)
    return copyto!(X, q / (p' * q) - p)
end

@doc raw"""
    inverse_retract(M, p, q, ::QRInverseRetraction)

Compute the inverse retraction for the [`QRRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.QRRetraction), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q(p^\mathrm{H}q)^{-1} - p,
````
where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::QRInverseRetraction)

inverse_retract_qr!(::Grassmann, X, p, q) = copyto!(X, q / (p' * q) - p)

function Base.isapprox(M::Grassmann, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_vector(M, p), X - Y)), 0; kwargs...)
end
Base.isapprox(M::Grassmann, p, q; kwargs...) = isapprox(distance(M, p, q), 0.0; kwargs...)

@doc raw"""
    log(M::Grassmann, p, q)

Compute the logarithmic map on the [`Grassmann`](@ref) `M`$ = \mathcal M=\mathrm{Gr}(n,k)$,
i.e. the tangent vector `X` whose corresponding [`geodesic`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.geodesic-Tuple{AbstractManifold,%20Any,%20Any}) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = V\cdot \operatorname{atan}(S) \cdot U^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
The matrices $U$ and $V$ are the unitary matrices, and $S$ is the diagonal matrix
containing the singular values of the SVD-decomposition

````math
USV = (q^\mathrm{H}p)^{-1} ( q^\mathrm{H} - q^\mathrm{H}pp^\mathrm{H}).
````

In this formula the $\operatorname{atan}$ is meant elementwise.
"""
log(::Grassmann, ::Any...)

function log!(::Grassmann{n,k}, X, p, q) where {n,k}
    z = q' * p
    At = q' - z * p'
    Bt = z \ At
    d = svd(Bt')
    return X .= view(d.U, :, 1:k) * Diagonal(atan.(view(d.S, 1:k))) * view(d.Vt, 1:k, :)
end

@doc raw"""
    manifold_dimension(M::Grassmann)

Return the dimension of the [`Grassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k) = k(n-k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}) of `𝔽`.
"""
manifold_dimension(::Grassmann{n,k,𝔽}) where {n,k,𝔽} = k * (n - k) * real_dimension(𝔽)

"""
    mean(
        M::Grassmann,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Grassmann{n,k} where {n,k}, ::Any...)

function default_estimation_method(::Grassmann, ::typeof(mean))
    return GeodesicInterpolationWithinRadius(π / 4)
end

@doc raw"""
    project(M::Grassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}X,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
project(::Grassmann, ::Any...)

project!(::Grassmann, Y, p, X) = copyto!(Y, X - p * p' * X)

@doc raw"""
    rand(M::Grassmann; σ::Real=1.0, vector_at=nothing)

When `vector_at` is `nothing`, return a random point `p` on [`Grassmann`](@ref) manifold `M`
by generating a random (Gaussian) matrix with standard deviation `σ` in matching
size, which is orthonormal.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_p\mathrm{Gr}(n,k)`` with mean zero and standard deviation `σ` by projecting a random
Matrix onto the tangent space at `vector_at`.
"""
rand(M::Grassmann; σ::Real=1.0)

function Random.rand!(
    M::Grassmann{n,k,𝔽},
    pX;
    σ::Real=one(real(eltype(pX))),
    vector_at=nothing,
) where {n,k,𝔽}
    if vector_at === nothing
        V = σ * randn(𝔽 === ℝ ? Float64 : ComplexF64, (n, k))
        pX .= qr(V).Q[:, 1:k]
    else
        Z = σ * randn(eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
        pX .= pX ./ norm(pX)
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::Grassmann{n,k,𝔽},
    pX;
    σ::Real=one(real(eltype(pX))),
    vector_at=nothing,
) where {n,k,𝔽}
    if vector_at === nothing
        V = σ * randn(rng, 𝔽 === ℝ ? Float64 : ComplexF64, (n, k))
        pX .= qr(V).Q[:, 1:k]
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
        pX .= pX ./ norm(pX)
    end
    return pX
end

@doc raw"""
    representation_size(M::Grassmann{n,k})

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M`, i.e. $(n,k)$ for both the real-valued and the complex value case.
"""
@generated representation_size(::Grassmann{n,k}) where {n,k} = (n, k)

@doc raw"""
    retract(M::Grassmann, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction) on the
[`Grassmann`](@ref) `M`. With $USV = p + X$ the retraction reads
````math
\operatorname{retr}_p X = UV^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
retract(::Grassmann, ::Any, ::Any, ::PolarRetraction)

function retract_polar!(::Grassmann, q, p, X)
    s = svd(p + X)
    return mul!(q, s.U, s.Vt)
end

@doc raw"""
    retract(M::Grassmann, p, X, ::QRRetraction )

Compute the QR-based retraction [`QRRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.QRRetraction) on the
[`Grassmann`](@ref) `M`. With $QR = p + X$ the retraction reads
````math
\operatorname{retr}_p X = QD,
````
where D is a $m × n$ matrix with
````math
D = \operatorname{diag}\left( \operatorname{sgn}\left(R_{ii}+\frac{1}{2}\right)_{i=1}^n \right).
````
"""
retract(::Grassmann, ::Any, ::Any, ::QRRetraction)

function retract_qr!(::Grassmann{N,K}, q, p, X) where {N,K}
    qrfac = qr(p + X)
    d = diag(qrfac.R)
    D = Diagonal(sign.(d .+ 1 // 2))
    mul!(q, Array(qrfac.Q), D)
    return q
end

function Base.show(io::IO, ::Grassmann{n,k,𝔽}) where {n,k,𝔽}
    return print(io, "Grassmann($(n), $(k), $(𝔽))")
end

"""
    uniform_distribution(M::Grassmann{n,k,ℝ}, p)

Uniform distribution on given (real-valued) [`Grassmann`](@ref) `M`.
Specifically, this is the normalized Haar measure on `M`.
Generated points will be of similar type as `p`.

The implementation is based on Section 2.5.1 in [^Chikuse2003];
see also Theorem 2.2.2(iii) in [^Chikuse2003].

[^Chikuse2003]:
    > Y. Chikuse: "Statistics on Special Manifolds", Springer New York, 2003,
    > doi: [10.1007/978-0-387-21540-2](https://doi.org/10.1007/978-0-387-21540-2).
"""
function uniform_distribution(M::Grassmann{n,k,ℝ}, p) where {n,k}
    μ = Distributions.Zeros(n, k)
    σ = one(eltype(p))
    Σ1 = Distributions.PDMats.ScalMat(n, σ)
    Σ2 = Distributions.PDMats.ScalMat(k, σ)
    d = MatrixNormal(μ, Σ1, Σ2)

    return ProjectedPointDistribution(M, d, (M, q, p) -> (q .= svd(p).U), p)
end

@doc raw"""
    vector_transport_to(M::Grassmann,p,X,q,::ProjectionTransport)

compute the projection based transport on the [`Grassmann`](@ref) `M` by
interpreting `X` from the tangent space at `p` as a point in the embedding and
projecting it onto the tangent space at q.
"""
vector_transport_to(::Grassmann, ::Any, ::Any, ::Any, ::ProjectionTransport)

@doc raw"""
    zero_vector(M::Grassmann, p)

Return the zero tangent vector from the tangent space at `p` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `p`.
"""
zero_vector(::Grassmann, ::Any...)

zero_vector!(::Grassmann, X, p) = fill!(X, 0)
