@doc raw"""
    ProjectorPoint <: AbstractManifoldPoint

A type to represent points on a manifold that are orthogonal projectors.
"""
struct ProjectorPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

@doc raw"""
    ProjectorTVector <: TVector

A type to represent tangent vectors to points on a manifold that are orthogonal projectors.
"""
struct ProjectorTVector{T<:AbstractMatrix} <: TVector
    value::T
end

@manifold_element_forwards ProjectorPoint value
@manifold_vector_forwards ProjectorTVector value

@doc raw"""
    check_point(::Grassmann{n,k}, p::ProjectorPoint; kwargs...)

Check whether a orthogonal projector is a point from the [`Grassmann`](@ref)`(n,k)` manifold,
i.e. the [`ProjectorPoint`](@ref) ``p ∈ \mathbb R^{n×n}`` has to fulfill
``p^{\mathrm{T}} = p``, ``p^2=p``, and ``\operatorname{rank} p = k`.
"""
function check_point(M::Grassmann{n,k,𝔽}, p::ProjectorPoint; kwargs...) where {n,k,𝔽}
    mpv = check_point(Euclidean(n, n; field=𝔽), p.value; kwargs...)
    mpv === nothing || return mpv
    c = p.value * p.value
    if !isapprox(c, p.value; kwargs...)
        return DomainError(
            norm(c - p.value),
            "The point $(p) is not equal to its square $c, so it does not lie on $M.",
        )
    end
    if !isapprox(p.value, transpose(p.value); kwargs...)
        return DomainError(
            norm(c - p),
            "The point $(p) is not equal to its transpose, so it does not lie on $M.",
        )
    end
    k2 = rank(p.value; kwargs...)
    if k2 != k
        return DomainError(
            k2,
            "The point $(p) is a projector of rank $k2 and not of rank $k, so it does not lie on $(M).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(::Grassmann{n,k,𝔽}, p::ProjectorPoint, X::ProjectorTVector; kwargs...) where {n,k,𝔽}

Check whether the [`ProjectorTVector`]('ref) `X` is from the tangent space ``T_p\operatorname{Gr}(n,k) ``
at the [`ProjectorPoint`](@ref) `p` on the [`Grassmann`](@ref) manifold ``\operatorname{Gr}(n,k)``.

This means that `X` has to be symmetric and that

```math
Xp + pX = X
```
must hold, where the `kwargs` can be used to check both for symmetrix of ``X```
and this equality up to a certain tolerance.
"""
function check_vector(
    M::Grassmann{n,k,𝔽},
    p::ProjectorPoint,
    X::ProjectorTVector;
    kwargs...,
) where {n,k,𝔽}
    if !isapprox(norm(X.value - X.value'), 0.0; kwargs...)
        return DomainError(
            norm(X.value - X.value'),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    if !isapprox(X.value * p.value + p.value * X.value, X.value; kwargs...)
        return DomainError(
            norm(X.value * p.value + p.value * X.value - X.value),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since X*p + p*X is not equal to X.",
        )
    end
    return nothing
end

Base.show(io::IO, p::ProjectorPoint) = print(io, "ProjectorPoint($(p.value))")
Base.show(io::IO, X::ProjectorTVector) = print(io, "ProjectorTVector($(X.value))")
