raw"""
    ProjectorPoint <: AbstractManifoldPoint

A type to represent points on a manifold that are orthogonal projectors.
"""
struct ProjectorPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

raw"""
    ProjectorTVector <: TVector

A type to represent tangent vectors to points on a manifold that are orthogonal projectors.
"""
struct ProjectorTVector{T<:AbstractMatrix} <: TVector
    value::T
end

@manifold_element_forwards ProjectorPoint value
@manifold_vector_forwards ProjectorTVector value

raw"""
    check_point(::Grassmann{n,k}, p::ProjectorPoint)

Check whether a orthogonal projector is a point from the [`Grassmann`](@ref)`(n,k)` manifold,
i.e. the [`ProjectorPoint`](@ref) ``p ∈ \mathbb R^{n×n}`` has to fulfill
``p^{\mathrm{T}} = p``, ``p^2=p``, and ``\operatorname{rank} p = k`.
"""
function check_point(::Grassmann{n,k,𝔽}, p::ProjectorPoint) where {n,k,𝔽}
    mpv = check_point(Euclidean(n, n; field=𝔽), p; kwargs...)
    mpv === nothing || return mpv
    c = p.value * p.value
    if !isapprox(c, p; kwargs...)
        return DomainError(
            norm(c - p),
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
            "The point $(p) is a projector of rank $k2 and not of rank $k, so it does not lie on $M.",
        )
    end
    return nothing
end

Base.show(io::IO, p::ProjectorPoint) = print(io, "ProjectorPoint($(p.value))")
Base.show(io::IO, X::ProjectorTVector) = print(io, "ProjectorTVector($(X.value))")
