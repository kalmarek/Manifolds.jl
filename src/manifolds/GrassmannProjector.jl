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

_GrassmannTantentTypes = [ProjectorTVector]
_GrassmannPointTypes = [ProjectorPoint]
_GrassmannTypes = [_GrassmannPointTypes..., _GrassmannTantentTypes...]

for T in _GrassmannTantentTypes
    @eval begin
        Base.:*(v::$T, s::Number) = $T(v.value * s)
        Base.:*(s::Number, v::$T) = $T(s * v.value)
        Base.:/(v::$T, s::Number) = $T(v.value / s)
        Base.:\(s::Number, v::$T) = $T(s \ v.value)
        Base.:+(v::$T, w::$T) = $T(v.value + w.value)
        Base.:-(v::$T, w::$T) = $T(v.value - w.value)
        Base.:-(v::$T) = $T(-v.value)
        Base.:+(v::$T) = $T(v.value)
    end
end

for T in _GrassmannTypes
    @eval begin
        Base.:(==)(v::$T, w::$T) = (v.value == w.value)

        allocate(p::$T) = $T(allocate(p.value))
        allocate(p::$T, ::Type{P}) where {P} = $T(allocate(p.value, P))
        allocate(p::$T, ::Type{P}, dims::Tuple) where {P} = $T(allocate(p.value, P, dims))

        @inline Base.copy(p::$T) = $T(copy(p.value))
        function Base.copyto!(q::$T, p::$T)
            copyto!(q.value, p.value)
            return q
        end

        Base.similar(p::$T) = $T(similar(p.value))

        function Broadcast.BroadcastStyle(::Type{<:$T})
            return Broadcast.Style{$T}()
        end
        function Broadcast.BroadcastStyle(
            ::Broadcast.AbstractArrayStyle{0},
            b::Broadcast.Style{$T},
        )
            return b
        end

        Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{$T},Nothing}) = bc
        function Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{$T}})
            Broadcast.check_broadcast_axes(bc.axes, bc.args...)
            return bc
        end

        Broadcast.broadcastable(v::$T) = v

        @inline function Base.copy(bc::Broadcast.Broadcasted{Broadcast.Style{$T}})
            return $T(Broadcast._broadcast_getindex(bc, 1))
        end

        Base.@propagate_inbounds Broadcast._broadcast_getindex(v::$T, I) = v.value

        Base.axes(v::$T) = axes(v.value)

        @inline function Base.copyto!(
            dest::$T,
            bc::Broadcast.Broadcasted{Broadcast.Style{$T}},
        )
            axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
            # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
            if bc.f === identity && bc.args isa Tuple{$T} # only a single input argument to broadcast!
                A = bc.args[1]
                if axes(dest) == axes(A)
                    return copyto!(dest, A)
                end
            end
            bc′ = Broadcast.preprocess(dest, bc)
            # Performance may vary depending on whether `@inbounds` is placed outside the
            # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
            copyto!(dest.value, bc′[1])
            return dest
        end
    end
end

for (P, T) in zip(_GrassmannPointTypes, _GrassmannTantentTypes)
    @eval allocate(p::$P, ::Type{$T}) = $T(allocate(p.value))
    @eval allocate_result_type(::Grassmann, ::typeof(log), ::Tuple{$P,$P}) = $T
    @eval allocate_result_type(::Grassmann, ::typeof(inverse_retract), ::Tuple{$P,$P}) = $T
end

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

for T in _GrassmannPointTypes
    @eval function isapprox(::Grassmann, p::$T, q::$T; kwargs...)
        return isapprox(p.value, q.value; kwargs...)
    end
end
for (P, T) in zip(_GrassmannPointTypes, _GrassmannTantentTypes)
    @eval function isapprox(::Grassmann, ::$P, X::$T, Y::$T; kwargs...)
        return isapprox(X.value, Y.value; kwargs...)
    end
end
for T in _GrassmannTypes
    @eval Base.show(io::IO, p::$T) = print(io, "$($T)($(p.value))")
end
