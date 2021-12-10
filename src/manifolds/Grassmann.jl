@doc raw"""
    Grassmann{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned by $k$ linear independent
vectors $𝔽^n$, where $𝔽  ∈ \{ℝ, ℂ\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $ℝ^n$ for the real-valued case and all $2k$-dimensional subspaces
of $ℂ^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n,k) := \bigl\{ \operatorname{span}(p) : p ∈ 𝔽^{n × k}, p^\mathrm{H}p = I_k\},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and
$I_k$ is the $k × k$ identity matrix. This means, that the columns of $p$
form an unitary basis of the subspace, that is a point on
$\operatorname{Gr}(n,k)$, and hence the subspace can actually be represented by
a whole equivalence class of representers.
Another interpretation is, that

````math
\operatorname{Gr}(n,k) = \operatorname{St}(n,k) / \operatorname{O}(k),
````

i.e the Grassmann manifold is the quotient of the [`Stiefel`](@ref) manifold and
the orthogonal group $\operatorname{O}(k)$ of orthogonal $k × k$ matrices.

The tangent space at a point (subspace) $x$ is given by

````math
T_x\mathrm{Gr}(n,k) = \bigl\{
X ∈ 𝔽^{n × k} :
X^{\mathrm{H}}p + p^{\mathrm{H}}X = 0_{k} \bigr\},
````

where $0_k$ is the $k × k$ zero matrix.

Note that a point $p ∈ \operatorname{Gr}(n,k)$ might be represented by
different matrices (i.e. matrices with unitary column vectors that span
the same subspace). Different representations of $p$ also lead to different
representation matrices for the tangent space $T_p\mathrm{Gr}(n,k)$

For a representation of points as orthogonal projectors see [`ProjectorPoint`](@ref)
and [`ProjectorTVector`](@ref).

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,field=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case `field = ℝ` is the default.
"""
struct Grassmann{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

Grassmann(n::Int, k::Int, field::AbstractNumbers=ℝ) = Grassmann{n,k,field}()

function allocation_promotion_function(::Grassmann{n,k,ℂ}, f, ::Tuple) where {n,k}
    return complex
end

decorated_manifold(::Grassmann{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field=𝔽)

@doc raw"""
    injectivity_radius(M::Grassmann)
    injectivity_radius(M::Grassmann, p)

Return the injectivity radius on the [`Grassmann`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Grassmann) = π / 2
injectivity_radius(::Grassmann, ::ExponentialRetraction) = π / 2
injectivity_radius(::Grassmann, ::Any) = π / 2
injectivity_radius(::Grassmann, ::Any, ::ExponentialRetraction) = π / 2
eval(
    quote
        @invoke_maker 1 AbstractManifold injectivity_radius(
            M::Grassmann,
            rm::AbstractRetractionMethod,
        )
    end,
)

include("GrassmannStiefel.jl")
include("GrassmannProjector.jl")

#
# Conversion
#
function convert(::Type{ProjectorPoint}, p::AbstractMatrix)
    return ProjectorPoint(p * p')
end
function convert(T::Type{ProjectorPoint}, p::GrassmannBasisPoint)
    return convert(T, p.value)
end
