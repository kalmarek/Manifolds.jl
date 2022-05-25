@doc raw"""
    Grassmann{n,k,𝔽} <: AbstractDecoratorManifold{𝔽}

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
struct Grassmann{n,k,𝔽} <: AbstractDecoratorManifold{𝔽} end

Grassmann(n::Int, k::Int, field::AbstractNumbers=ℝ) = Grassmann{n,k,field}()

active_traits(f, ::Grassmann, args...) = merge_traits(IsIsometricEmbeddedManifold())

function allocation_promotion_function(M::Grassmann{n,k,ℂ}, f, args::Tuple) where {n,k}
    return complex
end

@doc raw"""
    distance(M::Grassmann, p, q)

Compute the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = p^\mathrm{H}q$ denote the SVD decomposition of
$p^\mathrm{H}q$, where $\cdot^{\mathrm{H}}$ denotes the complex
conjugate transposed or Hermitian. Then the distance is given by
````math
d_{\mathrm{Gr}(n,k)}(p,q) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

````math
b_{i}=\begin{cases}
0 & \text{if} \; S_i ≥ 1\\
\arccos(S_i) & \, \text{if} \; S_i<1.
\end{cases}
````
"""
function distance(::Grassmann, p, q)
    p ≈ q && return zero(real(eltype(p)))
    a = svd(p' * q).S
    return sqrt(sum(x -> abs2(acos(clamp(x, -1, 1))), a))
end

embed(::Grassmann, p) = p
embed(::Grassmann, p, X) = X

@doc raw"""
    exp(M::Grassmann, p, X)

Compute the exponential map on the [`Grassmann`](@ref) `M`$= \mathrm{Gr}(n,k)$ starting in
`p` with tangent vector (direction) `X`. Let $X = USV$ denote the SVD decomposition of $X$.
Then the exponential map is written using

````math
z = p V\cos(S)V^\mathrm{H} + U\sin(S)V^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian and the
cosine and sine are applied element wise to the diagonal entries of $S$. A final QR
decomposition $z=QR$ is performed for numerical stability reasons, yielding the result as

````math
\exp_p X = Q.
````
"""
exp(::Grassmann, ::Any...)

function exp!(M::Grassmann, q, p, X)
    norm(M, p, X) ≈ 0 && return copyto!(q, p)
    d = svd(X)
    z = p * d.V * Diagonal(cos.(d.S)) * d.Vt + d.U * Diagonal(sin.(d.S)) * d.Vt
    return copyto!(q, Array(qr(z).Q))
end

function get_embedding(::Grassmann{N,K,𝔽}) where {N,K,𝔽}
    return Stiefel(N, K, 𝔽)
end

@doc raw"""
    injectivity_radius(M::Grassmann)
    injectivity_radius(M::Grassmann, p)

Return the injectivity radius on the [`Grassmann`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Grassmann) = π / 2
injectivity_radius(::Grassmann, p) = π / 2
injectivity_radius(::Grassmann, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::Grassmann, p, ::AbstractRetractionMethod) = π / 2

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
