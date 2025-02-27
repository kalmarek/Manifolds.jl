module Manifolds

import ManifoldsBase:
    @trait_function,
    _access_nested,
    _get_basis,
    _injectivity_radius,
    _inverse_retract,
    _inverse_retract!,
    _read,
    _retract,
    _retract!,
    _write,
    active_traits,
    allocate,
    allocate_coordinates,
    allocate_result,
    allocate_result_type,
    allocation_promotion_function,
    array_value,
    base_manifold,
    check_point,
    check_size,
    check_vector,
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    decorated_manifold,
    distance,
    dual_basis,
    embed,
    embed!,
    exp,
    exp!,
    get_basis,
    get_basis_default,
    get_basis_diagonalizing,
    get_basis_orthogonal,
    get_basis_orthonormal,
    get_basis_vee,
    get_component,
    get_coordinates,
    get_coordinates!,
    get_coordinates_diagonalizing,
    get_coordinates_diagonalizing!,
    get_coordinates_orthogonal,
    get_coordinates_orthonormal,
    get_coordinates_orthogonal!,
    get_coordinates_orthonormal!,
    get_coordinates_vee!,
    get_embedding,
    get_iterator,
    get_vector,
    get_vector!,
    get_vector_diagonalizing,
    get_vector_diagonalizing!,
    get_vector_orthogonal,
    get_vector_orthonormal,
    get_vector_orthogonal!,
    get_vector_orthonormal!,
    get_vectors,
    gram_schmidt,
    hat,
    hat!,
    injectivity_radius,
    _injectivity_radius,
    injectivity_radius_exp,
    inner,
    isapprox,
    is_point,
    is_vector,
    inverse_retract,
    inverse_retract!,
    _inverse_retract,
    _inverse_retract!,
    inverse_retract_caley!,
    inverse_retract_embedded!,
    inverse_retract_nlsolve!,
    inverse_retract_pade!,
    inverse_retract_polar!,
    inverse_retract_project!,
    inverse_retract_qr!,
    inverse_retract_softmax!,
    log,
    log!,
    manifold_dimension,
    mid_point,
    mid_point!,
    norm,
    number_eltype,
    number_of_coordinates,
    parallel_transport_along,
    parallel_transport_along!,
    parallel_transport_direction,
    parallel_transport_direction!,
    parallel_transport_to,
    parallel_transport_to!,
    parent_trait,
    power_dimensions,
    project,
    project!,
    representation_size,
    retract,
    retract!,
    retract_caley!,
    retract_exp_ode!,
    retract_pade!,
    retract_polar!,
    retract_project!,
    retract_qr!,
    retract_softmax!,
    set_component!,
    vector_space_dimension,
    vector_transport_along, # just specified in Euclidean - the next 5 as well
    vector_transport_along_diff,
    vector_transport_along_project,
    vector_transport_along!,
    vector_transport_along_diff!,
    vector_transport_along_project!,
    vector_transport_direction,
    vector_transport_direction_diff,
    vector_transport_direction!,
    vector_transport_direction_diff!,
    vector_transport_to,
    vector_transport_to_diff,
    vector_transport_to_project,
    vector_transport_to!,
    vector_transport_to_diff!,
    vector_transport_to_project!, # some overwrite layer 2
    _vector_transport_direction,
    _vector_transport_direction!,
    _vector_transport_to,
    _vector_transport_to!,
    vee,
    vee!,
    zero_vector,
    zero_vector!,
    CotangentSpace,
    TangentSpace
import Base:
    copyto!,
    convert,
    foreach,
    identity,
    in,
    inv,
    isempty,
    length,
    ndims,
    showerror,
    size,
    transpose

using Base.Iterators: repeated
using Colors: RGBA
using Distributions
using Einsum: @einsum
using HybridArrays
using Kronecker
using Graphs
using LinearAlgebra
using ManifoldsBase:
    ℝ,
    ℂ,
    ℍ,
    AbstractBasis,
    AbstractDecoratorManifold,
    AbstractInverseRetractionMethod,
    AbstractManifold,
    AbstractManifoldPoint,
    AbstractNumbers,
    AbstractOrthogonalBasis,
    AbstractOrthonormalBasis,
    AbstractPowerManifold,
    AbstractPowerRepresentation,
    AbstractRetractionMethod,
    AbstractTrait,
    AbstractVectorTransportMethod,
    AbstractLinearVectorTransportMethod,
    ApproximateInverseRetraction,
    ApproximateRetraction,
    CachedBasis,
    CayleyRetraction,
    CayleyInverseRetraction,
    ComplexNumbers,
    ComponentManifoldError,
    CompositeManifoldError,
    CotangentSpaceType,
    CoTFVector,
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    DefaultOrDiagonalizingBasis,
    DiagonalizingBasisData,
    DiagonalizingOrthonormalBasis,
    DifferentiatedRetractionVectorTransport,
    EmbeddedManifold,
    EmptyTrait,
    ExponentialRetraction,
    FVector,
    IsIsometricEmbeddedManifold,
    IsEmbeddedManifold,
    IsEmbeddedSubmanifold,
    IsExplicitDecorator,
    LogarithmicInverseRetraction,
    ManifoldsBase,
    NestedPowerRepresentation,
    NestedReplacingPowerRepresentation,
    TraitList,
    NLSolveInverseRetraction,
    ODEExponentialRetraction,
    OutOfInjectivityRadiusError,
    PadeRetraction,
    PadeInverseRetraction,
    ParallelTransport,
    PolarInverseRetraction,
    PolarRetraction,
    PoleLadderTransport,
    PowerManifold,
    PowerManifoldNested,
    PowerManifoldNestedReplacing,
    ProjectedOrthonormalBasis,
    ProjectionInverseRetraction,
    ProjectionRetraction,
    ProjectionTransport,
    QuaternionNumbers,
    QRInverseRetraction,
    QRRetraction,
    RealNumbers,
    ScaledVectorTransport,
    SchildsLadderTransport,
    SoftmaxRetraction,
    SoftmaxInverseRetraction,
    TangentSpaceType,
    TCoTSpaceType,
    TFVector,
    TVector,
    ValidationManifold,
    ValidationMPoint,
    ValidationTVector,
    VectorSpaceType,
    VeeOrthogonalBasis,
    @invoke_maker,
    _euclidean_basis_vector,
    combine_allocation_promotion_functions,
    default_inverse_retraction_method,
    geodesic,
    merge_traits,
    next_trait,
    number_system,
    real_dimension,
    rep_size_to_colons,
    shortest_geodesic,
    size_to_tuple,
    trait
using Markdown: @doc_str
using MatrixEquations: lyapc
using Random
using RecipesBase
using RecipesBase: @recipe, @series
using RecursiveArrayTools: ArrayPartition
using Requires
using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, get_weight
using StaticArrays
using Statistics
using StatsBase
using StatsBase: AbstractWeights

include("utils.jl")

include("product_representations.jl")
include("differentiation/differentiation.jl")
include("differentiation/riemannian_diff.jl")
include("differentiation/embedded_diff.jl")

# Main Meta Manifolds
include("manifolds/ConnectionManifold.jl")
include("manifolds/MetricManifold.jl")
include("manifolds/VectorBundle.jl")
include("groups/group.jl")

# Features I: Which are extended on Meta Manifolds
include("distributions.jl")
include("projected_distribution.jl")
include("statistics.jl")

# Meta Manifolds II: Products
include("manifolds/ProductManifold.jl")

METAMANIFOLDS = [
    AbstractManifold,
    AbstractDecoratorManifold,
    AbstractPowerManifold,
    PowerManifoldNested,
    PowerManifoldNestedReplacing,
    ProductManifold,
    TangentSpaceAtPoint,
    ValidationManifold,
    VectorBundle,
]

# Features II: That require metas
include("atlases.jl")
include("cotangent_space.jl")

# Meta Manifolds II: Power Manifolds
include("manifolds/PowerManifold.jl")
include("manifolds/GraphManifold.jl")

#
# Manifolds
#
include("manifolds/Euclidean.jl")
include("manifolds/Lorentz.jl")

include("manifolds/CenteredMatrices.jl")
include("manifolds/CholeskySpace.jl")
include("manifolds/Circle.jl")
include("manifolds/Elliptope.jl")
include("manifolds/FixedRankMatrices.jl")
include("manifolds/GeneralizedGrassmann.jl")
include("manifolds/GeneralizedStiefel.jl")
include("manifolds/Grassmann.jl")
include("manifolds/Hyperbolic.jl")
include("manifolds/MultinomialDoublyStochastic.jl")
include("manifolds/MultinomialSymmetric.jl")
include("manifolds/ProbabilitySimplex.jl")
include("manifolds/PositiveNumbers.jl")
include("manifolds/ProjectiveSpace.jl")
include("manifolds/Rotations.jl")
include("manifolds/SkewHermitian.jl")
include("manifolds/Spectrahedron.jl")
include("manifolds/Stiefel.jl")
include("manifolds/StiefelEuclideanMetric.jl")
include("manifolds/StiefelCanonicalMetric.jl")
include("manifolds/Sphere.jl")
include("manifolds/SphereSymmetricMatrices.jl")
include("manifolds/Symmetric.jl")
include("manifolds/SymmetricPositiveDefinite.jl")
include("manifolds/SymmetricPositiveDefiniteBuresWasserstein.jl")
include("manifolds/SymmetricPositiveDefiniteGeneralizedBuresWasserstein.jl")
include("manifolds/SymmetricPositiveDefiniteLinearAffine.jl")
include("manifolds/SymmetricPositiveDefiniteLogCholesky.jl")
include("manifolds/SymmetricPositiveDefiniteLogEuclidean.jl")
include("manifolds/SymmetricPositiveSemidefiniteFixedRank.jl")
include("manifolds/Tucker.jl")
include("manifolds/Symplectic.jl")
include("manifolds/SymplecticStiefel.jl")

# Product or power based manifolds
include("manifolds/Torus.jl")
include("manifolds/Multinomial.jl")
include("manifolds/Oblique.jl")
include("manifolds/EssentialManifold.jl")

#
# Group Manifolds
include("groups/GroupManifold.jl")

# a) generics
include("groups/addition_operation.jl")
include("groups/multiplication_operation.jl")
include("groups/connections.jl")
include("groups/metric.jl")
include("groups/group_action.jl")
include("groups/group_operation_action.jl")
include("groups/validation_group.jl")
include("groups/product_group.jl")
include("groups/semidirect_product_group.jl")

# Special Group Manifolds
include("groups/general_linear.jl")
include("groups/special_linear.jl")
include("groups/translation_group.jl")
include("groups/special_orthogonal.jl")
include("groups/circle_group.jl")
include("groups/heisenberg.jl")

include("groups/translation_action.jl")
include("groups/rotation_action.jl")

include("groups/special_euclidean.jl")

@doc raw"""
    Base.in(p, M::AbstractManifold; kwargs...)
    p ∈ M

Check, whether a point `p` is a valid point (i.e. in) a [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
This method employs [`is_point`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.is_point) deactivating the error throwing option.
"""
Base.in(p, M::AbstractManifold; kwargs...) = is_point(M, p, false; kwargs...)

@doc raw"""
    Base.in(p, TpM::TangentSpaceAtPoint; kwargs...)
    X ∈ TangentSpaceAtPoint(M,p)

Check whether `X` is a tangent vector from (in) the tangent space $T_p\mathcal M$, i.e.
the [`TangentSpaceAtPoint`](@ref) at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
This method uses [`is_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.is_vector) deactivating the error throw option.
"""
function Base.in(X, TpM::TangentSpaceAtPoint; kwargs...)
    return is_vector(base_manifold(TpM), TpM.point, X, false; kwargs...)
end

function __init__()
    @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" begin
        using .FiniteDifferences
        include("differentiation/finite_differences.jl")
    end

    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        using .OrdinaryDiffEq: ODEProblem, AutoVern9, Rodas5, solve
        include("differentiation/ode.jl")
    end

    @require NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56" begin
        using .NLsolve: NLsolve
        include("nlsolve.jl")
    end

    @require Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40" begin
        using .Test: Test
        include("tests/tests_general.jl")
        export test_manifold
        include("tests/tests_group.jl")
        export test_group, test_action
    end

    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        using RecipesBase: @recipe, @series
        using Colors: RGBA
        include("recipes.jl")
    end

    @require RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01" begin
        @require Colors = "5ae59095-9a9b-59fe-a467-6f913c188581" begin
            using .RecipesBase: @recipe, @series
            using Colors: RGBA
            include("recipes.jl")
        end
    end

    return nothing
end

#
export CoTVector, AbstractManifold, AbstractManifoldPoint, TVector
export AbstractSphere, AbstractProjectiveSpace
export Euclidean,
    ArrayProjectiveSpace,
    ArraySphere,
    CenteredMatrices,
    CholeskySpace,
    Circle,
    Elliptope,
    EssentialManifold,
    FixedRankMatrices,
    GeneralizedGrassmann,
    GeneralizedStiefel,
    Grassmann,
    HeisenbergGroup,
    Hyperbolic,
    Lorentz,
    MultinomialDoubleStochastic,
    MultinomialMatrices,
    MultinomialSymmetric,
    Oblique,
    PositiveArrays,
    PositiveMatrices,
    PositiveNumbers,
    PositiveVectors,
    ProbabilitySimplex,
    ProjectiveSpace,
    Rotations,
    SkewHermitianMatrices,
    SkewSymmetricMatrices,
    Spectrahedron,
    Sphere,
    SphereSymmetricMatrices,
    Stiefel,
    SymmetricMatrices,
    SymmetricPositiveDefinite,
    SymmetricPositiveSemidefiniteFixedRank,
    Symplectic,
    SymplecticStiefel,
    SymplecticMatrix,
    Torus,
    Tucker
export HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint, SVDMPoint, TuckerPoint
export HyperboloidTVector,
    PoincareBallTVector, PoincareHalfSpaceTVector, UMVTVector, TuckerTVector
export AbstractNumbers, ℝ, ℂ, ℍ

# decorator manifolds
export AbstractDecoratorManifold
export IsIsometricEmbeddedManifold, IsEmbeddedManifold, IsEmbeddedSubmanifold
export IsDefaultMetric, IsDefaultConnection, IsMetricManifold, IsConnectionManifold
export ValidationManifold, ValidationMPoint, ValidationTVector, ValidationCoTVector
export CotangentBundle,
    CotangentSpaceAtPoint, CotangentBundleFibers, CotangentSpace, FVector
export AbstractPowerManifold,
    AbstractPowerRepresentation,
    ArrayPowerRepresentation,
    NestedPowerRepresentation,
    NestedReplacingPowerRepresentation,
    PowerManifold
export ProductManifold, EmbeddedManifold
export GraphManifold, GraphManifoldType, VertexManifold, EdgeManifold
export ProjectedPointDistribution, ProductRepr, TangentBundle, TangentBundleFibers
export TangentSpace, TangentSpaceAtPoint, VectorSpaceAtPoint, VectorSpaceType, VectorBundle
export VectorBundleFibers
export AbstractVectorTransportMethod,
    DifferentiatedRetractionVectorTransport, ParallelTransport, ProjectedPointDistribution
export PoleLadderTransport, SchildsLadderTransport
export ProductVectorTransport
export AbstractAffineConnection,
    AbstractConnectionManifold, ConnectionManifold, LeviCivitaConnection
export AbstractCartanSchoutenConnection,
    CartanSchoutenMinus, CartanSchoutenPlus, CartanSchoutenZero
export AbstractMetric,
    RiemannianMetric,
    LorentzMetric,
    BuresWassersteinMetric,
    EuclideanMetric,
    GeneralizedBuresWassersteinMetric,
    LinearAffineMetric,
    LogCholeskyMetric,
    LogEuclideanMetric,
    MinkowskiMetric,
    PowerMetric,
    ProductMetric,
    RealSymplecticMetric,
    ExtendedSymplecticMetric,
    CanonicalMetric,
    MetricManifold
export AbstractAtlas, RetractionAtlas
export AbstractVectorTransportMethod, ParallelTransport, ProjectionTransport
export AbstractRetractionMethod,
    CayleyRetraction,
    ExponentialRetraction,
    QRRetraction,
    PolarRetraction,
    ProjectionRetraction,
    SoftmaxRetraction,
    ODEExponentialRetraction,
    PadeRetraction,
    ProductRetraction,
    PowerRetraction
export AbstractInverseRetractionMethod,
    ApproximateInverseRetraction,
    ApproximateLogarithmicMap,
    CayleyInverseRetraction,
    LogarithmicInverseRetraction,
    QRInverseRetraction,
    PolarInverseRetraction,
    ProjectionInverseRetraction,
    SoftmaxInverseRetraction
export AbstractEstimationMethod,
    GradientDescentEstimation,
    CyclicProximalPointEstimation,
    GeodesicInterpolation,
    GeodesicInterpolationWithinRadius,
    ExtrinsicEstimation
export CachedBasis,
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    InducedBasis,
    ProjectedOrthonormalBasis
export ComponentManifoldError, CompositeManifoldError
export ×,
    allocate,
    allocate_result,
    base_manifold,
    bundle_projection,
    change_metric,
    change_metric!,
    change_representer,
    change_representer!,
    check_point,
    check_vector,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    convert,
    complex_dot,
    decorated_manifold,
    det_local_metric,
    distance,
    dual_basis,
    einstein_tensor,
    embed,
    embed!,
    exp,
    exp!,
    flat,
    flat!,
    gaussian_curvature,
    geodesic,
    get_default_atlas,
    get_component,
    get_embedding,
    grad_euclidean_to_manifold,
    grad_euclidean_to_manifold!,
    hat,
    hat!,
    identity_element,
    identity_element!,
    induced_basis,
    incident_log,
    injectivity_radius,
    inner,
    inverse_local_metric,
    inverse_retract,
    inverse_retract!,
    isapprox,
    is_default_connection,
    is_default_metric,
    is_group_manifold,
    is_identity,
    is_point,
    is_vector,
    kurtosis,
    local_metric,
    local_metric_jacobian,
    log,
    log!,
    log_local_metric_density,
    manifold_dimension,
    metric,
    mean,
    mean!,
    mean_and_var,
    mean_and_std,
    median,
    median!,
    mid_point,
    mid_point!,
    minkowski_metric,
    moment,
    norm,
    normal_tvector_distribution,
    number_eltype,
    one,
    power_dimensions,
    project,
    project!,
    projected_distribution,
    real_dimension,
    ricci_curvature,
    ricci_tensor,
    representation_size,
    retract,
    retract!,
    riemann_tensor,
    set_component!,
    sharp,
    sharp!,
    shortest_geodesic,
    skewness,
    std,
    sym_rem,
    symplectic_inverse_times,
    symplectic_inverse_times!,
    submanifold,
    submanifold_component,
    submanifold_components,
    uniform_distribution,
    var,
    vector_space_dimension,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    vee,
    vee!,
    zero_vector,
    zero_vector!
# Lie group types & functions
export AbstractGroupAction,
    AbstractGroupOperation,
    ActionDirection,
    AdditionOperation,
    CircleGroup,
    GeneralLinear,
    GroupManifold,
    GroupOperationAction,
    Identity,
    InvariantMetric,
    LeftAction,
    LeftInvariantMetric,
    MultiplicationOperation,
    ProductGroup,
    ProductOperation,
    RealCircleGroup,
    RightAction,
    RightInvariantMetric,
    RotationAction,
    SemidirectProductGroup,
    SpecialEuclidean,
    SpecialLinear,
    SpecialOrthogonal,
    TranslationGroup,
    TranslationAction
export AbstractInvarianceTrait
export IsMetricManifold, IsConnectionManifold
export IsGroupManifold,
    HasLeftInvariantMetric, HasRightInvariantMetric, HasBiinvariantMetric
export adjoint_action,
    adjoint_action!,
    affine_matrix,
    apply,
    apply!,
    apply_diff,
    apply_diff!,
    base_group,
    center_of_orbit,
    has_approx_invariant_metric,
    compose,
    compose!,
    direction,
    exp_lie,
    exp_lie!,
    group_manifold,
    geodesic,
    get_coordinates_lie,
    get_coordinates_lie!,
    get_coordinates_orthogonal,
    get_coordinates_orthonormal,
    get_coordinates_orthogonal!,
    get_coordinates_orthonormal!,
    get_vector_diagonalizing!,
    get_vector_lie,
    get_vector_lie!,
    get_vector_orthogonal,
    get_vector_orthonormal,
    get_coordinates_vee!,
    has_biinvariant_metric,
    has_invariant_metric,
    identity_element,
    identity_element!,
    inv,
    inv!,
    inverse_apply,
    inverse_apply!,
    inverse_apply_diff,
    inverse_apply_diff!,
    inverse_translate,
    inverse_translate!,
    inverse_translate_diff,
    inverse_translate_diff!,
    lie_bracket,
    lie_bracket!,
    log_lie,
    log_lie!,
    optimal_alignment,
    optimal_alignment!,
    screw_matrix,
    switch_direction,
    translate,
    translate!,
    translate_diff,
    translate_diff!
# Orthonormal bases
export AbstractBasis,
    AbstractOrthonormalBasis,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    ProjectedOrthonormalBasis,
    CachedBasis,
    DiagonalizingBasisData,
    ProductBasisData,
    PowerBasisData
export OutOfInjectivityRadiusError
export get_basis,
    get_coordinates, get_coordinates!, get_vector, get_vector!, get_vectors, number_system

# atlases and charts
export get_point, get_point!, get_parameters, get_parameters!

end # module
