# Contributing to `Manifolds.jl`

First, thanks for taking the time to contribute.
Any contribution is appreciated and welcome.

The following is a set of guidelines to [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/).

#### Table of Contents

- [Contributing to `Manifolds.jl`](#contributing-to-manifoldsjl)
      - [Table of Contents](#table-of-contents)
  - [I just have a question](#i-just-have-a-question)
  - [How can I file an issue?](#how-can-i-file-an-issue)
  - [How can I contribute?](#how-can-i-contribute)
    - [Add a missing method](#add-a-missing-method)
    - [Provide a new manifold](#provide-a-new-manifold)
    - [Code style](#code-style)

## I just have a question

The developers can most easily be reached in the Julia Slack channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z).
You can apply for the Julia Slack workspace [here](https://julialang.org/slack/) if you haven't joined yet.
You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org).

## How can I file an issue?

If you found a bug or want to propose a feature, we track our issues within the [GitHub repository](https://github.com/JuliaManifolds/Manifolds.jl/issues).

## How can I contribute?

### Add a missing method

Not all methods from our interface [`ManifoldsBase.jl`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html) have been implemented for every manifold.
If you notice a method missing and can contribute an implementation, please do so!
Even providing a single new method is a good contribution.

### Provide a new manifold

A main contribution you can provide is another manifold that is not yet included in the
package.
A manifold is a concrete type of [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold) from [`ManifoldsBase.jl`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html).
This package also provides the main set of functions a manifold can/should implement.
Don't worry if you can only implement some of the functions.
If the application you have in mind only requires a subset of these functions, implement those.
The [`ManifoldsBase.jl`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html) interface provides concrete error messages for the remaining unimplemented functions.

One important detail is that the interface usually provides a mutating as well as a non-mutating variant
See for example [exp!](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.exp!-Tuple{AbstractManifold,Any,Any,Any}) and [exp](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Base.exp-Tuple{AbstractManifold,Any,Any}).
The non-mutating one (e.g. `exp`) always falls back to use the mutating one, so in most cases it should
suffice to implement the mutating one (e.g. `exp!`).

Note that since the first argument is _always_ the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold), the mutated argument is always the second one in the signature.
In the example we have `exp(M, p, X)` for the exponential map and `exp!(M, q, X, p)` for the mutating one, which stores the result in `q`.

On the other hand, the user will most likely look for the documentation of the non-mutating version, so we recommend adding the docstring for the non-mutating one, where all different signatures should be collected in one string when reasonable.
This can best be achieved by adding a docstring to the method with a general signature with the first argument being your manifold:

````julia
struct MyManifold <: AbstractManifold end

@doc raw"""
    exp(M::MyManifold, p, X)

Describe the function.
"""
exp(::MyManifold, ::Any...)
````

### Code style

We try to follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from the Julia documentation as well as [Blue Style](https://github.com/invenia/BlueStyle).
We run [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) on the repo in the way set in the `.JuliaFormatter.toml` file, which enforces a number of conventions consistent with the Blue Style.

We also follow a few internal conventions:

- It is preferred that the `AbstractManifold`'s struct contain a reference to the general theory.
- Any implemented function should be accompanied by its mathematical formulae if a closed form exists.
- Within the source code of one manifold, the type of the manifold should be the first element of the file, and an alphabetical order of the functions is preferable.
- The above implies that the mutating variant of a function follows the non-mutating variant.
- There should be no dangling `=` signs.
- Always add a newline between things of different types (struct/method/const).
- Always add a newline between methods for different functions (including mutating/nonmutating variants).
- Prefer to have no newline between methods for the same function; when reasonable, merge the docstrings.
- All `import`/`using`/`include` should be in the main module file.
