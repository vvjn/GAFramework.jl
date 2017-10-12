# GAFramework: a genetic algorithm framework with multi-threading

[![Build Status](https://travis-ci.org/vvjn/GAFramework.jl.svg?branch=master)](https://travis-ci.org/vvjn/GAFramework.jl)

[![Coverage Status](https://coveralls.io/repos/vvjn/GAFramework.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/vvjn/GAFramework.jl?branch=master)

[![codecov.io](http://codecov.io/github/vvjn/GAFramework.jl/coverage.svg?branch=master)](http://codecov.io/github/vvjn/GAFramework.jl?branch=master)


GAFramework is a framework for writing genetic algorithms that is very
customizable. It also supports multi-threading, calculating crossovers
and fitness values in parallel.

## Installation

`Pkg.clone("https://github.com/vvjn/GAFramework.jl")`

This requires the JLD and StaticArrays packages.

## Creating a genetic algorithm

To create a GA for a specific problem, we need to create concrete
sub-types of the abstract types `GAModel` and `GACreature`, and then
create relevant functions for the sub-types as explained below.

Here, we create a GA for optimizing a function over a rectangle in a
multi-dimensional space, i.e., a function `f : R^n -> R`.

First, we import the `GAFramework` module and import relevant
functions. We also represent a point as a static vector `SVector`, and
so we also import the `StaticArrays` package.

```julia
using GAFramework
import GAFramework: fitness, genauxga, crossover, mutate, selection,
randcreature,printfitness, savecreature

using StaticArrays
```

Here, we create a sub-type of `GAModel`, which contains the corners of
the rectangle (`xmin` and `xmax`) and the span of the rectangle
(`xspan`). We also add the `clamp` field: if `clamp = tru` then we
will clamp mutated points back into the rectangle, so that our
solutions will be inside the rectangle; otherwise, our solutions will
not restricted to the rectangle. We store the function `F` inside the
type since it does not change throughout the algorithm.

```julia
immutable EuclideanModel{F,T} <: GAModel
    xmin::T
    xmax::T
    xspan::T # xmax-xmin
    clamp::Bool
end

function EuclideanModel(F,xmin,xmax,clamp::Bool=true)
    D = length(xmin)
    ymin = SVector{D}(xmin)
    ymax = SVector{D}(xmax)
    yspan = ymax .- ymin
    EuclideanModel{F,typeof(yspan)}(ymin,ymax,yspan,clamp)
end
```

Here, we create a sub-type of `GACreature`, which contains the
"chromosomes" of the creature (`value`) and the objective value of the
function (`objvalue`). We can calculate the objective value using the
`EuclideanModel{F,T}` type.

```julia
immutable EuclideanCreature{T} <: GACreature
    value :: T
    objvalue :: Float64
end

EuclideanCreature(value::T, model::EuclideanModel{F,T}) where {F,T} =
    EuclideanCreature{T}(value, F(value))
```

Since we are minimizing the objective value, we set `fitness` to be
negative of the objective value.

```julia
fitness(x::EuclideanCreature{T}) where {T} = -x.objvalue
```

This creates a `EuclideanCreature` object when given
`EuclideanModel{F,T}`. Here, we create a random point drawn with uniform
probability from the rectangle. Note: `aux` is used to store auxiliary
scratch space for when we are trying to minimize memory
allocations. `aux` can be created by overloading the
`genauxga(model::EuclideanModel)` function, which is used to produce
memory-safe (due to multi-threading) auxiliary scratch space. In this
example, we do not need any scratch space.

```julia
randcreature(m::EuclideanModel{F,T}, aux, rng) where {F,T} =
    EuclideanCreature(m.xmin .+ m.xspan .* rand(rng,T), m)
```

This defines the crossover operator. Here, we define a crossover as
the average of two points. Note: we can optionally re-use memory from
the `z` object in order to create the new `EuclideanCreature`. We do
not do this since we are using `SVector`s but it can be done if using
`Vector`s.

```julia
crossover(x::EuclideanCreature{T}, y::EuclideanCreature{T},
          m::EuclideanModel{F,T}, aux,
          z::EuclideanCreature{T}, rng) where {F,T} =
              EuclideanCreature(0.5 .* (x.value.+y.value), m)
```

This defines the mutation operator. Here, we draw a vector from a
circular normal distribution, scale it by the rectangle, and shift the
original point with the drawn vector. Clamping is optionally done to
restrict points to be inside the rectangle.

```julia
function mutate(x::EuclideanCreature{T}, m::EuclideanModel{F,T},
                aux, rng) where {F,T}
    yvalue = x.value .+ 0.25 .* m.xspan .* randn(rng,T)
    if m.clamp
        yvalue = max.(yvalue, m.xmin)
        yvalue = min.(yvalue, m.xmax)
    end
    EuclideanCreature(yvalue, m)
end
```

We use tournament selection as our selection operator.

```julia
selection(pop::Vector{<:EuclideanCreature{T}}, n::Integer, rng) where {T} =
    selection(TournamentSelection(2), pop, n, rng)
```

This defines how to print details of our creature.

```julia
printfitness(curgen::Integer, x::EuclideanCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")
```

This defines how to save our creature to file. `GAFramework` will save
the best creature to file using this function.

```julia
savecreature(file_name_prefix::AbstractString, curgen::Integer,
             creature::EuclideanCreature, model::EuclideanModel) =
    save("$(file_name_prefix)_creature_$(curgen).jld", "creature", creature)
```

That takes care of how to define our problem using `GAFramework`. Now,
we run the GA.

```julia
using StaticArrays

model = EuclideanModel(x -> exp(norm(x-SVector(0.5,0.5,0.5,0.5,0.5))),
                         [-1.,-1.,-1.,-1.,-1], # minimum corner
                         [1.,1.,1.,1.,1]) # maximum corner in rectangle
```

This creates the GA state, with population size 6000, maximum number
of generations 500, fraction of elite creatures 0.1, crossover rate
0.9, and mutation rate 0.9. This also prints the objective value every
iteration. Primarily, this generates the population.

```julia
state = GAState(model, ngen=500, npop=6_000, elite_fraction=0.1,
                       crossover_rate=0.9, mutation_rate=0.9,
                       print_fitness_iter=1)
```

This runs the GA and we are done.

```julia
ga!(state)
````

`state.pop[1]` gives you the creature with the best fitness.

## Restarting

After we finish a GA run using `ga!(state)`, and we decide that we
want to continue optimizing for a few more generations, we can do the
following.  Here, we change maximum number of generations to 1000, and
then restart the GA.

```julia
state.ngen = 1000

ga!(state)
```

## Saving creature to file

We can save the creature to file every 10 iterations using the following.

```julia
state = GAState(m, ngen=500, npop=6_000, elite_fraction=0.1,
                crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1,
                save_creature_iter=10, file_name_prefix="minexp_6000")
```

After we finish a GA run using `ga!(state)`, and we decide that we
want to save the best creature to file afterwards, we can do the following.

```julia
savecreature("minexp_6000", state.ngen, state.pop[1], model)
```

## Saving GA state to file

This save the full GA state to file every 100 iterations using the
following. Note: unfortunately, this doesn't work with
`EuclideanModel{F,T}` since it contains the function `F`. It should
work for other types that doesn't contain functions.

```julia
state = GAState(m, ngen=500, npop=6_000, elite_fraction=0.1,
                crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1,
                save_state_iter=100, file_name_prefix="minexp_6000")
```

If something happens during the middle of running `ga!(state)`, we can
reload the state from file from the 200th iteration as follows, and
then restart the GA.

```julia
state = loadgastate("minexp_6000_state_100.jld")

ga!(state)
```

We can also save the state using the following.

```julia
savegastate("minexp_6000", state.ngen, state)
```
