module CoordinateGA

using Random
using LinearAlgebra: dot
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature

export CoordinateCreature, FunctionModel, SumModel

# This is a CoordinateCreature, T is some coordinate type like Vector
# and objvalue is some objective value
struct CoordinateCreature{T}
    value :: T
    objvalue :: Float64
end

# Negative, since we are minimizing the objective value
fitness(x::CoordinateCreature{T}) where {T} = -x.objvalue

# We can use multiple models to optimize this creature.
# To demonstrate, below we use the FunctionModel model
# as well as the SumModel

# The following are some crossover functions for the CoordinateCreature
# Each model can choose its own crossover function
struct AverageCrossover end
function crossover!(::AverageCrossover, z::CoordinateCreature{T},
    x::CoordinateCreature{T}, y::CoordinateCreature{T},
    st::GAState, aux, rng::AbstractRNG) where {T}
    z.value .= 0.5 .* (x.value .+ y.value)
    CoordinateCreature(z.value, st.model)
end

struct SinglePointCrossover end
function crossover!(::SinglePointCrossover, z::CoordinateCreature{T},
    x::CoordinateCreature{T}, y::CoordinateCreature{T},
    st::GAState, aux, rng::AbstractRNG) where {T}
    N = length(x.value)
    i = rand(rng, 1:N)
    if rand(rng) < 0.5
        z.value[1:i] = x.value[1:i]
        z.value[i+1:end] = y.value[i+1:end]
    else
        z.value[1:i] = y.value[1:i]
        z.value[i+1:end] = x.value[i+1:end]
    end
    CoordinateCreature(z.value, st.model)
end

struct TwoPointCrossover end
function crossover!(::TwoPointCrossover, z::CoordinateCreature{T},
    x::CoordinateCreature{T}, y::CoordinateCreature{T},
    st::GAState, aux, rng::AbstractRNG) where {T}
    N = length(x.value)
    i,j = rand(rng, 1:N, 2)
    i,j = i > j ? (j,i) : (i,j)
    if rand(rng) < 0.5
        z.value[:] = x.value
        z.value[i+1:j] = y.value[i+1:j]
    else
        z.value[:] = y.value
        z.value[i+1:j] = x.value[i+1:j]
    end
    CoordinateCreature(z.value, st.model)
end

printfitness(curgen::Int, x::CoordinateCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")


# The following is the SumModel
# It finds a d-dimensional coordinate x such that dot(x, 1:length(x)) ≈ y
"""
    model = SumModel(5, 42.)
    state = GAState(model, ngen=500, npop=6_000, elite_fraction=0.1,
                       params=Dict(:mutation_rate=>0.9, :print_fitness_iter=>1))
    ga!(state)
"""
struct SumModel{T} <: GAModel
    d::Int
    target::T
end

function CoordinateCreature(value::Vector{T}, m::SumModel{T}) where {T}
    objval = Float64(abs(dot(value, 1:length(value)) - m.target))
    CoordinateCreature(value, objval)
end

function randcreature(m::SumModel{T}, aux, rng::AbstractRNG) where {T}
    value = rand(rng, m.d)
    CoordinateCreature(value, m)
end

function mutation!(x::CoordinateCreature,
    st::GAState{SumModel{T}}, aux, rng::AbstractRNG) where {T}
    gp = st.params
    if rand(rng) < get(gp, :mutation_rate, 0.1)
        x.value[rand(rng, 1:length(x.value))] += st.model.target * randn(rng,T)
        CoordinateCreature(x.value, st.model)
    else
        x
    end
end

function crossover!(z::CoordinateCreature,
    x::CoordinateCreature, y::CoordinateCreature,
    st::GAState{SumModel{T}}, aux, rng::AbstractRNG) where {T}
    crossover!(TwoPointCrossover(), z, x, y, st, aux, rng)
end

# The following is the FunctionModel
# It's a more complicated model than above
# It minimizes the objective value using a given objective function
"""
# E.g.
    model = FunctionModel(x -> abs(x[1] - 20.0), [-200.0], [200.0])
    state = GAState(model, ngen=500, npop=6_000, elite_fraction=0.1,
                       params=Dict(:mutation_rate=>0.9, :print_fitness_iter=>1))
    ga!(state)

    type T has to have properties
        y-x :: T
        0.25 .* (x+y) :: T
        randn(T) :: T
        z + 0.25*(y-x)*randn(T) :: T

"""
struct FunctionModel{F,T} <: GAModel
    f::F
    xmin::T
    xmax::T
    xspan::T # xmax-xmin
    clamp::Bool
end
function FunctionModel(f::F,xmin,xmax,clamp::Bool=true) where {F}
    xmin,xmax = promote(xmin,xmax)
    ET = eltype(xmin)
    N = length(xmin)
    xspan = xmax .- xmin
    # check that f(xmin), f(xmax) can be converted to Float64 without error
    z1 = Float64(f(xmin))
    z2 = Float64(f(xmax))
    # and that f(xspan), f(xmin), and f(xmax) has sane values maybe
    # z1!=Inf && z2!=Inf && !isnan(z1) && !isnan(z2) ||
    #    error("f(xmin) or f(xmax) objective function is either NaN or Inf")
    all(xspan .>= zero(ET)) || error("xmax[i] < xmin[i] for some i")
    FunctionModel{F,typeof(xspan)}(f,xmin,xmax,xspan,clamp)
end

CoordinateCreature(value::T, m::FunctionModel{F,T}) where {F,T} =
    CoordinateCreature(value, m.f(value))

function randcreature(m::FunctionModel{F,T}, aux, rng::AbstractRNG) where {F,T}
    ET = eltype(T)
    if T <: Vector
        xvalue = m.xmin .+ m.xspan .* rand(rng, ET, length(m.xspan))
    else
        xvalue = m.xmin .+ m.xspan .* rand(rng, T)
    end
    CoordinateCreature(xvalue, m)
end

function crossover!(z::CoordinateCreature{T},
    x::CoordinateCreature{T}, y::CoordinateCreature{T},
    st::GAState{FunctionModel{F,T}}, aux, rng::AbstractRNG) where {F,T}
    crossover!(TwoPointCrossover(), z, x, y, st, aux, rng)
end

# Mutate over all dimensions
function mutatenormal!(x::CoordinateCreature{T}, temp::Real,
    model::FunctionModel{F,T}, rng::AbstractRNG) where {F,T}
    x.value .+= temp .* model.xspan .* randn(rng,length(x.value))
    model.clamp && (x.value .= clamp.(x.value, model.xmin, model.xmax))
    CoordinateCreature(x.value, model)
end

# Mutate through a single dimension
function mutatenormaldim!(x::CoordinateCreature{T}, temp::Real, dim::Integer,
    model::FunctionModel{F,T}, rng::AbstractRNG) where {F,T}
    ET = eltype(T)
    x.value[dim] += temp * model.xspan[dim] * randn(rng,ET)
    model.clamp && (x.value[dim] = clamp(x.value[dim], model.xmin[dim], model.xmax[dim]))
    CoordinateCreature(x.value, model)
end

function mutation!(x::CoordinateCreature{T},
    st::GAState{FunctionModel{F,T}}, aux, rng::AbstractRNG) where {F,T}
    gp = st.params
    if rand(rng) < get(gp, :mutation_rate, 0.1)
        if rand(rng) < get(gp, :sa_rate, 0.0)
            sa(x, st.model, gp[:sa_k], gp[:sa_lambda],
                gp[:sa_maxiter], st.curgen, aux, rng)
        else
            N = length(x.value)
            mutatenormaldim!(x, 0.1, rand(rng, 1:N), st.model, rng)
        end
    else
        x
    end
end

# export sa,satemp,saprob,mutatenormal
function sa(x::CoordinateCreature{T}, model::FunctionModel{F,T},
    k::Real, lambda::Real, maxiter::Integer, curgen::Int,
    aux, rng::AbstractRNG) where {F,T}
    N = length(x.value)
    y = x
    numnoups = 0 # number of consecutive
    #curgen = 0
    iter = curgen*maxiter + 0
    while true
        temp = satemp(iter, k, lambda) + numnoups/maxiter
        #temp = satemp(iter, k, lambda) + lambda * log(1+numnoups)
        dim = rand(rng,1:N)
        yvdim_old = y.value[dim]
        yov_old = y.objvalue
        fitness_old = fitness(y)
        y = mutatenormaldim!(y, temp, dim, model, rng)
        diffe = fitness(y) - fitness_old
        if diffe >= 0
        elseif rand(rng) < saprob(diffe, temp)
        else
            y.value[dim] = yvdim_old
            y = CoordinateCreature(y.value, yov_old)
        end
        #println("temp: $temp newy: $(newy.objvalue) diffe: $diffe prob: $(saprob(diffe, temp))")
        #numnoups = ifelse(diffe > 0, 0, ifelse(diffe < 0, numnoups+1, numnoups))
        numnoups = ifelse(diffe > 0, 0, numnoups + 1)
        iter += 1
        iter > curgen*maxiter + maxiter && break
    end
    #println("pre: $(x.objvalue) post: $(y.objvalue)")
    y
end

satemp(iter::Integer, k::Real, lambda::Real) = k * exp(-lambda * iter)
# diff = newscore - oldscore
saprob(diff::Real, iter::Integer, k::Real, lambda::Real) =
    exp(diff / satemp(iter, k, lambda))
saprob(diff::Real, temp::Real) = exp(diff / temp)

function selection(pop::Vector, n::Integer,
    st::GAState{FunctionModel{F,T}}, rng::AbstractRNG) where {F,T}
    selection(TournamentSelection(2), pop, n, st, rng)
end

end # CoordinateGA

