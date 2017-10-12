using StaticArrays

export CoordinateModel, CoordinateCreature

"""
# E.g.
minimizes function
    model = CoordinateModel(x -> abs(x[1] - 20.0), -200.0, 200.0)
    state = GAState(model, ngen=500, npop=6_000, elite_fraction=0.1,
                       crossover_rate=0.9, mutation_rate=0.9,
                       print_fitness_iter=1)
    ga(state)

    type T has to have properties
        y-x :: T     
        0.25 .* (x+y) :: T
        randn(T) :: T
        z + 0.25*(y-x)*randn(T) :: T

Since we are using StaticArrays here, this would work well (speed-wise)
for input arrays of size less than around 12 according to
the StaticArrays documentation    
"""
immutable CoordinateModel{F,T} <: GAModel
    f::F
    xmin::T
    xmax::T
    xspan::T # xmax-xmin
    clamp::Bool
end
function CoordinateModel(f::F,xmin,xmax,clamp::Bool=true) where {F}
    D = length(xmin)
    ymin = SVector{D}(xmin)
    ymax = SVector{D}(xmax)
    yspan = ymax .- ymin
    # check that F(ymin), F(ymax) can be converted to Float64 (fitness value)
    Float64(f(ymin))
    Float64(f(ymax))
    # and that F(yspan), F(ymin), and F(ymax) has sane values maybe
    #z1!=Inf && z2!=Inf && !isnan(z1) && !isnan(z2) ||
    #    error("F(xmin) or F(xmax) objective function is either NaN or Inf")
    all(yspan .>= zero(eltype(yspan))) || error("ymax[i] > ymin[i] for some i")
    CoordinateModel{F,typeof(yspan)}(f,ymin,ymax,yspan,clamp)
end

immutable CoordinateCreature{T} <: GACreature
    value :: T
    objvalue :: Float64
end
CoordinateCreature(value::T, m::CoordinateModel{F,T}) where {F,T} =
    CoordinateCreature{T}(value, m.f(value))

fitness(x::CoordinateCreature{T}) where {T} = -x.objvalue

randcreature(m::CoordinateModel{F,T}, aux, rng) where {F,T} =
    CoordinateCreature(m.xmin .+ m.xspan .* rand(rng,T), m)

immutable AverageCrossover end
crossover(::AverageCrossover, z::CoordinateCreature{T}, x::CoordinateCreature{T},
          y::CoordinateCreature{T}, m::CoordinateModel{F,T}, aux, rng) where {F,T} =
              CoordinateCreature(0.5 .* (x.value.+y.value), m)

immutable SinglePointCrossover end
function crossover(::SinglePointCrossover, z::CoordinateCreature{T}, x::CoordinateCreature{T},
                   y::CoordinateCreature{T}, m::CoordinateModel{F,T}, aux, rng) where {F,T}
    N = length(T)
    i = rand(rng, 1:N)
    zvec = Vector(x.value)
    if rand(rng) < 0.5
        zvec[i+1:end] = y.value[i+1:end]
    else
        zvec[1:i] = y.value[1:i]
    end
    zvalue = SVector{N}(zvec)
    m.clamp && (zvalue = clamp.(zvalue, m.xmin, m.xmax))
    CoordinateCreature(zvalue, m)
end

immutable TwoPointCrossover end
function crossover(::TwoPointCrossover, z::CoordinateCreature{T}, x::CoordinateCreature{T},
                   y::CoordinateCreature{T}, m::CoordinateModel{F,T}, aux, rng) where {F,T}
    N = length(T)
    i,j = rand(rng, 1:N, 2)
    i,j = i > j ? (j,i) : (i,j)
    if rand(rng) < 0.5
        zvec = Vector(x.value)
        zvec[i+1:j] = y.value[i+1:j]
    else
        zvec = Vector(y.value)
        zvec[i+1:j] = z.value[i+1:j]
    end
    zvalue = SVector{N}(zvec)
    m.clamp && (zvalue = clamp.(zvalue, m.xmin, m.xmax))
    CoordinateCreature(zvalue, m)
end

function crossover(z::CoordinateCreature{T}, x::CoordinateCreature{T}, y::CoordinateCreature{T},
                   m::CoordinateModel{F,T}, aux, rng) where {F,T}
    crossover(TwoPointCrossover(), z, x, y, m, aux, rng)
end

function mutate(x::CoordinateCreature{T}, m::CoordinateModel{F,T}, aux, rng) where {F,T}
    yvalue = x.value .+ 0.25 .* m.xspan .* randn(rng,T)
    m.clamp && (yvalue = clamp.(yvalue, m.xmin, m.xmax))
    CoordinateCreature(yvalue, m)
end

selection(pop::Vector{<:CoordinateCreature{T}}, n::Integer, rng) where {T} =
    selection(TournamentSelection(2), pop, n, rng)

printfitness(curgen::Integer, x::CoordinateCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")
