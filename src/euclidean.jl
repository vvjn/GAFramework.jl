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

crossover(x::CoordinateCreature{T}, y::CoordinateCreature{T},
          m::CoordinateModel{F,T}, aux,
          z::CoordinateCreature{T}, rng) where {F,T} =
              CoordinateCreature(0.5 .* (x.value.+y.value), m)

function mutate(x::CoordinateCreature{T}, m::CoordinateModel{F,T},
                aux, rng) where {F,T}
    yvalue = x.value .+ 0.25 .* m.xspan .* randn(rng,T)
    if m.clamp
        yvalue = max.(yvalue, m.xmin)
        yvalue = min.(yvalue, m.xmax)
    end
    CoordinateCreature(yvalue, m)
end

selection(pop::Vector{<:CoordinateCreature{T}}, n::Integer, rng) where {T} =
    selection(TournamentSelection(), pop, n, rng)

printfitness(curgen::Integer, x::CoordinateCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")
