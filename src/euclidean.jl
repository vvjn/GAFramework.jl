using StaticArrays

export EuclideanModel, EuclideanCreature, getobjective

"""
# E.g.
minimizes function
    model = EuclideanModel(x -> abs(x[1] - 20.0), -200.0, 200.0)
    state = GAState(model, ngen=500, npop=6_000, elite_fraction=0.1,
                       crossover_rate=0.9, mutation_rate=0.9,
                       print_fitness_iter=1)
    ga!(state)

    type T has to have properties
        y-x :: T     
        0.25 .* (x+y) :: T
        randn(T) :: T
        z + 0.25*(y-x)*randn(T) :: T

Since we are using StaticArrays here, this would work well (speed-wise)
for input arrays of size less than around 12 according to
the StaticArrays documentation    
"""
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
    # check that F(xmin), F(xmax) can be converted to Float64 (fitness value)
    # and that F(yspan), F(xmin), and F(xmax) has sane values
    Float64(F(xmin))
    Float64(F(xmax))
    #z1!=Inf && z2!=Inf && !isnan(z1) && !isnan(z2) ||
    #    error("F(xmin) or F(xmax) objective function is either NaN or Inf")
    all(yspan .>= zero(eltype(yspan))) || error("ymax[i] > ymin[i] for some i")
    EuclideanModel{F,typeof(yspan)}(ymin,ymax,yspan,clamp)
end
getobjective(::EuclideanModel{F,T}) where {F,T} = F

immutable EuclideanCreature{T} <: GACreature
    value :: T
    objvalue :: Float64
end
EuclideanCreature(value::T, m::EuclideanModel{F,T}) where {F,T} =
    EuclideanCreature{T}(value, F(value))

fitness(x::EuclideanCreature{T}) where {T} = -x.objvalue

randcreature(m::EuclideanModel{F,T}, aux, rng) where {F,T} =
    EuclideanCreature(m.xmin .+ m.xspan .* rand(rng,T), m)

crossover(x::EuclideanCreature{T}, y::EuclideanCreature{T},
          m::EuclideanModel{F,T}, aux,
          z::EuclideanCreature{T}, rng) where {F,T} =
              EuclideanCreature(0.5 .* (x.value.+y.value), m)

function mutate(x::EuclideanCreature{T}, m::EuclideanModel{F,T},
                aux, rng) where {F,T}
    yvalue = x.value .+ 0.25 .* m.xspan .* randn(rng,T)
    if m.clamp
        yvalue = max.(yvalue, m.xmin)
        yvalue = min.(yvalue, m.xmax)
    end
    EuclideanCreature(yvalue, m)
end

selection(pop::Vector{<:EuclideanCreature{T}}, n::Integer, rng) where {T} =
    selection(TournamentSelection(), pop, n, rng)

printfitness(curgen::Integer, x::EuclideanCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")
