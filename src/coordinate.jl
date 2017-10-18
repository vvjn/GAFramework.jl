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
    T = promote_type(eltype(xmin),eltype(xmax))
    N = length(xmin)
    ymin = SVector{N,T}(xmin)
    ymax = SVector{N,T}(xmax)
    yspan = ymax .- ymin
    # check that F(ymin), F(ymax) can be converted to Float64 (fitness value)
    Float64(f(ymin))
    Float64(f(ymax))
    # and that F(yspan), F(ymin), and F(ymax) has sane values maybe
    # z1!=Inf && z2!=Inf && !isnan(z1) && !isnan(z2) ||
    #    error("F(xmin) or F(xmax) objective function is either NaN or Inf")
    all(yspan .>= zero(T)) || error("ymax[i] < ymin[i] for some i")
    CoordinateModel{F,SVector{N,T}}(f,ymin,ymax,yspan,clamp)
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
crossover(::AverageCrossover, z::CoordinateCreature{T},
          x::CoordinateCreature{T}, y::CoordinateCreature{T},
          m::CoordinateModel{F,T}, params, curgen::Integer,
          aux, rng) where {F,T} =
              CoordinateCreature(0.5 .* (x.value.+y.value), m)

immutable SinglePointCrossover end
function crossover(::SinglePointCrossover, z::CoordinateCreature{T},
                   x::CoordinateCreature{T}, y::CoordinateCreature{T},
                   m::CoordinateModel{F,T}, params, curgen::Integer,
                   aux, rng) where {F,T}
    N = length(T)
    i = rand(rng, 1:N)
    zvec = Vector(x.value)
    if rand(rng) < 0.5
        zvec[i+1:end] = y.value[i+1:end]
    else
        zvec[1:i] = y.value[1:i]
    end
    zvalue = SVector{N}(zvec)
    CoordinateCreature(zvalue, m)
end

immutable TwoPointCrossover end
function crossover(::TwoPointCrossover, z::CoordinateCreature{T},
                   x::CoordinateCreature{T}, y::CoordinateCreature{T},
                   m::CoordinateModel{F,T}, params, curgen::Integer,
                   aux, rng) where {F,T}
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
    CoordinateCreature(zvalue, m)
end

function crossover(z::CoordinateCreature{T},
                   x::CoordinateCreature{T}, y::CoordinateCreature{T},
                   m::CoordinateModel{F,T}, params, curgen::Integer,
                   aux, rng) where {F,T}
    crossover(TwoPointCrossover(), z, x, y, m, nothing, curgen, aux, rng)
end

# Mutate over all dimensions
function mutatenormal(temp::Real, x::CoordinateCreature{T},
                      model::CoordinateModel{F,T}, rng) where {F,T}
    yvalue = x.value .+ temp .* model.xspan .* randn(rng,T)
    model.clamp && (yvalue = clamp.(yvalue, model.xmin, model.xmax))
    CoordinateCreature(yvalue, model)
end

# Mutate through a single dimension
function mutatenormaldim(temp::Real, x::CoordinateCreature{T}, dim::Integer,
                         model::CoordinateModel{F,T}, rng) where {F,T}
    ET = eltype(T)
    N = length(T)
    yvaluedim = x.value[dim] + temp * model.xspan[dim] * randn(rng,ET)
    model.clamp && (yvaluedim = clamp.(yvaluedim, model.xmin[dim], model.xmax[dim]))
    yvalue = x.value + SVector{N,ET}(vcat(zeros(ET,dim-1), yvaluedim, zeros(ET,N-dim)))
    CoordinateCreature(yvalue, model)
end

function mutate(x::CoordinateCreature{T}, model::CoordinateModel{F,T},
                params, curgen::Integer, aux, rng) where {F,T}
    if rand(rng) < params[:rate]
        if rand(rng) < get(params, :sa_rate, 0.0)
            sa(x,model,params[:k], params[:lambda],
               params[:maxiter], curgen, aux, rng)
        else
            N = length(T)
            mutatenormaldim(0.1, x, rand(1:N), model, rng)
        end
    else
        x
    end
end

export sa,satemp,saprob,mutatenormal
function sa(x::CoordinateCreature{T}, model::CoordinateModel{F,T},
            k::Real, lambda::Real, maxiter::Integer, curgen::Integer,
            aux, rng) where {F,T}
    y = x
    numnoups = 0 # number of consecutive
    #curgen = 0
    iter = curgen*maxiter + 0
    while true
        temp = satemp(iter, k, lambda) + numnoups/maxiter#lambda * log(1+numnoups)
        newy = mutatenormal(temp, y, model, rng)
        diffe = fitness(newy) - fitness(y)
        if diffe >= 0
            y = newy
        elseif rand(rng) < saprob(diffe, temp)
            y = newy
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
saprob(diff::Real, iter::Integer, k::Real, lambda::Real) = exp(diff / satemp(iter, k, lambda))
saprob(diff::Real, temp::Real) = exp(diff / temp)

selection(pop::Vector{<:CoordinateCreature{T}}, n::Integer, rng) where {T} =
    selection(TournamentSelection(2), pop, n, rng)

printfitness(curgen::Integer, x::CoordinateCreature{T}) where {T} =
    println("curgen: $curgen value: $(x.value) obj. value: $(x.objvalue)")

function deepcopy!(a::Vector{<:CoordinateCreature}, ixs::UnitRange,
                   b::Vector{<:CoordinateCreature}, jxs::UnitRange)
    for (i,j) in zip(ixs,jxs)
        a[i] = b[j]
    end
    a
end
