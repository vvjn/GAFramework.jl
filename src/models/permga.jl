module CayleyCrossover
using Random

export cayley_crossover!

# See MAGNA network alignment paper
# for each cycle in permutation
# cut off half off the cycle
function halfperm!(r, rng, visited = fill(false,length(r)))    
    i = 1
    while i <= length(r)
        if visited[i]
            i += 1
            continue
        end
        # traverse cycle
        clen = 1
        cstart = i
        clast = cstart
        while true
            visited[clast] = true
            ccur = r[clast]
            if ccur == cstart; break; end
            clen += 1
            clast = ccur
        end
        cbreak = rand(rng,1:clen)
        # go part way into cycle
        for _ = 1:cbreak; clast = r[clast]; end
        # cut off rest of cycle and stitch it back
        cstart = clast
        clast = r[clast]
        for _ = 1:div(clen,2)
            ccur = r[clast]
            r[clast] = clast
            clast = ccur
        end
        r[cstart] = clast
        i += 1        
    end
    r
end
function rdivperm!(r, p, q) # p q^-1
    for i = 1:length(p)
        r[q[i]] = p[i]
    end
    r
end

function cayley_crossover!(s::AbstractVector, p::AbstractVector, q::AbstractVector,
    rng::AbstractRNG,
    r::AbstractVector=similar(s),
    visited::AbstractVector=fill(false, length(s)))
    rdivperm!(r, p, q) # r = p q^-1
    halfperm!(r, rng, visited)
    # s[:] = r[q] # permute!(z, q) #
    @inbounds for i in 1:length(r)
        s[i] = r[q[i]]
    end
    s
end

end

module PermGA

using SparseArrays
using Random
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature, printfitness
using ..CayleyCrossover

export PermCreature, NetalignModel

# Represents a permutation, i.e. 1-1 mapping from one set to another
struct PermCreature
    f :: Vector{Int}
    score :: Float64 # alignment score
end

fitness(x::PermCreature) where {T} = x.score

printfitness(curgen::Int, x::PermCreature) =
    println("curgen: $curgen value: $(x.f) score: $(x.score)")

"""
Model to find network alignment by optimizing S3 (see MAGNA paper)
A network alignment is represented using a permutation
Slow implementation but fine  for small graphs
    using GAFramework, GAFramework.PermGA
    using LightGraphs
    using Random
    mv1 = 8
    me1 = 15
    mv2 = 10
    me2 = 15
    G2 = adjacency_matrix(LightGraphs.erdos_renyi(mv2,me2))
    perm = randperm(mv2)
    G1 = G2[perm[1:mv1],perm[1:mv1]]
    # G1.|V| <= G2.|V|
    model = NetalignModel(G1,G2)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f[1:mv1] .== perm[1:mv1])/mv1)

    G2 = sparse([1, 1, 2, 3, 4], [2, 3, 3, 4, 5], ones(Int,5), 5, 5) 
    G2 = min.(1, (G2 + G2')) 
    G1 = G2[1:4,1:4]
    model = NetalignModel(G1,G2)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f[1:4] .== [1,2,3,4])/4)
    println("accuracy = ", sum(st.pop[1].f[1:4] .== [2,1,3,4])/4)
    st.pop[1], PermCreature(perm, model)
"""
struct NetalignModel <: GAModel
    G1   :: SparseMatrixCSC{Int,Int}
    G2   :: SparseMatrixCSC{Int,Int}
end

# aux = genauxga(..) can be used to reduce allocations here
# by using auxiliary space instead of reallocating
function PermCreature(f, m::NetalignModel)
    # Assumes that all edge weights are 1
    h = f[1:size(m.G1,1)] # view(f, 1:size(m.G1,1))
    K = m.G1 + m.G2[h,h]
    w = nonzeros(K)
    Nc = count(x->x==2, w)
    Nn = length(w) - Nc
    Nc,cr = divrem(Nc,2)
    Nn,nr = divrem(Nn,2) # need this & above so it works well with sim. ann. code
    if cr!=0 || nr!=0 error("G1 and G2 need to be symmetric"); end
    score = Nc/(Nc + Nn)
    PermCreature(f, score)
end

function randcreature(m::NetalignModel, aux, rng::AbstractRNG) where {T}
    f = randperm(rng, size(m.G2,1))
    PermCreature(f, m)
end

function crossover!(z, x, y,
    st::GAState{NetalignModel}, aux, rng::AbstractRNG) where {T}
    cayley_crossover!(z.f, x.f, y.f, rng)
    PermCreature(z.f, st.model)
end

end # PermGA
