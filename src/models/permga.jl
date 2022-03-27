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

export NetalignCreature, NetalignModel

# Represents a permutation, i.e. 1-1 mapping from one set to another
struct NetalignCreature
    f :: Vector{Int}
    edge_score :: Float64
    node_score :: Float64
    score :: Float64 # alignment score
end

fitness(x::NetalignCreature) where {T} = x.score

printfitness(curgen::Int, x::NetalignCreature) =
    println("curgen: $curgen value: $(x.f) score: $(x.score)")

"""
Model to find network alignment by optimizing S3 (see MAGNA++ paper)
A network alignment is represented using a permutation
Slow implementation but fine  for small graphs
    using GAFramework, GAFramework.PermGA
    using Graphs
    using Random
    mv = 8
    me = 15
    G2 = adjacency_matrix(Graphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    G1 = G2[perm[1:mv],perm[1:mv]]
    # G1.|V| <= G2.|V|
    S = zeros(mv,mv)
    alpha = 1.0
    model = NetalignModel(G1,G2,S,alpha)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f[1:mv] .== perm[1:mv])/mv)
    st.pop[1]
"""
struct NetalignModel <: GAModel
    G1   :: SparseMatrixCSC{Int,Int}
    G2   :: SparseMatrixCSC{Int,Int}
    S :: Matrix{Float64}
    alpha :: Float64
    function NetalignModel(G1::SparseMatrixCSC,G2::SparseMatrixCSC,S::Matrix,alpha::Number)
        (size(G1,1)==size(G1,2) && size(G2,1)==size(G2,1) &&
            size(S,1)==size(G1,1) && size(S,2)==size(G2,1)) || error("input sizes")
        new(G1,G2,S,alpha)
    end
end

# aux = genauxga(..) can be used to reduce allocations here
# by using auxiliary space instead of reallocating
function NetalignCreature(f, m::NetalignModel)
    length(f)==size(m.G2,1) || error("length of h")
    # Assumes that all edge weights are 1
    h = f[1:size(m.G1,1)]
    K = m.G1 + m.G2[h,h]
    w = nonzeros(K)
    Nc = count(x->x==2, w)
    Nn = length(w) - Nc
    edge_score = Nc/(Nc + Nn)

    node_score = 0.0
    for i in 1:size(m.G1,1)
        node_score += m.S[i, f[i]]
    end
    node_score /= size(m.G1,1)

    score = m.alpha*edge_score + (1.0-m.alpha) * node_score
    
    NetalignCreature(f, edge_score, node_score, score)
end

function randcreature(m::NetalignModel, aux, rng::AbstractRNG) where {T}
    f = randperm(rng, size(m.G2,1))
    NetalignCreature(f, m)
end

function crossover!(z, x, y,
    st::GAState{NetalignModel}, aux, rng::AbstractRNG) where {T}
    cayley_crossover!(z.f, x.f, y.f, rng)
    NetalignCreature(z.f, st.model)
end

end # PermGA
