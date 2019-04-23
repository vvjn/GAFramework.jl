module PermGA

using SparseArrays
using Random
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature

export PermCreature, NetalignModel

# Represents a permutation, i.e. 1-1 mapping from one set to another
struct PermCreature
    f :: Vector{Int}
    score :: Float64 # alignment score
end

fitness(x::PermCreature) where {T} = x.score

struct CayleyCrossover end
# See MAGNA network alignment paper
# for each cycle in permutation
# cut off half off the cycle
function halfperm!(r, rng)
    visited = fill(false,length(r))
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
function crossover!(::CayleyCrossover, z::P, x::P, y::P,
    st::GAState, aux, rng::AbstractRNG) where {P <: PermCreature}
    rdivperm!(z.f, x.f, y.f) # r = p q^-1
    halfperm!(z.f, rng)
    z.f[:] = z.f[y.f] # permute!(z.f, y.f) # 
    PermCreature(z.f, st.model)
end

printfitness(curgen::Int, x::PermCreature) =
    println("curgen: $curgen value: $(x.f) score: $(x.score)")


"""
Model to find network alignment by optimizing S3 (see MAGNA paper)
A network alignment is represented using a permutation
Slow implementation but fine  for small graphs
    using GAFramework, GAFramework.PermGA
    using LightGraphs
    using Random
    mv = 20
    me = 40
    G1 = adjacency_matrix(LightGraphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    G2 = G1[invperm(perm),invperm(perm)]
    # G1.|V| <= G2.|V|
    model = NetalignModel(G1,G2)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f .== perm)/mv)
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
    h = view(f, 1:size(m.G1,1))
    w = nonzeros(m.G1 + m.G2[h,h])
    Nc = count(x->x==2, w)
    Nn = length(w) - Nc
    Nc,cr = divrem(Nc,2)
    Nn,nr = divrem(Nn,2) # need this & above so it works well with sim. ann. code
    if cr!=0 || nr!=0 error("G1 and G2 need to be symmetric"); end
    score = Nc/(Nc + Nn)
    PermCreature(f, score)
end

function randcreature(m::NetalignModel, aux, rng::AbstractRNG) where {T}
    f = randperm(size(m.G2,1))
    PermCreature(f, m)
end

function crossover!(z, x, y,
    st::GAState{NetalignModel}, aux, rng::AbstractRNG) where {T}
    crossover!(CayleyCrossover(), z, x, y, st, aux, rng)
end

end # PermGA
