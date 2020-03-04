module MagnaGA

using SparseArrays
using Random
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature, genauxga, printfitness
using ..CayleyCrossover
using ..Spmap

export MagnaCreature, MagnaModel

# Represents a permutation, i.e. 1-1 mapping from one set to another
struct MagnaCreature
    f :: Vector{Int}
    edge_score :: Float64
    node_score :: Float64
    score :: Float64 # alignment score
end

fitness(x::MagnaCreature) where {T} = x.score

"""
Model to find network alignment by optimizing S3 (see MAGNA paper)
A network alignment is represented using a permutation
Faster implementation, slightly slower than the C++ code from the MAGNA++ paper
    using GAFramework, GAFramework.MagnaGA
    using LightGraphs
    using Random
    mv = 100
    me = 400
    G1 = adjacency_matrix(LightGraphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    G2 = G1[invperm(perm),invperm(perm)]
    # G1.|V| <= G2.|V|
    S = zeros(mv,mv)
    alpha = 1.0
    model = MagnaModel(G1,G2,S,alpha)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f .== perm)/mv)
    st.pop[1]
"""
struct MagnaModel <: GAModel
    G1   :: SparseMatrixCSC{Int,Int}
    G2   :: SparseMatrixCSC{Int,Int}
    S :: Matrix{Float64}
    alpha :: Float64
    function MagnaModel(G1::SparseMatrixCSC,G2::SparseMatrixCSC,S::Matrix,alpha::Number)
        (size(G1,1)==size(G1,2) && size(G2,1)==size(G2,1) &&
            size(S,1)==size(G1,1) && size(S,2)==size(G2,1)) || error("input sizes")
        new(G1,G2,S,alpha)
    end
end

"""
    using GAFramework, GAFramework.MagnaGA, GAFramework.PermGA
    using LightGraphs, SparseArrays
    using Random
    GAFramework.setthreads(false)
    nv = 100
    ne = 400
    G2 = adjacency_matrix(LightGraphs.erdos_renyi(nv,ne))
    perm = randperm(nv)
    mv = 75
    G1 = G2[perm[1:mv], perm[1:mv]]
    S = rand(Float64, mv, nv)/10
    for i in 1:mv
        S[i,i] = 0.95
    end
    alpha = 0.5
    model1 = MagnaModel(G1,G2,S,alpha)
    model2 = NetalignModel(G1,G2)
    aux1 = genauxga(model1)
    f = randperm(size(G2,1));
    c1 = MagnaCreature(f, model1, aux1)
    c2 = PermCreature(f, model2)
    c1.score, c2.score

    st1 = GAState(model1, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));
    st2 = GAState(model2, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));

    @time ga!(st1)
    @time ga!(st2)

"""

function genauxga(m::MagnaModel)
    crossover_aux = randperm(size(m.G2,1)), fill(false, size(m.G2,1))
    Spmap.map_index_aux(m.G1, m.G2), crossover_aux
end

# alpha * S_E + (1-alpha) * S_N
function MagnaCreature(h::AbstractVector, m::MagnaModel, aux)
    length(h)==size(m.G2,1) || error("length of h")
    # aux = genauxga(..) can be used to reduce allocations here
    # by using auxiliary space instead of reallocating
    ((K, hinv, hinvrowvals), _) = aux
    # Assumes that all edge weights are 1
    Spmap.plus_getindex!(K, m.G1, m.G2, h, hinv, hinvrowvals) # K = G1 + G2[h,h][1:m,1:m]
    
    w = nonzeros(K)
    Nc = count(x->x==2, w)
    Nn = count(x->x==1, w)

    Nc,cr = divrem(Nc,2)
    Nn,nr = divrem(Nn,2) # need this & above so it works well with sim. ann. code
    if cr!=0 || nr!=0 error("G1 and G2 need to be symmetric"); end
    edge_score = Nc/(Nc + Nn)

    node_score = 0.0
    if m.alpha != 1.0
        S = m.S
        @inbounds for i in 1:size(m.G1,1)
            node_score += S[i, h[i]]
        end
        node_score /= size(m.G1,1)
    end

    score = m.alpha*edge_score + (1.0-m.alpha) * node_score

    MagnaCreature(h, edge_score, node_score, score)
end

function randcreature(m::MagnaModel, aux, rng::AbstractRNG) where {T}
    f = randperm(rng, size(m.G2,1))
    MagnaCreature(f, m, aux)
end

function crossover!(z, x, y, st::GAState{MagnaModel}, aux, rng::AbstractRNG) where {T}
    (_, (r, visited)) = aux
    visited .= false
    cayley_crossover!(z.f, x.f, y.f, rng, r, visited)
    MagnaCreature(z.f, st.model, aux)
end

function printfitness(curgen::Int, x::MagnaCreature)
    println("curgen: $(curgen) edge_score: $(x.edge_score) node_score: $(x.node_score) score: $(x.score)")
end

end # MagnaGA
