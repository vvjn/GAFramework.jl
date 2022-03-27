module MagnaGA

using SparseArrays
using Random
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature, genauxga, printfitness
using ..CayleyCrossover

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
Model to find network alignment by optimizing S3 (see MAGNA++ paper)
A network alignment is represented using a permutation
Faster implementation
    using GAFramework, GAFramework.MagnaGA
    using Graphs
    using Random
    mv = 100
    me = 400
    G1 = adjacency_matrix(Graphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    G1 = G2[perm[1:mv],perm[1:mv]]
    # G1.|V| <= G2.|V|
    S = zeros(mv,mv)
    alpha = 1.0
    model = MagnaModel(G1,G2,S,alpha)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f[1:mv] .== perm[1:mv])/mv)
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
    using Graphs, SparseArrays
    using Random
    GAFramework.setthreads(false)
    nv = 100
    ne = 400
    G2 = adjacency_matrix(Graphs.erdos_renyi(nv,ne))
    perm = randperm(nv)
    mv = 75
    G1 = G2[perm[1:mv], perm[1:mv]]
    S = rand(Float64, mv, nv)/10
    for i in 1:mv
        S[i,i] = 0.95
    end
    alpha = 0.5
    model1 = MagnaModel(G1,G2,S,alpha)
    model2 = NetalignModel(G1,G2,S,alpha)
    aux1 = genauxga(model1)
    f = randperm(size(G2,1));
    c1 = MagnaCreature(f, model1, aux1)
    c2 = NetalignCreature(f, model2)
    c1.score, c2.score

    st1 = GAState(model1, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));
    st2 = GAState(model2, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));

    @time ga!(st1)
    @time ga!(st2)

"""

function genauxga(m::MagnaModel)
    crossover_aux = randperm(size(m.G2,1)), fill(false, size(m.G2,1))

    n1 = size(m.G1, 1)
    n2 = size(m.G2, 1)
    finv = randperm(n2)
    v_store = zeros(Int, n1+n2)
    v_count = zeros(Int, n1)
    model_aux = (finv, v_store, v_count)

    model_aux, crossover_aux
end

function calculate_s3(G1, G2, f, aux=nothing)
    length(f)==size(G2,1) || error("length of f")
    n1 = size(G1,1)
    n2 = size(G2,1)

    if isnothing(aux)
        finv = randperm(n2)
        v_store = zeros(Int, n1+n2)
        v_count = zeros(Int, n1)
    else
        ((finv, v_store, v_count), ) = aux
    end
    
    @inbounds for j in 1:n2
        finv[f[j]] = j
    end

    rows1 = rowvals(G1)
    rows2 = rowvals(G2)
    # Map G2 back onto G1 axes and count edges
    n_induced = 0
    n_conserved = 0
    @inbounds for u1 in 1:n1
        # Accumulate the edges that are mapped back
        r = 1
        for i1 in nzrange(G1, u1)
            v1 = rows1[i1]
            v_store[r] = v1
            r += 1
        end
        u2 = f[u1]
        for i2 in nzrange(G2, u2)
            v2 = rows2[i2]
            v1 = finv[v2]
            if v1 <= n1
                v_store[r] = v1
                r += 1
                n_induced += 1
            end
        end
        n_neighbors = r-1
        # Count the edges that are mapped back and duplicated
        for r in 1:n_neighbors
            v_count[v_store[r]] = 0
        end
        for r in 1:n_neighbors
            v_count[v_store[r]] += 1
        end
        for r in 1:n_neighbors
            i = v_store[r]
            if v_count[i] == 2
                n_conserved += 1
                v_count[i] = 0
            end
        end
    end
    n_nonconserved = nnz(G1) + n_induced - 2n_conserved
    edge_score = n_conserved / (n_nonconserved + n_conserved)
    return n_conserved, n_nonconserved, edge_score
end

# alpha * S_E + (1-alpha) * S_N
function MagnaCreature(f::AbstractVector, m::MagnaModel, aux)
    # aux = genauxga(..) can be used to reduce allocations here
    # by using auxiliary space instead of reallocating

    _, _, edge_score = calculate_s3(m.G1, m.G2, f, aux)

    node_score = 0.0
    if m.alpha != 1.0
        S = m.S
        @inbounds for i in 1:size(m.G1,1)
            node_score += S[i, f[i]]
        end
        node_score /= size(m.G1,1)
    end

    score = m.alpha*edge_score + (1.0-m.alpha) * node_score

    MagnaCreature(f, edge_score, node_score, score)
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
