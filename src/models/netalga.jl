# Faster version of permga.jl

module Spmap

using SparseArrays
import SparseArrays: indtype, getcolptr
using Random

@inline colstartind(A::SparseMatrixCSC, j) = getcolptr(A)[j]
@inline colboundind(A::SparseMatrixCSC, j) = getcolptr(A)[j + 1]
@inline setcolptr!(A::SparseMatrixCSC, j, val) = getcolptr(A)[j] = val

function trimstorage!(A::SparseMatrixCSC, maxstored)
    resize!(rowvals(A), maxstored)
    resize!(nonzeros(A), maxstored)
    return maxstored
end
function expandstorage!(A::SparseMatrixCSC, maxstored)
    length(rowvals(A)) < maxstored && resize!(rowvals(A), maxstored)
    length(nonzeros(A)) < maxstored && resize!(nonzeros(A), maxstored)
    return maxstored
end

# map!(+, C, A, B[h,h][1:size(A,1),1:size(A,1)]), where size(C) == size(A)
# This version assumes that all values in A and B are the value 1
function map_index_plus!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC,
    h::AbstractVector, hinv::AbstractVector, hinvrowvals::AbstractVector)

    spaceC::Int = min(length(rowvals(C)), length(nonzeros(C)))
    rowsentinelA = convert(indtype(A), size(C,1) + 1)
    rowsentinelB = convert(indtype(B), size(C,1) + 1)

    (length(hinvrowvals) < length(rowvals(B))) && error("length hinvrowvals")
    @inbounds for hi in 1:length(rowvals(B))
        hinvrowvals[hi] = min(rowsentinelB, hinv[rowvals(B)[hi]])
    end
    @inbounds for j in 1:size(B,2)
        Bk, stopBk = colstartind(B, j), colboundind(B, j)
        sort!(hinvrowvals, Bk, stopBk-1, QuickSort, Base.Order.Forward)
    end

    Ck = 1
    @inbounds for j in 1:size(C,2)
        bj = h[j]
        setcolptr!(C, j, Ck)
        Ak, stopAk = colstartind(A, j), colboundind(A, j)
        Bk, stopBk = colstartind(B, bj), colboundind(B, bj)
        Ai = Ak < stopAk ? rowvals(A)[Ak] : rowsentinelA

        #Bi = Bk < stopBk ? hinv[rowvals(B)[Bk]] : rowsentinelB
        Bi = Bk < stopBk ? hinvrowvals[Bk] : rowsentinelB
        while true
            #println("j $j bj $bj Ak $Ak stopAk $stopAk Ai $Ai Bk $Bk stopBk $stopBk Bi $Bi Ck $Ck")
            if Ai == Bi
                Ai == rowsentinelA && break # column complete
                Cx, Ci::indtype(C) = 2, Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? rowvals(A)[Ak] : rowsentinelA
                #Bk += oneunit(Bk); Bi = Bk < stopBk ? hinv[rowvals(B)[Bk]] : rowsentinelB
                Bk += oneunit(Bk); Bi = Bk < stopBk ? hinvrowvals[Bk] : rowsentinelB
            elseif Ai < Bi
                Cx, Ci = 1, Ai
                Ak += oneunit(Ak); Ai = Ak < stopAk ? rowvals(A)[Ak] : rowsentinelA
            else # Bi < Ai
                Cx, Ci = 1, Bi
                #Bk += oneunit(Bk); Bi = Bk < stopBk ? hinv[rowvals(B)[Bk]] : rowsentinelB
                Bk += oneunit(Bk); Bi = Bk < stopBk ? hinvrowvals[Bk] : rowsentinelB
            end
            # NOTE: The ordering of the conditional chain above impacts which matrices this
            # method performs best for. In the map situation (arguments have same shape, and
            # likely same or similar stored entry pattern), the Ai == Bi and termination
            # cases are equally or more likely than the Ai < Bi and Bi < Ai cases.
            if !iszero(Cx)
                if Ck > spaceC
                    error("Ck > spaceC")
                    spaceC = expandstorage!(C, Ck + (nnz(A) - (Ak - 1)) + (nnz(B) - (Bk - 1)))
                end
                rowvals(C)[Ck] = Ci
                nonzeros(C)[Ck] = Cx
                Ck += 1
            end
        end
    end
    @inbounds setcolptr!(C, size(C,2) + 1, Ck)
    # trimstorage!(C, Ck - 1)
    return C
end

function invperm!(b::AbstractVector, a::AbstractVector)
    # b = zero(a) # similar vector of zeros
    b .= 0
    n = length(a)
    (length(b) == n) || throw(ArgumentError("vector lengths"))
    @inbounds for (i, j) in enumerate(a)
        ((1 <= j <= n) && b[j] == 0) ||
            throw(ArgumentError("argument is not a permutation"))
        b[j] = i
    end
    b
end

# Auxiliary space to reduce allocations
function map_index_aux(A::SparseMatrixCSC, B::SparseMatrixCSC)
    S = size(A)
    IT = promote_type(SparseArrays.indtype(A), SparseArrays.indtype(B))
    ET = promote_type(eltype(A), eltype(B))
    
    maxnnz = nnz(A)+nnz(B)
    pointers = ones(IT, S[2] + 1)
    storedinds = Vector{IT}(undef, maxnnz)
    storedvals = Vector{ET}(undef, maxnnz)
    C = SparseMatrixCSC(S..., pointers, storedinds, storedvals)    

    hinv = randperm(size(B,1))
    hinvrowvals = similar(hinv, nnz(B))

    (C, hinv, hinvrowvals)
end

function plus_getindex!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC,
    h::AbstractVector, hinv::AbstractVector, hinvrowvals::AbstractVector)
    invperm!(hinv, h)
    rowvals(C) .= 0
    nonzeros(C) .= 0
    map_index_plus!(C, A, B, h, hinv, hinvrowvals)
end

end

module NetalGA

using SparseArrays
using Random
using ..GAFramework
import ..GAFramework: fitness, crossover!, mutation!, selection, randcreature, genauxga, printfitness
using ..CayleyCrossover
using ..Spmap

export NetalCreature, NetalModel

# Represents a permutation, i.e. 1-1 mapping from one set to another
struct NetalCreature
    f :: Vector{Int}
    score :: Float64 # alignment score
end

fitness(x::NetalCreature) where {T} = x.score

printfitness(curgen::Int, x::NetalCreature) =
    println("curgen: $curgen value: $(x.f) score: $(x.score)")

"""
Model to find network alignment by optimizing S3 (see MAGNA paper)
A network alignment is represented using a permutation
Faster implementation, slightly slower than the C++ code from the MAGNA++ paper
    using GAFramework, GAFramework.NetalGA
    using LightGraphs
    using Random
    mv = 100
    me = 400
    G1 = adjacency_matrix(LightGraphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    G2 = G1[invperm(perm),invperm(perm)]
    # G1.|V| <= G2.|V|
    model = NetalModel(G1,G2)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f .== perm)/mv)
    st.pop[1]
"""
struct NetalModel <: GAModel
    G1   :: SparseMatrixCSC{Int,Int}
    G2   :: SparseMatrixCSC{Int,Int}
end

"""
    using GAFramework, GAFramework.NetalGA, GAFramework.PermGA
    using LightGraphs, SparseArrays
    using Random
    GAFramework.setthreads(false)
    mv = 100
    me = 400
    G2 = adjacency_matrix(LightGraphs.erdos_renyi(mv,me))
    perm = randperm(mv)
    nv = 75
    G1 = G2[perm[1:nv], perm[1:nv]]
    model1 = NetalModel(G1,G2)
    model2 = NetalignModel(G1,G2)
    aux1 = genauxga(model1)
    f = randperm(size(G2,1));
    c1 = NetalCreature(f, model1, aux1)
    c2 = PermCreature(f, model2)
    c1.score, c2.score

    st1 = GAState(model1, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));
    st2 = GAState(model2, ngen=10, npop=60_000, elite_fraction=0.1, rng=MersenneTwister(10));

    @time ga!(st1)
    @time ga!(st2)

    G2 = sparse([1, 1, 2, 3, 4], [2, 3, 3, 4, 5], ones(Int,5), 5, 5) 
    G2 = min.(1, (G2 + G2')) 
    G1 = G2[1:4,1:4]
    model1 = NetalModel(G1,G2)
    model2 = NetalignModel(G1,G2)

    aux1 = genauxga(model1)
    f = randperm(size(G2,1))
    c1 = NetalCreature(f, model1, aux1)
    c2 = PermCreature(f, model2)
    c1.score, c2.score

    st = GAState(model1, ngen=100, npop=6_000, elite_fraction=0.1,
        params=Dict(:print_fitness_iter=>1))
    ga!(st)
    println("accuracy = ", sum(st.pop[1].f[1:4] .== [1,2,3,4])/4)
    println("accuracy = ", sum(st.pop[1].f[1:4] .== [2,1,3,4])/4)

    # using Profile
    # @profile NetalCreature(f, model1, aux1)
    # open("/home/vvijayan/home/gaprof.txt", "w") do fd Profile.print(fd) end

"""

function genauxga(m::NetalModel)
    crossover_aux = randperm(size(m.G2,1)), fill(false, size(m.G2,1))
    Spmap.map_index_aux(m.G1, m.G2), crossover_aux
end

# aux = genauxga(..) can be used to reduce allocations here
# by using auxiliary space instead of reallocating
function NetalCreature(h, m::NetalModel, aux)
    # Assumes that all edge weights are 1
    ((K, hinv, hinvrowvals), _) = aux
    Spmap.plus_getindex!(K, m.G1, m.G2, h, hinv, hinvrowvals) # K = G1 + G2[h,h][1:m,1:m]
    
    w = nonzeros(K)
    Nc = count(x->x==2, w)
    Nn = count(x->x==1, w)

    Nc,cr = divrem(Nc,2)
    Nn,nr = divrem(Nn,2) # need this & above so it works well with sim. ann. code
    if cr!=0 || nr!=0 error("G1 and G2 need to be symmetric"); end
    score = Nc/(Nc + Nn)
    NetalCreature(h, score)
end

function randcreature(m::NetalModel, aux, rng::AbstractRNG) where {T}
    f = randperm(rng, size(m.G2,1))
    NetalCreature(f, m, aux)
end

function crossover!(z, x, y, st::GAState{NetalModel}, aux, rng::AbstractRNG) where {T}
    (_, (r, visited)) = aux
    visited .= false
    cayley_crossover!(z.f, x.f, y.f, rng, r, visited)
    NetalCreature(z.f, st.model, aux)
end

end # NetalGA
