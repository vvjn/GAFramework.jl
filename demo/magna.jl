using GAFramework, GAFramework.MagnaGA
using LightGraphs, SparseArrays
using Random
using Statistics
using DelimitedFiles
#GAFramework.setthreads(false)

"""
Return line after skipping if pred(line) is true
Default pred: after skipping empty lines and commented lines
"""
function readskipping(fd::IO,
    pred=line -> length(line)==0 || line[1]=='#' || line[1]=='%')
    while !eof(fd)
        line = strip(readline(fd))
        if pred(line); continue; end
        return line
    end
    return ""
end

"""
    readgw(fd::IO)
    readgw(file::AbstractString) -> SparseMatrixCSC, node list
Reads LEDA format file describing a network. Outputs an undirected network.
An example of a LEDA file is in the examples/ directory.
"""
function readgw(fd::IO)
    line = readskipping(fd)
    strip(line)=="LEDA.GRAPH" || error("Error in line: $line")
    for i = 1:3; readskipping(fd); end
    nverts = parse(Int,readskipping(fd))
    vertices = Array{String}(undef, nverts)
    for i = 1:nverts
        line = readskipping(fd)
        vertices[i] = match(r"\|{(.*?)}\|",line).captures[1]
    end
    nedges = parse(Int,readskipping(fd))
    I = Vector{Int}(undef, nedges)
    J = Vector{Int}(undef, nedges)
    for i = 1:nedges
        line = readskipping(fd)
        caps = match(r"(\d+) (\d+) 0 \|{.*}\|",line).captures
        n1 = parse(Int,caps[1])
        n2 = parse(Int,caps[2])
        n1==n2 && continue
        I[i] = n1
        J[i] = n2
    end
    G = sparse(vcat(I,J), vcat(J,I), 1, nverts, nverts, max)
    return G, vertices
end
readgw(file::AbstractString, args...) = open(fd -> readgw(fd,args...), file, "r")


function main()
    n1 = ARGS[1]
    n2 = ARGS[2]
    sim = ARGS[3]
    alpha = parse(Float64, ARGS[4])
    npop = parse(Int, ARGS[5])
    ngen = parse(Int, ARGS[6])
    elite_fraction = parse(Float64, ARGS[7])
    out = ARGS[8]

    G1,verts1 = readgw(n1)
    G2,verts2 = readgw(n2)

    S,_ = readdlm(sim, header=true)

    model = MagnaModel(G1,G2,S,alpha)
    st = GAState(model, ngen=2, npop=2, elite_fraction=0.5)
    ga!(st)

    model = MagnaModel(G1,G2,S,alpha)
    st = GAState(model, ngen=ngen, npop=npop, elite_fraction=elite_fraction)
    @time ga!(st)

    r = st.pop[1]
    verts2map = verts2[r.f]
    correctness = mean(verts1 .== verts2map[1:length(verts1)])
    open("$(out)_final_alignment.txt", "w") do fd
        for i in 1:length(verts1)
            println(fd, "$(verts1[i]) $(verts2map[i])")
        end
    end
    function print_stats(fd)
        println(fd, "generation final")
        println(fd, "score $(r.score)")
        println(fd, "node_score $(r.node_score)")
        println(fd, "S3_score $(r.edge_score)")
        println(fd, "node_correctness $(correctness)")
    end
    print_stats(stdout)
    open(print_stats, "$(out)_final_stats.txt", "w")
end

main()
