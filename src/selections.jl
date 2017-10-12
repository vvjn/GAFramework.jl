immutable RouletteWheelSelection end
# selection(pop::Vector{<:GACreature}, n::Integer, rng)
function selection(::RouletteWheelSelection,
                   pop::Vector{<:GACreature}, n::Integer, rng=Base.GLOBAL_RNG)    
    wmin,wmax = extrema(fitness(c) for c in pop)
    weight = wmax - wmin
    function stochasticpick()
        #iter = 0
        while true
            i = rand(rng,1:length(pop))
            if weight * rand(rng) < fitness(pop[i]) - wmin
                #println("stochastic pick: $n $(iter)")
                return i
            end
            #iter += 1
            #iter > length(pop) || error("infinite loop in selection")
        end
    end
    parents = Vector{Tuple{Int,Int}}(n)
    for k = 1:n
        i = stochasticpick()
        j = i
        while i==j
            j = stochasticpick()
        end
        parents[k] = (i,j)
    end
    parents
end

immutable TournamentSelection
    k::Int # if k=2 then binary tournament selection
    TournamentSelection(k=2) = new(k)
end
function selection(sel::TournamentSelection,
                   pop::Vector{<:GACreature}, n::Integer, rng=Base.GLOBAL_RNG)    
    function stochasticpick()
        si = rand(rng,1:length(pop))
        weighti = fitness(pop[si])
        for _ = 1:sel.k-1
            sj = rand(rng,1:length(pop))
            weightj = fitness(pop[sj])
            if weightj > weighti
                si = sj
                weighti = weightj
            end
        end
        si
    end
    parents = Vector{Tuple{Int,Int}}(n)
    for k = 1:n
        i = stochasticpick()
        j = i
        while i==j
            j = stochasticpick()
        end
        parents[k] = (i,j)
    end
    parents
end
