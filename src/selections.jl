struct RouletteWheelSelection end
function selection(::RouletteWheelSelection,
    pop::Vector, n::Integer, st::GAState, rng::AbstractRNG)    
    wmin,wmax = extrema(fitness(c) for c in pop)
    weight = wmax - wmin
    function stochasticpick()
        while true
            i = rand(rng,1:length(pop))
            if wmin + weight * rand(rng,typeof(weight)) <= fitness(pop[i])
                return i
            end
        end
    end
    parents = Vector{Tuple{Int,Int}}(undef, n)
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

struct TournamentSelection
    k::Int # if k=2 then binary tournament selection
    TournamentSelection(k=2) = new(k)
end
function selection(sel::TournamentSelection,
    pop::Vector, n::Integer, st::GAState, rng::AbstractRNG)    
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
    parents = Vector{Tuple{Int,Int}}(undef, n)
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
