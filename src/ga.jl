"""
   function to create a population for GAState or ga
"""
function initializepop(model::GAModel,npop::Integer,sort=true)
    # each thread gets its own auxiliary scratch space
    nthreads = Threads.nthreads()
    rngs = randjump(Base.GLOBAL_RNG, nthreads)
    aux = map(i -> genauxga(model), 1:nthreads)
    # initialize population
    pop1 = randcreature(model,aux[1],rngs[1])
    pop = Vector{typeof(pop1)}(npop)
    pop[1] = pop1
    Threads.@threads for i = 2:npop
        threadid = Threads.threadid()
        pop[i] = randcreature(model,aux[threadid],rngs[threadid])
    end
    sort && sort!(pop,by=fitness,rev=true)
    return pop,aux,rngs
end

type GAState
    model::GAModel
    pop::Vector
    ngen::Integer
    curgen::Integer
    npop::Integer
    elite_fraction::Real
    crossover_rate::Real
    mutation_rate::Real
    print_fitness_iter::Real
    save_creature_iter::Real
    save_state_iter::Integer    
    file_name_prefix::AbstractString
end
function GAState(model::GAModel;
                 ngen=10,
                 npop=100,
                 elite_fraction=0.2,
                 crossover_rate=0.9,
                 mutation_rate=0.1,
                 print_fitness_iter=1,
                 save_creature_iter=0,
                 save_state_iter=0,
                 file_name_prefix="gamodel")
    pop,aux,rngs = initializepop(model, npop)
    return GAState(model,pop,ngen,0,npop,
                   elite_fraction,
                   crossover_rate,
                   mutation_rate,
                   print_fitness_iter,
                   save_creature_iter,
                   save_state_iter,
                   file_name_prefix)
end

"""
       Saves ga state to file
       Doesn't support GAModels or GACreatures containing functions (e.g. EuclideanModel)
       since JLD doesn't support saving functions
"""       
function savegastate(file_name_prefix::AbstractString, curgen::Integer, state::GAState)
    filename = "$(file_name_prefix)_state_$(curgen).jld"
    println("Saving state to file $filename")
    save(filename,"state",state)
end

function loadgastate(filename::AbstractString)
    println("Load state from file $filename")
    load(filename, "state")
end

"""
    ga! function
        x create state using state = GAState(...) and run using ga!(state)
        - this allows us to restart from a saved state
        x does crossover & mutation
        - elite part of population is kept for next generation
            - crossover selects from all creatures in population
            - children created using crossover replaces only non-elite part of population
            - mutation mutates only non-elite part of population
            x saves state every save_state_iter iterations to file
            - restart using state = loadgastate(filename) & ga!(state)
            x outputs creature every save_creature_iter iterations to file
            x prints fitness value every print_fitness_iter iterations to screen
            """
function ga!(state::GAState)
    # load from state
    model = state.model
    pop = state.pop
    ngen = state.ngen
    curgen = state.curgen
    npop = state.npop
    elite_fraction = state.elite_fraction
    crossover_rate = state.crossover_rate
    mutation_rate = state.mutation_rate
    print_fitness_iter = state.print_fitness_iter
    save_creature_iter = state.save_creature_iter
    save_state_iter = state.save_state_iter
    file_name_prefix = state.file_name_prefix

    # keep elites for next generation. elite fraction cutoff index
    elite_cf = Int(floor(elite_fraction*npop))
    0 <= elite_cf <= npop || error("elite fraction")
    nchildren = npop-elite_cf

    println("Running genetic algorithm with\n population size $npop,\n generation number $ngen,\n elite fraction $elite_fraction,\n children created $nchildren,\n crossover rate $crossover_rate,\n mutation rate $mutation_rate,\n printing fitness every $print_fitness_iter iteration(s),\n saving creature to file every $save_state_iter iteration(s),\n saving state every $save_state_iter iteration(s),\n with file name prefix $file_name_prefix.")

    # initialization
    children,aux,rngs = initializepop(model, nchildren, false)

    # main loop:
    # 1. select parents
    # 2. crossover parents & create children
    # 3. replace non-elite population with children
    # 4. mutate non-elite population
    # 5. sort population
    for curgen = curgen+1:ngen
        # crossover. uses multi-threading when available
        if crossover_rate > 0
            parents = selection(pop, nchildren, rngs[1])
            Threads.@threads for i = 1:nchildren
                ip = elite_cf+i
                if rand() < crossover_rate
                    threadid = Threads.threadid()
                    p1,p2 = parents[i]
                    children[i] = crossover(pop[p1], pop[p2], model,
                                            aux[threadid], children[i],
                                            rngs[threadid])
                else
                    # if not crossing then prepare to just keep pop[ip]
                    # we'll be swapping this back later
                    children[i], pop[ip] = pop[ip], children[i]
                end
            end
            # swaps space between non-elites and children
            # non-elites will get overwritten in the next generation
            for i = 1:nchildren
                ip = elite_cf+i
                children[i], pop[ip] = pop[ip], children[i]
            end
        end
        # mutate
        if mutation_rate > 0
            Threads.@threads for ip = elite_cf+1:npop
                if rand() < mutation_rate
                    threadid = Threads.threadid()
                    pop[ip] = mutate(pop[ip], model, aux[threadid], rngs[threadid])
                end
            end
        end
        sort!(pop,by=fitness,rev=true)

        if print_fitness_iter>0 && mod(curgen,print_fitness_iter)==0
            printfitness(curgen, pop[1])
        end

        if save_creature_iter>0 && mod(curgen,save_creature_iter)==0
            savecreature(file_name_prefix, curgen, pop[1], model)
        end

        if save_state_iter>0 && mod(curgen,save_state_iter)==0
            state = GAState(model,pop,ngen,curgen,npop,
                            elite_fraction,
                            crossover_rate,
                            mutation_rate,
                            print_fitness_iter,
                            save_creature_iter,
                            save_state_iter,
                            file_name_prefix)
            savegastate(file_name_prefix, curgen, state)
        end
    end
    state.curgen = curgen
    pop[1]
end

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
