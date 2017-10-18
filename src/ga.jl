import Base.Threads: @threads
macro threads(ex) :($(esc(ex))) end 

"""
   function to create a population for GAState or ga
"""
function initializepop(model::GAModel,npop::Integer,baserng,sort=true)
    # each thread gets its own auxiliary scratch space
    nthreads = Threads.nthreads()
    rngs = randjump(baserng, nthreads)
    aux = map(i -> genauxga(model), 1:nthreads)
    # initialize population
    pop1 = randcreature(model,aux[1],rngs[1])
    pop = Vector{typeof(pop1)}(npop)
    pop[1] = pop1
    @threads for i = 2:npop
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
    crossover_params
    mutation_params
    print_fitness_iter::Real
    save_creature_iter::Real
    save_state_iter::Integer    
    file_name_prefix::AbstractString
    baserng::AbstractRNG
end
function GAState(model::GAModel;
                 ngen=10,
                 npop=100,
                 elite_fraction=0.2,
                 crossover_params=nothing,
                 mutation_params=Dict(:rate=>0.1),
                 print_fitness_iter=1,
                 save_creature_iter=0,
                 save_state_iter=0,
                 file_name_prefix="gamodel",
                 baserng=Base.GLOBAL_RNG)
    pop,aux,rngs = initializepop(model, npop, baserng)
    return GAState(model,pop,ngen,0,npop,
                   elite_fraction,
                   crossover_params,
                   mutation_params,
                   print_fitness_iter,
                   save_creature_iter,
                   save_state_iter,
                   file_name_prefix,
                   baserng)
end

"""
       Saves ga state to file
       Doesn't support GAModels or GACreatures containing functions (e.g. CoordinateModel)
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
    ga function
        x create state using state = GAState(...) and run using ga(state)
        - this allows us to restart from a saved state
        x does crossover & mutation
        - elite part of population is kept for next generation
            - crossover selects from all creatures in population
            - children created using crossover replaces only non-elite part of population
            - mutation mutates only non-elite part of population
            x saves state every save_state_iter iterations to file
            - restart using state = loadgastate(filename) & ga(state)
            x outputs creature every save_creature_iter iterations to file
            x prints fitness value every print_fitness_iter iterations to screen
            """
function ga(state::GAState)
    # load from state
    model = state.model
    pop = state.pop
    ngen = state.ngen
    curgen = state.curgen
    npop = state.npop
    elite_fraction = state.elite_fraction
    crossover_params = state.crossover_params
    mutation_params = state.mutation_params
    print_fitness_iter = state.print_fitness_iter
    save_creature_iter = state.save_creature_iter
    save_state_iter = state.save_state_iter
    file_name_prefix = state.file_name_prefix
    baserng = state.baserng

    # keep elites for next generation. elite fraction cutoff index
    nelites = Int(floor(elite_fraction*npop))
    0 <= nelites <= npop || error("elite fraction")
    nchildren = npop-nelites

    println("Running genetic algorithm with
            population size $npop,
            generation number $ngen,
            elite fraction $elite_fraction,
            children created $nchildren,
            crossover_params $crossover_params,
            mutation_params $mutation_params,
            printing fitness every $print_fitness_iter iteration(s),
            saving creature to file every $save_state_iter iteration(s),
            saving state every $save_state_iter iteration(s),
            with file name prefix $file_name_prefix.")

    # initialization
    children,aux,rngs = initializepop(model, nchildren, baserng, false)

    # main loop:
    # 1. select parents
    # 2. crossover parents & create children
    # 4. replace non-elites in current generation with children
    # 3. mutate population
    # 5. sort population
    for curgen = curgen+1:ngen
        # crossover. uses multi-threading when available
        parents = selection(pop, nchildren, rngs[1])
        @threads for i = 1:nchildren
            threadid = Threads.threadid()
            p1,p2 = parents[i]
            children[i] = crossover(children[i], pop[p1], pop[p2],
                                    model, crossover_params, curgen,
                                    aux[threadid], rngs[threadid])
        end
        # moves children and elites to current pop
        for i = 1:nchildren
            ip = nelites+i
            pop[ip], children[i] = children[i], pop[ip]
        end
        # mutate pop (including elites)
        # range i from 2:npop if you want monotonocity
        @threads for i = 1:npop
            threadid = Threads.threadid()
            pop[i] = mutate(pop[i], model, mutation_params, curgen,
                            aux[threadid], rngs[threadid])
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
                            crossover_params,
                            mutation_params,
                            print_fitness_iter,
                            save_creature_iter,
                            save_state_iter,
                            file_name_prefix,
                            baserng)
            savegastate(file_name_prefix, curgen, state)
        end
    end
    state.curgen = curgen
    pop[1]
end
