import Future
import Base.Threads: @threads
#macro threads(ex) :($(esc(ex))) end 

"""
   function to create a population for GAState or ga
"""
function initializepop(model::GAModel,npop::Integer,
                       nelites::Integer, baserng=Random.GLOBAL_RNG, sortpop=true)
    # each thread gets its own auxiliary scratch space
    # and each thread gets its own random number generator
    nthreads = Threads.nthreads()
    rngs = accumulate(Future.randjump, fill(big(10)^20, nthreads), init=baserng)
    aux = map(i -> genauxga(model), 1:nthreads)
    if npop > 0
        # initialize population
        pop1 = randcreature(model,aux[1],rngs[1])
        pop = Vector{typeof(pop1)}(undef, npop)
        pop[1] = pop1
        @threads for i = 2:npop
            threadid = Threads.threadid()
            pop[i] = randcreature(model, aux[threadid], rngs[threadid])
        end
        if sortpop
            sort!(pop,by=fitness,rev=true,
                  alg=PartialQuickSort(max(1,nelites)))
        end
    else
        pop = nothing
    end
    return pop,aux,rngs
end

# this holds the full state of the genetic algorithm
# so that it can be stored to file
mutable struct GAState
    model::GAModel
    # vector of the population
    pop::Vector
    # number of generations
    ngen::Int
    # current generation
    curgen::Int
    # size of the population
    npop::Int
    # fraction of population that goes to the next generation regardless
    elite_fraction::Real
    # parameters for crossover function
    crossover_params
    # parameters for mutation function
    mutation_params
    # print the fitness of fittest creature every n iteration
    print_fitness_iter::Int
    # save the fittest creature to file every n iteration
    save_creature_iter::Int
    # save the entire state of the GA (i.e. this struct) to file every n iteration
    save_state_iter::Int
    # prefix for the files to be save
    file_name_prefix::AbstractString
    # random number generator for replication purposes
    baserng::AbstractRNG
end
function GAState(model::GAModel;
                 ngen=10,
                 npop=100,
                 elite_fraction=0.01,
                 crossover_params=nothing,
                 mutation_params=Dict(:rate=>0.1),
                 print_fitness_iter=1,
                 save_creature_iter=0,
                 save_state_iter=0,
                 file_name_prefix="gamodel",
                 baserng=Random.GLOBAL_RNG)
    0 <= elite_fraction <= 1 || error("elite_fraction bounds")
    nelites = Int(floor(elite_fraction*npop))
    pop,aux,rngs = initializepop(model, npop, nelites, baserng)
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

struct GAIterable
    state :: GAState
    children :: Vector
    aux :: Vector
    rngs :: Vector
end

function ga!(st::GAState)  
    println("Running genetic algorithm with
            population size $(st.npop),
            generation number $(st.ngen),
            elite fraction $(st.elite_fraction),
            crossover_params $(repr((st.crossover_params))),
            mutation_params $(repr((st.mutation_params))),
            printing fitness every $(st.print_fitness_iter) iteration(s),
            saving creature to file every $(st.save_state_iter) iteration(s),
            saving state every $(st.save_state_iter) iteration(s),
            with file name prefix $(st.file_name_prefix).")
    it = GAIterable(st)
    foreach(identity, savegastate(savecreature(printfitness(it))))
    st.pop[1]
end

#        - logging
#            x saves state every save_state_iter iterations to file
#                - restart using state = loadgastate(filename) & ga!(state)
#            x outputs creature every save_creature_iter iterations to file
#            x prints fitness value every print_fitness_iter iterations to screen
#
function printfitness(it)
    f(st) = if st.print_fitness_iter > 0 && mod(st.curgen, st.print_fitness_iter) == 0
        printfitness(st.curgen, st.pop[1])
    end
    ((f(st); st) for st in it)
end

function savecreature(it)
    f(st) = if st.save_creature_iter > 0 && mod(st.curgen, st.save_creature_iter) == 0
        savecreature(st.file_name_prefix, st.curgen, st.pop[1], st.model)
    end
    ((f(st); st) for st in it)
end

function savegastate(it)
    f(st) = if st.save_state_iter > 0 && mod(st.curgen, st.save_state_iter) == 0
        savegastate(st.file_name_prefix, st.curgen, st)
    end
    ((f(st); st) for st in it)
end

function GAIterable(st::GAState)
    (_, aux, rngs) = initializepop(st.model, 0, 0, st.baserng)
    nelites = Int(floor(st.elite_fraction * st.npop))
    children = deepcopy(st.pop[nelites+1:end])
    GAIterable(st, children, aux, rngs)
end

"""
    ga function
    x in each generation, the following is done
        - select parents from all creatures in population
        - create children using crossover
        - replace non-elites in population with children
        - mutate all creatures (both elites and children) in population
"""
function Base.iterate(it::GAIterable, iteration::Int=it.state.curgen)
    st = it.state
    nelites = Int(floor(st.elite_fraction * st.npop))
    nchildren = st.npop - nelites

    if iteration > st.ngen
        return nothing
    end

   # main loop:
    # 1. select parents
    # 2. crossover parents & create children
    # 4. replace non-elites in current generation with children
    # 3. mutate population
    # 5. sort population
    parents = selection(st.pop, nchildren, it.rngs[1])
    for i = 1:nchildren
        threadid = Threads.threadid()
        p1,p2 = parents[i]
        it.children[i] = crossover(it.children[i], st.pop[p1], st.pop[p2],
            st.model, st, it.aux[threadid], it.rngs[threadid])
    end
    # moves children and elites to current pop
    st.pop[nelites+1:end] = it.children
    
    # mutate pop, including elites (except for the most elite creature
    # in order to preserve monotonocity wrt best fitness)
    for i = 2:st.npop
        threadid = Threads.threadid()
        st.pop[i] = mutation(st.pop[i], st.model, st,
            it.aux[threadid], it.rngs[threadid])
    end
    sort!(st.pop, by=fitness, rev=true, alg=PartialQuickSort(max(1,nelites)))

    st.curgen += 1

    st, iteration+1
end

"""
       Saves ga state to file
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
