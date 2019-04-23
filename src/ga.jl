import Future
#import Base.Threads: @threads

dothreads = false

function setthreads(p)
    global dothreads = p
end

macro threads(ex)
    if ex.head === :for
        ret = quote
            global dothreads
            if dothreads
                Threads.@threads($ex)
            else
                $ex
            end
        end
        esc(ret)
    else
        throw(ArgumentError("unrecognized argument to @threads"))
    end
end

"""
Creates initial population as well as auxiliary structures for GA.
"""
function initializepop(model::GAModel, npop::Integer,
    nelites::Integer, baserng=Random.GLOBAL_RNG, sortpop=true)
    # each thread gets its own auxiliary scratch space
    # and each thread gets its own random number generator
    nthreads = Threads.nthreads()
    rngs = accumulate(Future.randjump, fill(big(10)^20, nthreads), init=baserng)
    aux = map(i -> genauxga(model), 1:nthreads)
    if npop > 0
        # initialize population
        pop1 = randcreature(model, aux[1], rngs[1])
        pop = Vector{typeof(pop1)}(undef, npop)
        pop[1] = pop1
        @threads for i = 2:npop
            threadid = Threads.threadid()
            pop[i] = randcreature(model, aux[threadid], rngs[threadid])
        end
        if sortpop
            sort!(pop, by=fitness, rev=true, alg=PartialQuickSort(max(1,nelites)))
        end
    else
        pop = nothing
    end
    return pop,aux,rngs
end

# this holds the full state of the genetic algorithm
# so that it can be stored to file
mutable struct GAState{GAM <: GAModel}
    model::GAM
    # vector of the population
    pop::Vector
    # number of generations
    ngen::Int
    # size of the population
    npop::Int
    # fraction of population that goes to the next generation regardless
    elite_fraction::Real
    # parameters for mutation!, crossover!, selection, logiteration, etc.
    params::Dict
    # random number generator for replication purposes
    baserng::AbstractRNG
    # current generation
    curgen::Int
end
function GAState(model::GAM;
    ngen=10, npop=100, elite_fraction=0.01,
    params=Dict(:mutation_rate=>0.1,
        :print_fitness_iter=>1,
        :save_creature_iter=>0,
        :save_state_iter=>0,
        :file_name_prefix=>"gamodel"),
    baserng=Random.GLOBAL_RNG) where {GAM <: GAModel}
    0 <= elite_fraction <= 1 || error("elite_fraction bounds")
    nelites = Int(floor(elite_fraction*npop))
    pop,aux,rngs = initializepop(model, npop, nelites, baserng)
    return GAState{GAM}(model, pop, ngen, npop, elite_fraction,
        params, baserng, 0)
end

struct GAIterable
    state :: GAState
    children :: Vector
    aux :: Vector
    rngs :: Vector
end

# Create space to store children
# as well as auxiliary space and rngs for each thread
function GAIterable(st::GAState)
    nelites = Int(floor(st.elite_fraction * st.npop))
    nchildren = st.npop - nelites
    (children, aux, rngs) = initializepop(st.model, nchildren, 0, st.baserng, false)
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
    if stopcondition(st)
        return nothing
    end
    st.curgen += 1

    nelites = Int(floor(st.elite_fraction * st.npop))
    nchildren = st.npop - nelites
    # main loop:
    # 1. select parents
    # 2. crossover parents & create children
    # 4. replace non-elites in current generation with children
    # 3. mutate population
    # 5. sort population
    parents = selection(st.pop, nchildren, st, it.rngs[1])
    @threads for i = 1:nchildren
        threadid = Threads.threadid()
        p1,p2 = parents[i]
        it.children[i] = crossover!(it.children[i], st.pop[p1], st.pop[p2], st,
            it.aux[threadid], it.rngs[threadid])
    end
    # moves children and elites to current pop
    for i = 1:nchildren
        j = nelites+i
        # need to swap here because each array element is a reference
        # and crossover! and mutation! overwrites the creatures
        st.pop[j], it.children[i] = it.children[i], st.pop[j]
    end

    # mutate pop, including elites (except for the most elite creature,
    # so that monotonocity of best fitness wrt generation number is preserved)
    @threads for i = 2:st.npop
        threadid = Threads.threadid()
        st.pop[i] = mutation!(st.pop[i], st, it.aux[threadid], it.rngs[threadid])
    end
    sort!(st.pop, by=fitness, rev=true, alg=PartialQuickSort(max(1,nelites)))

    logiteration(st)

    # @assert map(x -> CoordinateGA.CoordinateCreature(x.value, st.model).objvalue, st.pop) ≈ map(x -> x.objvalue, st.pop)

    st, iteration+1
end

function ga!(st::GAState)  
    println("Running genetic algorithm with population size $(st.npop), generation number $(st.ngen), elite fraction $(st.elite_fraction).")
    it = GAIterable(st)
    for _ in it; end
    st.pop[1]
end
