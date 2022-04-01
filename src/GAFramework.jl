__precompile__()

module GAFramework

using Random
using FileIO

export GAModel, ga!,
fitness, genauxga, crossover!, mutation!, selection, randcreature,
printfitness, savecreature,
GAState, GAIterable, loadgastate, savegastate,
RouletteWheelSelection, TournamentSelection

"""
To create a GA with a specific GAModel, import this module,
make a GAModel with the following interface functions:
    fitness (has default)
    genauxga  (has default)
    crossover! (no default)
    mutation! (has identity function as default)
    selection (has default)
    randcreature (no default)
    printfitness (has default)
    savecreature (has default)
    stopcondition (has default)
 """
abstract type GAModel end

const DEFAULT_GASTATE_PARAMS = Dict(:mutation_rate=>0.1,
    :print_fitness_iter=>1,
    :save_creature_iter=>0,
    :save_state_iter=>0,
    :file_name_prefix=>"gamodel")

include("ga.jl")

"""
Fitness function.
    fitness(x) is maximized always
    To minimize x.objvalue, dispatch fitness(x) to -x.objvalue for your Creature
    Recommended to make this either x.objvalue to maximize or -x.objvalue to minimize

    Since fitness(x) used for selecting the fittest creature, elites, and parents,
    all the computationally expensive part of calculating the fitness value should
    be implemented in the randcreature method.
"""
fitness(x) = x.objvalue

"""
    genauxga(model::GAModel) :: GAModel auxiliary structure

Given model GAM <: GAModel, generate auxiliary scratch space for calculating fitness scores
    model = GAM(G1,G2)
    aux = genauxga(model)
The purpose is to not allocate memory every time you calculate fitness for a new creature.    
"""
genauxga(model::GAModel) = nothing

"""
    crossover!(z, x,y, model::GAModel, aux, rng)

Crosses over x and y to create a child. Optionally use space in z as a
scratch space or to create the child. aux is more scratch space. rng is random number generator.
    model = GAM(G1,G2)
    aux = genauxga(model)
    x = randcreature(model,aux)
    y = randcreature(model,aux)
    z = randcreature(model,aux)
    child = crossover(z,x,y,model,aux,rng)
"""
crossover!(z, x, y, st::GAState, aux, rng::AbstractRNG) =
    error("crossover not implemented for $(typeof(z))")

"""
    Mutates a incoming creature and outputs mutated creature
"""
mutation!(creature, st::GAState, aux, rng::AbstractRNG) = creature

"""
    selection(pop::Vector, n::Integer, rng)

    Generate a vector of n tuples (i,j) where i and j are
    indices into pop, and where pop[i] and pop[j] are the
    selected parents.
    Uses binary tournament selection by default. 
"""    
selection(pop::Vector, n::Integer, st::GAState, rng::AbstractRNG) =
    selection(TournamentSelection(2), pop, n, st, rng)

"""
    randcreature(model::GAModel, aux)

    Create a random instance of a creature, given a GAModel.
    There is always a creature associated with a GAModel    
"""    
randcreature(model::GAModel, aux, rng::AbstractRNG) =
    error("randcreature not implemented for $(typeof(model))")

"""
Logging
    * saves state every save_state_iter iterations to file
        - restart using state = loadgastate(filename) & ga!(state)
    * outputs creature every save_creature_iter iterations to file
    * prints fitness value every print_fitness_iter iterations to screen

    print the fitness of fittest creature every n iteration
        print_fitness_iter::Int
    save the fittest creature to file every n iteration
        save_creature_iter::Int
    save the entire state of the GA (i.e. this struct) to file every n iteration
        save_state_iter::Int
    prefix for the files to be save
        file_name_prefix::AbstractString
"""
printfitness(curgen::Int, x::Any) =
    println("curgen: $(curgen) fitness: $(fitness(x))")
savecreature(file_name_prefix::AbstractString, curgen::Int, x) =
    save("$(file_name_prefix)_creature_$(curgen).bson", "creature", x)

stopcondition(st::GAState) = st.curgen > st.ngen

"""
    Saves ga state to file
"""       
function savegastate(file_name_prefix::AbstractString, curgen::Integer, state::GAState)
    filename = "$(file_name_prefix)_state_$(curgen).bson"
    println("Saving state to file $filename")
    save(filename,"state", state)
end

function loadgastate(filename::AbstractString)
    println("Load state from file $filename")
    load(filename, "state")
end

include("selections.jl")

include("models/coordinatega.jl")
include("models/permga.jl")
include("models/magnaga.jl")

function logiteration(st::GAState)
    creature = st.pop[1]
    gp = st.params
    file_name_prefix = gp[:file_name_prefix]
    if gp[:print_fitness_iter] > 0 && mod(st.curgen, gp[:print_fitness_iter]) == 0
        printfitness(st.curgen, creature)
    end
    if gp[:save_creature_iter] > 0 && mod(st.curgen, gp[:save_creature_iter]) == 0
        savecreature(file_name_prefix, st.curgen, creature)
    end
    if gp[:save_state_iter] > 0 && mod(st.curgen, gp[:save_state_iter]) == 0
        filename = "$(file_name_prefix)_state_$(st.curgen).bson"
        save(filename, "state", st)
    end
end

end
