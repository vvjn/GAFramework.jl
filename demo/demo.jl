using GAFramework

# Solves for function a + 2b + 3c + 4d + 5e = 42
# by minimizing abs(a + 2b + 3c + 4d + 5e - 42)
m = CoordinateModel(x -> abs(x[1] + 2x[2] + 3x[3] + 4x[4] + 5x[5] - 42),
                  [-100.,-100.,-100.,-100.,-100], # minimum corner
                  [100.,100,100.,100.,100]) # maximum corner in rectangle
# Prints fitness every iteration
state = GAState(m, ngen=100, npop=3_000, elite_fraction=0.1,
                crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1)
println(ga(state))

# Minimizes exp(|x|)
m = CoordinateModel(x -> exp(dot(x,x)),
                  [-1.,-1.,-1.,-1.,-1], # minimum corner
                  [1.,1.,1.,1.,1]) # maximum corner in rectangle
# Saves creature to file every 10 iterations
state = GAState(m, ngen=100, npop=3_000, elite_fraction=0.1,
                crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1,
                save_creature_iter=10, file_name_prefix="minexp_3000")
println(ga(state))

# Change maximum number of generations and then restart
state.ngen = 200
ga(state)

# It's possible to also change the other fields but
# it doesn't really make too much sense to

if false
    # The following doesn't work b/c JLD doesn't support types w/ functions
    # Minimizes |tan(x + 1000)|
    m = CoordinateModel(x -> abs(tanh(x[1]+1000)), -2000., 2000.)
    # Saves full state to file every 10 iterations 
    state = GAState(m, ngen=100, npop=3_000, elite_fraction=0.1,
                    crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1,
                    save_state_iter=25, file_name_prefix="tan1000_3000")
    println(ga(state))

    # Reload state from file and restart
    state = loadgastate("tan1000_3000_state_25.jld")

    ga(state)

end
