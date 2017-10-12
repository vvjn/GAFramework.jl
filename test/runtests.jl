using GAFramework
using Base.Test

function test1()
    model = CoordinateModel(x -> x[1]==0.0 ? 0.0 : x[1] * sin(1/x[1]), -1.0, 1.0)
    state = GAState(model, ngen=100, npop=3_000, elite_fraction=0.1,
                    crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1)
    best = ga(state)
    x = abs(best.value[1])
    y = x==0 ? 0.0 : x * sin(1/x)
    println("$best $x $y")
    abs(x - 0.222549) < 0.01 && abs(y - (-0.217234)) < 0.01 && abs(best.objvalue - y) < 0.01
end

function test2()
    model = CoordinateModel(x -> any(x.==0) ? 0.0 : dot(x, sin.(1./x)), [-1.,-1.], [1.,1.])
    state = GAState(model, ngen=100, npop=3_000, elite_fraction=0.1,
                    crossover_rate=0.9, mutation_rate=0.9, print_fitness_iter=1)
    best = ga(state)
    x = abs.(best.value)
    y = any(x.==0) ? 0.0 : dot(x, sin.(1./x))
    println("$best $x $y")
    all(abs.(x - 0.222549) .< 0.1) && abs(y - (-0.4344604)) < 0.01 && abs(best.objvalue - y) < 0.01
end

@test test1()
@test test2()
