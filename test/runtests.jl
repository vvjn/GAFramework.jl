using Test
using LinearAlgebra
using SparseArrays

using GAFramework
using GAFramework.CoordinateGA
using GAFramework.PermGA
using GAFramework.MagnaGA

function test_fm1()
    model = FunctionModel(x -> x[1]==0.0 ? 0.0 : x[1] * sin(1/x[1]), [-1.0], [1.0])

    state = GAState(model, ngen=50, npop=300, elite_fraction=0.01,
                    params=Dict(:mutation_rate=>0.1, :crossover=>SinglePointCrossover(), :print_fitness_iter=>10))
    best = ga!(state)
    x = best.value[1]
    y = best.objvalue
    println("$best $x $y")
    @test abs(abs(x) - 0.223126) < 0.1 && abs(y - (-0.217219)) < 0.1

    state = GAState(model, ngen=50, npop=300, elite_fraction=0.01,
                    params=Dict(:mutation_rate=>0.1, :crossover=>TwoPointCrossover(), :print_fitness_iter=>10))
    best = ga!(state)
    x = best.value[1]
    y = best.objvalue
    println("$best $x $y")
    @test abs(abs(x) - 0.223126) < 0.1 && abs(y - (-0.217219)) < 0.1

    state = GAState(model, ngen=50, npop=300, elite_fraction=0.01,
                    params=Dict(:mutation_rate=>0.1, :crossover=>AverageCrossover(), :print_fitness_iter=>10))
    best = ga!(state)
    x = best.value[1]
    y = best.objvalue
    println("$best $x $y")
    @test abs(abs(x) - 0.223126) < 0.1 && abs(y - (-0.217219)) < 0.1

end

function test_fm2()
    model = FunctionModel(x -> any(z -> z==0.0, x) ? 0.0 : dot(x, sin.(1 ./x)),
                            -ones(Float64,15), ones(Float64,15))
    # do simulated annealing when mutating
    state = GAState(model, ngen=500, npop=300, elite_fraction=0.01,
        params=Dict(
            :mutation_rate=>0.1,
            :sa_rate=>0.1,:sa_k=>1, :sa_lambda=>1/1000,:sa_maxiter=>1000,
            :print_fitness_iter=>50))
    best = ga!(state)
    x = best.value
    y = best.objvalue
    println("$best $x $y")
    @test all(abs.(abs.(x) .- 0.222549) .< 0.1) && abs(y - (-0.21723*15)) < 0.1
end

function test_fm3()
    model = FunctionModel(x -> 10length(x) + sum(z -> z^2 - 10cos(2pi*z), x),
                            -5.12 .* ones(15), 5.12 .* ones(15))
    state = GAState(model, ngen=500, npop=300, elite_fraction=0.01,
        params=Dict(:mutation_rate=>0.1, :sa_rate=>0.1, :sa_k=>1,
            :sa_lambda=>1/1000, :sa_maxiter=>1000, :print_fitness_iter=>50))
    best = ga!(state)
    x = best.value
    y = best.objvalue
    println("$best $x $y")
    @test all(abs.(x) .< 0.1) && abs(y) < 0.1
end

function test_na1()
    rows = [3, 4, 5, 7, 9, 3, 7, 8, 9, 10, 1, 2, 6, 7, 8, 1, 6, 1, 6, 7, 9, 10, 3, 4, 5, 8, 1, 2, 3, 5, 9, 2, 3, 6, 1, 2, 5, 7, 2, 5]
    cols = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10]
    G2 = sparse(rows, cols, ones(length(rows)))
    f = [3, 2, 4, 8, 6, 1, 5, 10, 9, 7]
    G1 = G2[f[1:7],f[1:7]]
    S = rand(7,10)/10
    for i in 1:7 S[i,f[i]] = 1.0 end
    model = NetalignModel(G1,G2,S,0.5)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params = Dict(:print_fitness_iter=>20))
    best = ga!(st)
    @test best.f[1:7] == f[1:7]
end

function test_na2()
    rows = [3, 4, 5, 7, 9, 3, 7, 8, 9, 10, 1, 2, 6, 7, 8, 1, 6, 1, 6, 7, 9, 10, 3, 4, 5, 8, 1, 2, 3, 5, 9, 2, 3, 6, 1, 2, 5, 7, 2, 5]
    cols = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10]
    G2 = sparse(rows, cols, ones(length(rows)))
    f = [3, 2, 4, 8, 6, 1, 5, 10, 9, 7]
    G1 = G2[f[1:7],f[1:7]]
    S = rand(7,10)/10
    for i in 1:7 S[i,f[i]] = 1.0 end
    model = MagnaModel(G1,G2,S,0.5)
    st = GAState(model, ngen=100, npop=6_000, elite_fraction=0.1,
        params = Dict(:print_fitness_iter=>20))
    best = ga!(st)
    @test best.f[1:7] == f[1:7]
end

test_na1()
test_na2()
test_fm1()
test_fm2()
test_fm3()
