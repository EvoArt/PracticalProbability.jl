@testset "Markov chains" begin
    weather = MarkovChain([1//2 1//2; 1//4 3//4]; states=[:sun, :rain])
    @test stationary(weather) == [1//3, 2//3]
    @test eltype(stationary(weather)) <: Rational
    @test isirreducible(weather) && isaperiodic(weather)
    @test mean_return_time(weather, :sun) == 3          # 1 / π(sun)
    @test hitting_time(weather, :sun) == [0, 4]
    @test nstep(weather, 2) == [3//8 5//8; 5//16 11//16]
    @test distribution(weather, [1, 0], 2) == [3//8, 5//8]

    # gambler's ruin with a fair coin, N = 4
    N = 4
    T = zeros(Rational{Int}, N + 1, N + 1)
    T[1, 1] = 1; T[N+1, N+1] = 1
    for i in 2:N
        T[i, i-1] = 1//2; T[i, i+1] = 1//2
    end
    ruin = MarkovChain(T; states=0:N)
    @test hitting_probability(ruin, N) == [0, 1//4, 1//2, 3//4, 1]      # i / N
    @test hitting_time(ruin, [0, N]) == [0, 3, 4, 3, 0]                 # i (N - i)
    @test !isirreducible(ruin)
    cls = communicating_classes(ruin)
    @test count(c -> c.recurrent, cls) == 2
    transient = only(filter(c -> !c.recurrent, cls))
    @test sort(transient.states) == [1, 2, 3] && transient.period == 2
    @test_throws ArgumentError stationary(ruin)          # two absorbing states

    # a biased coin: P(reach N from i) = (1 - r^i) / (1 - r^N), r = q / p
    p, q = 2//3, 1//3
    T2 = zeros(Rational{Int}, N + 1, N + 1)
    T2[1, 1] = 1; T2[N+1, N+1] = 1
    for i in 2:N
        T2[i, i-1] = q; T2[i, i+1] = p
    end
    r = q / p
    @test hitting_probability(MarkovChain(T2; states=0:N), N) == [(1 - r^i) / (1 - r^N) for i in 0:N]

    # a periodic chain and a never-reached state
    flip = MarkovChain([0 1; 1 0])
    @test !isaperiodic(flip) && only(communicating_classes(flip)).period == 2
    @test stationary(flip) == [1//2, 1//2]
    @test hitting_time(ruin, 0)[end] == Inf

    # the chain as a Bayesian network over its first steps
    @rv W
    bn = BayesNet(weather, W, 3; init=[1, 0])
    @test compute(P(W[3](:rain) | W[1](:sun)), bn) == 5//8
    @test compute(P(W[2](:sun) | W[3](:rain)), bn) == compute(P(W[2](:sun), W[3](:rain)), bn) / (5//8)
    bn2 = BayesNet(weather, W, 3; init=[1//2, 1//2])
    @test verify(P(W[3] | W[1], W[2]), P(W[3] | W[2]), bn2; atol=0)     # the Markov property


    # floats work too
    fl = MarkovChain([0.9 0.1; 0.5 0.5])
    @test stationary(fl) ≈ [5 / 6, 1 / 6]
    @test_throws ArgumentError MarkovChain([0.5 0.6; 0.5 0.5])
end
