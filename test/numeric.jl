@testset "numeric" begin
    @rv A B C

    bn = BayesNet(
        A => [0.3, 0.7],
        (B | A) => [0.9 0.2; 0.1 0.8];
        domains=Dict(A => [:no, :yes]),
    )
    @test domain(bn, A) == [:no, :yes] && domain(bn, B) == [1, 2]
    @test compute(P(A(:yes)), bn) ≈ 0.7
    @test compute(P(B(1) | A(:no)), bn) ≈ 0.9
    pB1 = 0.3 * 0.9 + 0.7 * 0.2
    @test compute(P(B(1)), bn) ≈ pB1
    @test compute(P(A(:no) | B(1)), bn) ≈ 0.3 * 0.9 / pB1
    @test compute(Σ(A, P(A, B(1))), bn) ≈ pB1

    f = compute(P(A | B), bn)
    @test f isa Factor && f.vars == [A, B]       # targets first
    @test f[A=:no, B=1] ≈ 0.3 * 0.9 / pB1
    @test f(A => :yes, B => 2) ≈ 0.7 * 0.8 / (1 - pB1)
    @test sumout(f, A)[B=1] ≈ 1

    # do(⋅): no back-door here, so intervening on A is conditioning; on B it cuts A off
    @test isapprox(compute(P(B | Do(A)), bn), compute(P(B | A), bn))
    @test isapprox(compute(P(A | Do(B(1))), bn), compute(P(A), bn))

    # Tables.jl rows: first variable slowest
    cols = PracticalProbability.Tables.columns(f)
    @test cols.A == [:no, :no, :yes, :yes] && cols.B == [1, 2, 1, 2]
    @test sum(cols.value) ≈ 2

    @test_throws ArgumentError BayesNet(A => [0.5, 0.6])                 # doesn't sum to 1
    @test_throws ArgumentError BayesNet((B | A) => [0.5 0.5; 0.5 0.5])   # A has no table
    @test_throws ArgumentError compute(P(A(:maybe)), bn)

    # verify compares as functions: P(C | A, B) = P(C | B) on a chain
    g = @dag A → B → C
    nb = rand_bayesnet(g; rng=RNG)
    @test verify(P(C | A, B), P(C | B), nb)
    @test verify(Equation(P(A), Σ(B, P(A, B))), nb)
end
