@testset "expectations" begin
    @rv X Y Z
    @event A
    Value = PracticalProbability.Value

    @testset "building and display" begin
        @test E(X + 2Y) isa Expect && 𝔼(X) == E(X)
        @test sprint(show, E(X + 2Y | Z)) == "E[X + 2Y | Z]"
        @test sprint(show, E(X^2)) == "E[X²]"
        @test sprint(show, Var(X)) == "Var(X)"
        @test sprint(show, Cov(X, Y | Z)) == "Cov(X, Y | Z)"
        @test sprint(show, E(X; name=:q)) == "E_q[X]"
        @test sprint(show, E(X)^2) == "E[X]²"
        @test latex(E(X | Y); env=:raw) == raw"\mathbb{E}\left[X \mid Y\right]"
        @test latex(Var(X); env=:raw) == raw"\operatorname{Var}\left(X\right)"
        @test latex(E(X^2); env=:raw) == raw"\mathbb{E}\left[X^{2}\right]"
        @test X * X == X^2 && 2X - 2X == PracticalProbability.RConst(0)
        @test Cov(X, Y) == Cov(Y, X)
    end

    @testset "rules" begin
        @test simplify(E(X + 2Y + 3)) == E(X) + 2 * E(Y) + 3
        @test simplify(E(3)) == Literal(3)
        @test simplify(E(𝟙(A) | Z)) == P(A | Z)
        @test simplify(E(A)) == P(A)                         # an event means its indicator
        @test sprint(show, simplify(E(X * Z | Z))) == "Z E[X | Z]"   # taking out what is known
        @test simplify(E(Z | Z)) == Value(PracticalProbability.RVal(Z))
        @test simplify(E(E(X | Y, Z) | Z)) == E(X | Z)       # tower
        @test simplify(E(E(X | Y))) == E(X)
        @test simplify(Var(2X + 1)) == 4 * Var(X)
        @test simplify(Var(X + Y)) == Var(X) + Var(Y) + 2 * Cov(X, Y)
        @test simplify(Cov(X, X)) == Var(X)
        @test simplify(Cov(2X, 3Y)) == 6 * Cov(X, Y)
        @test simplify(Var(5)) == Literal(0)

        gi = @dag begin
            X → Z
            Y → Z
        end                                                   # X ⟂ Y
        @test simplify(Var(X + Y), gi) == Var(X) + Var(Y)
        @test simplify(E(X * Y), gi) == E(X) * E(Y)
        @test simplify(E(X | Y), gi) == E(X)
        @test simplify(E(X * Y | Z), gi) == E(X * Y | Z)     # dependent given the collider
        @test simplify(Cov(X, Y), assume(X ⟂ Y)) == Literal(0)
        @test simplify(E(X * Y), assume(X ⟂ Y)) == E(X) * E(Y)
    end

    @testset "numbers" begin
        bn = rand_bayesnet(@dag(X → Y → Z); card=3, rng=RNG)
        @test compute(E(X), bn) ≈ sum(k * compute(P(X(k)), bn) for k in 1:3)
        @test compute(Var(3), bn) ≈ 0 atol = 1e-12
        pairs = ((E(X + 2Y), E(X) + 2 * E(Y)), (Var(X), definition(Var(X))), (E(X), definition(E(X))),
                 (E(Y), tower(E(Y), X)), (E(Z | X), tower(E(Z | X), Y)),
                 (Var(Z), total_variance(Var(Z), Y)), (Cov(X, Y), definition(Cov(X, Y))),
                 (E(X^2 | Y), definition(E(X^2 | Y))))
        for (a, b) in pairs
            @test verify(a, b, bn)
        end
        for e in (Var(2X + Y + 1), E(X * Y + Z | Y), Cov(X + Y, Z), E(E(Z | Y, X) | X))
            @test verify(simplify(e; trace=true), bn)
        end
        @test verify(simplify(E(Z | X, Y), @dag(X → Y → Z); trace=true), bn)

        bi = rand_bayesnet(@dag(begin
            X → Z
            Y → Z
        end); card=3, rng=RNG)
        @test isapprox(compute(Cov(X, Y), bi), 0; atol=1e-12)
        cz = compute(Cov(X, Y | Z), bi)                      # explaining away: dependent given Z
        @test maximum(abs, cz.values) > 1e-6

        eb = BayesNet(A => 0.3)
        @test compute(E(A), eb) ≈ 0.3
        @test compute(Var(A), eb) ≈ 0.21
        nb = BayesNet(X => [0.5, 0.5]; domains=Dict(X => [:a, :b]))
        @test_throws ArgumentError compute(E(X), nb)
    end
end
