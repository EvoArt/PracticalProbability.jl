@testset "named distributions" begin
    @rv x y D X Y Z X1 X2 Y2
    @density π q

    @test π isa Density
    j = π(x, y | D)
    @test j isa Prob && j.name === :π
    @test sprint(show, j) == "π(x, y | D)"
    @test latex(j; env=:raw) == raw"\pi(x, y \mid D)"
    @test Density(:f)(x) == P(x; name=:f)
    @test π(x) != P(x) && π(x) != q(x)
    @test hash(π(x)) != hash(q(x))

    # rules stay within one distribution
    @test chainrule(π(x, y | D)) == π(x | y, D) * π(y | D)
    @test bayes(π(x | y)) == π(y | x) * π(x) / π(y)
    @test simplify(π(x | y) * π(y)) == π(x, y)
    @test simplify(π(x | y) * q(y)) == π(x | y) * q(y)
    @test simplify(π(x, y) / π(y)) == π(x | y)
    @test simplify(P(x, y) / q(y)) == P(x, y) / q(y)
    @test simplify(P(x) / q(x)) == P(x) / q(x)      # p/q is an importance weight, not 1
    @test simplify(Σ(x, q(x | y))) == Literal(1)

    @testset "a graph belongs to one distribution" begin
        g = @dag Z → X → Y                           # Y ⟂ Z | X
        @test simplify(P(Y | X, Z), g) == P(Y | X)
        @test simplify(q(Y | X, Z), g) == q(Y | X, Z)                 # q is any kernel
        @test simplify(q(Y | X, Z); dag=Dict(:q => g)) == q(Y | X)
        @test simplify(P(Y | X, Z) * q(Y | X, Z), g) == P(Y | X) * q(Y | X, Z)
        @test expand(π(Y | Z), g) isa PExpr
        @test all(p -> p.name === :π, PracticalProbability._factors(expand(π(Y, Z), g)))
    end

    @testset "variable elimination with the graph" begin
        g = @dag X1 → X2 → Y2
        d = simplify(Σ(X2, P(X2 | X1) * P(Y2 | X2)), g; trace=true)
        @test last(d) == P(Y2 | X1)
        @test any(r -> occursin("d-separation), chain rule", r), d.rules)
        bn = rand_bayesnet(g; card=3, rng=RNG)
        @test verify(d, bn)
        # without the graph it stays a sum (still correct)
        @test simplify(Σ(X2, P(X2 | X1) * P(Y2 | X2))) isa Marginal

        # widening by the other factor's targets: P(Y1 | X1) P(Y2, X1) = P(Y1, Y2, X1)
        @rv Y1
        h = @dag begin
            X1 → (X2, Y1)
            X2 → Y2
        end
        e = Σ((X1, X2), P(X1) * P(X2 | X1) * P(Y1 | X1) * P(Y2 | X2))
        d2 = simplify(e, h; trace=true)
        @test last(d2) == P(Y1, Y2)
        @test verify(d2, rand_bayesnet(h; card=3, rng=RNG))
        # and not when the extra variable is not separated: Y1 depends on X2 here
        h2 = @dag begin
            X1 → (X2, Y1)
            X2 → (Y1, Y2)
        end
        @test PracticalProbability._graph_merge(P(Y1 | X1), P(Y2, X1, X2),
            PracticalProbability.SimplifyContext(PracticalProbability._dags(h2), true,
                                                 PracticalProbability._facts(nothing))) === nothing
    end

    @testset "one network per distribution" begin
        g = @dag X → Y
        bp = rand_bayesnet(g; card=3, rng=RNG)
        bq = rand_bayesnet(g; card=3, rng=RNG)
        nets = Dict(:P => bp, :q => bq)
        w = P(X) / q(X)                                # importance weight
        @test compute(Σ(X, q(X) * w), nets) ≈ 1
        f = compute(P(Y | X), bp)
        @test isapprox(compute(Σ(X, q(X) * w * P(Y | X)), nets), compute(P(Y), bp))
        @test verify(Σ(X, q(X) * w), Literal(1), nets)
        @test_throws ArgumentError compute(π(X), Dict(:q => bq))
        @test compute(q(X(1)), nets) ≈ compute(P(X(1)), bq)
    end

    @testset "models" begin
        gm = @generative function lin(x, y)
            a ~ Normal(0, 1)
            y ~ Normal(a * x, 1)
        end
        eq = posterior(gm; density=:π)
        @test sprint(show, eq) == "π(a | y, x) ∝ π(a) π(y | a, x)"
        @test factorize(gm; name=:π) == eq.rhs
    end
end
