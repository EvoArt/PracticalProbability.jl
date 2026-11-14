@testset "rules" begin
    @rv A B C D X Y Z W

    # A generic network on four variables in which everything depends on
    # everything: identities from the axioms of probability must hold here.
    full = @dag begin
        A → (B, C, D)
        B → (C, D)
        C → D
    end
    bn = rand_bayesnet(full; card=3, rng=RNG)

    @testset "explicit transformations" begin
        @test chainrule(P(A, B, C)) == P(A | B, C) * P(B | C) * P(C)
        @test chainrule(P(A, B | C); order=[B, A]) == P(B | A, C) * P(A | C)
        @test bayes(P(A | B)) == P(B | A) * P(A) / P(B)
        @test bayes(P(A | B, C); swap=B) == P(B | A, C) * P(A | C) / P(B | C)
        @test marginalize(P(A | C), B) == Σ(B, P(A, B | C))
        @test condition_on(P(A), B) == Σ(B, P(A | B) * P(B))

        for e in (P(A, B, C | D), P(A, B | C))
            @test verify(e, chainrule(e), bn)
        end
        @test verify(P(A | B, C), bayes(P(A | B, C)), bn)
        @test verify(P(A | B, C), bayes(P(A | B, C); swap=B), bn)
        @test verify(P(A | B), bayes(P(A | B); expand=true), bn)
        @test verify(P(A | C), marginalize(P(A | C), [B, D]), bn)
        @test verify(P(A | D), condition_on(P(A | D), B), bn)
        @test verify(P(A(2) | B(1)), bayes(P(A(2) | B(1)); expand=true), bn)
    end

    @testset "simplify without a graph" begin
        @test simplify(bayes(P(A | B))) == P(A | B)
        @test simplify(chainrule(P(A, B, C))) == P(A, B, C)
        @test simplify(Σ(B, P(A, B | C))) == P(A | C)
        @test simplify(Σ(B, P(B | C))) == Literal(1)
        @test simplify(Σ([A, B], P(A, B))) == Literal(1)
        @test simplify(Σ(B, P(A | B) * P(B))) == P(A)
        @test simplify(P(A, B | A)) == P(B | A)
        @test simplify(P(A | A)) == Literal(1)
        @test simplify(P(A, B | C) / P(B | C)) == P(A | B, C)
        @test simplify((P(A) * P(B)) / P(B)) == P(A)
        # factors that don't involve the summed variable leave the sum
        @test simplify(Σ(B, P(C) * P(B | A)); combine=false) == P(C)
        # nothing to do: unchanged
        @test simplify(P(A | B) * P(C)) == P(A | B) * P(C)

        # every simplification is an identity
        for e in (bayes(P(A | B, C); expand=true), chainrule(P(A, B, C | D)),
                  condition_on(P(A | C), B), Σ(D, P(A | B, C, D) * P(D | B, C)))
            d = simplify(e; trace=true)
            @test verify(d, bn)
        end
    end

    @testset "derivations" begin
        d = simplify(bayes(P(A | B)); trace=true)
        @test d isa Derivation
        @test first(d) == bayes(P(A | B)) && last(d) == P(A | B)
        @test length(d) == 3
        @test occursin("chain rule", d.rules[2])

        d2 = derivation(P(A | B))
        apply!(d2, bayes, "Bayes' rule")
        apply!(d2, simplify)        # a traced simplify appends its own steps
        @test last(d2) == P(A | B) && d2.rules[2] == "Bayes' rule"
        @test Equation(d2).lhs == P(A | B)
        @test verify(d2, bn)
    end

    @testset "d-separation" begin
        g = @dag A → B → C
        @test simplify(P(C | A, B), g) == P(C | B)
        @test simplify(P(A | B, C), g) == P(A | B)
        @test simplify(P(C | A), g) == P(C | A)
        nb = rand_bayesnet(g; rng=RNG)
        @test verify(P(C | A, B), P(C | B), nb)
        @test !verify(P(C | A), P(C), nb)      # sanity: a dependence is detected

        d = simplify(P(C | A, B), g; trace=true)
        @test occursin("d-separation", d.rules[end])
    end

    @testset "expand" begin
        g = @dag begin
            A → B → C
            A → D
        end
        nb = rand_bayesnet(g; card=2, rng=RNG)
        for q in (P(C), P(A | C), P(B | C, D), P(A, C), P(D | C), P(C(1) | A(2)))
            d = expand(q, g; trace=true)
            @test verify(d, nb)
            # D is barren for every query that doesn't mention it
            D in PracticalProbability.allvars(q) || @test !(D in PracticalProbability.allvars(last(d)))
        end
        # barren descendants drop out entirely
        @test expand(P(A), g) == P(A)
        @test expand(P(B), g) == P(B)
    end

    @testset "do-calculus and identification" begin
        # X → Y: intervening is conditioning (rule 2)
        g1 = @dag X → Y
        @test simplify(P(Y | Do(X)), g1) == P(Y | X)
        # Y → X: intervening on an effect does nothing (rule 3)
        g2 = @dag Y → X
        @test simplify(P(Y | Do(X)), g2) == P(Y)

        # confounder: back-door adjustment
        g3 = @dag begin
            Z → (X, Y)
            X → Y
        end
        e = identify(P(Y | Do(X)), g3)
        @test e == Σ(Z, P(Z) * P(Y | X, Z))
        nb3 = rand_bayesnet(g3; card=3, rng=RNG)
        @test verify(P(Y | Do(X)), e, nb3)
        @test !verify(P(Y | Do(X)), P(Y | X), nb3)   # confounding is real
        @test verify(identify(P(Y(2) | Do(X(1))), g3; trace=true), nb3)

        # front-door-free mediator chain with a confounded treatment, all observed
        g4 = @dag begin
            Z → (X, Y)
            X → W → Y
        end
        nb4 = rand_bayesnet(g4; card=2, rng=RNG)
        for q in (P(Y | Do(X)), P(W | Do(X)), P(Y | Do(W)), P(Y | Do(X), Z), P(Y | Do(X), W))
            d = identify(q, g4; trace=true)
            @test !has_do(last(d))
            @test verify(d, nb4)
        end
    end
end
