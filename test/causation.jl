@testset "probabilities of causation, nested counterfactuals, mediation" begin
    @rv X Y W M Z U U_Y
    nobi = Tuple{RV,RV}[]

    @testset "PNS, PN, PS bounds" begin
        q = pns(X, Y, 1, 2, 1, 2)
        @test q == P(Y(Do(X(1)))(1), Y(Do(X(2)))(2))
        b = bounds(q)
        @test b isa CFBounds && length(b.lower) == 4 && length(b.upper) == 4
        @test occursin("≤", sprint(show, b))
        g = @dag begin
            W → X → Y
            W → Y
        end
        for _ in 1:6
            scm = rand_scm(g; rng=RNG)
            for query in (pns(X, Y, 1, 2, 1, 2), pn(X, Y, 1, 2, 1, 2), ps(X, Y, 1, 2, 1, 2))
                lo, hi = compute(bounds(query), scm)
                truth = compute(query, scm)
                @test lo - 1e-10 <= truth <= hi + 1e-10
            end
            # with the graph, the experimental terms are identified
            bg = bounds(q; dag=g)
            @test compute(bg, scm)[1] ≈ compute(b, scm)[1] atol = 1e-12
            @test compute(bg, scm)[2] ≈ compute(b, scm)[2] atol = 1e-12
        end
    end

    @testset "monotone outcome: point values" begin
        # u = 1: never, u = 2: Y = 1 exactly when X = 1, u = 3: always
        T = zeros(2, 2, 3)
        T[2, :, 1] .= 1
        T[1, 1, 2] = 1; T[2, 2, 2] = 1
        T[1, :, 3] .= 1
        scm = BayesNet(X => [0.4, 0.6], U_Y => [0.2, 0.5, 0.3], (Y | (X, U_Y)) => T)
        for query in (pns(X, Y, 1, 2, 1, 2), pn(X, Y, 1, 2, 1, 2), ps(X, Y, 1, 2, 1, 2))
            @test compute(bounds(query; monotone=true), scm) ≈ compute(query, scm)
        end
        @test compute(pns(X, Y, 1, 2, 1, 2), scm) ≈ 0.5
        @test_throws ArgumentError bounds(P(Y(1) | X(1)))
    end

    @testset "free values" begin
        g = @dag begin
            W → X → Y
            W → Y
        end
        scm = rand_scm(g; rng=RNG)
        q = P(Y(Do(X(1))), X)                       # a function of y and x
        @test compute(q, scm) isa Factor
        e = identify(q, g)
        @test issetequal(freevars(e), [X, Y])
        @test verify(q, e, scm)
        q2 = P(Y(Do(X(1))) | X)                     # treatment on the treated, for every x
        @test compute(q2, scm) isa Factor
        e2 = identify(q2, g)
        @test verify(q2, e2, scm)
        # both potential outcomes together: not identifiable (like the PNS)
        @test !isidentifiable(P(Y(Do(X(1))), Y(Do(X(2)))), ADMG(g, nobi))
    end

    @testset "nested counterfactuals" begin
        y = Y(Do(X(1)), M(Do(X(2))))
        @test sprint(show, y(1)) == "Y_{M_{X=2},X=1}=1"
        med = @dag begin
            X → M → Y
            X → Y
        end
        scm = rand_scm(med; rng=RNG)
        # composition: with the mediator at its natural value the nesting does nothing
        @test compute(P(Y(Do(X(1)), M(Do(X(1))))(1)), scm) ≈ compute(P(Y(Do(X(1)))(1)), scm)
        e = identify(P(y(1)), med)
        @test compute(e, scm) ≈ compute(P(y(1)), scm)
        # the mediation formula
        @test verify(e, Σ(M, P(Y(1) | X(1), M) * P(M | X(2))), scm)
        # confounded mediator and outcome: the natural direct effect is not identifiable
        conf = @admg begin
            X → M → Y
            X → Y
            M ↔ Y
        end
        @test_throws NotIdentifiable identify(P(y(1)), conf)
        # the recanting witness: Z on both paths from X
        rw = @dag begin
            X → Z → M → Y
            Z → Y
            X → Y
        end
        @test !isidentifiable(P(y(1)), ADMG(rw, nobi))
    end

    @testset "mediation" begin
        med = @dag begin
            W → X → M → Y
            W → (M, Y)
            X → Y
        end
        m = mediation(Y(1), X, M, 1, 2)
        @test keys(m) == (:total, :direct, :indirect)
        for _ in 1:3
            scm = rand_scm(med; rng=RNG)
            for k in keys(m)
                @test compute(identify(m[k], med), scm) ≈ compute(m[k], scm) atol = 1e-12
            end
        end
    end
end
