@testset "counterfactuals" begin
    @rv X Y W M U
    nobi = Tuple{RV,RV}[]

    @testset "atoms" begin
        y = Y(Do(X(1)))
        @test y isa Counterfactual && !iseventlike(y) && iseventlike(y(0))
        @test sprint(show, y(0)) == "Y_{X=1}=0"
        @test sprint(show, P(y(0) | X(1))) == "P(Y_{X=1}=0 | X=1)"
        @test Y(Do(X(1), W(0))) == Y(Do(W(0), X(1)))
        @test_throws ArgumentError Y(Do(X))
        @test_throws ArgumentError y(0)(1)
        @test latex(P(y(0)); env=:raw) == "P(Y_{X = 1} = 0)"
    end

    backdoor = @dag begin
        W → X → Y
        W → Y
    end
    scm = rand_scm(backdoor; rng=RNG)

    @testset "computing in a structural model" begin
        for x in 1:2, y in 1:2
            @test compute(P(Y(Do(X(x)))(y)), scm) ≈ compute(P(Y(y) | Do(X(x))), scm)
        end
        # consistency: Y_x = Y when X = x
        @test compute(P(Y(Do(X(1)))(2) | X(1)), scm) ≈ compute(P(Y(2) | X(1)), scm)
        @test compute(P(X(Do(X(1)))(1)), scm) ≈ 1
        # the potential outcomes under both treatments, summed, give the factual margin
        @test sum(compute(P(Y(Do(X(1)))(a), Y(Do(X(2)))(b)), scm) for a in 1:2, b in 1:2) ≈ 1
        # a network that is not structural cannot answer counterfactuals
        @test_throws ArgumentError compute(P(Y(Do(X(1)))(1)), rand_bayesnet(backdoor; rng=RNG))
    end

    @testset "effect of treatment on the treated" begin
        g = ADMG(backdoor, nobi)
        for x in 1:2, x′ in 1:2, y in 1:2
            q = P(Y(Do(X(x)))(y) | X(x′))
            e = identify(q, g)
            @test compute(e, scm) ≈ compute(q, scm)
        end
        @test identify(P(Y(Do(X(1)))(1)), g) == identify(P(Y(1) | Do(X(1))), g)
        # with a hidden confounder of X and Y but a front door through M
        full = @dag begin
            U → (X, Y)
            X → M → Y
        end
        fd = latent_projection(full, U)
        for _ in 1:2
            sf = rand_scm(full; rng=RNG)
            q = P(Y(Do(X(1)))(2) | X(2))
            @test compute(identify(q, fd), sf) ≈ compute(q, sf)
        end
        d = identify(P(Y(Do(X(1)))(2) | X(2)), fd; trace=true)
        @test d isa Derivation && occursin("IDC*", d.rules[2])
    end

    @testset "trivial and impossible" begin
        g = ADMG(backdoor, nobi)
        @test identify(P(X(Do(X(1)))(1)), g) == Literal(1)
        @test identify(P(X(Do(X(1)))(2)), g) == Literal(0)
        @test identify(P(Y(Do(X(1)))(1), Y(Do(X(1)))(2)), g) == Literal(0)
    end

    @testset "not identifiable" begin
        gx = ADMG(@dag(X → Y), nobi)
        pns = P(Y(Do(X(1)))(2), Y(Do(X(2)))(1))       # probability of necessity and sufficiency
        @test_throws NotIdentifiable identify(pns, gx)
        @test !isidentifiable(pns, gx)
        bow = @admg begin
            X → Y
            X ↔ Y
        end
        @test_throws NotIdentifiable identify(P(Y(Do(X(1)))(1) | X(2)), bow)
        # with x′ = x it is just P(y | x), by consistency
        e = identify(P(Y(Do(X(1)))(1) | X(1)), bow)
        @test e == P(Y(1) | X(1))
    end
end
