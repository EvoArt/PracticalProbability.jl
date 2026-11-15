@testset "adjustment sets" begin
    @rv X Y Z A B M W

    # every valid set gives P(Y | do(X)) on a random network (with the hidden variables of an ADMG)
    function checks(g, X, Y)
        bn = rand_bayesnet(g isa ADMG ? canonical_dag(g) : g; rng=RNG, exact=true)
        truth = compute(P(Y | Do(X)), bn)
        return Z -> isapprox(compute(adjustment_formula(P(Y | Do(X)), Z), bn), truth)
    end

    conf = @dag begin
        Z → (X, Y)
        X → Y
    end
    @test adjustment_sets(conf, X, Y) == [[Z]]
    @test isadjustment(conf, X, Y, [Z]) && !isadjustment(conf, X, Y, RV[])
    @test optimal_adjustment(conf, X, Y) == [Z]
    @test adjustment_formula(P(Y | Do(X)), Z) == Σ(Z, P(Y | X, Z) * P(Z))
    @test adjustment_formula(P(Y | Do(X)), RV[]) == P(Y | X)
    @test checks(conf, X, Y)([Z])
    @test_throws ArgumentError adjustment_formula(P(Y | Do(X)), RV[]; dag=conf)

    # M-bias: adjusting for the collider M alone opens a path
    mbias = @dag begin
        A → (X, M)
        B → (M, Y)
        X → Y
    end
    @test adjustment_sets(mbias, X, Y) == [RV[]]
    @test length(adjustment_sets(mbias, X, Y; minimal=false)) == 7
    @test !isadjustment(mbias, X, Y, [M]) && isadjustment(mbias, X, Y, [M, A])
    ok = checks(mbias, X, Y)
    @test all(ok, adjustment_sets(mbias, X, Y; minimal=false))
    @test !ok([M])                                   # generically wrong

    # mediators and descendants of the outcome are forbidden; other descendants of X are fine
    med = @dag begin
        Z → (X, Y)
        X → M → Y
        Y → W
        X → B
    end
    @test !isadjustment(med, X, Y, [Z, M]) && !isadjustment(med, X, Y, [Z, W])
    @test isadjustment(med, X, Y, [Z, B])
    @test adjustment_sets(med, X, Y) == [[Z]]
    @test all(checks(med, X, Y), adjustment_sets(med, X, Y; minimal=false))

    # the optimal set keeps precision variables and drops instruments
    prec = @dag begin
        A → X → Y
        B → Y
        Z → (X, Y)
    end
    @test optimal_adjustment(prec, X, Y) == [B, Z]
    @test adjustment_sets(prec, X, Y) == [[Z]]

    # hidden confounding
    fd = @admg begin
        X → M → Y
        X ↔ Y
    end
    @test isempty(adjustment_sets(fd, X, Y))
    @test_throws ArgumentError optimal_adjustment(fd, X, Y)
    obs = @admg begin
        Z → (X, Y)
        X → Y
        Z ↔ Y
    end
    @test adjustment_sets(obs, X, Y) == [[Z]]
    @test checks(obs, X, Y)([Z])
end
