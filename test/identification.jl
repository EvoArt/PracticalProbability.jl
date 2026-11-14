@testset "identification with hidden confounders" begin
    @rv X Y M Z W1 W2 U U1 U2

    frontdoor = @admg begin
        X → M → Y
        X ↔ Y
    end
    full = @dag begin
        U → (X, Y)
        X → M → Y
    end

    @testset "ADMGs" begin
        @test bidirected(frontdoor) == [(X, Y)]
        @test siblings(frontdoor, X) == [Y]
        @test districts(frontdoor) == [[X, Y], [M]]
        @test latent_projection(full, U) == frontdoor
        @test length(nodes(canonical_dag(frontdoor))) == 4
        @test !d_separated(frontdoor, X, Y, M)
        collider = @admg begin
            X → Z
            Z ↔ Y
        end
        @test d_separated(collider, X, Y)
        @test !d_separated(collider, X, Y, Z)
        @test holds(X ⟂ Y, collider)
        @test sprint(show, frontdoor) == "ADMG(3 nodes, 2 directed, 1 bidirected edges)"
        # projection keeps directed paths through latent variables
        chain = @dag U → X → U1 → Y
        @test edges(latent_projection(chain, [U, U1])) == [X => Y]
    end

    @testset "do-calculus on an ADMG" begin
        @test simplify(P(M | Do(X)), frontdoor) == P(M | X)
        @test simplify(P(Y | Do(M), X), frontdoor) == P(Y | M, X)
    end

    @testset "front door" begin
        e = identify(P(Y | Do(X)), frontdoor)
        @test !has_do(e)
        @test occursin("X′", sprint(show, e))          # the summed copy of X is primed
        bn = rand_bayesnet(full; rng=RNG)
        @test verify(P(Y | Do(X)), e, bn)
        @test identify(P(Y | Do(X)), full; latent=[U]) == e
        e1 = identify(P(Y(1) | Do(X(2))), frontdoor)
        @test compute(e1, bn) ≈ compute(P(Y(1) | Do(X(2))), bn)
        d = identify(P(Y | Do(X)), frontdoor; trace=true)
        @test d isa Derivation && verify(d, bn)
    end

    @testset "napkin" begin
        napkin_full = @dag begin
            W1 → W2 → X → Y
            U1 → (W1, X)
            U2 → (W1, Y)
        end
        napkin = latent_projection(napkin_full, [U1, U2])
        @test issetequal(bidirected(napkin), [(W1, X), (W1, Y)])
        e = identify(P(Y | Do(X)), napkin)
        @test !has_do(e)
        for _ in 1:3
            @test verify(P(Y | Do(X)), e, rand_bayesnet(napkin_full; rng=RNG))
        end
    end

    @testset "backdoor without hidden variables" begin
        g = @dag begin
            Z → X → Y
            Z → Y
        end
        e = identify(P(Y | Do(X)), ADMG(g, Tuple{RV,RV}[]))
        @test verify(P(Y | Do(X)), e, rand_bayesnet(g; rng=RNG))
    end

    @testset "not identifiable" begin
        bow = @admg begin
            X → Y
            X ↔ Y
        end
        @test_throws NotIdentifiable identify(P(Y | Do(X)), bow)
        @test !isidentifiable(P(Y | Do(X)), bow)
        @test hedge(P(Y | Do(X)), bow) == ([X, Y], [Y])
        @test hedge(P(M | Do(X)), frontdoor) === nothing
        g = @admg begin
            X → Z → Y
            X ↔ Z
        end
        @test !isidentifiable(P(Y | Do(X)), g)
        @test isidentifiable(P(Y | Do(Z)), g)
        err = try
            identify(P(Y | Do(X)), bow)
        catch e
            e
        end
        @test occursin("hedge", sprint(showerror, err))
    end

    @testset "IDC: conditional effects" begin
        g = @admg begin
            X → Z → Y
            X ↔ Y
        end
        gfull = @dag begin
            U → (X, Y)
            X → Z → Y
        end
        e = identify(P(Y | Do(X), Z), g)
        @test !has_do(e)
        bn = rand_bayesnet(gfull; rng=RNG)
        @test verify(P(Y | Do(X), Z), e, bn)
        # a covariate that stays conditioned on
        g2 = @dag begin
            U → (X, Y)
            W1 → X → M → Y
            W1 → Y
        end
        e2 = identify(P(Y | Do(X), W1), g2; latent=[U])
        @test verify(P(Y | Do(X), W1), e2, rand_bayesnet(g2; rng=RNG))
    end
end
