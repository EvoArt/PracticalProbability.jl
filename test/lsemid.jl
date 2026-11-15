@testset "identification in linear SEMs" begin
    @rv X Y Z W M A B C
    RVal = PracticalProbability.RVal

    # random parameters on the edges of g; Σ; and a check that every identified
    # coefficient formula gives back its coefficient
    function recovers(g, res; trials=3)
        vs = nodes(g)
        idx = Dict(v => i for (i, v) in enumerate(vs))
        G = g isa ADMG ? g : ADMG(g, Tuple{RV,RV}[])
        ok = true
        for _ in 1:trials
            n = length(vs)
            Λ = zeros(n, n)
            for v in vs, p in parents(G, v)
                Λ[idx[p], idx[v]] = 0.5 + rand(RNG)
            end
            Ω = Matrix{Float64}(2I_(n))
            for (a, b) in bidirected(G)
                Ω[idx[a], idx[b]] = Ω[idx[b], idx[a]] = 0.3 + 0.4rand(RNG)
            end
            Σ = PracticalProbability.implied_covariance(g, Λ, Ω)
            vals = covariance_values(vs, Σ)
            for ((p, v), f) in res.coefficients
                ok &= isapprox(evaluate(f, vals...), Λ[idx[p], idx[v]]; rtol=1e-8)
            end
        end
        return ok
    end
    I_(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

    iv = @admg begin
        Z → X → Y
        X ↔ Y
    end
    @testset "instrumental variables" begin
        @test instruments(iv, X, Y) == [Z]
        @test isinstrument(iv, Z, X, Y) && !isinstrument(iv, X, X, Y)
        f = iv_estimand(iv, Z, X, Y)
        @test f == simplify(RVal(covariance_symbol(Y, Z)) / RVal(covariance_symbol(X, Z)))
        @test recovers(iv, (coefficients=Dict((X, Y) => f),))
        @test covariance_symbol(X, Z) == covariance_symbol(Z, X)
        # a conditional instrument
        civ = @admg begin
            W → (Z, Y)
            Z → X → Y
            X ↔ Y
        end
        @test !isinstrument(civ, Z, X, Y) && isinstrument(civ, Z, X, Y; given=[W])
        @test recovers(civ, (coefficients=Dict((X, Y) => iv_estimand(civ, Z, X, Y; given=[W])),))
        @test_throws ArgumentError iv_estimand(civ, Z, X, Y)
    end

    @testset "half-trek criterion" begin
        r = htc_identify(iv)
        @test r.identifiable && length(r.coefficients) == 2
        @test recovers(iv, r)
        bow = @admg begin
            X → Y
            X ↔ Y
        end
        rb = htc_identify(bow)
        @test !rb.identifiable && rb.unsolved == [Y]
        fd = @admg begin
            X → M → Y
            X ↔ Y
        end
        rf = htc_identify(fd)
        @test rf.identifiable && recovers(fd, rf)
        # a bigger graph with two confounded pairs

        big = @admg begin
            A → B → C → D
            A → C
            A ↔ D
            B ↔ D
        end
        rbig = htc_identify(big)
        @test rbig.identifiable && recovers(big, rbig)
        dag = @dag begin
            A → (B, C)
            B → C
        end
        rd = htc_identify(dag)
        @test rd.identifiable && recovers(dag, rd)
        @test occursin("generically identifiable", sprint(show, MIME"text/plain"(), rd))
        @test occursin("undetermined", sprint(show, MIME"text/plain"(), rb))
    end
end
