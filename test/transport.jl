@testset "transportability and selection bias" begin
    @rv X Y Z U W M
    @event S

    @testset "transport" begin
        full = @dag begin
            U → (X, Y)
            Z → Y
            X → Y
        end
        g = latent_projection(full, U)
        e = transport(P(Y | Do(X)), g, [Z])
        @test occursin("π", sprint(show, e))
        @test !isidentifiable(P(Y | Do(X)), g)          # the target alone is not enough
        for _ in 1:3
            src = rand_bayesnet(full; rng=RNG)
            other = rand_bayesnet(full; rng=RNG)
            tgt = BayesNet(src.dag, merge(src.cpts, Dict(Z => other.cpts[Z])))
            @test verify(P(Y | Do(X)), e, Dict(:P => tgt, :π => src))
        end
        # the mechanism of Y differs: no experiment in the source helps
        @test !istransportable(P(Y | Do(X)), g, [Y])
        @test_throws NotTransportable transport(P(Y | Do(X)), g, [Y])
        @test transport(P(Y | Do(X)), full, [Z]; latent=[U]) == e
        # identifiable in the target: target terms only
        bd = @dag begin
            Z → X → Y
            Z → Y
        end
        e2 = transport(P(Y | Do(X)), bd, [Z])
        @test !occursin("π", sprint(show, e2))
        @test verify(P(Y | Do(X)), e2, rand_bayesnet(bd; rng=RNG))
        # nothing differs: the source experiment transfers directly
        bow = @admg begin
            X → Y
            X ↔ Y
        end
        @test transport(P(Y | Do(X)), bow, RV[]; source=:q) == Prob([Y], [Do(X)], :q)
    end

    @testset "selection bias" begin
        g1 = @dag begin
            X → Y
            X → S
        end
        e = recover(P(Y | X), g1, S)
        @test e == P(Y | X, S)
        @test verify(P(Y | X), e, rand_bayesnet(g1; rng=RNG))
        g2 = @dag X → Y → S
        @test !isrecoverable(P(Y | X), g2, S)
        @test_throws NotRecoverable recover(P(Y | X), g2, S)

        g3 = @dag begin
            Z → X → Y
            Z → Y
            Z → S
        end
        @test !isrecoverable(P(Y | Do(X)), g3, S)
        e3 = recover(P(Y | Do(X)), g3, S; external=[Z])
        @test verify(P(Y | Do(X)), e3, rand_bayesnet(g3; rng=RNG))

        g4 = @dag begin
            Z → X → Y
            Z → Y
            W → S
        end
        e4 = recover(P(Y | Do(X)), g4, S)
        @test occursin("S", sprint(show, e4))
        @test verify(P(Y | Do(X)), e4, rand_bayesnet(g4; rng=RNG))

        g5 = @dag begin
            Z → Y
            Z → S
            X → Y
        end
        @test !isrecoverable(P(Y | X), g5, S)
        e5 = recover(P(Y | X), g5, S; external=[X, Z])
        @test verify(P(Y | X), e5, rand_bayesnet(g5; rng=RNG))
        @test verify(P(Y(1) | X(2)), recover(P(Y(1) | X(2)), g5, S; external=[X, Z]), rand_bayesnet(g5; rng=RNG))
        @test_throws ArgumentError recover(P(Y | X), g5, Z)
    end

    @testset "surrogate experiments and several sources" begin
        @rv U1 U2 A B C D
        g = @admg begin
            Z → X → Y
            Z ↔ Y
            X ↔ Z
        end
        full = @dag begin
            U1 → (Z, Y)
            U2 → (X, Z)
            Z → X → Y
        end
        @test latent_projection(full, [U1, U2]) == g
        @test !isidentifiable(P(Y | Do(X)), g)
        e = zidentify(P(Y | Do(X)), g, [Z])
        @test occursin("do(Z)", sprint(show, e))
        for _ in 1:3
            @test verify(P(Y | Do(X)), e, rand_bayesnet(full; rng=RNG))
        end
        @test iszidentifiable(P(Y | Do(X)), full, [Z]; latent=[U1, U2])
        bow = @admg begin
            X → Y
            X ↔ Y
        end
        @test !iszidentifiable(P(Y | Do(X)), bow, RV[])
        @test zidentify(P(Y | Do(X)), bow, [X]) == P(Y | Do(X))

        # transport through the new interface: all experiments in the source (sID)
        gt = latent_projection(@dag(begin
            U → (X, Y)
            Z → Y
            X → Y
        end), U)
        fullt = @dag begin
            U → (X, Y)
            Z → Y
            X → Y
        end
        e2 = transport(P(Y | Do(X)), gt; sources=[Domain(:π; differs=[Z], experiments=:all)])
        # and with an experiment on X only
        e3 = transport(P(Y | Do(X)), gt; sources=[Domain(:π; differs=[Z], experiments=[[X]])])
        for _ in 1:3
            src = rand_bayesnet(fullt; rng=RNG)
            other = rand_bayesnet(fullt; rng=RNG)
            tgt = BayesNet(src.dag, merge(src.cpts, Dict(Z => other.cpts[Z])))
            nets = Dict(:P => tgt, :π => src)
            @test verify(P(Y | Do(X)), e2, nets)
            @test verify(P(Y | Do(X)), e3, nets)
        end
        # a difference in Y's own mechanism cannot be transported
        @test_throws NotTransportable transport(P(Y | Do(X)), gt;
                                                sources=[Domain(:π; differs=[Y], experiments=:all)])
        # two sources: the first differs where it matters, the second does not
        e4 = transport(P(Y | Do(X)), gt; sources=[Domain(:π₁; differs=[Y], experiments=:all),
                                                  Domain(:π₂; differs=[Z], experiments=[[X]])])
        @test occursin("π₂", sprint(show, e4)) && !occursin("π₁", sprint(show, e4))

        # soundness on random models: whenever an answer comes back it is right
        rng = Xoshiro(7)
        obs = [A, B, C, D]
        answered = 0
        for trial in 1:40
            es = Pair{RV,RV}[]
            for i in 1:4, j in i+1:4
                rand(rng) < 0.5 && push!(es, obs[i] => obs[j])
            end
            for u in (U1, U2)
                a, b = sortperm(rand(rng, 4))[1:2]
                push!(es, u => obs[a], u => obs[b])
            end
            fg = DAG(es...; nodes=vcat(obs, [U1, U2]))
            ag = latent_projection(fg, [U1, U2])
            exps = [RV[v] for v in obs[2:3] if rand(rng) < 0.6]
            differs = RV[v for v in obs if rand(rng) < 0.3]
            q = P(D | Do(A))
            r = try
                transport(q, ag; experiments=exps, sources=[Domain(:π; differs, experiments=:all)])
            catch err
                err isa NotTransportable || rethrow()
                nothing
            end
            r === nothing && continue
            answered += 1
            src = rand_bayesnet(fg; rng)
            other = rand_bayesnet(fg; rng)
            tgt = BayesNet(src.dag, merge(src.cpts, Dict(v => other.cpts[v] for v in differs)))
            @test verify(q, r, Dict(:P => tgt, :π => src))
        end
        @test answered >= 10
    end

    @testset "recovery through identification" begin
        # front door with selection on the treatment
        g = @dag begin
            U → (X, Y)
            X → M → Y
            X → S
        end
        @test !isrecoverable(P(Y | Do(X)), g, S; latent=[U])
        e = recover(P(Y | Do(X)), g, S; external=[X], latent=[U])
        @test occursin("S", sprint(show, e))
        for _ in 1:3
            @test verify(P(Y | Do(X)), e, rand_bayesnet(g; rng=RNG))
        end
    end

    @testset "selection bias: complete criteria (Bareinboim–Tian–Pearl 2014, Bareinboim–Tian 2015)" begin
        @rv W W1 W2 U2
        # P(y | x) without external data: Y ⟂ S | X is necessary and sufficient
        g = @dag begin
            W → X → Y
            W → Y
            W → S
        end
        err = try
            recover(P(Y | X), g, S)
        catch e
            e
        end
        @test err isa NotRecoverable && occursin("Theorem 1", sprint(showerror, err))
        # Theorems 2–3: external data on X and the separator W
        e = recover(P(Y | X), g, S; external=[X, W])
        @test verify(P(Y | X), e, rand_bayesnet(g; rng=RNG))
        # extra external variables are pruned to a minimal separator
        g2 = @dag begin
            W → X → Y
            W → Y
            W → S
            M → X
        end
        e2 = recover(P(Y | X), g2, S; external=[X, W, M])
        @test !occursin("M", sprint(show, e2))
        @test verify(P(Y | X), e2, rand_bayesnet(g2; rng=RNG))
        # Corollaries 3–4: external data on Y and W, not on X
        e3 = recover(P(Y | X), g, S; external=[Y, W])
        @test verify(P(Y | X), e3, rand_bayesnet(g; rng=RNG))
        # Corollary 5: external data on W only, through the joint
        e4 = recover(P(Y | X), g, S; external=[W])
        @test verify(P(Y | X), e4, rand_bayesnet(g; rng=RNG))
        # biased data that do not cover the separator
        @test !isrecoverable(P(Y | X), g, S; external=[X, W], measured=[X, Y])

        # causal effects without hidden variables: complete (Bareinboim–Tian Theorem 2)
        gm = @dag begin
            X → Z → Y
            W → (Z, Y)
            X → S
        end
        e5 = recover(P(Y | Do(X)), gm, S)
        for _ in 1:3
            @test verify(P(Y | Do(X)), e5, rand_bayesnet(gm; rng=RNG))
        end
        gbad = @dag begin
            W → X → Y
            W → Z → Y
            W → S
        end
        err = try
            recover(P(Y | Do(X)), gbad, S)
        catch e
            e
        end
        @test err isa NotRecoverable && occursin("complete", sprint(showerror, err))
        @test isrecoverable(P(Y | Do(X)), gbad, S; external=[W])      # s-backdoor with P(w) external

        # hidden variables: RC recovers what no adjustment can (BT 2015, Fig. 3(a))
        gr = @dag begin
            W1 → W2 → X → Y
            U2 → (W2, Y)
            W1 → S
        end
        er = recover(P(Y | Do(X)), gr, S; latent=[U2])
        for _ in 1:3
            @test verify(P(Y | Do(X)), er, rand_bayesnet(gr; rng=RNG))
        end
        # every c-component holding Y also holds an ancestor of S: RC fails
        gf = @dag begin
            Z → X → Y
            U2 → (Z, Y)
            Y → S
        end
        @test isidentifiable(P(Y | Do(X)), gf; latent=[U2])
        @test !isrecoverable(P(Y | Do(X)), gf, S; latent=[U2])
    end
end
