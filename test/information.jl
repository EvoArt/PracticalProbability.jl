@testset "information theory" begin
    @rv X Y Z W
    @event A
    @density q
    g = @dag X → Y → Z
    bn = rand_bayesnet(g; card=3, rng=RNG)
    qbn = rand_bayesnet(g; card=3, rng=RNG)
    nets = Dict(:P => bn, :q => qbn)

    @testset "building and display" begin
        @test sprint(show, H(X, Y | Z)) == "H(X, Y | Z)"
        @test sprint(show, MI(X, Y | Z)) == "I(X; Y | Z)"
        @test sprint(show, MI((X, W), Y)) == "I(X, W; Y)"
        @test sprint(show, H(X; name=:q)) == "H_q(X)"
        @test sprint(show, KL(P(X | Z), q(X | Z))) == "KL(P(X | Z) ‖ q(X | Z))"
        @test sprint(show, chisq(P(X), q(X))) == "χ²(P(X) ‖ q(X))"
        @test sprint(show, renyi(P(X), q(X), 2)) == "D_2(P(X) ‖ q(X))"
        @test sprint(show, crossentropy(P(X), q(X))) == "H(P(X), q(X))"
        @test sprint(show, log(P(X))) == "log(P(X))"
        @test MI(X, Y) == MI(Y, X) && H(X, Y) == H(Y, X) && ℍ(X) == H(X)
        @test renyi(P(X), q(X), 1) == KL(P(X), q(X))
        @test latex(MI(X, Y | Z); env=:raw) == raw"I\left(X; Y \mid Z\right)"
        @test latex(H(X); env=:raw) == raw"H\left(X\right)"
        @test latex(KL(P(X), q(X)); env=:raw) ==
              raw"D_{\mathrm{KL}}\left(P(X) \,\middle\|\, q(X)\right)"
        @test_throws ArgumentError H(X(1))
        @test_throws ArgumentError KL(P(X), q(Y))
        @test information(X ⟂ Y | Z) == MI(X, Y | Z)
        @test (MI(X, Y) >= 0) isa Inequality
        @test sprint(show, MI(X, Z) <= MI(X, Y)) == "I(X; Z) ≤ I(X; Y)"
    end

    @testset "numbers" begin
        px = compute(P(X), bn)
        @test compute(H(X), bn) ≈ -sum(p * log(p) for p in px.values)
        for e in (H(X), H(Y | X), H(X, Y | Z), MI(X, Z), MI(X, Z | Y), MI((X, Y), Z), H(Z | Y(1)))
            @test compute(e, bn) ≈ compute(definition(e), bn) atol = 1e-12
        end
        @test compute(MI(X, Z | Y), bn) ≈ 0 atol = 1e-12       # X → Y → Z
        @test compute(MI(X, Y), bn) > 0
        @test verify(MI(X, Y | Z), entropies(MI(X, Y | Z)), bn)
        @test verify(MI(X, Y | Z), entropies(MI(X, Y | Z); joint=true), bn)
        @test verify(H(X | Y), entropies(H(X | Y); joint=true), bn)
        # chain rules
        @test verify(H(X, Y, Z), chainrule(H(X, Y, Z)), bn)
        h4 = H(X, Y, Z | W(1))
        @test verify(chainrule(h4; order=[Z, X, Y]), h4, rand_bayesnet(@dag(W → X → Y → Z); rng=RNG))
        @test verify(MI((X, Y), Z), chainrule(MI((X, Y), Z)), bn)
        # an interventional entropy is a function of the intervened value
        @test compute(H(Z | Do(X)), bn) isa Factor
    end

    @testset "divergences" begin
        k = compute(KL(P(X, Y), q(X, Y)), nets)
        @test k > 0
        @test k ≈ compute(definition(KL(P(X, Y), q(X, Y))), nets)
        @test compute(KL(P(X, Y), P(X, Y)), nets) ≈ 0 atol = 1e-12
        # pointwise in the conditioning, averaged with E(⋯)
        @test compute(KL(P(Y | X), q(Y | X)), nets) isa Factor
        @test compute(E(KL(P(Y | X), q(Y | X))), nets) isa Real
        @test verify(KL(P(X, Y), q(X, Y)), chainrule(KL(P(X, Y), q(X, Y))), nets)
        @test verify(KL(P(X, Y), q(X, Y)), chainrule(KL(P(X, Y), q(X, Y)); order=[Y, X]), nets)
        # cross-entropy = H + KL
        @test compute(crossentropy(P(X), q(X)), nets) ≈ compute(H(X), nets) + compute(KL(P(X), q(X)), nets)
        @test compute(crossentropy(P(X), q(X)), nets) ≈ compute(definition(crossentropy(P(X), q(X))), nets)
        # Rényi: D_2 = log(1 + χ²), D_α → KL as α → 1
        c = compute(chisq(P(X), q(X)), nets)
        @test c ≈ compute(definition(chisq(P(X), q(X))), nets)
        @test compute(renyi(P(X), q(X), 2), nets) ≈ log(1 + c)
        @test compute(renyi(P(X), q(X), 3), nets) ≈ compute(definition(renyi(P(X), q(X), 3)), nets)
        @test compute(renyi(P(X), q(X), 1.0001), nets) ≈ compute(KL(P(X), q(X)), nets) atol = 1e-3

        # importance sampling: relative ESS = 1 / (1 + χ²(target ‖ proposal))
        @rv θ D
        gi = @dag θ → D
        bi = rand_bayesnet(gi; card=3, rng=RNG)
        iw = importance_weight(P(θ | D(2)), P(θ); dag=gi)
        @test relative_ess(iw, bi) ≈ 1 / (1 + compute(chisq(P(θ | D(2)), P(θ)), bi))
    end

    @testset "rules" begin
        @test simplify(H(X | X, Z)) == Literal(0)
        @test simplify(H(X, Y | X)) == H(Y | X)
        @test simplify(MI(X, X | Z)) == H(X | Z)
        @test simplify(MI(X, Y | Y)) == Literal(0)
        @test simplify(MI(X, Z | Y), g) == Literal(0)
        @test simplify(H(Z | Y, X), g) == H(Z | Y)
        @test simplify(MI(X, Y), assume(X ⟂ Y)) == Literal(0)
        gi = @dag begin
            X → Y
            W
        end
        @test simplify(H(X, W), gi) == H(X) + H(W)
        @test simplify(MI(X, Y | W), gi) == MI(X, Y)
        # the chain rules backwards inside sums
        @test simplify(H(X) - H(X | Y)) == MI(X, Y)
        @test simplify(H(X, Y | Z) - H(Y | Z)) == H(X | Y, Z)
        @test simplify(H(X) + H(Y | X)) == H(X, Y)
        @test simplify(chainrule(H(X, Y, Z))) == H(X, Y, Z)
        @test simplify(H(X) + H(Y) - H(X, Y)) == MI(X, Y)
        @test simplify(entropies(MI(X, Y | Z))) == MI(X, Y | Z)
        @test simplify(H(X | Z) - MI(X, Y | Z)) == H(X | Y, Z)
        @test simplify(MI(X, Y) + MI(X, Z | Y)) == MI(X, (Y, Z))
        @test simplify(2 * H(X) - 2 * H(X | Y)) == 2 * MI(X, Y)
        @test simplify(KL(P(X), P(X))) == Literal(0)
        @test simplify(crossentropy(P(X), P(X))) == H(X)
        @test simplify(log(P(X) * P(Y))) == log(P(X)) + log(P(Y))
        @test simplify(log(P(X) / P(Y))) == log(P(X)) - log(P(Y))
        d = simplify(H(X) + H(Y) - H(X, Y); trace=true)
        @test verify(d, bn)
    end

    @testset "data processing" begin
        ineq = data_processing(g, X, Y, Z)
        @test ineq.lhs == MI(X, Z) && ineq.rhs == MI(X, Y)
        @test verify(ineq, bn)
        @test_throws ArgumentError data_processing(g, X, Z, Y)
        @test verify(MI(X, Y | Z) >= 0, bn)
        @test !verify(H(X) <= MI(X, Z), bn)
    end

    @testset "variational inference" begin
        @rv y z
        gm = @dag z → y
        pbn = rand_bayesnet(gm; card=3, rng=RNG)
        qbn2 = BayesNet(y => [0.2, 0.3, 0.5], (z | y) => [0.1 0.6 0.3; 0.2 0.3 0.4; 0.7 0.1 0.3])
        vn = Dict(:P => pbn, :q => qbn2)
        d = evidence_decomposition(P(y, z), q(z | y))
        @test first(d) == log(P(y))
        @test last(d) == elbo(P(y, z), q(z | y)) + KL(q(z | y), P(z | y))
        @test verify(d, vn)
        @test verify(elbo(P(y, z), q(z | y)) <= log(P(y)), vn)
        # with q the exact posterior the bound is tight
        exact = Dict(:P => pbn, :q => pbn)
        @test verify(elbo(P(y, z), q(z | y)), log(P(y)), exact)
    end

    @testset "closed forms" begin
        @rv μ::continuous σ::continuous
        k = KL(Family(:Normal, 0, 1), Family(:Normal, 1, 2))
        @test evaluate(k) ≈ log(2) + (1 + 1) / 8 - 0.5
        @test evaluate(KL(Family(:Normal, μ, 1), Family(:Normal, 0, 1)), μ => 2.0) ≈ 2.0
        @test evaluate(KL(Family(:Normal, μ, σ), Family(:Normal, μ, σ)), μ => 0.3, σ => 1.7) ≈ 0 atol = 1e-12
        @test evaluate(H(Family(:Normal, 0, σ)), σ => 2.0) ≈ 0.5 * log(2π * ℯ * 4)
        @test evaluate(KL(Family(:Exponential, 2), Family(:Exponential, 1))) ≈ log(2) + 0.5 - 1
        @test evaluate(H(Family(:Exponential, 2))) ≈ 1 - log(2)
        # Poisson against a truncated sum
        λ1, λ2 = 2.0, 3.5
        pmf(λ, k) = exp(-λ + k * log(λ) - sum(log, 1:k; init=0.0))
        @test evaluate(KL(Family(:Poisson, λ1), Family(:Poisson, λ2))) ≈
              sum(pmf(λ1, k) * log(pmf(λ1, k) / pmf(λ2, k)) for k in 0:60)
        # Bernoulli against the exact computation on events
        ebn = Dict(:P => BayesNet(A => 0.3), :q => BayesNet(A => 0.6))
        @test evaluate(KL(Family(:Bernoulli, 0.3), Family(:Bernoulli, 0.6))) ≈ compute(KL(P(A), q(A)), ebn)
        @test evaluate(H(Family(:Bernoulli, 0.3))) ≈ compute(H(A), ebn)
        @test_throws ArgumentError H(Family(:Poisson, 1))
        @test_throws ArgumentError Family(:Cauchy, 0, 1)
        @test sprint(show, Family(:Normal, μ, σ)) == "Normal(μ, σ)"

        # special functions
        lg = PracticalProbability._loggamma
        dg = PracticalProbability._digamma
        @test lg(0.5) ≈ 0.5 * log(π)
        @test lg(10) ≈ log(362880)
        @test lg(0.1) ≈ 2.252712651734206
        @test dg(1) ≈ -0.5772156649015329
        @test dg(0.5) ≈ -0.5772156649015329 - 2 * log(2)
        @test dg(7.3) ≈ (lg(7.3 + 1e-6) - lg(7.3 - 1e-6)) / 2e-6 atol = 1e-6

        # continuous families against numerical integration (Simpson's rule)
        function simpson(f, a, b; n=20_000)
            h = (b - a) / n
            s = f(a) + f(b)
            for i in 1:n-1
                s += (isodd(i) ? 4 : 2) * f(a + i * h)
            end
            return s * h / 3
        end
        xlogy(p, q) = p == 0 ? 0.0 : p * log(q)
        Hnum(pdf, a, b) = -simpson(x -> xlogy(pdf(x), pdf(x)), a, b)
        KLnum(p, q, a, b) = simpson(x -> xlogy(p(x), p(x)) - xlogy(p(x), q(x)), a, b)
        gpdf(α, β) = x -> x <= 0 ? 0.0 : exp(α * log(β) - lg(α) + (α - 1) * log(x) - β * x)
        bpdf(a, b) = x -> (x <= 0 || x >= 1) ? 0.0 : exp((a - 1) * log(x) + (b - 1) * log(1 - x) - (lg(a) + lg(b) - lg(a + b)))
        lpdf(m, s) = x -> exp(-abs(x - m) / s) / (2s)
        @test evaluate(H(Family(:Gamma, 2.5, 1.5))) ≈ Hnum(gpdf(2.5, 1.5), 0, 60) atol = 1e-6
        @test evaluate(KL(Family(:Gamma, 2.5, 1.5), Family(:Gamma, 3.0, 0.7))) ≈
              KLnum(gpdf(2.5, 1.5), gpdf(3.0, 0.7), 0, 80) atol = 1e-6
        @test evaluate(H(Family(:Beta, 2.0, 3.5))) ≈ Hnum(bpdf(2.0, 3.5), 0, 1) atol = 1e-6
        @test evaluate(KL(Family(:Beta, 2.0, 3.5), Family(:Beta, 4.0, 1.5))) ≈
              KLnum(bpdf(2.0, 3.5), bpdf(4.0, 1.5), 0, 1) atol = 1e-5
        @test evaluate(H(Family(:Laplace, 0.3, 1.7))) ≈ Hnum(lpdf(0.3, 1.7), -60, 60) atol = 1e-6
        # (split at the two kinks, where Simpson's rule loses its accuracy)
        @test evaluate(KL(Family(:Laplace, 0.3, 1.7), Family(:Laplace, -1.0, 2.5))) ≈
              sum(KLnum(lpdf(0.3, 1.7), lpdf(-1.0, 2.5), a, b) for (a, b) in ((-80, -1), (-1, 0.3), (0.3, 80))) atol = 1e-7
        @test evaluate(KL(Family(:Uniform, 0, 1), Family(:Uniform, -1, 3))) ≈ log(4)
        @test evaluate(H(Family(:Uniform, 2, 5))) ≈ log(3)
        # Dirichlet with two components is a Beta
        @test evaluate(H(Family(:Dirichlet, [2.0, 3.5]))) ≈ evaluate(H(Family(:Beta, 2.0, 3.5)))
        @test evaluate(KL(Family(:Dirichlet, [2.0, 3.5]), Family(:Dirichlet, [4.0, 1.5]))) ≈
              evaluate(KL(Family(:Beta, 2.0, 3.5), Family(:Beta, 4.0, 1.5)))
        # Geometric against a truncated sum
        gp(p, k) = (1 - p)^(k - 1) * p
        @test evaluate(KL(Family(:Geometric, 0.3), Family(:Geometric, 0.6))) ≈
              sum(gp(0.3, k) * log(gp(0.3, k) / gp(0.6, k)) for k in 1:400)
        @test evaluate(H(Family(:Geometric, 0.3))) ≈ -sum(gp(0.3, k) * log(gp(0.3, k)) for k in 1:400)
        # multivariate Normal: diagonal case is a sum of univariate ones
        @test evaluate(KL(Family(:MvNormal, [0.0, 1.0], [1.0 0; 0 4.0]), Family(:MvNormal, [1.0, 0.0], [2.0 0; 0 1.0]))) ≈
              evaluate(KL(Family(:Normal, 0, 1), Family(:Normal, 1, sqrt(2)))) + evaluate(KL(Family(:Normal, 1, 2), Family(:Normal, 0, 1)))
        @test evaluate(H(Family(:MvNormal, [0.0, 0.0], [1.0 0; 0 4.0]))) ≈
              evaluate(H(Family(:Normal, 0, 1))) + evaluate(H(Family(:Normal, 0, 2)))
        # symbolic parameters: the special functions appear by name
        @rv α::continuous β::continuous
        hg = H(Family(:Gamma, α, β))
        @test occursin("loggamma(α)", sprint(show, hg)) && occursin("digamma(α)", sprint(show, hg))
        @test evaluate(hg, α => 2.5, β => 1.5) ≈ evaluate(H(Family(:Gamma, 2.5, 1.5)))
        @test occursin("\\psi", latex(hg; env=:raw))
    end

    @testset "sequences" begin
        @index t 1:n
        te = transfer_entropy(X, Y, t)
        @test te isa MutualInformation
        @test sprint(show, te) == "I(Y[t]; X[t-1] | Y[t-1])"
        @test length(transfer_entropy(X, Y, t; lag=2).X) == 1
        di = directed_information(X, Y, 2)
        x1, x2, y1, y2 = X[1], X[2], Y[1], Y[2]
        gd = DAG(x1 => y1, x1 => x2, y1 => x2, x2 => y2, y1 => y2)
        bd = rand_bayesnet(gd; rng=RNG)
        @test compute(di, bd) ≥ 0
        @test compute(di, bd) ≈ compute(MI(x1, y1) + MI((x1, x2), y2 | y1), bd)
        # directed information is at most the mutual information
        @test compute(di, bd) <= compute(MI((x1, x2), (y1, y2)), bd) + 1e-12
    end
end
