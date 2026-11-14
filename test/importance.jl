@testset "importance sampling" begin
    @rv θ Y X1 X2 Y1 Y2
    @density q

    g = @dag θ → Y
    bn = rand_bayesnet(g; card=3, rng=RNG)
    qbn = BayesNet(θ => [0.2, 0.3, 0.5])
    nets = Dict(:P => bn, :q => qbn)

    @testset "constant factors" begin
        e = (P(θ) * P(Y | θ) / P(Y)) / q(θ)
        eq = proportional(e, θ)
        @test eq.relation === :propto
        @test eq.rhs == P(θ) * P(Y | θ) / q(θ)
        @test proportional(P(Y) * P(θ | Y), θ).rhs == P(θ | Y)
    end

    @testset "prior as proposal: likelihood weighting" begin
        iw = importance_weight(P(θ | Y), P(θ); dag=g)
        @test weight(iw) == P(Y | θ)
        @test iw.sampled == [θ] && iw.data == [Y]
        @test verify(iw, bn)
        @test verify(iw.steps, bn)
        r = relative_ess(iw, bn)
        @test r isa Factor && all(x -> 0 < x <= 1 + 1e-12, r.values)
        # with the data fixed to a value the moments are numbers
        iw2 = importance_weight(P(θ | Y(2)), P(θ); dag=g)
        @test weight(iw2) == P(Y(2) | θ)
        @test verify(iw2, bn)
        @test 0 < relative_ess(iw2, bn) <= 1
    end

    @testset "named proposal" begin
        # no graph: definition of conditional probability, then drop P(Y)
        iw = importance_weight(P(θ | Y), q(θ))
        @test weight(iw) == P(θ, Y) / q(θ)
        @test iw.constant == 1 / P(Y)
        @test verify(iw, nets)
        # with the graph: the joint is factorised
        iw = importance_weight(P(θ | Y), q(θ); dag=g)
        @test weight(iw) == P(θ) * P(Y | θ) / q(θ)
        @test verify(iw, nets)
        @test check_support(iw, nets)
        # normalized = true keeps the evidence
        iwn = importance_weight(P(θ | Y), q(θ); dag=g, normalized=true)
        @test iwn.constant == Literal(1)
        @test verify(iwn, nets)
        m1, _ = weight_moments(iwn, nets)        # E_q[π/q] = 1 for each y
        @test all(≈(1), m1.values)
        # p(θ) / q(θ) never cancels across distributions
        @test weight(importance_weight(P(θ), q(θ))) == P(θ) / q(θ)
    end

    @testset "support" begin
        bad = Dict(:P => bn, :q => BayesNet(θ => [0.5, 0.5, 0.0]))
        iw = importance_weight(P(θ | Y), q(θ); dag=g)
        @test !check_support(iw, bad)
        @test !verify(iw, bad)
    end

    @testset "the target as proposal" begin
        iw = importance_weight(P(θ | Y), P(θ | Y); dag=g)
        @test weight(iw) == Literal(1)
        @test all(≈(1), relative_ess(iw, bn).values)
    end

    @testset "sequential Monte Carlo: incremental weights" begin
        h = @dag begin
            X1 → (X2, Y1)
            X2 → Y2
        end
        bnh = rand_bayesnet(h; card=2, rng=RNG)
        target = P(X2 | X1) * P(Y2 | X2)

        boot = importance_weight(target, P(X2 | X1); dag=h, data=[Y2])
        @test weight(boot) == P(Y2 | X2)
        @test boot.sampled == [X2]
        @test verify(boot, bnh)

        opt = importance_weight(target, P(X2 | X1, Y2); dag=h, data=[Y2])
        @test weight(opt) == P(Y2 | X1)               # depends on the previous state only
        @test verify(opt, bnh)
        @test all(≈(1), relative_ess(opt, bnh).values) # no variance given X1
        @test all(relative_ess(boot, bnh).values .<= 1 + 1e-12)

        qg = @dag begin
            X1 → X2
            Y2 → X2
        end
        qn = rand_bayesnet(qg; card=2, rng=RNG)
        arb = importance_weight(target, q(X2 | X1, Y2); dag=h, data=[Y2])
        @test weight(arb) == P(X2 | X1) * P(Y2 | X2) / q(X2 | X1, Y2)
        @test verify(arb, Dict(:P => bnh, :q => qn))
    end

    @testset "output" begin
        iw = importance_weight(P(θ | Y), q(θ); dag=g)
        s = sprint(show, MIME"text/plain"(), iw)
        @test occursin("∝", s) && occursin("collect into one fraction", s)
        @test sprint(show, iw) == "w(θ) ∝ [P(θ) P(Y | θ)] / q(θ)"
        @test occursin(raw"\propto", latex(iw; env=:raw))
        @test Equation(iw).relation === :propto
    end

    @testset "generative models" begin
        gm = @generative function iw_linreg(x, y)
            α ~ Normal(0, 10)
            β ~ Normal(0, 10)
            σ ~ Exponential(1)
            μ = α .+ β .* x
            y .~ Normal.(μ, σ)
        end
        prior_q = @generative function iw_prior()
            α ~ Normal(0, 10)
            β ~ Normal(0, 10)
            σ ~ Exponential(1)
        end
        local_q = @generative function iw_local()
            α ~ Normal(1, 0.1)
            β ~ Normal(2, 0.1)
            σ ~ Exponential(1)
        end
        α, β, σ, x, y = (RV(s, :continuous) for s in (:α, :β, :σ, :x, :y))

        iw = importance_weight(gm, prior_q)
        @test weight(iw) == P(y | (α, β, σ, x))
        @test weight(iw) == likelihood(gm)
        iw2 = importance_weight(gm, local_q)
        @test weight(iw2) == P(α) * P(β) * P(y | (α, β, σ, x)) / (q(α) * q(β))

        @test_throws ArgumentError importance_weight(gm, @generative function iw_partial()
            α ~ Normal(0, 1)
        end)
        @test_throws ArgumentError logweight_expr(importance_weight(P(θ | Y), q(θ)))
        @test logweight_expr(iw) isa Expr

        if HAVE_PB
            M = Core.eval(Main, :(module IWTest
            using Distributions
            end))
            xs = collect(range(0, 1; length=12))
            ys = 1 .+ 2 .* xs .+ 0.3 .* sin.(7 .* xs)
            θv = (α=0.8, β=2.1, σ=0.4)
            loglik = sum(M.logpdf.(M.Normal.(θv.α .+ θv.β .* xs, θv.σ), ys))
            lw = logweight(iw; mod=M)
            @test lw(θv, (x=xs, y=ys)) ≈ loglik
            lw2 = logweight(iw2; mod=M)
            @test lw2(θv, (x=xs, y=ys)) ≈ loglik +
                  M.logpdf(M.Normal(0, 10), θv.α) + M.logpdf(M.Normal(0, 10), θv.β) -
                  M.logpdf(M.Normal(1, 0.1), θv.α) - M.logpdf(M.Normal(2, 0.1), θv.β)
            @test_throws ArgumentError lw(θv, (x=xs,))
            # @logweight compiles in the module it is called from
            mac = Expr(:macrocall, GlobalRef(PracticalProbability, Symbol("@logweight")),
                       LineNumberNode(@__LINE__, Symbol(@__FILE__)), iw2)
            lw3 = Core.eval(M, mac)
            @test lw3(θv, (x=xs, y=ys)) ≈ lw2(θv, (x=xs, y=ys))
        end
    end

    @testset "sequential Monte Carlo on plates" begin
        @rv x y
        @index t 1:T
        @index s 1:t
        γ = ∏(s, P(x[s] | x[s-1]) * P(y[s] | x[s]))       # state-space model, x[0] the initial state
        g = DAG([x[0], x[1], x[2], x[3], y[1], y[2], y[3]],
                Dict(x[1] => [x[0]], x[2] => [x[1]], x[3] => [x[2]],
                     y[1] => [x[1]], y[2] => [x[2]], y[3] => [x[3]]))
        bn = rand_bayesnet(g; card=2, rng=RNG)

        boot = smc_weight(γ, P(x[t] | x[t-1]))
        @test boot isa SMCWeight
        @test weight(boot) == P(y[t] | x[t])
        @test verify(boot, bn; at=1:3)
        @test path_weight(boot) == ∏(s, P(y[s] | x[s]))
        @test all(r -> 0 < r <= 1 + 1e-12, relative_ess(boot, bn; at=2).values)

        opt = smc_weight(γ, P(x[t] | x[t-1], y[t]))
        @test weight(opt) == P(y[t] | x[t-1])
        @test verify(opt, bn; at=1:3)
        @test all(≈(1), relative_ess(opt, bn; at=3).values)

        arb = smc_weight(γ, q(x[t] | x[t-1], y[t]))
        @test weight(arb) == P(x[t] | x[t-1]) * P(y[t] | x[t]) / q(x[t] | x[t-1], y[t])
        qn = rand_bayesnet(DAG([x[1], y[2], x[2]], Dict(x[2] => [x[1], y[2]])); card=2, rng=RNG)
        @test verify(arb, Dict(:P => bn, :q => qn); at=2)

        s_ = sprint(show, MIME"text/plain"(), opt)
        @test occursin("peel the last step", s_) && occursin("path weight", s_)
        @test sprint(show, boot) == "w_t = P(y[t] | x[t])"
        @test occursin(raw"\begin{aligned}", latex(opt; env=:raw))
    end
end
