@testset "generative models" begin
    gm = @generative function linreg(x, y)
        α ~ Normal(0, 10)
        β ~ Normal(0, 10)
        σ ~ Exponential(1)
        μ = α .+ β .* x
        y .~ Normal.(μ, σ)
    end
    @test gm isa GenerativeModel
    nm(vs) = Set(v.name for v in vs)
    @test nm(gm.latent) == Set([:α, :β, :σ])
    @test nm(gm.observed) == Set([:y])
    @test nm(gm.inputs) == Set([:x])
    y = only(gm.observed)
    # μ is deterministic: y depends on what μ is computed from
    @test nm(parents(dag(gm), y)) == Set([:α, :β, :σ, :x])
    @test all(iscontinuous, gm.latent)

    eq = posterior(gm)
    @test eq.relation === :propto
    @test nm(PracticalProbability.freevars(eq.lhs)) == Set([:α, :β, :σ, :y, :x])
    @test length(eq.rhs.factors) == 4
    @test sprint(show, eq) == "p(α, β, σ | y, x) ∝ p(α) p(β) p(σ) p(y | α, β, σ, x)"
    @test posterior(gm; normalized=true).rhs isa Quotient
    @test likelihood(gm) == eq.rhs.factors[end]
    @test prior(gm) == eq.rhs.factors[1] * eq.rhs.factors[2] * eq.rhs.factors[3]

    s = latex(gm; env=:raw)
    @test occursin(raw"\alpha &\sim \mathrm{Normal}\left( 0, 10 \right)", s)
    @test occursin(raw"\mu &=", s)
    @test occursin(raw"\propto", s)
    @test occursin("observed", tabulate(gm))

    # loops, indexed statements, discrete distributions, observe/unobserve
    hier = @generative function hier(J, y, sigma)
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5); lower=0)
        for j in 1:J
            theta[j] ~ Normal(mu, tau)
            y[j] ~ Normal(theta[j], sigma[j])
        end
    end
    @test nm(hier.latent) == Set([:mu, :tau, :theta])
    @test nm(hier.inputs) == Set([:sigma])
    th = only(v for v in hier.latent if v.name === :theta)
    @test nm(parents(dag(hier), th)) == Set([:mu, :tau])
    @test occursin(raw"j = 1, \ldots, J", latex(hier; env=:raw))
    @test nm(unobserve(hier, :y).latent) == Set([:mu, :tau, :theta, :y])
    yv = only(hier.observed)
    @test nm(parents(dag(hier), yv)) == Set([:theta, :sigma])

    # a container allocated first and then filled by `~` is a random variable,
    # not a function of its size (PracticalBayes style)
    hier2 = @generative function hier2(J, y, sigma)
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5); lower=0)
        theta = Vector{paramtype(__mode__)}(undef, J)
        for j in 1:J
            theta[j] ~ Normal(mu, tau)
            y[j] ~ Normal(theta[j], sigma[j])
        end
    end
    y2 = only(hier2.observed)
    @test nm(parents(dag(hier2), y2)) == Set([:theta, :sigma])
    @test nm(hier2.inputs) == Set([:sigma])
    s2 = latex(hier2; env=:raw)               # renders, and leaves the allocation out
    @test !occursin("paramtype", s2) && occursin(raw"_{j} &\sim", s2)
    @test occursin(raw"\theta_{j}", s2) && !occursin(raw"\theta\left(", s2)
    @test PracticalProbability._lxcode(:(Vector{Float64}(undef, n))) isa String

    coin = @generative function coin(n)
        p ~ Beta(1, 1)
        k ~ Binomial(n, p)
    end
    @test isdiscrete(only(v for v in coin.latent if v.name === :k))
    @test nm(observe(coin, :k).observed) == Set([:k])
    @test sprint(show, posterior(observe(coin, :k)).lhs) == "p(p | k, n)"

    # AR(1): a variable that depends on its own earlier elements is one node
    ar = @generative function ar(T)
        ρ ~ Uniform(-1, 1)
        x[1] ~ Normal(0, 1)
        for t in 2:T
            x[t] ~ Normal(ρ * x[t-1], 1)
        end
    end
    xr = only(v for v in ar.latent if v.name === :x)
    @test nm(parents(dag(ar), xr)) == Set([:ρ])

    @test_throws ErrorException PracticalProbability.generative(:(function f(x)
        return x
    end))
end
