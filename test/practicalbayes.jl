# Included at top level of runtests.jl's testset only when PracticalBayes loads.
# Models are defined with `@eval` at module scope: `@model` defines methods.

@eval Main @graphical PracticalBayes.@model function pp_linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    μ = α .+ β .* x
    y .~ Normal.(μ, σ)
end

@testset "PracticalBayes extension" begin
    f = Main.pp_linreg
    nm(vs) = Set(v.name for v in vs)
    x = collect(range(0, 1; length=10))
    y = 1 .+ 2 .* x

    @test generative(f) isa GenerativeModel
    m = f(x, y)
    @test m isa PracticalBayes.Model
    # the evaluator a Model carries was registered too
    @test generative(m.f) === generative(f)

    gm = generative(m)
    @test nm(gm.observed) == Set([:y]) && nm(gm.latent) == Set([:α, :β, :σ])
    # `missing` data makes y a parameter; conditioning makes it data again
    @test nm(generative(f(x, missing)).latent) == Set([:α, :β, :σ, :y])
    @test nm(generative(PracticalBayes.condition(f(x, missing), (; y=y))).observed) == Set([:y])

    @test sprint(show, posterior(m)) == "p(α, β, σ | y, x) ∝ p(α) p(β) p(σ) p(y | α, β, σ, x)"
    @test occursin(raw"\propto", latex(m; env=:raw))
    @test occursin("observed", tabulate(m))
    @test factorize(generative(m)) == factorize(gm)

    # @graphical leaves the model itself untouched
    @test isfinite(PracticalBayes.logjoint(m, (; α=1.0, β=2.0, σ=0.5)))
end
