@testset "distributions and transformations" begin
    @rv X::continuous Y::continuous U::continuous W::continuous t::continuous y::continuous
    @rv μ::continuous σ::continuous λ::continuous α::continuous β::continuous p::continuous
    RConst = PracticalProbability.RConst
    RVal = PracticalProbability.RVal
    Value = PracticalProbability.Value
    N(a, b) = Family(:Normal, a, b)
    vals = (μ => 0.7, σ => 1.3, λ => 2.1, α => 2.5, β => 1.7, p => 0.3)
    ev(r) = evaluate(r, vals...)

    @testset "moments" begin
        @test ev(E(N(μ, σ))) ≈ 0.7
        @test rawmoment(N(μ, σ), 2) == expand(μ^2 + σ^2)
        @test ev(rawmoment(N(μ, σ), 3)) ≈ 0.7^3 + 3 * 0.7 * 1.3^2
        @test ev(rawmoment(N(μ, σ), 4)) ≈ 0.7^4 + 6 * 0.7^2 * 1.3^2 + 3 * 1.3^4
        @test ev(centralmoment(N(μ, σ), 4)) ≈ 3 * 1.3^4
        @test ev(centralmoment(N(μ, σ), 3)) ≈ 0 atol = 1e-12
        @test ev(rawmoment(Family(:Exponential, λ), 3)) ≈ 6 / 2.1^3
        @test ev(rawmoment(Family(:Gamma, α, β), 2)) ≈ 2.5 * 3.5 / 1.7^2
        @test ev(rawmoment(Family(:Poisson, λ), 2)) ≈ 2.1 + 2.1^2
        @test ev(rawmoment(Family(:Poisson, λ), 3)) ≈ 2.1^3 + 3 * 2.1^2 + 2.1
        @test ev(Var(Family(:Geometric, p))) ≈ 0.7 / 0.09
        @test ev(rawmoment(Family(:Geometric, p), 2)) ≈ (2 - 0.3) / 0.09
        @test ev(rawmoment(Family(:Uniform, α, β), 2)) ≈ (1.7^3 - 2.5^3) / (3 * (1.7 - 2.5))
        @test ev(rawmoment(Family(:Binomial, 5, p), 2)) ≈ 5 * 0.3 * 0.7 + 25 * 0.09
        @test ev(rawmoment(Family(:Bernoulli, p), 3)) ≈ 0.3
        @test ev(rawmoment(Family(:LogNormal, μ, σ), 2)) ≈ exp(2 * 0.7 + 2 * 1.3^2)
        @test ev(rawmoment(Family(:Beta, α, β), 2)) ≈ 2.5 * 3.5 / (4.2 * 5.2)
        @test ev(rawmoment(Family(:Laplace, μ, σ), 2)) ≈ 0.7^2 + 2 * 1.3^2
        @test ev(cumulant(Family(:Poisson, λ), 4)) ≈ 2.1
        @test ev(cumulant(N(μ, σ), 3)) ≈ 0 atol = 1e-12
        @test ev(Var(Family(:LogNormal, μ, σ))) ≈ (exp(1.3^2) - 1) * exp(2 * 0.7 + 1.3^2)
        @test ev(Var(Family(:Beta, α, β))) ≈ 2.5 * 1.7 / (4.2^2 * 5.2)
        @test_throws ArgumentError E(Family(:Cauchy, 0, 1))
        @test_throws ArgumentError MGF(Family(:Cauchy, 0, 1), t)
    end

    @testset "laws of transformed variables" begin
        L = laws(X => N(1, 2), Y => N(-1, 3))
        @test law(2X + 3, L) == N(5, 4)
        @test law(X + Y, L) == N(0, sqrt(13))
        @test law(X - Y, L) == N(2, sqrt(13))
        @test law(exp(X), L) == Family(:LogNormal, 1, 2)
        @test law(log(exp(X)), L) == N(1, 2)
        @test law(X, L) == N(1, 2)
        Lu = laws(U ~ Family(:Uniform, 0, 1))
        @test law(-2U + 1, Lu) == Family(:Uniform, -1, 1)
        Lp = laws(X => Family(:Poisson, λ), Y => Family(:Poisson, 2))
        @test law(X + Y, Lp) == Family(:Poisson, λ + 2)
        Le = laws(X => Family(:Exponential, λ), Y => Family(:Exponential, λ))
        @test law(X + Y, Le) == Family(:Gamma, 2, λ)
        @test law(min(X, Y), Le) == Family(:Exponential, 2λ)
        @test law(3X, Le) == Family(:Exponential, λ / 3)
        Lz = laws(X => N(0, 1), Y => N(0, 1))
        @test law(X^2 + Y^2, Lz) == Family(:Gamma, 1, 1 // 2)       # χ²₂
        Lb = laws(X => Family(:Bernoulli, p), Y => Family(:Binomial, 4, p))
        @test law(X + Y, Lb) == Family(:Binomial, 5, p)
        Lc = laws(X => Family(:Cauchy, 0, 1), Y => Family(:Cauchy, 0, 1))
        @test law((X + Y) / 2, Lc) == Family(:Cauchy, 0, 1)           # the mean of Cauchys is Cauchy
        Lg = laws(X => Family(:Geometric, p), Y => Family(:Geometric, p))
        @test ev(law(min(X, Y), Lg).params[1]) ≈ 1 - 0.7^2
        @test_throws ArgumentError law(X * Y, L)
        @test_throws ArgumentError law(X^2 + X, L)
        @test_throws ArgumentError law(X + Y, laws(X => Family(:Gamma, 2, 1), Y => Family(:Gamma, 2, 3)))
        @test_throws ArgumentError law(X + Y, laws(X => N(0, 1), Y => N(0, 1); independent=false))
        @test sprint(show, Lu) == "Laws(U ~ Uniform(0.0, 1.0), independent)"
    end

    @testset "generating functions" begin
        @test evaluate(MGF(N(μ, σ), t), vals..., t => 0.4) ≈ exp(0.7 * 0.4 + 1.3^2 * 0.08)
        L = laws(X => N(1, 2), Y => N(-1, 3))
        for s in (-0.3, 0.2, 0.5)
            @test evaluate(MGF(2X + Y + 1, L, t), t => s) ≈ evaluate(MGF(law(2X + Y + 1, L), t), t => s)
        end
        @test MGF(X + Y, L, 0.2).value ≈ exp(0.0 * 0.2 + 13 * 0.04 / 2)
        @test MGF(2X + Y + 1, L, t) isa PracticalProbability.RFun          # one merged exponential
        @test MGF(2X + Y, L, t) == exp(t + 12.5 * t^2)
        @test evaluate(PGF(Family(:Poisson, λ), y), vals..., y => 0.5) ≈ exp(2.1 * (0.5 - 1))
        Lp = laws(X => Family(:Poisson, 1.5), Y => Family(:Poisson, 2.5))
        @test evaluate(PGF(X + Y, Lp, y), y => 0.5) ≈ exp(4 * (0.5 - 1))
        @test evaluate(PGF(Family(:Binomial, 3, p), y), vals..., y => 0.5) ≈ (0.7 + 0.15)^3
        @test evaluate(CF(Family(:Cauchy, 0, 1), t), t => 2.0) ≈ exp(-2)
        @test CF(N(1, 2), 0.5).value ≈ exp(0.5im - 2 * 0.25)
        @test evaluate(CF(X + Y, L, t), t => 0.3) ≈ exp(0.0im - 13 * 0.09 / 2)
    end

    @testset "densities and distribution functions" begin
        @test PDF(N(0, 1), 0).value ≈ 1 / sqrt(2π)
        @test CDF(N(0, 1), 1.96).value ≈ 0.9750021048517795 atol = 1e-9
        @test CDF(N(0, 1), -3.5).value ≈ 2.3262907903552504e-4 rtol = 1e-7
        @test CDF(Family(:Exponential, 2), 1).value ≈ 1 - exp(-2)
        @test CDF(Family(:Cauchy, 0, 1), 1).value ≈ 0.75
        @test PDF(Family(:Poisson, 2.1), 3).value ≈ 2.1^3 * exp(-2.1) / 6
        @test PDF(Family(:Binomial, 5, 0.3), 2).value ≈ 10 * 0.09 * 0.343
        @test PDF(Family(:Geometric, 0.3), 3).value ≈ 0.3 * 0.49
        # the derivative of the CDF is the density
        for f in (N(μ, σ), Family(:Exponential, λ), Family(:Cauchy, μ, σ), Family(:LogNormal, μ, σ))
            @test evaluate(derivative(CDF(f, y), y), vals..., y => 1.1) ≈ evaluate(PDF(f, y), vals..., y => 1.1)
        end
        # densities integrate to one (trapezoid rule)
        for (f, lo, hi) in ((Family(:Gamma, 2.5, 1.7), 0.0, 40.0), (Family(:Beta, 2.5, 1.7), 0.0, 1.0),
                            (Family(:Laplace, 0.5, 1.2), -40.0, 40.0), (N(0.7, 1.3), -15.0, 15.0))
            d = PDF(f, y)
            xs = range(lo, hi; length=20001)
            fx = [evaluate(d, y => x) for x in xs]
            @test sum((fx[1:end-1] .+ fx[2:end]) ./ 2) * step(xs) ≈ 1 atol = 1e-3
        end
    end

    @testset "change of variables" begin
        d = change_of_variables(N(0, 1), X, exp(X), y)
        @test evaluate(d, y => 1.7) ≈ evaluate(PDF(Family(:LogNormal, 0, 1), y), y => 1.7)
        d2 = change_of_variables(Family(:Exponential, λ), X, 3X + 1, y)
        @test evaluate(d2, vals..., y => 2.0) ≈ 2.1 / 3 * exp(-2.1 * (2.0 - 1) / 3)
        d3 = change_of_variables(Family(:Uniform, 0, 1), U, -log(U), y)     # Exponential(1)
        @test evaluate(d3, y => 0.8) ≈ exp(-0.8)
        @test evaluate(jacobian(X^2, X, y), y => 4.0) ≈ 1 / 4
        @test_throws ArgumentError change_of_variables(N(0, 1), X, X^2 + X, y)
    end

    @testset "order statistics" begin
        m = order_statistic(Family(:Exponential, λ), 1, 3, y)       # the minimum of three
        @test evaluate(m, vals..., y => 0.4) ≈ 3 * 2.1 * exp(-3 * 2.1 * 0.4)
        @test m == 3 * λ * exp(-3 * y * λ)                                # exponentials folded and merged


        @test evaluate(order_statistic(Family(:Uniform, 0, 1), 2, 2, y), y => 0.3) ≈ 0.6
        @test evaluate(order_statistic(Family(:Uniform, 0, 1), 2, 3, y), y => 0.3) ≈ 6 * 0.3 * 0.7
    end

    @testset "moments in simplify" begin
        L = laws(X => N(μ, σ), Y => Family(:Poisson, λ))
        @test simplify(E(X), L) == Value(RVal(μ))
        @test simplify(E(X^2 + 2Y), L) == Value(expand(μ^2 + σ^2)) + 2 * Value(RVal(λ))
        @test simplify(Var(X + Y); laws=L) == Value(σ^2) + Value(RVal(λ))
        @test simplify(E(3X + 1), L) == 3 * Value(RVal(μ)) + 1
        s = simplify(E(exp(2X)), L)
        @test s isa Value && evaluate(s.f, vals...) ≈ exp(2 * 0.7 + 2 * 1.3^2)
        @test simplify(E(X * Y), L) == Value(RVal(μ)) * Value(RVal(λ))
        d = simplify(E(X); laws=L, trace=true)
        @test occursin("mean", d.rules[end])
    end
end
