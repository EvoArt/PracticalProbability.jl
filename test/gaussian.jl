@testset "Gaussian algebra" begin
    @rv X::continuous Y::continuous Z::continuous a::continuous b::continuous c::continuous
    @rv μx::continuous μy::continuous sxx::continuous syy::continuous sxy::continuous x0::continuous
    RConst = PracticalProbability.RConst
    RVal = PracticalProbability.RVal
    ωX, ωZ = RV(:ω_X, :continuous), RV(:ω_Z, :continuous)

    @testset "conditioning a bivariate Normal" begin
        G = Gaussian([X, Y], [μx, μy], [sxx sxy; sxy syy])
        cnd = conditional(G, [X], [Y])
        vals = (μx => 1.0, μy => -0.5, sxx => 2.0, syy => 3.0, sxy => 1.2, Y => 0.4)
        @test evaluate(cnd.mean[1], vals...) ≈ 1.0 + 1.2 / 3.0 * (0.4 + 0.5)
        @test evaluate(cnd.cov[1, 1], vals...) ≈ 2.0 - 1.2^2 / 3.0
        @test marginal(G, Y).mean == [RVal(μy)]
        @test law(G, X) == Family(:Normal, μx, sqrt(sxx))
        @test_throws ArgumentError Gaussian([X, Y], [0, 0], [1 2; 3 1])     # not symmetric
    end

    @testset "numeric conditioning is exact" begin
        Σ = [4 2 1; 2 3 1; 1 1 2]
        G = Gaussian([X, Y, Z], [1, 2, 3], Σ)
        cnd = conditional(G, X, [Y => 1, Z => 0])
        @test cnd.mean[1] == RConst(-1 // 5)
        @test cnd.cov[1, 1] == RConst(13 // 5)
    end

    @testset "linear SEMs" begin
        sem = linear_sem(Z => 0, X => a * Z, Y => b * X + c * Z)
        @test dag(sem) isa DAG && parents(dag(sem), Y) == [X, Z]
        @test total_effect(sem, X, Y) == RVal(b)
        @test total_effect(sem, Z, Y) == expand(a * b + c)               # two directed paths
        vars, Σ = implied_covariance(sem)
        @test vars == [Z, X, Y]
        @test Σ[2, 1] == expand(a * ωZ)                                  # Cov(X, Z) = a ω_Z
        # adjusting for the confounder recovers the causal effect exactly
        coefs, icpt = regression(Gaussian(sem), Y, [X, Z])
        @test coefs == [RVal(b), RVal(c)] && icpt == RConst(0)
        # without it the slope is biased: b + a c ω_Z / (a² ω_Z + ω_X)
        β, _ = regression(Gaussian(sem), Y, [X])
        v = (a => 0.8, b => 1.5, c => -0.7, ωX => 0.5, ωZ => 2.0)
        @test evaluate(β[1], v...) ≈ 1.5 + 0.8 * (-0.7) * 2.0 / (0.64 * 2.0 + 0.5)
        # do(X = x₀): E[Y] = b x₀
        Gdo = Gaussian(intervene(sem, X => x0))
        @test Gdo.mean[findfirst(==(Y), Gdo.vars)] == expand(b * x0)
        @test Gdo.cov[findfirst(==(X), Gdo.vars), findfirst(==(X), Gdo.vars)] == RConst(0)
        @test_throws ArgumentError linear_sem(X => a * X)
        @test_throws ArgumentError linear_sem(X => Y * Z, Y => 0, Z => 0)
    end

    @testset "implied covariance against the matrix formula" begin
        sem = linear_sem(Z => 0, X => 0.8Z, Y => 1.5X - 0.7Z; noise=Dict(Z => 2.0, X => 0.5, Y => 1.0))
        vars, Σ = implied_covariance(sem)
        B = [0 0 0; 0.8 0 0; -0.7 1.5 0]        # in the order Z, X, Y
        T = inv([1.0 0 0; 0 1 0; 0 0 1] - B)
        @test [x.value for x in Σ] ≈ T * [2.0 0 0; 0 0.5 0; 0 0 1.0] * T'
    end

    @testset "Kalman filter step" begin
        k = kalman_step(m=0, P=1, a=1, q=1, c=1, r=1, y=2)
        @test k.predicted_var == RConst(2)
        @test k.gain == RConst(2 // 3)
        @test k.mean == RConst(4 // 3) && k.var == RConst(2 // 3)
        @rv m::continuous Pv::continuous q::continuous r::continuous yv::continuous
        ks = kalman_step(m=m, P=Pv, a=a, q=q, c=c, r=r, y=yv)
        v = (m => 0.3, Pv => 1.7, a => 0.9, q => 0.4, c => 1.2, r => 0.6, yv => 1.1)
        Pm = 0.9^2 * 1.7 + 0.4
        K = 1.2 * Pm / (1.2^2 * Pm + 0.6)
        @test evaluate(ks.gain, v...) ≈ K
        @test evaluate(ks.mean, v...) ≈ 0.9 * 0.3 + K * (1.1 - 1.2 * 0.9 * 0.3)
        @test evaluate(ks.var, v...) ≈ (1 - K * 1.2) * Pm
    end
end
