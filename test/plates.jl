@testset "plates" begin
    @rv x y θ
    @index i 1:n
    @index t 1:T

    @testset "indexed variables" begin
        @test i isa Index && i.hi === :n && t.lo == 1
        @test sprint(show, P(y[i] | θ)) == "P(y[i] | θ)"
        @test sprint(show, P(x[t] | x[t-1])) == "P(x[t] | x[t-1])"
        @test sprint(show, P(x[1:t])) == "P(x[1:t])"
        @test sprint(show, P(y[whole(i)])) == "P(y[1:n])"
        @test latex(P(x[t] | x[1:t-1]); env=:raw) == raw"P(x_{t} \mid x_{1:t-1})"
        @test y[3] == RV(:y_3) && x[1:3] == [x[1], x[2], x[3]]
        @test x[t] != x[t-1] && x[t-1] == x[t-1] && (t + 1) - 1 == t
        @index s 1:t                                             # bounds can be indices
        @test s.hi == t
    end

    @testset "sequences" begin
        q = sequence(P(x[whole(t)]), t)
        @test sprint(show, q) == "∏_{t=1}^{T} P(x[t] | x[1:t-1])"
        @test latex(q; env=:raw) == raw"\prod_{t=1}^{T} P(x_{t} \mid x_{1:t-1})"
        @test peel(P(x[1:t])) == P(x[t] | x[1:t-1]) * P(x[1:t-1])
        @test simplify(P(x[t] | x[1:t-1]), assume(markov(x))) == P(x[t] | x[t-1])
        @test simplify(P(y[whole(i)] | θ), assume(iid(y, θ))) == ∏(i, P(y[i] | θ))
        @test simplify(P(y[i] | y[1:i-1], θ), assume(iid(y, θ))) == P(y[i] | θ)

        @test unroll(q, :T => 3) == P(x[1]) * P(x[2] | x[1]) * P(x[3] | x[1], x[2])
        chain = DAG(x[1] => x[2], x[2] => x[3])
        bn = rand_bayesnet(chain; rng=RNG)
        @test verify(P(x[1:3]), unroll(q, :T => 3), bn)
        @test verify(unroll(simplify(P(x[t] | x[1:t-1]), assume(markov(x))), t => 3), P(x[3] | x[1], x[2]), bn)
        @test verify(unroll(peel(P(x[1:t])), t => 3), P(x[1:3]), bn)
        # peeling a product: ∏_{s}^{t} f(s) = [∏^{t-1} f(s)] f(t)
        @index s 2:t
        w = ∏(s, P(x[s] | x[s-1]))
        pw = peel(w)
        @test sprint(show, pw) == "∏_{s=2}^{t-1} P(x[s] | x[s-1]) P(x[t] | x[t-1])"
        @test unroll(pw, t => 3) == unroll(w, t => 3) == P(x[2] | x[1]) * P(x[3] | x[2])
        pq = peel(q)                                             # symbolic end T
        @test sprint(show, pq) == "∏_{t=1}^{T - 1} P(x[t] | x[1:t-1]) P(x[T] | x[1:T-1])"
        @test unroll(pq, :T => 3) == unroll(q, :T => 3)
        @test verify(unroll(pq, :T => 4), P(x[1:4]), rand_bayesnet(DAG(x[1] => x[2], x[2] => x[3], x[3] => x[4]); rng=RNG))
        @test_throws ArgumentError compute(q, bn)
        @test_throws ArgumentError unroll(q)                     # T has no value

        # iid given θ, checked on an unrolled network
        star = DAG(θ => y[1], θ => y[2], θ => y[3])
        sb = rand_bayesnet(star; rng=RNG)
        e = simplify(P(y[whole(i)] | θ), assume(iid(y, θ)))
        @test verify(unroll(e, :n => 3), P(y[1:3] | θ), sb)
    end

    @testset "models with loops" begin
        hier = @generative function hier(J, y, sigma)
            mu ~ Normal(0, 5)
            tau ~ truncated(Cauchy(0, 5); lower=0)
            for j in 1:J
                theta[j] ~ Normal(mu, tau)
                y[j] ~ Normal(theta[j], sigma[j])
            end
        end
        @test sprint(show, posterior(hier)) ==
              "p(mu, tau, theta | y, sigma) ∝ p(mu) p(tau) ∏_{j=1}^{J} p(theta[j] | mu, tau) ∏_{j=1}^{J} p(y[j] | theta[j], sigma[j])"
        @test sprint(show, posterior(hier; plates=false)) ==
              "p(mu, tau, theta | y, sigma) ∝ p(mu) p(tau) p(theta | mu, tau) p(y | theta, sigma)"
        @test occursin(raw"\prod_{j=1}^{J}", latex(hier; env=:raw))
        pl = plates(hier)
        byname = Dict(v.name => [ix.name for ix in ixs] for (v, ixs) in pl)
        @test byname[:theta] == [:j] && byname[:y] == [:j] && byname[:sigma] == [:j]
        @test isempty(byname[:mu]) && isempty(byname[:tau])
        @test only(pl[only(v for v in keys(pl) if v.name === :y)]) == Index(:j, 1, :J)

        ar = @generative function ar(T)
            ρ ~ Uniform(-1, 1)
            x[1] ~ Normal(0, 1)
            for t in 2:T
                x[t] ~ Normal(ρ * x[t-1], 1)
            end
        end
        @test sprint(show, factorize(ar)) == "p(ρ) p(x_1) ∏_{t=2}^{T} p(x[t] | x[t-1], ρ)"
    end
end
