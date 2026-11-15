# PracticalProbability

Computer algebra for probability, in Julia. Write probability expressions as you
would on paper: `P(A, B | C)`, `P(A ∪ Bᶜ)`, `π(x, y | D)`, `P(Y | Do(X))`.
Transform them with the chain rule, Bayes' rule, marginalisation, complements,
De Morgan and inclusion–exclusion, simplify them with a causal graph
(d-separation, do-calculus), and check every step numerically on a Bayesian
network. Results print as LaTeX or as PrettyTables tables. `@graphical` reads a
[PracticalBayes](https://github.com/EvoArt/PracticalBayes.jl) (or Turing)
`@model` into the same algebra.

This is a work in progress, part of the PracticallyBayesian family of packages.

## Markov-chain and HMM diagrams

```julia
using PracticalProbability
import PracticalBayesPlot as PBP

mc = MarkovChain([3//4 1//4; 1//3 2//3]; states=["Sun", "Rain"])
markovplot(mc)                  # weighted arrows, including self-transitions
markovplot(mc; minprob=0.3)     # hide small edges without renormalising

@rv X Y
hmmplot(X, Y, 5)               # hidden X₁ → … → X₅; emissions below
hmmplot(X, Y, 5; observed=[true, true, false, true, true])
PBP.save("hmm.svg", hmmplot(X, Y, 5))  # also .tex and .typ
```

Markov arrows follow the row-to-column convention of `transition_matrix(mc)`.
Labels preserve rational probabilities; absorbing states are shaded. A custom
`positions` vector follows the chain's state order. `markov_view(mc)` exposes
the layout and original transition weights.

HMM diagrams accept base variables and a step count, or explicit hidden and
emission variable vectors. `hmmplot(bn, hidden, emissions)` also checks that a
`BayesNet` has exactly the supplied HMM structure. `observed` is a Boolean or
mask controlling emission shading; these diagrams do not perform inference.
`hmm_view` returns the same `DAGView` used by `dagplot`.

Both diagrams support the current `plotbackend()` (`:pbp` or `:makie`),
`figure`/`axis` options, and `markovplot!`/`hmmplot!` for existing axes.
Runnable examples are in [examples/markov-plots.jl](examples/markov-plots.jl).

## Symbolic probability

```julia
using PracticalProbability

@rv A B C                      # discrete random variables; `θ::continuous` for densities

bayes(P(A | B))                # [P(B | A) P(A)] / P(B)
bayes(P(A | B); expand=true)   # [P(B | A) P(A)] / [∑_A P(B | A) P(A)]
chainrule(P(A, B, C))          # P(A | B, C) P(B | C) P(C)

simplify(bayes(P(A | B)); trace=true)
```

```
Derivation with 2 steps
┌───┬────────────────────────┬───────────────────────────────────────┐
│ # │ expression             │ justification                         │
├───┼────────────────────────┼───────────────────────────────────────┤
│ 0 │ [P(B | A) P(A)] / P(B) │                                       │
│ 1 │ = P(B, A) / P(B)       │ chain rule: P(B | A) P(A) = P(B, A)   │
│ 2 │ = P(A | B)             │ definition of conditional probability │
└───┴────────────────────────┴───────────────────────────────────────┘
```

## Events

Named events combine with `∪`, `∩` and complement (`A'`, `!A`; shown `Aᶜ`), with
`Ω` and `∅` for the sure and impossible events. `X(1)` is the event `X = 1` of an
ordinary variable, and combines the same way.

```julia
@event A B C

P(A ∪ B | C)
P(A | B ∪ C)                       # the union is the condition: P(A | (B ∪ C))
inclusion_exclusion(P(A ∪ B))      # P(A) + P(B) - P(A ∩ B)
complement_rule(P(A | C))          # 1 - P(Aᶜ | C)
demorgan(P((A ∪ B)'))              # P(Aᶜ ∩ Bᶜ)
total_probability(P(C), A)         # P(C | A) P(A) + P(C | Aᶜ) P(Aᶜ)
bayes(P(A | C); expand=true)       # P(C | A) P(A) / [P(C | A) P(A) + P(C | Aᶜ) P(Aᶜ)]
simplify(P(A ∩ B) + P(A ∩ B'))     # P(A)
simplify(1 - P(A))                 # P(Aᶜ)
```

Sums and differences of terms are ordinary expressions (`+`, `-`), with like
terms collected.

### Venn diagrams

`venn` draws an expression about events as Venn diagrams, one panel per
probability term (see [Diagrams](#diagrams) for how figures are drawn). In
`P(E | C)` the cells of `E ∩ C` are filled, the rest of `C` is tinted, and the
sample space outside `C` is greyed out, so conditioning visibly shrinks the
sample space. Sums, products,
fractions, equations and derivations keep their shape, with the operators
between the panels:

```julia
@rv X
venn(P(A | B))                                            # one panel
venn(Equation(P(A ∪ B), inclusion_exclusion(P(A ∪ B))))  # P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
venn(Equation(P(A | B), bayes(P(A | B))))                 # Bayes' rule as a fraction of panels
venn(Equation(P(C), total_probability(P(C), A)))
venn(P(X(1) ∪ X(2)))                 # X = 1 and X = 2 never overlap: that cell is hatched

vennplot!(ax, P(A ∩ B' | C); highlightcolor = :orange)    # into your own Axis
```

Only events have Venn diagrams, and up to three per figure. `P(X)` for a plain
variable, `do(⋅)` and sums over variables raise an `ArgumentError`.
`venn_layout(x)` gives the highlighted cells and positions without drawing
anything.

With numbers, the cells show their probabilities and each panel its value.
They can come from facts (anything `deduce` takes; cells the facts leave open
show `?`) or from draws, such as posterior samples, with events defined by
conditions:

```julia
venn(Equation(P(A ∪ B), inclusion_exclusion(P(A ∪ B)));
     facts = (P(A) => 0.3, P(B) => 0.5, P(A ∩ B) => 0.1))

se = @sampleevents chain begin       # any Tables.jl table, NamedTuple or Dict of draws
    Wide = σ > 0.2
    Pos  = μ > 0
end
venn(P(Wide | Pos); samples = se)    # circles labelled σ > 0.2 and μ > 0
compute(P(Wide | Pos), se)           # the share of draws with μ > 0 that have σ > 0.2
```

## Assumptions and deduction

State facts without a graph: independence (`A ⟂ B`, `A ⟂ B | C`),
`disjoint(A, B)`, `implies(A, B)` (A ⊆ B), and known values as pairs.
`simplify` uses the structural facts; `deduce` also uses the numbers, and
answers the kind of question found in exercises:

```julia
@event A B C

simplify(P(A ∩ B), assume(A ⟂ B))                 # P(A) P(B)
simplify(P(A ∪ B | C), assume(disjoint(A, B)))    # P(A | C) + P(B | C)
simplify(P(B | A), assume(implies(A, B)))         # 1

deduce(P(A ∩ B), P(A) => 0.3, P(B) => 0.5, P(A ∪ B) => 0.6)    # P(A ∩ B) = 0.2
deduce(P(A ∪ B), P(A) => 0.3, P(B) => 0.4, A ⟂ B)              # P(A ∪ B) = 0.58
deduce(P(A | B), P(B | A) => 0.9, P(A) => 0.01, P(B | A') => 0.05)
deduce(P(A ∩ B), P(A) => 0.3, P(B) => 0.5)       # … not determined: between 0 and 0.3
```

`deduce` works on the 2ⁿ cells of the events: known values are linear
equations on the cell probabilities, and independence adds product equations
as soon as enough is known. Inconsistent facts are an error.
`cell_probabilities(facts...)` gives every cell's probability (or `nothing`).

## Expectations

`E`, `Var` and `Cov` of random quantities built with ordinary arithmetic
(`X + 2Y`, `X * Y`, `X^2`, `exp(X)`), indicators `𝟙(A)`, and other
expectations:

```julia
@rv X Y Z

simplify(E(X + 2Y + 3))            # E[X] + 2 E[Y] + 3
simplify(E(X * Z | Z))             # Z E[X | Z]      (taking out what is known)
simplify(E(E(X | Y, Z) | Z))       # E[X | Z]        (tower property)
simplify(Var(2X + 1))              # 4 Var(X)
simplify(Var(X + Y), g)            # Var(X) + Var(Y) when g makes X ⟂ Y
definition(Var(X))                 # E[X²] - E[X]²
total_variance(Var(Y), X)          # E[Var(Y | X)] + Var(E[Y | X])
compute(Var(Y), bn)                # exact, for numeric values
```

### Algebra of random quantities

Random quantities have a canonical polynomial form, derivatives, and lowest
terms by polynomial division:

```julia
@rv a b c
expand((X + Y)^2)                  # X² + 2X Y + Y²
simplify((a*b + a*c) / a)          # b + c
derivative(log(X^2 + 1), X)        # 2 (X² + 1)⁻¹ X
substitute(exp(X * Y), Y => 0)     # 1
canonical(P(B, A | D, C))          # P(A, B | C, D): equal expressions print the same
simplify(e; search = true)         # search over rule orders for the smallest form
```

## Distributions, moments and generating functions

`laws` says how variables are distributed (independent unless stated
otherwise). Then `law` gives the distribution of a transformed variable,
`MGF`/`PGF`/`CF` its generating functions, and `simplify` fills in moments:

```julia
@rv X::continuous Y::continuous t::continuous y::continuous μ::continuous σ::continuous λ::continuous
L = laws(X => Family(:Normal, 1, 2), Y => Family(:Normal, -1, 3))
law(2X + Y + 3, L)                        # Normal(4.0, 5.0)
law(exp(X), L)                            # LogNormal(1.0, 2.0)
law(X^2 + Y^2, laws(X => Family(:Normal, 0, 1), Y => Family(:Normal, 0, 1)))   # Gamma(1, 0.5) = χ²₂
law(min(X, Y), laws(X => Family(:Exponential, λ), Y => Family(:Exponential, 2)))  # Exponential(λ + 2.0)

MGF(Family(:Normal, μ, σ), t)             # exp(μ t + 1/2 σ² t²)
MGF(2X + Y, L, t)                         # M_X(2t) M_Y(t) = exp(t + 12.5 t²)
CF(Family(:Cauchy, 0, 1), t)              # exp(-|t|)
rawmoment(Family(:Normal, μ, σ), 4)       # μ⁴ + 6μ² σ² + 3σ⁴, from the MGF
cumulant(Family(:Poisson, λ), 3)          # λ

simplify(E(X^2 + 3Y), laws(X => Family(:Normal, μ, σ), Y => Family(:Poisson, λ)))   # (μ² + σ²) + 3λ

PDF(Family(:Gamma, 2, 1), y)              # densities and CDFs as random quantities
CDF(Family(:Normal, 0, 1), 1.96)          # 0.975 (Φ)
change_of_variables(Family(:Normal, 0, 1), X, exp(X), y)   # the LogNormal density
order_statistic(Family(:Exponential, λ), 1, 3, y)        # the minimum of three: 3 λ exp(-3y λ)


```

## Gaussians and linear SEMs

Jointly Normal variables with symbolic means and covariances condition by the
Schur complement. A linear-Gaussian SEM gives its covariances by path tracing
and shows confounding bias exactly:

```julia
@rv X::continuous Y::continuous Z::continuous a::continuous b::continuous c::continuous
sem = linear_sem(Z => 0, X => a * Z, Y => b * X + c * Z)
total_effect(sem, X, Y)                   # b
total_effect(sem, Z, Y)                   # a b + c: two directed paths
regression(Gaussian(sem), Y, [X])         # slope b + a c ω_Z / (a² ω_Z + ω_X): biased
regression(Gaussian(sem), Y, [X, Z])      # slopes b and c: the back-door adjustment
Gaussian(intervene(sem, X => 1))          # the law under do(X = 1)
conditional(Gaussian([X, Y], [0, 0], [1 0.5; 0.5 1]), X, [Y => 2])
kalman_step(m = 0, P = 1, a = 1, q = 1, c = 1, r = 1, y = 2)   # gain 2/3, mean 4/3
```

## Plates and sequences

Indexed variables and products over an index, for models (`∏ᵢ p(yᵢ | θ)`) and
sequences (`x[t] | x[t-1]`):

```julia
@rv x y θ
@index i 1:n
@index t 1:T

sequence(P(x[whole(t)]), t)                         # ∏_{t=1}^{T} P(x[t] | x[1:t-1])
peel(P(x[1:t]))                                    # P(x[t] | x[1:t-1]) P(x[1:t-1])
simplify(P(x[t] | x[1:t-1]), assume(markov(x)))    # P(x[t] | x[t-1])
simplify(P(y[whole(i)] | θ), assume(iid(y, θ)))     # ∏_{i=1}^{n} P(y[i] | θ)
unroll(sequence(P(x[whole(t)]), t), :T => 3)        # P(x_1) P(x_2 | x_1) P(x_3 | x_1, x_2)
```

`y[3]` with a number is the ordinary variable `y_3`, so an `unroll`ed
expression can be checked with `compute`/`verify` on a network over
`y[1], y[2], …`. Models with loops are written with plates:

```julia
posterior(hier)
# p(mu, tau, theta | y, sigma) ∝ p(mu) p(tau) ∏_{j=1}^{J} p(theta[j] | mu, tau)
#                                ∏_{j=1}^{J} p(y[j] | theta[j], sigma[j])
```

## Named distributions

A distribution's name is part of a term: `π(x, y | D)` is the joint
distribution `π`, and `q(z)` is not `p(z)`. The rules only combine terms of the
same distribution, so a ratio like an importance weight `p(x) / q(x)` stays
put.

```julia
@density π q
@rv x y D

π(x, y | D)                        # π(x, y | D)
chainrule(π(x, y | D))             # π(x | y, D) π(y | D)
simplify(P(x) / q(x))              # P(x) / q(x): not 1
posterior(model; density = :π)     # π(θ | y) ∝ π(y | θ) π(θ)
```

## Importance weights

`importance_weight(target, proposal)` derives `w = target / proposal`. It
writes each term in its distribution's factorisation, cancels what the target
and proposal share, and drops factors that do not vary across particles. Every
step carries its justification.

```julia
@rv θ Y X1 X2 Y2
@density q
g = @dag θ → Y
importance_weight(P(θ | Y), P(θ); dag = g)   # w(θ) ∝ P(Y | θ)
importance_weight(P(θ | Y), q(θ); dag = g)   # w(θ) ∝ [P(θ) P(Y | θ)] / q(θ)

h = @dag X1 → X2 → Y2                        # one step of a particle filter
target = P(X2 | X1) * P(Y2 | X2)
importance_weight(target, P(X2 | X1); dag = h, data = [Y2])       # P(Y2 | X2)
importance_weight(target, P(X2 | X1, Y2); dag = h, data = [Y2])   # P(Y2 | X1)
```

`verify(iw, nets)` checks `q ⋅ w = π`, every step and the support exactly on
discrete networks. `relative_ess(iw, nets)` gives `E_q[w]² / E_q[w²]`.
Between two generative models, identical `~` statements cancel, and
`logweight(iw)` (or `@logweight iw` in a notebook or module) compiles the
remaining factors to `logw(θ, data)`.

For sequential Monte Carlo, write the path target on a plate. `smc_weight`
peels off the last step, so the past cancels and one time slice remains:

```julia
@rv x y
@index t 1:T
@index s 1:t
γ = ∏(s, P(x[s] | x[s-1]) * P(y[s] | x[s]))
smc_weight(γ, P(x[t] | x[t-1]))              # w_t = P(y[t] | x[t])
smc_weight(γ, P(x[t] | x[t-1], y[t]))        # w_t = P(y[t] | x[t-1])
```

`path_weight` gives `∏ₛ w_s`, and `verify(sw, nets; at = 1:3)` checks the
derivation at concrete times. See
[examples/importance.qmd](examples/importance.qmd).

## Graphs, independence and the do-calculus

```julia
g = @dag A → B → C
d_separated(g, A, C, B)          # true
holds(A ⟂ C | B, g)              # true    (⟂ is \perp<tab>)
independencies(g)                # a table of every implied independence
simplify(P(C | A, B), g)         # P(C | B)    "C ⟂ A | B (d-separation)"
expand(P(A | C), g)              # P(A) ∑_B P(B | A) P(C | B) / P(C)

g = @dag begin
    Z → (X, Y)
    X → Y
end
identify(P(Y | Do(X)), g)        # ∑_Z P(Z) P(Y | X, Z)    (back-door adjustment)
```

`identify` applies do-calculus rules 1–3 first and falls back to the truncated
factorisation (g-formula). That is complete when every variable is observed.
Graphs with latent variables are on the [roadmap](ROADMAP.md).

A graph describes one distribution. `simplify(expr, g)` applies `g` to the
default `P` only; give other distributions their own graphs with
`simplify(expr; dag = Dict(:P => g, :q => gq))`. A proposal `q(y | x, z)` is
then never simplified with `p`'s independencies.

### Adjustment, testable implications and bounds

```julia
@rv X Y Z A B M
g = @dag begin
    Z → (X, Y)
    X → Y
    A → X
    B → Y
end
adjustment_sets(g, X, Y)                 # [[Z]]: the minimal valid sets
isadjustment(g, X, Y, [Z, A])            # true (generalised adjustment criterion)
optimal_adjustment(g, X, Y)              # [B, Z]: the smallest-variance valid set
adjustment_formula(P(Y | Do(X)), [Z])    # ∑_Z P(Y | X, Z) P(Z)

markov_basis(g)                          # independences that imply all the others
cpdag(g); markov_equivalent(g, h); equivalence_class(g)
verma = @admg begin
    A → B → Z → Y
    B ↔ Y
end
verma_constraints(verma)                 # ∑_B P(Y | A, B, Z) P(B | A) does not depend on A

iv = @admg begin
    Z → X → Y
    X ↔ Y
end
lp_bounds(P(Y(1) | Do(X(1))), iv, bn)    # sharp instrumental-variable bounds (Balke–Pearl)
htc_identify(iv)                         # linear SEM: λ[X→Y] = σ[Y,Z] / σ[X,Z] (half-trek criterion)
iv_estimand(iv, Z, X, Y)                 # the same through the instrument
```

## Numbers


`compute` evaluates any expression exactly against a discrete Bayesian network.
Events of any shape go through their indicator functions, and `do(⋅)` is
handled by graph surgery. `verify` checks an identity, or every line of a
derivation.

```julia
@rv X Y
bn = BayesNet(
    X => [0.3, 0.7],
    (Y | X) => [0.9 0.2; 0.1 0.8];          # rows: Y, columns: X
    domains = Dict(X => [:no, :yes]),
)
compute(P(X(:yes) | Y(1)), bn)               # 0.3414…
compute(P(X | Y), bn)                        # a Factor, printed as a table
verify(identify(P(Y | Do(X)), g; trace=true), rand_bayesnet(g))   # true

# events: a table can give just the probability that the event happens
@event Disease Positive
test = BayesNet(Disease => 0.01, (Positive | Disease) => [0.95, 0.05])
compute(P(Disease | Positive), test)         # 0.161…

# one network per named distribution
compute(Σ(X, q(X) * P(X) / q(X)), Dict(:P => bn_p, :q => bn_q))   # 1.0
```

(It is `compute`, not `evaluate`, because PracticalBayes exports an `evaluate`.)

### Exact answers

Tables of integers or rationals stay exact through `compute`, and `deduce`
solves in rational arithmetic when the known values are rational:

```julia
@event H D
bn = BayesNet(H => 1//100, (D | H) => [9//10, 1//20])
compute(P(H | D), bn)                        # 2//13
deduce(P(H | D), P(D | H) => 9//10, P(H) => 1//100, P(D | H') => 1//20)   # P(H | D) = 2/13
rand_bayesnet(g; exact = true)               # identities checked with ==, not ≈
```

### Odds

Bayes' rule in odds form needs no normalising constant, and evidence adds on
the log scale:

```julia
bayes_odds(P(H | D))                  # [P(D | H) / P(D | Hᶜ)] [P(H) / P(Hᶜ)]
compute(odds(P(H | D)), bn)           # 2//11
likelihood_ratio(D, H)                # P(D | H) / P(D | Hᶜ)
log_odds(P(H | D))                    # log prior odds + weight_of_evidence(D, H)
probability_from_odds(2//11)          # 2//13
```

## Experiments

Describe a chance experiment and get its Bayesian network, with exact
probabilities:

```julia
@rv D1 D2 S Car Pick Host C1 C2
@event Doubles
dice = experiment(D1 => die(), D2 => die(), S => derived(+, D1, D2),
                  Doubles => derived(==, D1, D2))
compute(P(S(7)), dice)                       # 1//6
compute(P(Doubles | S(8)), dice)             # 1//5

monty = experiment(Car => uniform(1:3), Pick => uniform(1:3),
                   Host => depends_on((c, p) -> uniform(setdiff(1:3, [c, p])), Car, Pick))
compute(P(Car(2) | Pick(1), Host(3)), monty) # 2//3: switch

balls = experiment(draws(urn(:red => 3, :blue => 2), C1, C2))   # without replacement
compute(P(C1(:red) | C2(:blue)), balls)      # 3//4
experiment(draws(deck(), C1, C2))            # cards are (rank, suit)
```

## Markov chains

Finite chains, solved exactly:

```julia
weather = MarkovChain([1//2 1//2; 1//4 3//4]; states = [:sun, :rain])
stationary(weather)                    # [1//3, 2//3]
mean_return_time(weather, :sun)        # 3
communicating_classes(chain)           # states, recurrent or transient, period
hitting_probability(ruin, 4)           # gambler's ruin: [0, 1//4, 1//2, 3//4, 1]
hitting_time(ruin, [0, 4])             # [0, 3, 4, 3, 0]
bn = BayesNet(weather, W, 3; init = [1, 0])
compute(P(W[3](:rain) | W[1](:sun)), bn)   # 5//8
```


## Output

- `latex(x)` gives a `LaTeXString` for expressions, equations, derivations (as
  an `aligned` block with a justification per line) and generative models.
  Everything also has a `text/latex` display, so it renders in Jupyter, Pluto
  and VS Code, and `latexify` works too.
- `tabulate(x; backend = :text | :markdown | :html | :latex)` renders factors,
  CPTs, Bayesian networks, derivations, independence lists and generative
  models with PrettyTables. A `Factor` is also a Tables.jl table.

## Diagrams

Diagrams are drawn with
[PracticalBayesPlot](https://github.com/EvoArt/PracticalBayesPlot.jl), a
small plotting package with no dependencies, so they need nothing else
loaded. Figures display inline as SVG (VS Code, Jupyter, Pluto, Quarto), and
`PracticalBayesPlot.save("fig.tex", fig)` writes a TikZ picture whose labels
LaTeX typesets as mathematics. With a Makie backend loaded (`using
CairoMakie`), the same functions return Makie figures instead;
`plotbackend!(:pbp)` or `plotbackend!(:makie)` chooses explicitly. Besides
`venn`:

```julia
dagplot(g)                          # a DAG, in layers
dagplot(g, X ⟂ Y | Z)               # every path between X and Y: open (red) or blocked
                                    # (dashed), the blocking node ringed, and the verdict
dagplot(g, P(Y | Do(X)))            # back-door paths and the adjustment set identify uses,
                                    # next to the graph with the edges into X cut
dagplot(model)                      # a @graphical/@generative model in plate notation

probtree(P(Disease | Positive); bn = test)   # a probability tree, leaves coloured like
                                             # Venn panels, P(E | C) worked out from them
mosaic(P(Disease | Positive); bn = test)     # the same, area-proportional
probtree([A, B]; facts = (P(A) => 0.3, P(B | A) => 0.9, P(B | A') => 0.2))
mosaic(P(Wide | Pos); samples = se)

factorplot(compute(P(Y | X), bn))            # bars, grouped bars or (kind = :heatmap) a heatmap
barplot(compute(P(X), bn))                   # with Makie: plain barplot works on one-variable factors
```

Each has a `!` form that draws into an existing axis, a PracticalBayesPlot
`Axis` (with `aspect = DataAspect()`) or a Makie one, so diagrams and data
plots can share a figure. The layouts can be inspected without drawing with
`dag_view`, `dag_paths`, `tree_layout`, `mosaic_layout` and `venn_layout`.
[examples/plotting.qmd](examples/plotting.qmd) shows every diagram, the
posterior plots, and the TikZ output.

## Models from PracticalBayes

Put `@graphical` in front of `@model`. The model is defined exactly as before,
and its graph is read from the `~` statements:

```julia
using PracticalBayes, PracticalProbability, Distributions

@graphical @model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    μ = α .+ β .* x
    y .~ Normal.(μ, σ)
end

m = linreg(x, y)
posterior(m)     # p(α, β, σ | y, x) ∝ p(α) p(β) p(σ) p(y | α, β, σ, x)
latex(m)         # the model as aligned `~` statements, then the posterior
generative(m)    # a table of variables, roles, distributions and parents
```

Data are the model's non-`missing` arguments plus anything bound with
`condition`, the same rule PracticalBayes uses. Deterministic assignments
(`μ = …`) are followed through to the variables they are computed from.
`@generative function … end` reads a model without defining it.

## Design

Probability expressions use their own small term language, not Symbolics.jl.
The arguments of `P(⋅|⋅)` are sets and products are multisets, so rules match
by set operations. Every rewrite returns its justification, which is what makes
`trace = true` derivations possible. The tests check each rule against exact
enumeration on random networks.
