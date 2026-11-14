# Roadmap

## Long term: a full CAS

The eventual goal is a general-purpose computer algebra system for Julia that is
more feature-complete than Symbolics.jl. For now it will be built as separate
packages. PracticalProbability is the probability layer. Its term language
(`Prob`, `Product`, `Quotient`, `Marginal`) is self-contained on purpose, so it
can later sit on top of that CAS core without depending on Symbolics.

## PracticalProbability

### Algebra
- [x] Event algebra: `∪`, `∩`, complements, `Ω`/`∅`, De Morgan,
      inclusion–exclusion, complement rule, total probability; sums of terms.
- [x] Named distributions (`π(x, y | D)`, `q(z)`), with graphs per distribution.
- [ ] Events with inequalities and sets of values (`X > 0`, `X ∈ {1, 2}`).
- [x] Facts without a graph: independence (with decomposition and weak union),
      `disjoint`, `implies`, `iid`, `markov`; `deduce` for numeric exercises
      (linear algebra over the 2ⁿ cells, plus independence products).
- [ ] `deduce` bounds when the facts don't determine the answer (a small LP:
      "between 0.1 and 0.3"), and mutual vs pairwise independence of events.
- [ ] Distribution families in expressions (`p(x | θ) = Normal(x; μ, σ)`) and
      conjugate updates done symbolically.
- [x] Transformations of random variables: `laws(X => Family(…))` and `law(g(X, Y))`
      for affine maps, sums of independent variables (Normal, Cauchy, Poisson,
      Gamma/Exponential, Binomial/Bernoulli), `exp`/`log`, squares of centred
      Normals (χ²) and minima of Exponentials/Geometrics; densities and CDFs
      (`PDF`, `CDF`), `change_of_variables` and `jacobian`, `order_statistic`.
- [x] Generating functions: `MGF`, `PGF`, `CF` of families and of linear
      combinations of independent variables (`M_{X+Y} = M_X M_Y`); moments and
      cumulants from them (`rawmoment`, `centralmoment`, `cumulant`).
- [x] Gaussian algebra: `Gaussian` with symbolic means and covariances,
      `marginal`, `conditional` (Schur complement, brought to lowest terms),
      `regression`; linear-Gaussian SEMs (`linear_sem`): path tracing
      (`total_effect`, `implied_covariance`), `intervene`, confounding bias of a
      regression vs adjustment; the Kalman step as Gaussian conditioning.
- [x] Exact arithmetic: rational tables stay rational through `compute`,
      `deduce` solves in rational arithmetic, literals print as `1/6`;
      `rand_bayesnet(…; exact = true)` checks identities with `==`.
- [x] Odds form of Bayes' rule: `odds`, `likelihood_ratio`, `bayes_factor`,
      `bayes_odds` (posterior odds = likelihood ratio × prior odds), `log_odds`
      and `weight_of_evidence` (evidence adds on the log scale).
- [x] Discrete Markov chains (`MarkovChain`): stationary distributions,
      communicating classes and periods, hitting probabilities and expected
      hitting times (gambler's ruin exactly), the chain unrolled into a `BayesNet`.
- [x] Experiment builders: `experiment` with `die`, `coin`, `uniform`,
      `categorical`, `depends_on`, `derived`, urns and cards (`urn`, `deck`,
      `draws` with or without replacement): Monty Hall, dice sums and urn
      problems without writing tables.
- [x] Expectations `E[f | Y]`, `Var`, `Cov`: linearity, taking out what is
      known, tower property, independence, total variance; exact numbers.
- [x] Moments of standard distributions in `simplify` (`E[X] = μ` for
      `X ~ Normal(μ, σ)`, `E[X²]`, `Var(X)`, `E[exp(tX)]` from the MGF), with
      independence read from the laws.
- [ ] Expectations over continuous variables from PracticalBayes chains.
- [x] Canonical forms: `expand` and `canonical` (polynomials in atoms, sorted
      arguments and terms), `simplify(::RExpr)` to lowest terms by
      multivariate polynomial division, `derivative`, `substitute`; and
      `simplify(…; search = true)`, a beam search over rule orders for the
      smallest normal form instead of first match.
- [ ] Polynomial GCDs (lowest terms when the denominator doesn't divide the
      numerator), and a canonical form shared with the future CAS core.
- [x] Plates and indexed variables (`y[i]`, `x[t-1]`, `x[1:t]`, `∏ᵢ`),
      `sequence`/`peel`, `iid`/`markov` facts, `unroll`; models with loops.
- [ ] d-separation on plated graphs (templates) without unrolling.

### Causal inference (identification)
The symbolic side of causal inference: deciding whether a causal quantity is
determined by the observed distribution and, if so, writing it in observational
terms. Estimation (propensity scores, DML, TMLE) and structure learning (PC,
GES) are out of scope; CausalInference.jl covers those.
- [x] ADMGs: bidirected edges for hidden confounding (`@admg`, `X ↔ Y`), latent
      projection of a DAG with hidden variables, c-components (districts),
      m-separation, and the do-calculus rules in `simplify` on an ADMG.
- [x] The ID and IDC algorithms (Shpitser–Pearl) in `identify`, with a hedge as
      the certificate when a query is not identifiable. Front-door and napkin
      as test cases; every answer checked numerically against the full model
      with its latent variables.
- [x] Counterfactuals: potential outcomes `Y(Do(X(x)))(y)` as atoms, exact
      computation in a structural model through the parallel-worlds network
      (`rand_scm`), make-cg, ID* / IDC* (Shpitser–Pearl 2007). Effect of
      treatment on the treated; non-identifiability of the probability of
      necessity and sufficiency.
- [x] Bounds for non-identifiable counterfactuals: Tian–Pearl bounds on the
      probabilities of necessity and sufficiency (`bounds`, `pns`, `pn`, `ps`),
      their point values under monotonicity, experimental terms identified
      through a graph.
- [x] Counterfactual queries with free values (`P(Y_x, X)` as a function) and
      nested counterfactuals (`Y(Do(X(1)), M(Do(X(0))))`, unnested for ID*), with
      `mediation` giving the total, natural direct and natural indirect effects.
- [x] Transportability: selection diagrams and the sID algorithm
      (Bareinboim–Pearl), `transport`, with source experiments and target
      observations.
- [x] z-identifiability (`zidentify`) and transport from several sources with
      limited experiments (`Domain`, TR^mz), checked for soundness on random
      models.
- [x] Selection bias: `recover` with a selection event `S`: s-recoverability of
      `P(Y | X)`, and generalised adjustment for `P(Y | Do(X))` with external
      unbiased data on covariates.
- [x] Recoverability beyond adjustment: identify the query (hidden variables
      allowed) and recover each term from the biased data, external data or a
      term-level adjustment (front-door cases).
- [ ] Adjustment sets: enumerate the minimal valid back-door sets, and the
      optimal adjustment set (Henckel–Perković–Maathuis), for a DAG or ADMG.
- [ ] General LP bounds for non-identifiable queries (Balke–Pearl), e.g. the
      instrumental-variable bounds, generalising the Tian–Pearl bounds; the same
      LP gives `deduce` its "between 0.1 and 0.3" answers.
- [ ] Testable implications of a graph: its conditional independences, Verma
      constraints of an ADMG (nested Markov), and Markov equivalence (are two
      DAGs equivalent? the CPDAG). Symbolic only; no structure learning.
- [ ] Identification in linear SEMs: instrumental variables and the
      half-trek criterion, on top of the Gaussian algebra.
- [x] The papers' criteria in `recover`: P(y | x) without external data iff
      `Y ⟂ S | X` (BTP 2014, Thm 1); with external data the one-test
      separator conditions (Thms 2–3, Cors 3–5) and the selection-backdoor
      criterion (Thm 5); causal effects without hidden variables iff no
      variable of An(Y) in G[V∖X] is an ancestor of S (BT 2015, Thm 2); with
      hidden variables the RC algorithm (BT 2015, Fig. 4). Failures of the
      complete criteria say that no method can recover the query.

### Information theory
Entropy, mutual information and divergences are expectations of logs of the
probability terms the package already has, so they get the same treatment:
rewrite rules, links to independence facts and graphs, exact numbers. Estimators
from samples (entropy, kNN mutual information) and complexity measures
(sample/permutation entropy, Lempel–Ziv, fractal dimensions) are out of scope;
ComplexityMeasures.jl and Associations.jl cover those.
- [x] Terms `H(X | Z)`, `MI(X, Y | Z)`, `KL(p, q)`, `crossentropy`; `log` of an
      expression; `definition` as `E[-log P(⋯)]`; exact numbers by `compute`.
- [x] Chain rules for entropy, mutual information and KL; rewriting into joint
      entropies (`entropies`); recognising `H(X | Z) - H(X | Y, Z) = I(X; Y | Z)`
      and the chain rules backwards inside sums.
- [x] Independence ↔ information: `X ⟂ Y | Z` gives `I(X; Y | Z) = 0` and drops
      conditioning in `H`, by d-separation or assumed facts. The
      data-processing inequality read off a graph (`data_processing`).
- [x] Variational inference: `log p(y) = ELBO + KL(q(z | y) ‖ p(z | y))`
      derived step by step (`evidence_decomposition`, `elbo`).
- [x] Rényi and χ² divergences, and their link to importance sampling:
      relative ESS `= 1 / (1 + χ²(π ‖ q))` (tested).
- [x] Closed-form entropies and KL divergences for distribution families
      (`Family`: Normal, Exponential, Poisson, Bernoulli).
- [x] More families: Gamma, Beta, Dirichlet, Laplace, Uniform, Geometric
      (with `loggamma` and `digamma` in random quantities) and numeric
      multivariate Normal.
- [x] A Shannon-type inequality prover (`prove`, ITIP-style, by non-negative
      least squares over the elemental inequalities) that returns the proof as
      a weighted sum of elemental inequalities, also under independence
      constraints or a graph; a counterexample search when there is no proof.
- [x] Non-Shannon inequalities in the prover: Zhang–Yeung instances as
      generators (`nonshannon = true`) and the copy lemma (`copies`, or
      `copies = :auto`, which proves Zhang–Yeung itself).
- [x] Transfer entropy and directed information on indexed sequences
      (definitions and rewriting; no estimators).

### Numbers
- [ ] Variable elimination instead of full enumeration, for networks beyond a
      handful of variables.
- [x] Probabilities from draws for events defined by conditions
      (`@sampleevents chain begin A = σ > 0.2 end`; `compute(P(A | B), se)`).
- [ ] Continuous models: evaluate posterior queries from PracticalBayes chains
      directly (`P(θ > 0 | y)` as an expression, without naming the event first).
- [ ] Fitting CPTs from data (Tables.jl input).

### Models and PPLs
- [x] `@graphical` for PracticalBayes `@model`; `generative(::PracticalBayes.Model)`.
- [ ] Turing/DynamicPPL extension: `generative(::DynamicPPL.Model)` (registration
      by function already works for Turing's `@model`).
- [ ] PracticalBayesianRegression: the generative model of a fitted
      formula-based GLM, with LaTeX.
- [ ] Conjugacy detection from the graph (hand closed-form updates to PracticalBayes'
      `ConjugateGibbs`).
- [ ] A small PPL of its own, if the integrations need one.

### Importance sampling
- [x] Importance weights derived symbolically: `importance_weight(target, proposal)`
      factorises both, cancels shared terms, drops factors constant across
      particles (`∝`, self-normalised). Covers the prior, bootstrap and locally
      optimal proposals and arbitrary proposal kernels.
- [x] Exact checks on discrete networks: `verify` (`q ⋅ w = π`), `check_support`,
      `weight_moments`, `relative_ess`.
- [x] Proposal models: identical `~` lines cancel, and `logweight` compiles the
      rest to `logw(θ, data)`.
- [x] SMC on plates: `smc_weight(γ, q_t)` derives the incremental weight
      `γ_t / (γ_{t-1} q_t)` by peeling the path target, plus `path_weight` and
      exact checks at unrolled times.
- [ ] Tempering and resample-move targets (`π_t ∝ p(y | θ)^{β_t} p(θ)`), which
      need powers of terms; SMC weights between two plated generative models.
- [ ] Cancellation up to parameterisation (`Normal(μ, σ)` vs `Normal(μ, sqrt(v))`),
      and Jacobians for proposals in transformed spaces (Bijectors).
- [ ] A PracticalBayes sampler that uses `logweight` (SNIS / SMC samplers).

### Output
- [x] Venn diagrams of event expressions, equations and derivations (Makie
      extension: `venn`, `vennplot`).
- [x] Venn cell and panel probabilities from facts (`deduce`) or from draws.
- [ ] Venn diagrams with four events (ellipses), and cell probabilities from a
      `BayesNet` via `compute`.
- [x] DAG drawing (Makie): d-separation paths open/blocked, back-door paths and
      graph surgery for `P(Y | do(X))`, plate notation for models.
- [x] Probability trees and mosaic plots (area-proportional), coloured by a query
      like the Venn panels; `Factor` plots (bars, grouped bars, heatmaps).
- [ ] DAG drawing: TikZ (via TikzMakie); better layout for large graphs (crossing
      minimisation across plates, curved edges).
- [ ] Drawings for the front-door and latent-variable cases once ID/IDC exists.
- [ ] Documenter.jl site with worked examples (Monty Hall, Simpson's paradox,
      front-door, hierarchical models).

### Infrastructure
- [x] Version control (git).
- [ ] The package on GitHub with CI running the tests.

- [x] Static checks in the test suite: Aqua (ambiguities, unbound parameters,
      stale dependencies, compat, piracy) and JET (provable errors).
- [ ] Clear the 44 JET reports in the older code (cfbounds, counterfactual,
      transport, importance, dagplot, probtree, admg; mostly `findfirst` results
      used without a `nothing` check), then make the JET test strict.


