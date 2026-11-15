"""
    PracticalProbability

Computer algebra for probability: build expressions like `P(A, B | C)`,
`P(A ∪ Bᶜ)`, `π(x, y | D)` and `P(Y | Do(X))`; transform them (chain rule,
Bayes' rule, marginalisation, complements, De Morgan, inclusion–exclusion);
simplify them using a causal graph (d-separation, do-calculus); check them
numerically on a Bayesian network; and render everything as LaTeX or as
PrettyTables tables. `@graphical` reads a PracticalBayes/Turing `@model` into
the same algebra.
"""
module PracticalProbability

using LaTeXStrings: LaTeXString, @L_str
using Latexify: Latexify, latexify
# `factorize` gets methods for graphs and models rather than being a new
# function, so `using LinearAlgebra` alongside this package does not clash.
import LinearAlgebra: factorize
using LinearAlgebra: dot, nullspace, pinv, logdet, tr
using MacroTools: MacroTools
using PrettyTables: pretty_table
using Random: Random, AbstractRNG
using Tables: Tables

include("atoms.jl")
include("expr.jl")
include("linalg.jl")

include("graph.jl")
include("facts.jl")
include("rules.jl")
include("eventrules.jl")
include("printing.jl")
include("numeric.jl")
include("tables.jl")
include("generative.jl")
include("assume.jl")
include("expectation.jl")
include("plates.jl")
include("venn.jl")
include("importance.jl")
include("plots.jl")         # diagram functions and the backend switch
include("samples.jl")
include("dagplot.jl")
include("probtree.jl")
include("draw/common.jl")     # drawing with PracticalBayesPlot (the default backend)
include("draw/dag.jl")
include("draw/tree.jl")
include("draw/venn.jl")
include("draw/factor.jl")
include("information.jl")
include("canonical.jl")       # after information.jl: uses its special functions
include("transforms.jl")
include("gaussian.jl")
include("odds.jl")
include("markovchain.jl")
include("markovplot.jl")
include("draw/markov.jl")
include("experiments.jl")
include("admg.jl")
include("shannon.jl")
include("counterfactual.jl")
include("cfbounds.jl")
include("transport.jl")
include("adjustment.jl")      # after admg.jl and cfbounds.jl
include("implications.jl")
include("lp.jl")
include("lsemid.jl")


"""
    independencies(g; max_conditioning = 2) -> Independencies

Every pairwise independence `A ⟂ B | Z` the graph implies, for conditioning
sets `Z` of at most `max_conditioning` variables and with `Z` minimal (no
proper subset of `Z` also separates `A` and `B`). Prints as a table.
"""
independencies(g::DAG; max_conditioning::Integer=2) =
    Independencies(_independencies(g; max_conditioning))

# Not exported: Product, Quotient, Sum and Marginal (Distributions exports a
# `Product`); build them with *, /, +, - and Σ.

# terms
export Atom, RV, Event, Intervention, Do, @rv, rv, rvs, isdiscrete, iscontinuous
export PExpr, Prob, Literal, P, @P, Σ, Given, Density, @density
export freevars, substitute, mapprobs, Equation
# events
export @event, isevent, iseventlike, complement, Ω, ∅, @hold
export EventUnion, EventIntersection, Complement, SureEvent, ImpossibleEvent
export demorgan, complement_rule, inclusion_exclusion, union_bound, total_probability
# graphs
export DAG, @dag, nodes, parents, children, edges, ancestors, descendants, markov_blanket
export topological_order, remove_incoming, remove_outgoing, d_separated
export Independence, Independencies, ⟂, holds, independencies
# algebra
export chainrule, bayes, marginalize, condition_on, factorize, joint
export simplify, expand, identify
export Derivation, derivation, apply!
# numbers
export Factor, BayesNet, rand_bayesnet, cpt, compute, verify, sumout, domain
# output
export latex, latexify, LaTeXString, tabulate, tabulate_cpt
# models
export GenerativeModel, @graphical, @generative, generative, observe, unobserve
export posterior, prior, likelihood, dag
# assumptions and deduction
export Facts, assume, disjoint, implies, iid, markov, deduce, Solution, cell_probabilities
# expectations
export E, 𝔼, Var, Cov, 𝟙, indicator, RExpr, Expect, Variance, Covariance
export definition, tower, total_variance
# plates
export Index, @index, IndexedRV, whole, ∏, plate, PlateProduct, sequence, peel, unroll, plates
# venn (drawn with PracticalBayesPlot, src/draw/, or Makie, ext/)
export venn, vennplot, vennplot!, venn_layout, venn_sets, VennLayout
# importance
export importance_weight, ImportanceWeight, proportional, weight
export check_support, weight_moments, relative_ess, logweight, @logweight, logweight_expr
export smc_weight, SMCWeight, path_weight
# diagrams (drawn with PracticalBayesPlot, src/draw/, or Makie, ext/)
export venn_value, dagplot, dagplot!, dag_view, dag_paths, pathstring, DAGView, DAGPath
export probtree, probtree!, tree_layout, TreeLayout, mosaic, mosaic!, mosaic_layout, MosaicLayout
export factorplot, factorplot!, plotbackend, plotbackend!
# sample events (events defined on draws)
export SampleEvents, @sampleevents, sample_events
# information theory
export H, ℍ, MI, KL, crossentropy, renyi, chisq, Entropy, MutualInformation, Divergence, Log
export entropies, information, Inequality, data_processing, elbo, evidence_decomposition
export transfer_entropy, directed_information, Family, evaluate
export prove, isshannon, ShannonProof, proof_expr
# hidden confounders and identification
export ADMG, @admg, bidirected, siblings, subgraph, districts, c_components, canonical_dag
export latent_projection, NotIdentifiable, isidentifiable, hedge
export Counterfactual, rand_scm, CFBounds, bounds, pns, pn, ps, mediation
export transport, istransportable, NotTransportable, recover, isrecoverable, NotRecoverable
export Domain, zidentify, iszidentifiable
# canonical forms and the algebra of random quantities
export canonical, derivative, RExt
# distributions of transformed variables
export Laws, laws, law, MGF, PGF, CF, PDF, CDF, rawmoment, centralmoment, cumulant
export change_of_variables, jacobian, order_statistic
# Gaussian algebra
export Gaussian, marginal, conditional, regression, LinearSEM, linear_sem, total_effect
export implied_covariance, intervene, kalman_step
# odds
export odds, likelihood_ratio, bayes_factor, bayes_odds, log_odds, weight_of_evidence
export probability_from_odds, odds_from_probability
# Markov chains
export MarkovChain, transition_matrix, nstep, distribution, communicating_classes
export isirreducible, isaperiodic, stationary, hitting_probability, hitting_time, mean_return_time
export MarkovView, MarkovTransition, markov_view, markovplot, markovplot!, hmm_view, hmmplot, hmmplot!
# experiments
export experiment, Outcomes, uniform, die, coin, bernoulli, categorical, depends_on, derived
export Urn, urn, deck, draws
# adjustment sets
export isadjustment, adjustment_sets, optimal_adjustment, adjustment_formula
# testable implications and Markov equivalence
export markov_basis, verma_constraints, FunctionalConstraint, CPDAG, cpdag, vstructures
export markov_equivalent, equivalence_class
# bounds by linear programming
export LPBounds, lp_bounds
# identification in linear SEMs
export covariance_symbol, covariance_values, isinstrument, instruments, iv_estimand
export HTCResult, htc_identify




include("draw/precompile.jl")   # after everything: it uses the whole API

end
