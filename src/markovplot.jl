# Layouts for finite-state transition graphs and time-unrolled HMMs.

"A positioned transition, with its original (possibly exact rational) probability."
struct MarkovTransition
    from::Int
    to::Int
    probability::Real
    path::Vector{NTuple{2,Float64}}
    head::Vector{NTuple{2,Float64}}
    labelpos::NTuple{2,Float64}
end

"Backend-independent layout returned by `markov_view`."
struct MarkovView
    states::Vector{Any}
    positions::Vector{NTuple{2,Float64}}
    transitions::Vector{MarkovTransition}
    absorbing::Vector{Bool}
    bbox::NTuple{4,Float64}
end

"""
    markov_view(mc; minprob=0, selfloops=true, positions=nothing) -> MarkovView

Lay out a finite Markov chain. Arrows follow `P[i,j]` from state `i` to state
`j`; zero probabilities are omitted. `minprob` hides probabilities at or below
the threshold without renormalising them. Self-loops and reciprocal arrows
are curved. `positions` is an optional vector of `(x,y)` pairs in state order.
Absorbing states (exactly `P[i,i] == 1`) are shaded by `markovplot`.
"""
function markov_view(mc::MarkovChain; minprob=0, selfloops=true, positions=nothing)
    n = length(mc)
    n > 0 || throw(ArgumentError("cannot plot an empty chain"))
    isfinite(minprob) && 0 <= minprob <= 1 || throw(ArgumentError("minprob must be in [0,1]"))
    radius = max(1.0, n/4)
    pos = positions === nothing ? [(radius*cospi(2(i-1)/n), radius*sinpi(2(i-1)/n)) for i in 1:n] :
        [(Float64(p[1]), Float64(p[2])) for p in positions]
    length(pos) == n && all(p -> all(isfinite, p), pos) || throw(ArgumentError("give one finite position per state"))
    all(i == j || hypot((pos[i] .- pos[j])...) > 0.8 for i in 1:n, j in 1:n) ||
        throw(ArgumentError("node positions must be at least 0.8 units apart"))
    transitions = MarkovTransition[]
    centre = (sum(first, pos)/n, sum(last, pos)/n)
    for i in 1:n, j in 1:n
        prob = mc.P[i, j]
        (prob <= minprob || (!selfloops && i == j)) && continue
        a, b = pos[i], pos[j]
        if i == j
            # Point loops away from the graph so lower nodes' loops do not
            # collide with incoming edges or their probability labels.
            r = hypot((a .- centre)...)
            ux, uy = r < 1e-12 ? (0.0, 1.0) : (a .- centre) ./ r
            path = [(a[1]+(0.43+0.31sin(t))*ux+0.31cos(t)*uy,
                     a[2]+(0.43+0.31sin(t))*uy-0.31cos(t)*ux) for t in range(-pi/4, 5pi/4; length=40)]
            labelpos = (a[1]+0.85ux, a[2]+0.85uy)
        else
            dx, dy = b .- a
            len = hypot(dx, dy)
            ux, uy = dx/len, dy/len
            bend = mc.P[j, i] > minprob ? 0.65 : 0.0
            c = ((a[1]+b[1])/2-bend*uy, (a[2]+b[2])/2+bend*ux)
            ua = (c .- a) ./ hypot((c .- a)...)
            ub = (b .- c) ./ hypot((b .- c)...)
            start, stop = a .+ 0.3 .* ua, b .- 0.3 .* ub
            path = [((1-t)^2*start[1]+2t*(1-t)*c[1]+t^2*stop[1],
                     (1-t)^2*start[2]+2t*(1-t)*c[2]+t^2*stop[2]) for t in range(0, 1; length=32)]
            labelpos = ((a[1]+b[1])/2-(bend/2+0.12)*uy, (a[2]+b[2])/2+(bend/2+0.12)*ux)
        end
        tip, prev = path[end], path[end-1]
        ux, uy = (tip .- prev) ./ hypot((tip .- prev)...)
        base = tip .- 0.13 .* (ux, uy)
        head = [tip, (base[1]-0.06uy, base[2]+0.06ux), (base[1]+0.06uy, base[2]-0.06ux)]
        push!(transitions, MarkovTransition(i, j, prob, path, head, labelpos))
    end
    allpoints = copy(pos)
    for e in transitions
        append!(allpoints, e.path)
        push!(allpoints, e.labelpos)
    end
    xs, ys = first.(allpoints), last.(allpoints)
    bbox = (minimum(xs)-0.5, maximum(xs)+0.5, minimum(ys)-0.5, maximum(ys)+0.5)
    return MarkovView(copy(mc.states), pos, transitions, [isone(mc.P[i,i]) for i in 1:n], bbox)
end

"""
    markovplot(mc; minprob=0, selfloops=true, positions=nothing, labels=true,
               scale=140, figure=(;), axis=(;))

Draw a Markov chain's weighted transition graph with the current `plotbackend()`.
Probabilities keep exact rational labels; shaded nodes are absorbing states.
See `markov_view` for layout options. `markovplot!(ax, mc)` draws into an
existing axis and returns the layout. Both PracticalBayesPlot and Makie work.
"""
function markovplot(mc::MarkovChain; minprob=0, selfloops=true, positions=nothing, kwargs...)
    return _markovplot(_backend(), markov_view(mc; minprob, selfloops, positions); kwargs...)
end
markovplot(v::MarkovView; kwargs...) = _markovplot(_backend(), v; kwargs...)
function _markovplot end

"Draw a transition graph into an existing axis; return its `MarkovView`."
function markovplot!(ax, mc::MarkovChain; minprob=0, selfloops=true, positions=nothing, kwargs...)
    return markovplot!(ax, markov_view(mc; minprob, selfloops, positions); kwargs...)
end

"""
    hmm_view(hidden, emissions; observed=true) -> DAGView
    hmm_view(X::RV, Y::RV, n; observed=true)
    hmm_view(bn::BayesNet, hidden, emissions; observed=true)

A time-aligned HMM diagram: `X[t] → X[t+1]` across the upper row and
`X[t] → Y[t]` downwards. Give equally sized vectors of distinct random
variables, or base variables and a positive number of steps. `observed` is
`true` (shade every emission), `false`, or a Boolean mask over time.
When a BayesNet is supplied, its graph must match this HMM topology exactly.
This draws model structure; it does not fit an HMM or infer hidden states.
"""
function hmm_view(hidden::AbstractVector{RV}, emissions::AbstractVector{RV}; observed=true)
    n = length(hidden)
    n > 0 && length(emissions) == n || throw(ArgumentError("give equally sized, nonempty hidden and emission sequences"))
    length(unique(vcat(hidden, emissions))) == 2n || throw(ArgumentError("HMM variables must be distinct"))
    mask = observed isa Bool ? fill(observed, n) : collect(observed)
    length(mask) == n && all(v -> v isa Bool, mask) || throw(ArgumentError("observed must be a Boolean or a Boolean mask over time"))
    ns = DAGNode[]
    es = DAGEdge[]
    for t in 1:n
        x = 1.5(t-1)
        push!(ns, DAGNode(hidden[t], x, 0.0, :circle, :none, false, false))
        push!(ns, DAGNode(emissions[t], x, -1.3, :circle, mask[t] ? :observed : :none, false, false))
        push!(es, DAGEdge(hidden[t], emissions[t], :plain))
        t < n && push!(es, DAGEdge(hidden[t], hidden[t+1], :plain))
    end
    panel = DAGPanel(ns, es, DAGPlate[])
    return DAGView([panel], DAGText[], DAGPath[], _panelbox(panel))
end
function hmm_view(x::RV, y::RV, n::Integer; kwargs...)
    n >= 1 || throw(ArgumentError("an HMM diagram needs at least one step"))
    return hmm_view(x[1:n], y[1:n]; kwargs...)
end
function hmm_view(bn::BayesNet, hidden::AbstractVector{RV}, emissions::AbstractVector{RV}; kwargs...)
    v = hmm_view(hidden, emissions; kwargs...)
    expected = Set((e.from, e.to) for e in only(v.panels).edges)
    actual = Set((a, b) for (a, b) in edges(bn.dag))
    Set(nodes(bn.dag)) == Set(vcat(hidden, emissions)) && actual == expected ||
        throw(ArgumentError("BayesNet graph does not match the supplied HMM sequences"))
    return v
end

"""
    hmmplot(hidden, emissions; observed=true, kwargs...)
    hmmplot(X::RV, Y::RV, n; observed=true, kwargs...)
    hmmplot(bn::BayesNet, hidden, emissions; observed=true, kwargs...)

Draw a time-unrolled hidden Markov model using `hmm_view`. Remaining keywords
are those of `dagplot`. Uses the selected plotting backend.
"""
hmmplot(args...; observed=true, kwargs...) = dagplot(hmm_view(args...; observed); kwargs...)

"Draw a time-aligned HMM into an existing axis; return its `DAGView`."
hmmplot!(ax, args...; observed=true, kwargs...) = dagplot!(ax, hmm_view(args...; observed); kwargs...)
