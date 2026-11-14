| | PracticalBayesPlot route | CairoMakie route |
|:--|--:|--:|
| `using PracticalProbability` (+ `using CairoMakie`) | 1.7 s | 13 s |
| first `dagplot` | 0.06 s | 24 s |
| first `venn` (after a `dagplot`) | 0.02 s | — |
| writing the first figure to a file | 0.05 s (SVG, TikZ or CeTZ) | 3 s (PNG) |
| precompiling the plotting package | 19 s (PracticalBayesPlot) | several minutes (Makie, CairoMakie) |

PracticalBayesPlot on its own loads in about 0.1 s, and its first figure takes
under 0.1 s, because its code is compiled when it is installed. Its TikZ and
CeTZ outputs each add 0.1 to 0.2 s to that compilation and nothing measurable
to loading or to the first figure.
