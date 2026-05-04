# Byzantine-Resilient Consensus & Optimization — Alan John

Two expository papers and three paired open-source implementations covering
the core problems in resilient multi-agent systems:

| Problem | Algorithm | Paper file | Implementation |
|---------|-----------|------------|----------------|
| Resilient *constrained* consensus over complete graphs | Drop-furthest-then-project | `acc_algorithms.tex` (Part I) | `ACC22.py`, `ACC22.m` |
| Resilient *distributed optimization* on directed graphs | Tverberg-style aggregator + subgradient | `acc_algorithms.tex` (Part II) | `ACC23.py`, `ACC23.m` |
| Resilient *average consensus* with detection-and-recovery | Two-hop info sets + running-sum push-sum | `rac_algorithm.tex` | `RAC.py`, `RAC.m` |

Both papers are by **Alan John**:

- `acc_algorithms.tex` — October 22, 2024 (combined ACC22 + ACC23 exposition)
- `rac_algorithm.tex`  — April 9, 2025

Each is adapted from published source paper(s).  The algorithms, theorems,
and proofs are due to the original authors; this work contributes structured
expositions plus a unified, paper-faithful, open-source implementation.

## Papers

### 1. `acc_algorithms.tex` — Resilient Constrained Consensus and Distributed Optimization
An integrated exposition of two related ACC papers by Zhu, Lin, Velasquez,
and Liu:

- **Part I** — *Resilient Constrained Consensus over Complete Graphs via
  Feasibility Redundancy*, **ACC 2022**.  Each of `n` agents holds a state
  in `R^m` confined to a private convex set `X_i`. Up to `f` agents are
  Byzantine (per-recipient: a Byzantine `j` may send different `x_{ji}` to
  different recipients `i`).  Honest agents drop the `f` farthest received
  values, step toward the centroid of the residuals, and project onto
  `X_i`.  Theorem 2 gives an analytic exponential contraction rate
  `ρ = 1 − (μ²k − 4f − 2fμ² + μ²) α + 4|H|³ α² ∈ (0, 1)`
  provided the redundancy and step-size bounds are satisfied.

- **Part II** — *Resilient Distributed Optimization*, **ACC 2023**.  Honest
  agents minimize `f(x) = Σ f_i(x)` despite up to `β` Byzantine in-neighbors
  per agent.  The aggregator is **Tverberg-style**: for every
  `(d+1)β+1`-subset `A_ij` of `N_i^-`, the agent picks
  `y_ij ∈ ⋂_{S ∈ B_ij} conv(S)` over its `(dβ+1)`-subsets, then
  `v_i = (x_i + Σ y_ij)/(1+a_i)` and `x_i(t+1) = v_i − α(t) g_i(v_i)`.
  The exposition gives explicit closed-form constructions for **`d=1`,
  any `β`** (reduces to a trimmed mean) and **`d=2, β=1`** (Tverberg
  point of 4 points via crossing segments).

The two parts share preliminaries (graph notation, projection, threat
model) and a unified discussion of how they fit together in the resilient
multi-agent landscape.

### 2. `rac_algorithm.tex` — Resilient Average Consensus with Adversaries
Adapted from Yuan, Ishii — *Resilient Average Consensus with Adversaries
via Distributed Detection and Recovery*, **arXiv 2405.18752v1, 2024**.

A **two-phase** algorithm: detection + averaging.  Detection has two flavors
(Algorithm 2: sharing detection on undirected graphs; Algorithm 3: fully
distributed detection on directed graphs via majority voting on `2f+1`
two-hop paths).  Averaging is the running-sum push-sum
\[Hadjicostis et al. 2016\] augmented with two recovery equations: when a
malicious in-neighbor is newly detected, subtract its cumulative
contribution; when a malicious out-neighbor is newly detected, add back
the cumulative `λ_i, γ_i` that was sent to it.  Honest agents
asymptotically converge to the average over the protocol-conforming
initial values.

## Implementations

| File | Run | What it demonstrates |
|------|-----|---------------------|
| `ACC22.py` | `python ACC22.py` | Box / Polyhedral demo, V(t) Lyapunov, empirical ρ vs analytic prediction, Theorem-2 condition checker, two-faced Byzantine adversary |
| `ACC23.py` | `python ACC23.py` | Paper-exact Tverberg aggregator (d=1 trimmed-mean, d=2 β=1 closed-form), `(β,dβ)`-resilience brute-force checker, both `α(t) = c/√T` and `α(t) = a/(1+bt)` schedules, comparison aggregators (CWTM, Krum, geometric median, distance trim) |
| `RAC.py` | `python RAC.py` | Push-sum running-sum core, two detection algorithms (sharing + fully distributed), Cases 1 and 2 recovery, four attack policies (`naive`, `stealth_constant`, `delayed`, `pretend_normal`), small-graph constructors matching the source paper's figures |
| `ACC22.m` | (open in MATLAB ≥ R2018b, press Run) | MATLAB port of `ACC22.py` |
| `ACC23.m` | (open in MATLAB ≥ R2018b, press Run) | MATLAB port of `ACC23.py` |
| `RAC.m`   | (open in MATLAB ≥ R2018b, press Run) | MATLAB port of `RAC.py` |

### Sample commands

```bash
# ACC22 -- constrained consensus (paper Part I)
python ACC22.py                          # default Box demo, max_spread adversary
python ACC22.py --attack two_faced       # per-recipient Byzantine
python ACC22.py --demo polyhedral        # Corollary 1 (Hoffman regularity, μ=1)

# ACC23 -- distributed optimization (paper Part II)
python ACC23.py                          # paper aggregator, d=2, β=1, K_8
python ACC23.py --d 1 --n 6              # d=1 trimmed-mean reduction
python ACC23.py --aggregator cwtm        # CWTM comparison baseline
python ACC23.py --attack two_faced       # per-recipient adversary

# RAC -- average consensus with detection
python RAC.py                            # 6-node directed (paper Fig. 5(b))
python RAC.py --graph fig5a --algorithm sharing --f 2 --iters 30
python RAC.py --graph extreme8 --f 1 --iters 30      # 8-node, 5 malicious
python RAC.py --graph layered --layers 4 --f 1       # paper Fig. 4(a)
python RAC.py --plot                                 # save matplotlib trace
```

## How to compile the papers

The two `.tex` files are self-contained (single-file, all references
inline as `\bibitem`s) and target the **IEEEtran** journal class.  Two
options:

**Option A: Overleaf (no install).**
Create a free account at [overleaf.com](https://www.overleaf.com), upload
the `.tex` file, click *Compile*.  About a minute per paper.  Recommended.

**Option B: Local LaTeX install.**
Install MiKTeX (Windows) or TeX Live (macOS/Linux), then:
```bash
pdflatex acc_algorithms.tex
pdflatex acc_algorithms.tex   # second pass for cross-refs
pdflatex rac_algorithm.tex
pdflatex rac_algorithm.tex
```

## Folder map

```
.
├── acc_algorithms.tex       Alan's combined ACC22 + ACC23 exposition (LaTeX source)
├── rac_algorithm.tex        Alan's RAC exposition
├── ACC22.py / ACC22.m       Resilient constrained consensus implementation
├── ACC23.py / ACC23.m       Resilient distributed optimization
├── RAC.py   / RAC.m         Resilient average consensus
└── README.md                this file
```

## Author & attribution

**Alan John** — papers dated October 22, 2024 (ACC22 + ACC23 combined),
April 9, 2025 (RAC).  Each paper acknowledges the original source(s) on
its title page and in the abstract.  The exposition contributions are:

1. Self-contained re-derivations with full pseudocode and explicit
   theorem statements;
2. Open-source paper-faithful implementations in Python (with MATLAB
   ports);
3. Cross-referencing between the two expositions so that the reader can
   move between the related problems in the resilient-consensus
   landscape.

The original algorithms, theorems, and proofs are credited to:

- ACC22 + ACC23 (combined exposition):
  J.~Zhu, Y.~Lin, A.~Velasquez, J.~Liu (Stony Brook University /
  Air Force Research Laboratory / University of Colorado Boulder /
  The University of Tokyo).
- RAC: L.~Yuan (Hunan University), H.~Ishii (The University of Tokyo).
"# Byzantine-Resilient-Consensus" 
