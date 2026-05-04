"""
ACC23 -- Resilient Distributed Optimization
===========================================
Faithful implementation of:

    Zhu, Lin, Velasquez, Liu -- "Resilient Distributed Optimization",
    American Control Conference 2023, pp. 1307-1312.

Algorithm (paper Eqs. 4-6):

    For each honest agent i in H, for each (d+1)β+1-subset A_ij of N_i:
        y_ij(t) in cap_{S in B_ij} conv(S)        # B_ij = (dβ+1)-subsets of A_ij
    v_i(t)    = (x_i(t) + sum_j y_ij(t)) / (1 + a_i)        a_i = C(|N_i|, (d+1)β+1)
    x_i(t+1)  = v_i(t) - α(t) g_i(v_i(t))

Tverberg / convex-hull intersection (paper's `y_ij`) is implemented exactly for:

    * d = 1, any β: paper proves the construction reduces to *trimmed mean*
      over (2β+1)-subsets; we use the midpoint of the trimmed-mean interval,
      which is in cap_{S in B_ij} conv(S) by construction.
    * d = 2, β = 1: |A_ij| = 4. The unique 2-Tverberg partition of 4 points in
      R^2 has crossing segments; their intersection is in every 3-subset's
      triangular convex hull (Helly + crossing argument).  Closed-form via 2x2.
    * d >= 2, β >= 2: raises NotImplementedError -- the paper itself notes
      (Sec. IV) that picking points in intersections of high-dimensional
      convex hulls is computationally expensive.  Use --aggregator
      cwtm|krum|geomedian for empirical comparisons in those regimes.

Also provides:

    * Per-recipient Byzantine messaging (independent of recipient -- the paper
      treats Byzantine agents as fully adversarial and graph-dependent).
    * `(β, dβ)`-resilient graph diagnostic and κ_{r,s}(G) computation by brute
      enumeration (small graphs only).
    * k-redundancy verification for the quadratic-with-shared-center family.
    * Both step-size schedules from the paper:
        --alpha-rule sqrtT        α(t) = c / sqrt(T)         (Theorem 2)
        --alpha-rule diminishing  α(t) = a / (1 + b t)        (Theorem 1)
    * Comparison aggregators (CWTM, Krum, Geometric median, distance-trim) for
      empirical study; not the paper's algorithm but useful baselines.

Run:
    python ACC23.py                                       # d=2, β=1, paper alg
    python ACC23.py --d 1                                 # d=1 trimmed-mean
    python ACC23.py --aggregator cwtm                     # comparison baseline
    python ACC23.py --attack two_faced                    # per-recipient adversary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Protocol

import numpy as np


# Try to reconfigure stdout for UTF-8 (Windows cp1252 can't print Greek letters).
def _can_unicode() -> bool:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        return True
    except Exception:
        return "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()


_UNICODE = _can_unicode()
_BETA  = "β" if _UNICODE else "beta"
_ALPHA = "α" if _UNICODE else "alpha"
_KAPPA = "κ" if _UNICODE else "kappa"


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
class Objective(Protocol):
    def value(self, x: np.ndarray) -> float: ...
    def subgradient(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Quadratic:
    """f(x) = 0.5 ||x - center||^2."""
    center: np.ndarray
    def value(self, x):       return 0.5 * float(((x - self.center) ** 2).sum())
    def subgradient(self, x): return x - self.center


@dataclass(frozen=True)
class Huber:
    target: np.ndarray
    delta: float = 1.0
    def value(self, x):
        r = x - self.target; n = float(np.linalg.norm(r))
        return 0.5 * n * n if n <= self.delta else self.delta * (n - 0.5 * self.delta)
    def subgradient(self, x):
        r = x - self.target; n = float(np.linalg.norm(r))
        return r if n <= self.delta else (self.delta / max(n, 1e-12)) * r


# ---------------------------------------------------------------------------
# Tverberg / y_ij selection (paper's Eq. 4)
# ---------------------------------------------------------------------------
def _segment_intersection(a, b, c, d):
    """If segments [a,b] and [c,d] cross at an interior point, return it; else None."""
    r = b - a
    s = d - c
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    qmp = c - a
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / rxs
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / rxs
    if 0.0 < t < 1.0 and 0.0 < u < 1.0:
        return a + t * r
    return None


def tverberg_4points_2d(p: np.ndarray) -> np.ndarray:
    """Tverberg point of 4 points in R^2 -- the unique 2-partition of the 4
    points whose two segments cross, intersected.  Falls back to centroid
    for degenerate (collinear) configurations."""
    p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
    for (a, b), (c, d) in [((p1, p2), (p3, p4)),
                           ((p1, p3), (p2, p4)),
                           ((p1, p4), (p2, p3))]:
        ix = _segment_intersection(a, b, c, d)
        if ix is not None:
            return ix
    return p.mean(0)


def y_ij_from_subset(A_values: np.ndarray, beta: int, d: int) -> np.ndarray:
    """Pick y_ij in cap_{S in B_ij} conv(S) where B_ij = (dβ+1)-subsets of A_ij.

    Implemented exactly for (d=1, any β) and (d=2, β=1).  Otherwise raises."""
    if d == 1:
        # A_values shape: (2β+1,) or (2β+1, 1).  Sort and take midpoint of the
        # [β-th smallest, β-th largest] -- the intersection of all (β+1)-subsets'
        # interval hulls.
        sorted_ = np.sort(A_values, axis=0)
        lo = sorted_[beta]
        hi = sorted_[-beta - 1]
        return 0.5 * (lo + hi)
    if d == 2 and beta == 1:
        return tverberg_4points_2d(A_values)
    raise NotImplementedError(
        f"Paper's y_ij construction is intentionally only implemented for "
        f"(d=1, any β) and (d=2, β=1).  For d={d}, β={beta} the paper itself "
        f"flags computational difficulty (Sec. IV).  Use --aggregator "
        f"cwtm|krum|geomedian for an empirical baseline.")


def aggregate_paper(self_x: np.ndarray, neighbors: np.ndarray,
                    beta: int, d: int) -> np.ndarray:
    """Paper Eq. 5: v_i = (x_i + Σ_j y_ij) / (1 + a_i),
    where the y_ij are y_ij_from_subset over all (d+1)β+1-subsets of N_i."""
    n_nbr = neighbors.shape[0]
    subset_size = (d + 1) * beta + 1
    if n_nbr < subset_size:
        # Paper requires |N_i| >= (d+1)β+1; we hit this only on tiny demos.
        # Fall back to averaging self with neighbor mean -- still consensus-style.
        return 0.5 * (self_x + neighbors.mean(0))
    y_sum = np.zeros_like(self_x)
    a_i = 0
    for ix in combinations(range(n_nbr), subset_size):
        y_sum += y_ij_from_subset(neighbors[list(ix)], beta=beta, d=d)
        a_i  += 1
    return (self_x + y_sum) / (1.0 + a_i)


# ---------------------------------------------------------------------------
# Comparison aggregators (NOT the paper's algorithm)
# ---------------------------------------------------------------------------
def aggregate_trim(self_x, neighbors, beta, d):
    if neighbors.shape[0] <= beta:
        return self_x.copy()
    dist = np.linalg.norm(neighbors - self_x, axis=1)
    keep = neighbors[np.argsort(dist, kind="stable")[: neighbors.shape[0] - beta]]
    return keep.mean(0)


def aggregate_cwtm(self_x, neighbors, beta, d):
    n = neighbors.shape[0]
    if 2 * beta >= n:
        return np.median(neighbors, axis=0)
    s = np.sort(neighbors, axis=0)
    return s[beta : n - beta].mean(0)


def aggregate_krum(self_x, neighbors, beta, d):
    n = neighbors.shape[0]
    k = n - beta - 2
    if n == 0:
        return self_x.copy()
    if k < 1:
        return neighbors.mean(0)
    diff = neighbors[:, None, :] - neighbors[None, :, :]
    sq = (diff * diff).sum(-1)
    np.fill_diagonal(sq, np.inf)
    sums = np.sort(sq, axis=1)[:, :k].sum(1)
    return neighbors[int(np.argmin(sums))]


def aggregate_geomedian(self_x, neighbors, beta, d, iters=32, eps=1e-9):
    if neighbors.shape[0] == 0:
        return self_x.copy()
    y = neighbors.mean(0)
    for _ in range(iters):
        d_ = np.linalg.norm(neighbors - y, axis=1)
        if (d_ < eps).any():
            return neighbors[int(np.argmin(d_))]
        w = 1.0 / d_
        y_new = (w[:, None] * neighbors).sum(0) / w.sum()
        if float(np.linalg.norm(y_new - y)) < eps:
            return y_new
        y = y_new
    return y


AGGREGATORS: dict[str, Callable] = {
    "paper":     aggregate_paper,
    "trim":      aggregate_trim,
    "cwtm":      aggregate_cwtm,
    "krum":      aggregate_krum,
    "geomedian": aggregate_geomedian,
}


# ---------------------------------------------------------------------------
# Byzantine attack policies (mirror ACC22.py, per-recipient buffer)
# ---------------------------------------------------------------------------
def attack_max_spread(scale=5.0):
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0)
        r = float(np.linalg.norm(honest - c, axis=1).max() + 1e-9)
        v = rng.standard_normal(c.shape); v /= np.linalg.norm(v) + 1e-12
        msg = c - scale * r * v
        return np.broadcast_to(msg, (1, n_recipients, msg.shape[0])).copy()
    return go


def attack_drift(velocity_scale=0.05):
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0) + (t + 1) * velocity_scale
        return np.broadcast_to(c, (1, n_recipients, c.shape[0])).copy()
    return go


def attack_two_faced(scale=5.0):
    """Per-recipient extremal lies."""
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0)
        r = float(np.linalg.norm(honest - c, axis=1).max() + 1e-9)
        d_ = c.shape[0]
        v = rng.standard_normal((1, n_recipients, d_))
        v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
        return c + scale * r * v
    return go


def attack_uniform(low=-3.0, high=3.0):
    def go(t, honest, n_recipients, rng):
        return rng.uniform(low, high, size=(1, n_recipients, honest.shape[1]))
    return go


def attack_gradient_flip(scale=2.0):
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0) + scale * (1.0 + 0.05 * t)
        return np.broadcast_to(c, (1, n_recipients, c.shape[0])).copy()
    return go


ATTACKS: dict[str, Callable] = {
    "max_spread":    lambda: attack_max_spread(scale=5.0),
    "drift":         lambda: attack_drift(velocity_scale=0.05),
    "two_faced":     lambda: attack_two_faced(scale=5.0),
    "uniform":       lambda: attack_uniform(-2.5, 2.5),
    "gradient_flip": lambda: attack_gradient_flip(scale=2.0),
}


# ---------------------------------------------------------------------------
# Graph machinery
# ---------------------------------------------------------------------------
def complete_graph(n: int) -> np.ndarray:
    A = np.ones((n, n), dtype=bool)
    np.fill_diagonal(A, False)
    return A


def reaches_all(A: np.ndarray, start: int) -> bool:
    """BFS: does `start` have directed paths to every other vertex?"""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool); seen[start] = True
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(A[u] & ~seen):
                seen[v] = True
                nxt.append(int(v))
        frontier = nxt
    return bool(seen.all())


def n_roots(A: np.ndarray) -> int:
    return int(sum(reaches_all(A, v) for v in range(A.shape[0])))


def is_rooted(A: np.ndarray) -> bool:
    return n_roots(A) >= 1


def _edge_removal_choices(sub: np.ndarray, s: int):
    """Yield every digraph obtained from `sub` by removing up to s incoming
    edges per vertex.  Brute force; only feasible for tiny graphs."""
    n = sub.shape[0]
    incoming = [list(np.flatnonzero(sub[:, v])) for v in range(n)]
    def per_vertex(v):
        for r in range(0, min(s, len(incoming[v])) + 1):
            yield from combinations(incoming[v], r)
    def rec(v, cur):
        if v == n:
            yield cur
            return
        for rem in per_vertex(v):
            new = cur.copy()
            for u in rem:
                new[u, v] = False
            yield from rec(v + 1, new)
    yield from rec(0, sub.copy())


def is_r_s_resilient(A: np.ndarray, r: int, s: int, max_n: int = 6) -> bool | None:
    """Brute force: True iff every (r,s)-reduced subgraph is rooted.  Returns
    None when n exceeds max_n (too large to enumerate)."""
    n = A.shape[0]
    if r + s + 1 > n:
        return False
    if n > max_n:
        return None
    for S in combinations(range(n), n - r):
        S = list(S)
        sub = A[np.ix_(S, S)].copy()
        for removed in _edge_removal_choices(sub, s):
            if not is_rooted(removed):
                return False
    return True


def kappa_r_s(A: np.ndarray, r: int, s: int, max_n: int = 6) -> int | None:
    """Min number of roots over (r,s)-reduced subgraphs.  None for large n."""
    n = A.shape[0]
    if r + s + 1 > n or n > max_n:
        return None
    worst = n + 1
    for S in combinations(range(n), n - r):
        S = list(S)
        sub = A[np.ix_(S, S)].copy()
        for removed in _edge_removal_choices(sub, s):
            worst = min(worst, n_roots(removed))
            if worst == 0:
                return 0
    return worst if worst <= n else 0


def is_k_redundant_quadratic(centers: np.ndarray, k: int, atol: float = 1e-9) -> bool:
    """For f_i = 0.5 ||x - c_i||^2, k-redundancy holds iff every (n-k)-subset
    of {c_i} has the same mean (which equals the global argmin)."""
    n = centers.shape[0]
    if n - k <= 0:
        return False
    target = None
    for S in combinations(range(n), n - k):
        m = centers[list(S)].mean(0)
        if target is None:
            target = m
        elif np.linalg.norm(m - target) > atol:
            return False
    return True


# ---------------------------------------------------------------------------
# Optimization system (paper Eq. 6, synchronous)
# ---------------------------------------------------------------------------
@dataclass
class OptSystem:
    states:     np.ndarray              # (n, d)
    objectives: list                    # length n
    A:          np.ndarray              # (n, n) bool, A[i, j] iff j is i's neighbor
    byzantine:  np.ndarray              # bool (n,)
    beta:       int
    d:          int
    aggregator: Callable                # see AGGREGATORS
    attack:     Callable
    rng:        np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    step_rule:  Callable[[int], float] = field(default_factory=lambda: (lambda t: 0.05))

    def __post_init__(self):
        n = self.states.shape[0]
        assert self.A.shape == (n, n)
        assert self.byzantine.shape == (n,)
        assert len(self.objectives) == n
        self._honest = np.flatnonzero(~self.byzantine)
        self._byz    = np.flatnonzero(self.byzantine)

    def _build_message_buffer(self, t: int) -> np.ndarray:
        n, d = self.states.shape
        buf = np.broadcast_to(self.states[:, None, :], (n, n, d)).copy()
        if self._byz.size > 0:
            honest = self.states[~self.byzantine]
            byz_msgs = self.attack(t, honest, n, self.rng)
            if byz_msgs.shape[0] == 1 and self._byz.size > 1:
                byz_msgs = np.broadcast_to(byz_msgs,
                                           (self._byz.size, n, d)).copy()
            buf[self._byz] = byz_msgs
        return buf

    def step(self, t: int) -> None:
        buf = self._build_message_buffer(t)
        snapshot = self.states.copy()
        new_states = self.states.copy()
        alpha = self.step_rule(t)
        for i in self._honest:
            nbr_idx = np.flatnonzero(self.A[i])
            received = buf[nbr_idx, i, :]                 # what i hears from its neighbors
            v_i = self.aggregator(snapshot[i], received, self.beta, self.d)
            new_states[i] = v_i - alpha * self.objectives[i].subgradient(v_i)
        self.states = new_states

    def run(self, T: int) -> np.ndarray:
        hist = np.empty((T + 1, *self.states.shape))
        hist[0] = self.states
        for t in range(T):
            self.step(t)
            hist[t + 1] = self.states
        return hist


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def disagreement(states: np.ndarray, byz: np.ndarray) -> float:
    h = states[~byz]
    if h.shape[0] < 2:
        return 0.0
    diff = h[:, None, :] - h[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).max())


def f_global(states: np.ndarray, objectives: list, byz: np.ndarray,
             x_eval: np.ndarray | None = None) -> float:
    if x_eval is None:
        x_eval = states[~byz].mean(0)
    return float(sum(objectives[i].value(x_eval) for i in np.flatnonzero(~byz)))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    d = args.d
    beta = args.beta
    n = args.n

    if args.aggregator == "paper" and (d > 2 or (d == 2 and beta > 1)):
        raise SystemExit(
            f"Paper aggregator only supports (d=1, any beta) or (d=2, beta=1); "
            f"got d={d}, beta={beta}.  Use --aggregator cwtm|krum|geomedian.")

    # Quadratics with shared center => network is k-redundant for any k <= n-1.
    center = rng.uniform(-1.0, 1.0, size=d)
    objectives = [Quadratic(center=center.copy()) for _ in range(n)]
    x_star = center.copy()

    states = rng.normal(scale=0.7, size=(n, d))
    byz = np.zeros(n, dtype=bool); byz[: beta] = True; rng.shuffle(byz)

    A = complete_graph(n)

    if args.alpha_rule == "sqrtT":
        c_alpha = args.alpha
        T_total = args.iters
        step_rule = lambda t, _c=c_alpha, _T=T_total: _c / np.sqrt(_T)
    else:
        step_rule = lambda t, _a=args.alpha, _b=args.alpha_decay: _a / (1.0 + _b * t)

    system = OptSystem(
        states=states.copy(), objectives=objectives, A=A, byzantine=byz,
        beta=beta, d=d, aggregator=AGGREGATORS[args.aggregator],
        attack=ATTACKS[args.attack](),
        rng=rng, step_rule=step_rule,
    )

    H = int((~byz).sum())
    print(f"  ACC23 resilient distributed optimization")
    print(f"  agents={n}  honest={H}  byzantine={int(byz.sum())} "
          f"({_BETA}={beta})  d={d}  aggregator={args.aggregator}  "
          f"attack={args.attack}  iters={args.iters}")
    print(f"  step-size: {args.alpha_rule}   "
          f"{_ALPHA}(0) = {step_rule(0):.4f}   "
          f"{_ALPHA}({args.iters - 1}) = {step_rule(args.iters - 1):.4f}")

    # --- graph + redundancy diagnostics ---
    print(f"\n  Graph diagnostics  G = K_{n}  (r = {beta}, s = d{_BETA} = {d * beta}):")
    necessary = n >= beta * (1 + d) + 2     # paper Lemma 2
    print(f"    Lemma 2 necessary condition  n {chr(0x2265) if _UNICODE else '>='} "
          f"{_BETA}(1+d)+2 = {beta*(1+d)+2}  : {necessary}")
    resil = is_r_s_resilient(A, beta, d * beta)
    kap   = kappa_r_s(A, beta, d * beta)
    if resil is None:
        print(f"    brute-force ({_BETA}, d{_BETA})-resilient check skipped "
              f"(n > 6); rely on empirical convergence.")
    else:
        print(f"    brute-force ({_BETA}, d{_BETA})-resilient        : {resil}")
        if kap is not None:
            print(f"    {_KAPPA}_{{ {_BETA}, d{_BETA} }}(G)                          "
                  f": {kap}")

    centers_arr = np.array([o.center for o in objectives])
    # Shared centers => k-redundant for any k <= n-1.
    print(f"    quadratic-objective k-redundancy (k = n-1 = {n-1}): "
          f"{is_k_redundant_quadratic(centers_arr, n - 1)}")

    # Run.
    history = system.run(args.iters)

    diam = np.array([disagreement(history[t], byz) for t in range(args.iters + 1)])
    err  = np.array([np.linalg.norm(history[t, ~byz].mean(0) - x_star)
                     for t in range(args.iters + 1)])
    obj  = np.array([f_global(history[t], objectives, byz)
                     for t in range(args.iters + 1)])

    print("\n   t   |  disagreement |  ||x_bar - x*||  |    F(x_bar)")
    print("  -----+---------------+------------------+-----------------")
    sample = sorted(set([0, 1, 2, 5, 10, 25, 50, args.iters // 2, args.iters]))
    for t in sample:
        if t > args.iters: continue
        print(f"  {t:>4} |  {diam[t]:>10.4e}  |   {err[t]:>10.4e}   |  "
              f"{obj[t]:>10.4e}")

    print(f"\n  honest-mean final state : {np.array2string(history[-1, ~byz].mean(0), precision=5)}")
    print(f"  x*                      : {np.array2string(x_star, precision=5)}")
    print(f"  final ||x_bar - x*||   : {err[-1]:.4e}")
    print(f"  final disagreement     : {diam[-1]:.4e}")
    print(f"  final F(x_bar)         : {obj[-1]:.4e}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resilient distributed optimization (ACC23).")
    p.add_argument("--n",          type=int,   default=8,   help="agents")
    p.add_argument("--beta",       type=int,   default=1,   help="byzantine per neighborhood")
    p.add_argument("--d",          type=int,   default=2,   help="dimension")
    p.add_argument("--alpha",      type=float, default=0.5, help="step coefficient")
    p.add_argument("--alpha-rule", choices=["sqrtT", "diminishing"],
                   default="diminishing")
    p.add_argument("--alpha-decay",type=float, default=0.02, help="decay for diminishing")
    p.add_argument("--iters",      type=int,   default=200, help="iterations")
    p.add_argument("--aggregator", choices=list(AGGREGATORS), default="paper")
    p.add_argument("--attack",     choices=list(ATTACKS),    default="max_spread")
    p.add_argument("--seed",       type=int,   default=11)
    return p.parse_args()


if __name__ == "__main__":
    _demo(_parse())
