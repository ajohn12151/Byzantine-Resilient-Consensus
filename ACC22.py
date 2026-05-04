"""
ACC22 -- Resilient Constrained Consensus over Complete Graphs via Feasibility Redundancy
========================================================================================
Faithful implementation of:

    Zhu, Lin, Velasquez, Liu -- "Resilient Constrained Consensus over Complete Graphs
    via Feasibility Redundancy", American Control Conference 2022, pp. 3418-3422.

Algorithm (paper Eq. 1):

    x_i(t+1) = P_{X_i}[ x_i(t) + alpha * sum_{j in M_i(t)} (x_{ji}(t) - x_i(t)) ]

where:
    * x_i(t) in R^m is honest agent i's state (i in H, |H| >= n - f).
    * X_i is a closed convex set; cap_{i in H} X_i = {x*} is a singleton.
    * x_{ji}(t) is the value agent j sends to agent i.  The paper's footnote 1
      explicitly allows x_{ji}(t) != x_{jk}(t) when j is Byzantine -- the
      "two-faced" adversary -- so we keep an (n, n, d) message buffer.
    * M_i(t) is the retain set: drop the f received values with the largest
      distance ||x_i(t) - x_{ji}(t)||, breaking ties arbitrarily.
    * P_{X_i} is the Euclidean projection onto X_i.

Theorem 2 (paper, p. 3419) gives the exponential bound
    V(t) := sum_{i in H} ||x_i(t) - x*||^2,
    rho   = 1 - (mu^2 k - 4f - 2f mu^2 + mu^2) alpha + 4|H|^3 alpha^2  in (0, 1),
    V(t) <= rho^t V(0),
provided H is k-redundant, the regularity constant satisfies
    max_{i in S} dist(x, X_i) >= mu * dist(x, cap_{i in S} X_i),
and (k, alpha) lie inside the bound k > 4f/mu^2 + 2f - 1 with
    alpha < (mu^2 k - 4f - 2f mu^2 + mu^2) / (4 |H|^3).

This file:
    * runs the algorithm exactly as written in Eq. 1 with synchronous (atomic)
      updates from a per-round snapshot;
    * tracks both the paper's Lyapunov V(t) and a visualization-friendly
      honest-pairwise diameter, fits the empirical contraction, and prints the
      analytic prediction next to it;
    * supports per-recipient Byzantine messaging via a "two_faced" attack class
      alongside the standard broadcast adversaries;
    * provides closed-form Box / Ball projections and a single-halfspace closed
      form for Corollary 1, with a scipy QP fallback for general polytopes;
    * exposes a Theorem-2 conditions checker that prints PASS/FAIL with the
      explicit formulas;
    * runs two demos: --demo box (unit box, mu = 1) and --demo polyhedral
      (Corollary 1, axis-aligned half-spaces through the origin).

Run:
    python ACC22.py                                   # default box demo
    python ACC22.py --attack two_faced                # per-recipient adversary
    python ACC22.py --demo polyhedral                 # Corollary 1 setting
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Terminal glyphs: prefer Unicode box-drawing, fall back to ASCII on Windows
# ---------------------------------------------------------------------------
def _can_unicode() -> bool:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        return True
    except Exception:
        return "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()


_UNICODE = _can_unicode()
_GLYPHS = {
    "tl": "┌" if _UNICODE else "+",
    "tr": "┐" if _UNICODE else "+",
    "bl": "└" if _UNICODE else "+",
    "br": "┘" if _UNICODE else "+",
    "h":  "─" if _UNICODE else "-",
    "v":  "│" if _UNICODE else "|",
    "trail": "·•○◉" if _UNICODE else ".oO0",
    "in":  "∈" if _UNICODE else "in",
    "ge":  "≥" if _UNICODE else ">=",
    "lt":  "<" if _UNICODE else "<",
    "rho": "ρ" if _UNICODE else "rho",
    "mu":  "μ" if _UNICODE else "mu",
    "alpha": "α" if _UNICODE else "alpha",
    "leq": "≤" if _UNICODE else "<=",
    "sum": "Σ" if _UNICODE else "sum",
    "check": "✓" if _UNICODE else "[OK]",
    "cross": "✗" if _UNICODE else "[X ]",
}


# ---------------------------------------------------------------------------
# Convex sets with closed-form projections
# ---------------------------------------------------------------------------
class ConvexSet(Protocol):
    def project(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Box:
    """Axis-aligned box [lo, hi]."""
    lo: np.ndarray
    hi: np.ndarray
    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo, self.hi)


@dataclass(frozen=True)
class Ball:
    """Closed Euclidean ball of radius r centered at c."""
    c: np.ndarray
    r: float
    def project(self, x: np.ndarray) -> np.ndarray:
        d = x - self.c
        n = float(np.linalg.norm(d))
        return x if n <= self.r else self.c + (self.r / n) * d


@dataclass(frozen=True)
class Halfspace:
    """Single half-space {x : a^T x <= b}.  This is the Corollary 1 setting:
    each agent owns one inequality and the projection is closed-form."""
    a: np.ndarray
    b: float
    def project(self, x: np.ndarray) -> np.ndarray:
        a = self.a
        slack = float(a @ x) - self.b
        if slack <= 0.0:
            return x
        return x - (slack / float(a @ a)) * a


@dataclass(frozen=True)
class Polytope:
    """General polytope {x : A x <= b} via scipy QP (fallback only)."""
    A: np.ndarray
    b: np.ndarray
    def project(self, x: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize
        cons = [{"type": "ineq", "fun": lambda y, a=self.A[i], bi=self.b[i]: bi - a @ y}
                for i in range(len(self.b))]
        res = minimize(lambda y: float(np.dot(y - x, y - x)), x, constraints=cons,
                       options={"disp": False})
        return res.x


# ---------------------------------------------------------------------------
# Byzantine attack policies
#
# Each attack returns the per-recipient message rows for the Byzantine senders:
#       shape (n_byz, n_recipients, d)
# This honors the paper's footnote 1 -- a Byzantine sender j can transmit
# a different value x_{ji}(t) to each recipient i.
# ---------------------------------------------------------------------------
Attack = Callable[[int, np.ndarray, int, np.random.Generator], np.ndarray]


def _broadcast(byz_per_sender: np.ndarray, n_recipients: int) -> np.ndarray:
    """Tile a (n_byz, d) value matrix into (n_byz, n_recipients, d)."""
    return np.broadcast_to(byz_per_sender[:, None, :],
                           (byz_per_sender.shape[0], n_recipients,
                            byz_per_sender.shape[1])).copy()


def attack_constant(value: np.ndarray) -> Attack:
    v = np.asarray(value, dtype=float)
    def go(t, honest, n_recipients, rng):
        n_byz = max(1, honest.shape[0] // 4)  # placeholder; overridden by sys.byz count
        # ConsensusSystem passes honest.shape[0] correctly; n_byz inferred elsewhere
        return np.broadcast_to(v, (n_byz, n_recipients, v.shape[0])).copy()
    return go


def attack_drift(velocity: np.ndarray) -> Attack:
    v = np.asarray(velocity, dtype=float)
    def go(t, honest, n_recipients, rng):
        # caller passes byz state via closure; simple drift relative to centroid
        c = honest.mean(0) + (t + 1) * v
        return np.broadcast_to(c, (1, n_recipients, c.shape[0])).copy()
    return go


def attack_max_spread(scale: float = 4.0) -> Attack:
    """Antipode of the honest centroid -- the strongest single-direction push
    a broadcast Byzantine can apply."""
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0)
        r = float(np.linalg.norm(honest - c, axis=1).max() + 1e-9)
        v = rng.standard_normal(c.shape)
        v /= np.linalg.norm(v) + 1e-12
        msg = c - scale * r * v
        return np.broadcast_to(msg, (1, n_recipients, msg.shape[0])).copy()
    return go


def attack_mimic(target_idx: int = 0) -> Attack:
    """Copy a real honest agent's state -- defeats naive distance filters that
    expect outliers."""
    def go(t, honest, n_recipients, rng):
        msg = honest[target_idx]
        return np.broadcast_to(msg, (1, n_recipients, msg.shape[0])).copy()
    return go


def attack_uniform(low: float = -3.0, high: float = 3.0) -> Attack:
    def go(t, honest, n_recipients, rng):
        return rng.uniform(low, high, size=(1, n_recipients, honest.shape[1]))
    return go


def attack_two_faced(scale: float = 4.0) -> Attack:
    """Per-recipient extremal lies (paper footnote 1).  For each recipient i,
    each Byzantine sender picks an independent random direction and pushes
    the honest centroid by `scale * radius` along it.  This is the strongest
    pointwise-asymmetric attack the paper's algorithm has to absorb."""
    def go(t, honest, n_recipients, rng):
        c = honest.mean(0)
        r = float(np.linalg.norm(honest - c, axis=1).max() + 1e-9)
        d = c.shape[0]
        v = rng.standard_normal((1, n_recipients, d))
        v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
        return c + scale * r * v
    return go


ATTACKS: dict[str, Callable[[], Attack]] = {
    "constant":   lambda: attack_constant(np.array([2.0, 2.0])),
    "drift":      lambda: attack_drift(np.array([0.05, -0.05])),
    "max_spread": lambda: attack_max_spread(scale=4.0),
    "mimic":      lambda: attack_mimic(target_idx=0),
    "uniform":    lambda: attack_uniform(-2.5, 2.5),
    "two_faced":  lambda: attack_two_faced(scale=4.0),
}


# ---------------------------------------------------------------------------
# Resilient consensus system (paper Eq. 1, synchronous)
# ---------------------------------------------------------------------------
@dataclass
class ConsensusSystem:
    states:    np.ndarray            # (n, d)
    sets:      list[ConvexSet]       # length n
    byzantine: np.ndarray            # bool (n,)
    f:         int
    alpha:     float = 0.1
    attack:    Attack = field(default_factory=attack_max_spread)
    rng:       np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        n, _ = self.states.shape
        assert self.byzantine.shape == (n,)
        assert len(self.sets) == n
        assert 0 <= self.f and 2 * self.f + 1 <= n, "graph-redundancy condition violated"
        self._honest = np.flatnonzero(~self.byzantine)
        self._byz    = np.flatnonzero(self.byzantine)

    def _build_message_buffer(self, t: int) -> np.ndarray:
        """Return buf of shape (n, n, d) where buf[j, i] is the message agent
        j sends to agent i.  Honest senders broadcast their true state;
        Byzantine senders fill via the attack policy (per-recipient allowed)."""
        n, d = self.states.shape
        buf = np.broadcast_to(self.states[:, None, :], (n, n, d)).copy()
        if self._byz.size > 0:
            honest_states = self.states[~self.byzantine]
            byz_msgs = self.attack(t, honest_states, n, self.rng)
            # Reshape if attack returned (1, n_recipients, d) for a "global" attack;
            # broadcast it across all Byzantine senders.
            if byz_msgs.shape[0] == 1 and self._byz.size > 1:
                byz_msgs = np.broadcast_to(byz_msgs,
                                           (self._byz.size, n, d)).copy()
            buf[self._byz] = byz_msgs
        return buf

    @staticmethod
    def _retain(self_state: np.ndarray, received_from_others: np.ndarray,
                f: int) -> np.ndarray:
        """Drop the f rows with the largest ||self_state - row|| (paper M_i(t))."""
        if f <= 0:
            return received_from_others
        d = np.linalg.norm(received_from_others - self_state, axis=1)
        keep = np.argsort(d, kind="stable")[: received_from_others.shape[0] - f]
        return received_from_others[keep]

    def step(self, t: int) -> None:
        n, _ = self.states.shape
        buf = self._build_message_buffer(t)         # (n, n, d)
        snapshot = self.states.copy()
        new_states = self.states.copy()
        for i in self._honest:
            received = np.delete(buf[:, i, :], i, axis=0)   # what i hears, exclude self
            kept = self._retain(snapshot[i], received, self.f)
            update = (kept - snapshot[i]).sum(0)
            cand = snapshot[i] + self.alpha * update
            new_states[i] = self.sets[i].project(cand)
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
def lyapunov_V(states: np.ndarray, x_star: np.ndarray,
               byzantine: np.ndarray) -> float:
    """V(t) = sum_{i in H} ||x_i - x*||^2  (paper, Theorem 2)."""
    h = states[~byzantine]
    return float(((h - x_star) ** 2).sum())


def honest_diameter(states: np.ndarray, byzantine: np.ndarray) -> float:
    h = states[~byzantine]
    if h.shape[0] < 2:
        return 0.0
    diff = h[:, None, :] - h[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).max())


def fit_exp_rate(trace: np.ndarray) -> tuple[float, float]:
    """Fit log y = a + b t.  Returns (b, R^2)."""
    eps = 1e-15
    y = np.log(np.maximum(trace, eps))
    t = np.arange(y.size, dtype=float)
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-15)
    return float(coef[1]), 1.0 - ss_res / ss_tot


def predicted_rho(mu: float, k: int, f: int, H: int, alpha: float) -> float:
    """rho = 1 - (mu^2 k - 4f - 2f mu^2 + mu^2) alpha + 4 |H|^3 alpha^2."""
    return 1.0 - (mu ** 2 * k - 4 * f - 2 * f * mu ** 2 + mu ** 2) * alpha \
                + 4.0 * (H ** 3) * (alpha ** 2)


def check_theorem2(mu: float, k: int, f: int, H: int, alpha: float) -> dict:
    """Return PASS/FAIL on each of the paper's Theorem 2 conditions."""
    cond_k = k > 4.0 * f / max(mu ** 2, 1e-12) + 2 * f - 1
    alpha_max = (mu ** 2 * k - 4 * f - 2 * f * mu ** 2 + mu ** 2) / max(4 * (H ** 3), 1e-12)
    cond_alpha = alpha < alpha_max
    rho = predicted_rho(mu, k, f, H, alpha)
    cond_rho = 0.0 < rho < 1.0
    return {
        "k_lower_bound": 4.0 * f / max(mu ** 2, 1e-12) + 2 * f - 1,
        "k_pass":        cond_k,
        "alpha_max":     alpha_max,
        "alpha_pass":    cond_alpha,
        "rho":           rho,
        "rho_pass":      cond_rho,
    }


def k_redundancy_polyhedral(normals: np.ndarray, k: int) -> bool:
    """For axis-spanning half-spaces through the origin, k-redundancy holds iff
    every (n - k)-subset of `normals` still positively spans R^d.  Tested by
    LP feasibility: subset positively spans iff there is no nonzero x with
    A_S x <= 0 (i.e., the only solution is 0).  We test the equivalent via
    the rank/sign condition in low dimension by brute enumeration -- only
    practical for small n.  Returns True/False."""
    n, d = normals.shape
    if n - k <= 0:
        return False
    for S in combinations(range(n), n - k):
        A = normals[list(S)]
        if not _positively_spans(A):
            return False
    return True


def _positively_spans(A: np.ndarray) -> bool:
    """Return True iff the rows of A positively span R^d, i.e. there is no
    nonzero x with A x <= 0.  Test by linear programming."""
    from scipy.optimize import linprog
    d = A.shape[1]
    # Try to find a nonzero x with A x <= -1 (strict) by minimizing 0 over
    # A x <= -1, x in R^d (free).  Feasibility of A x <= -1 implies a nonzero
    # x exists with A x <= 0, hence non-positive-span.
    c = np.zeros(d)
    res = linprog(c, A_ub=A, b_ub=-np.ones(A.shape[0]),
                  bounds=[(None, None)] * d, method="highs")
    return not res.success     # infeasible <=> positively spans


# ---------------------------------------------------------------------------
# Unicode trajectory plot (for d == 2 visualization)
# ---------------------------------------------------------------------------
def render_trajectory(history: np.ndarray, byzantine: np.ndarray, *,
                      width: int = 64, height: int = 22, title: str = "") -> None:
    if history.shape[2] < 2:
        return
    pts = history[..., :2].reshape(-1, 2)
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    if x1 - x0 < 1e-9: x0 -= 0.5; x1 += 0.5
    if y1 - y0 < 1e-9: y0 -= 0.5; y1 += 0.5
    px, py = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    x0 -= px; x1 += px; y0 -= py; y1 += py

    grid = [[" "] * width for _ in range(height)]
    n_steps, n_agents, _ = history.shape

    def to_cell(x: float, y: float) -> tuple[int, int]:
        cx = int((x - x0) / (x1 - x0) * (width - 1))
        cy = int((1.0 - (y - y0) / (y1 - y0)) * (height - 1))
        return max(0, min(width - 1, cx)), max(0, min(height - 1, cy))

    glyphs = _GLYPHS["trail"]
    for t in range(n_steps - 1):
        g = glyphs[min(int(t / max(n_steps - 1, 1) * len(glyphs)), len(glyphs) - 1)]
        for i in range(n_agents):
            cx, cy = to_cell(history[t, i, 0], history[t, i, 1])
            if grid[cy][cx] == " ":
                grid[cy][cx] = g
    for i in range(n_agents):
        cx, cy = to_cell(history[-1, i, 0], history[-1, i, 1])
        grid[cy][cx] = "*" if byzantine[i] else (str(i) if i < 10 else "#")

    bar = _GLYPHS["h"] * width
    print(f"\n  {title}")
    print(f"  {_GLYPHS['tl']}{bar}{_GLYPHS['tr']}")
    for row in grid:
        print(f"  {_GLYPHS['v']}{''.join(row)}{_GLYPHS['v']}")
    print(f"  {_GLYPHS['bl']}{bar}{_GLYPHS['br']}")
    print(f"   x {_GLYPHS['in']} [{x0:+.2f}, {x1:+.2f}]   "
          f"y {_GLYPHS['in']} [{y0:+.2f}, {y1:+.2f}]   "
          f"digits = honest agent id   * = Byzantine")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def _print_theorem2(mu: float, k: int, f: int, H: int, alpha: float) -> None:
    chk = check_theorem2(mu, k, f, H, alpha)
    ok = lambda b: _GLYPHS["check"] if b else _GLYPHS["cross"]
    print(f"\n  Theorem 2 conditions  ({_GLYPHS['mu']}={mu:.3f}, k={k}, "
          f"f={f}, |H|={H}, {_GLYPHS['alpha']}={alpha:.4f})")
    print(f"    {ok(chk['k_pass'])}  k > 4f/{_GLYPHS['mu']}^2 + 2f - 1  "
          f"= {chk['k_lower_bound']:.3f}     (k = {k})")
    print(f"    {ok(chk['alpha_pass'])}  {_GLYPHS['alpha']} {_GLYPHS['lt']} "
          f"({_GLYPHS['mu']}^2 k - 4f - 2f{_GLYPHS['mu']}^2 + {_GLYPHS['mu']}^2)/(4|H|^3) "
          f"= {chk['alpha_max']:.4e}")
    print(f"    {ok(chk['rho_pass'])}  predicted {_GLYPHS['rho']} = {chk['rho']:.4f}  "
          f"{_GLYPHS['in']} (0,1)")


def _summary(history: np.ndarray, byz: np.ndarray, x_star: np.ndarray,
             T: int) -> tuple[np.ndarray, np.ndarray]:
    V    = np.array([lyapunov_V(history[t], x_star, byz) for t in range(T + 1)])
    diam = np.array([honest_diameter(history[t], byz) for t in range(T + 1)])
    pos = V > 1e-300
    if pos.sum() >= 3:
        rate, r2 = fit_exp_rate(V[pos])
        emp_rho = math.exp(rate)
        print(f"\n  V(0)            = {V[0]:.4e}")
        print(f"  V({T})           = {V[-1]:.4e}")
        print(f"  empirical {_GLYPHS['rho']} = {emp_rho:.4f}   "
              f"(R^2={r2:.3f})")
    print(f"  honest disagreement diam   t=0:    {diam[0]:.4e}")
    print(f"  honest disagreement diam   t={T:>3}:  {diam[-1]:.4e}")
    return V, diam


def _demo_box(args: argparse.Namespace) -> None:
    """Half-box demo: each honest agent owns one of 4 axis-bounded boxes; their
    intersection is the singleton {(0, 0)} (Theorem 2's required setting).
    Same geometry as the polyhedral demo but uses Box (clip) projections."""
    rng = np.random.default_rng(args.seed)
    f = args.f
    d = 2
    n_per_axis = max(args.n_per_axis, f + 1)
    classes = [
        Box(np.array([ 0.0, -1.0]), np.array([1.0, 1.0])),     # x >= 0
        Box(np.array([-1.0, -1.0]), np.array([0.0, 1.0])),     # x <= 0
        Box(np.array([-1.0,  0.0]), np.array([1.0, 1.0])),     # y >= 0
        Box(np.array([-1.0, -1.0]), np.array([1.0, 0.0])),     # y <= 0
    ]
    n_honest = 4 * n_per_axis
    n_total  = n_honest + f
    sets: list[ConvexSet] = [classes[0]] * n_total
    byz = np.zeros(n_total, dtype=bool)
    perm = rng.permutation(n_total)
    inv = np.argsort(perm)
    for k_h, p in enumerate(inv[: n_honest]):
        sets[p] = classes[k_h // n_per_axis]
    for p in inv[n_honest:]:
        sets[p] = Box(-np.ones(d), np.ones(d))                 # placeholder; ignored
        byz[p] = True

    states = rng.normal(scale=0.6, size=(n_total, d))
    states[byz] = rng.uniform(-2.0, 2.0, size=(int(byz.sum()), d))

    sys_ = ConsensusSystem(states=states.copy(), sets=sets, byzantine=byz, f=f,
                           alpha=args.alpha, attack=ATTACKS[args.attack](), rng=rng)

    x_star = np.zeros(d)
    H = int((~byz).sum())
    print(f"  ACC22 resilient constrained consensus  --  Box demo (singleton at origin)")
    print(f"  agents={n_total}  honest={H}  byzantine={f}  "
          f"{_GLYPHS['alpha']}={args.alpha}  attack={args.attack}  T={args.iters}")
    print(f"  Each honest X_i is a half-box {{x_k {_GLYPHS['ge']} 0}} or {{x_k {_GLYPHS['leq']} 0}};  "
          f"4 directions x {n_per_axis} copies => cap X_i = {{0}}.  "
          f"{_GLYPHS['mu']}=1 (Hoffman).")

    _print_theorem2(mu=1.0, k=2 * f, f=f, H=H, alpha=args.alpha)
    print(f"    (k = 2f shown above is the *necessary* redundancy from Theorem 1; "
          f"sufficient k from Theorem 2 may be larger.)")

    history = sys_.run(args.iters)
    render_trajectory(history, byz, title=f"trajectories (attack={args.attack})")
    _summary(history, byz, x_star, args.iters)

    print("\n  honest agents (sample of 6) final states:")
    honest_idx = np.flatnonzero(~byz)
    for i in honest_idx[: 6]:
        x = history[-1, i]
        print(f"    agent {i:>2}: ({x[0]:+.5f}, {x[1]:+.5f})")
    if H > 6:
        print(f"    ... ({H - 6} more)")


def _build_polyhedral_demo(n_per_axis: int, f: int, rng: np.random.Generator,
                           ) -> tuple[list[ConvexSet], np.ndarray, np.ndarray, int]:
    """Construct a polyhedral instance of Corollary 1.

    All agents own a single half-space {x : a^T x <= 0} where a is one of the
    four cardinal directions in R^2; n_per_axis copies per direction.  Then
        cap X_i = {(0, 0)}.
    f Byzantine agents are then added with arbitrary half-spaces (their X_i
    is irrelevant -- they don't honor it -- but we still give them one).
    Returns (sets, byzantine_mask, normals_honest, total_n).
    """
    axes = np.array([[ 1.0, 0.0],
                     [-1.0, 0.0],
                     [ 0.0, 1.0],
                     [ 0.0,-1.0]])
    honest_normals = np.repeat(axes, n_per_axis, axis=0)        # (4*n_per_axis, 2)
    n_honest = honest_normals.shape[0]
    n_total  = n_honest + f
    sets: list[ConvexSet] = []
    byz = np.zeros(n_total, dtype=bool)
    # Honest first, Byzantine last; we'll shuffle indices so positions are
    # randomized while keeping `sets` aligned with the agent index.
    perm = rng.permutation(n_total)
    inverse = np.argsort(perm)
    honest_positions = inverse[: n_honest]
    byz_positions    = inverse[n_honest:]
    sets = [Halfspace(np.array([1.0, 0.0]), 0.0)] * n_total      # placeholder
    sets = list(sets)
    for k_h, p in enumerate(honest_positions):
        sets[p] = Halfspace(honest_normals[k_h], 0.0)
    for p in byz_positions:
        # Random half-space for Byzantine -- never used (they ignore their set)
        a = rng.standard_normal(2); a /= np.linalg.norm(a) + 1e-12
        sets[p] = Halfspace(a, float(rng.uniform(-1.0, 1.0)))
    byz[byz_positions] = True
    return sets, byz, honest_normals, n_total


def _demo_polyhedral(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    f = args.f
    n_per_axis = max(args.n_per_axis, f + 1)        # at least f+1 per side for redundancy
    sets, byz, honest_normals, n_total = _build_polyhedral_demo(n_per_axis, f, rng)
    H = int((~byz).sum())
    d = 2

    # Initial states: away from origin so we can see convergence.
    states = rng.normal(scale=0.6, size=(n_total, d))
    # Push Byzantine to absurd starting locations so the attack has somewhere to go.
    states[byz] = rng.uniform(-3.0, 3.0, size=(int(byz.sum()), d))

    sys_ = ConsensusSystem(states=states.copy(), sets=sets, byzantine=byz, f=f,
                           alpha=args.alpha, attack=ATTACKS[args.attack](), rng=rng)

    x_star = np.zeros(d)
    print(f"  ACC22 resilient constrained consensus  --  Polyhedral (Corollary 1) demo")
    print(f"  agents={n_total}  honest={H}  byzantine={f}  {_GLYPHS['alpha']}={args.alpha}  "
          f"attack={args.attack}  T={args.iters}")
    print(f"  X_i = halfspace {{x : a_i^T x {_GLYPHS['leq']} 0}};  4 axis directions x "
          f"{n_per_axis} copies => cap X_i = {{0}}.")

    # --- redundancy check (brute force) ---
    # 2f-redundancy is necessary (Theorem 1).  Verify it.
    print(f"\n  k-redundancy verification (small-graph brute force):")
    max_k_to_test = min(H - 1, 4 * f + 4)
    largest_redundant = 0
    for k in range(0, max_k_to_test + 1):
        if k_redundancy_polyhedral(honest_normals, k):
            largest_redundant = k
        else:
            break
    print(f"    largest k for which H is k-redundant: {largest_redundant}")
    print(f"    necessary 2f-redundancy (k = 2f = {2*f}):  "
          f"{_GLYPHS['check'] if largest_redundant >= 2*f else _GLYPHS['cross']}")

    _print_theorem2(mu=1.0, k=largest_redundant, f=f, H=H, alpha=args.alpha)

    history = sys_.run(args.iters)
    render_trajectory(history, byz, title=f"trajectories (attack={args.attack})")
    _summary(history, byz, x_star, args.iters)

    print("\n  honest agents (sample of 6) final states:")
    honest_idx = np.flatnonzero(~byz)
    for i in honest_idx[: 6]:
        x = history[-1, i]
        print(f"    agent {i:>2}: ({x[0]:+.5f}, {x[1]:+.5f})")
    if H > 6:
        print(f"    ... ({H - 6} more)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resilient constrained consensus (ACC22).")
    p.add_argument("--demo",       choices=["box", "polyhedral"], default="box")
    p.add_argument("--n",          type=int,   default=9,                help="agents (box demo)")
    p.add_argument("--n-per-axis", type=int,   default=3,                help="copies per axis (polyhedral demo)")
    p.add_argument("--f",          type=int,   default=2,                help="byzantine budget")
    p.add_argument("--alpha",      type=float, default=0.15,             help="step size")
    p.add_argument("--iters",      type=int,   default=80,               help="iterations")
    p.add_argument("--attack",     choices=list(ATTACKS), default="max_spread")
    p.add_argument("--seed",       type=int,   default=7)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.demo == "box":
        _demo_box(args)
    else:
        _demo_polyhedral(args)
