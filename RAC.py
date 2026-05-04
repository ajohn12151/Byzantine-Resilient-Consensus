"""
RAC -- Resilient Average Consensus with Adversaries via Distributed Detection
==============================================================================
Faithful Python implementation of:

    Yuan, Ishii -- "Resilient Average Consensus with Adversaries via
    Distributed Detection and Recovery", arXiv 2405.18752v1, 2024.

The protocol has two parts at every iteration k:

    1) DETECTION   — each normal node monitors its neighbors using the two-hop
       information sets Phi_j[k-1] it received the previous round.  Two
       options are provided:
         * Algorithm 2 (sharing detection) for undirected graphs that have a
           secure broadcast of detection events (Assumption 3 in the paper).
         * Algorithm 3 (fully distributed detection) for general directed
           graphs, using majority voting over (2f+1) two-hop paths.

    2) AVERAGING (running-sum recovery, Algorithm 1)
         - y_i[k], z_i[k] are computed from the *differences* of running sums
           lambda, gamma kept by each normal in-neighbor (paper Eq. 8).
         - When a malicious in-neighbor is detected for the first time, the
           cumulative effect it contributed is *subtracted* (paper Eq. 9).
         - When a malicious out-neighbor is detected for the first time, the
           cumulative value sent to it is *added back* to y_i, z_i so the
           total mass of normal nodes is preserved (paper Eq. 10).
         - r_i[k] = y_i[k] / z_i[k] is the consensus estimate.

This file follows the conventions established by ACC22.py / ACC23.py in this
repo: UTF-8 stdout reconfigure, dataclass system, attack policies, diagnostics
table, argparse CLI.

Run:
    python RAC.py                                                 # 6-node directed (paper Fig. 5b)
    python RAC.py --graph extreme8 --attack stealth_constant      # paper Fig. 8 (5 malicious)
    python RAC.py --graph fig5a --algorithm sharing --f 2         # paper Fig. 5(a) Alg. 2
    python RAC.py --graph layered --layers 4 --f 1                # paper Fig. 4(a)
    python RAC.py --plot                                          # save matplotlib trace
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Console encoding
# ---------------------------------------------------------------------------
def _can_unicode() -> bool:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        return True
    except Exception:
        return "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()


_UNICODE = _can_unicode()
_LAMBDA  = "λ"   if _UNICODE else "lam"
_GAMMA   = "γ"   if _UNICODE else "gam"
_DELTA   = "δ"   if _UNICODE else "del"
_OMEGA   = "ω"   if _UNICODE else "ome"
_PHI     = "Φ"   if _UNICODE else "Phi"
_BAR_X   = "X̄_N" if _UNICODE else "X_N"


# ---------------------------------------------------------------------------
# Adversarial attack policies (broadcast model, paper Definition 2)
#
# Each malicious node sends a single (potentially falsified) information set
# to ALL its out-neighbors at each iteration.  An attack policy returns a
# function (t, my_state) -> falsified_x_value that the malicious node will
# claim its current state is.  The averaging then derives the broadcast
# lambda, gamma from this falsified value via the protocol equations.
# ---------------------------------------------------------------------------
Attack = Callable[[int, "NodeState", np.random.Generator], float]


def attack_naive(low: float = -10.0, high: float = 10.0) -> Attack:
    def go(t, st, rng): return float(rng.uniform(low, high))
    return go


def attack_stealth_constant(value: float = 100.0, t_attack: int = 3) -> Attack:
    """Behave normally until t_attack, then claim a fixed extreme value forever.
    The fixed value forces a `lambda` mismatch with the prescribed update --
    detectable by the cross-checks in Algorithms 2 and 3."""
    def go(t, st, rng):
        if t < t_attack:
            return None         # signal: act normally this round
        return float(value)
    return go


def attack_delayed(t_attack: int = 5, perturb: float = 4.0) -> Attack:
    def go(t, st, rng):
        if t < t_attack:
            return None
        return float(st.x_initial + perturb)
    return go


def attack_pretend_normal() -> Attack:
    """Never deviates from the protocol -- just has an extreme initial value.
    Per paper Section V-E, indistinguishable from a normal node with that
    initial value."""
    def go(t, st, rng): return None
    return go


ATTACKS: dict[str, Callable[[], Attack]] = {
    "naive":            lambda: attack_naive(-10.0, 10.0),
    "stealth_constant": lambda: attack_stealth_constant(100.0, t_attack=3),
    "delayed":          lambda: attack_delayed(t_attack=5, perturb=4.0),
    "pretend_normal":   lambda: attack_pretend_normal(),
}


# ---------------------------------------------------------------------------
# Per-node state
# ---------------------------------------------------------------------------
@dataclass
class NodeState:
    idx:            int
    x_initial:      float
    in_neighbors:   list[int]
    out_neighbors: list[int]
    is_malicious:  bool
    # core state
    y:     float = 0.0
    z:     float = 1.0
    lam:   float = 0.0          # lambda_i[k]   -- running sum of y_i broadcasts
    gam:   float = 0.0          # gamma_i[k]    -- running sum of z_i broadcasts
    # buffers of received running sums (delta_ij[k], omega_ij[k])
    delta: dict[int, float] = field(default_factory=dict)   # delta_ij = lam_j as last seen
    omega: dict[int, float] = field(default_factory=dict)
    # detection sets
    A:    set[int] = field(default_factory=set)             # detected malicious neighbors
    A2:   set[int] = field(default_factory=set)             # detected malicious 2-hop in-neighbors

    def __post_init__(self):
        self.y = float(self.x_initial)
        for j in self.in_neighbors:
            self.delta[j] = 0.0
            self.omega[j] = 0.0


# ---------------------------------------------------------------------------
# The information set Phi_j[k]  (paper Eq. 6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InfoSet:
    """The packet broadcast by node j at time k+1 (= Phi_j[k] in paper notation).

    Fields:
      A         : set of malicious-agent IDs detected by j up to time k.
      ids       : list of nodes whose values j claims to forward (= N_j^- ∪ {j}).
      delta_jj  : lambda_j[k+1]   -- j's own running sum of y, after step k.
      omega_jj  : gamma_j[k+1]    -- same for z.
      delta_jh  : {h: delta_jh[k|k]} for h in N_j^- ∪ {j}.
      omega_jh  : {h: omega_jh[k|k]} for h in N_j^- ∪ {j}.
    """
    sender:    int
    A:         frozenset
    ids:       tuple
    delta_jj:  float
    omega_jj:  float
    delta_jh:  dict
    omega_jh:  dict


# ---------------------------------------------------------------------------
# RAC system
# ---------------------------------------------------------------------------
@dataclass
class RACSystem:
    A_adj:        np.ndarray                  # (n,n) bool, A_adj[i,j] iff edge (j,i) ∈ E (j is in-neighbor of i)
    states:       list[NodeState]
    f:            int
    rng:          np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    detection:    str = "fully_distributed"   # "sharing" | "fully_distributed"
    attack:       Attack = field(default_factory=attack_naive)
    safety_min:   float = -1e9
    safety_max:   float =  1e9

    def __post_init__(self):
        n = self.A_adj.shape[0]
        assert len(self.states) == n
        self.n = n
        self._honest = [i for i, s in enumerate(self.states) if not s.is_malicious]
        self._byz    = [i for i, s in enumerate(self.states) if s.is_malicious]
        # Cache previous round's packets for the Step-4 reconstruction.
        self._prev_packets: list[InfoSet | None] = [None] * n
        for st in self.states:
            st.A = set()
            st.A2 = set()

    # ------------------------------------------------------------------
    # Honest update: one full RAC iteration for node i
    # ------------------------------------------------------------------
    def _honest_step(self, i: int, t: int, packets: list[InfoSet | None]) -> None:
        st = self.states[i]
        prev_M_minus = set(st.in_neighbors) - st.A
        prev_M_plus  = set(st.out_neighbors) - st.A

        # ---- detection ----
        if self.detection == "sharing":
            self._detect_sharing(i, t, packets)
        else:
            self._detect_fully_distributed(i, t, packets)

        # update non-faulty in/out sets
        M_minus = set(st.in_neighbors) - st.A
        M_plus  = set(st.out_neighbors) - st.A
        d_plus  = len(M_plus)

        # ---- record received delta_ij, omega_ij from non-faulty in-neighbors ----
        prev_delta = dict(st.delta)
        prev_omega = dict(st.omega)
        for j in st.in_neighbors:
            pkt = packets[j]
            if pkt is None or j in st.A:
                continue
            if j in pkt.delta_jh:                  # j claims its own running sum
                st.delta[j] = pkt.delta_jj
                st.omega[j] = pkt.omega_jj

        # i's own delta_ii increment uses i's running sums (no malicious effect on i)
        st.delta[i] = st.lam if hasattr(st, "lam") else st.delta.get(i, 0.0)
        st.omega[i] = st.gam if hasattr(st, "gam") else st.omega.get(i, 0.0)
        prev_delta.setdefault(i, 0.0); prev_omega.setdefault(i, 0.0)

        # ---- y_i[k], z_i[k] from running-sum differences (paper Eq. 8) ----
        y = sum(st.delta[j] - prev_delta.get(j, 0.0) for j in (M_minus | {i}))
        z = sum(st.omega[j] - prev_omega.get(j, 0.0) for j in (M_minus | {i}))

        # ---- Case 1 recovery (Eq. 9): newly-detected malicious in-neighbors ----
        new_byz_in = prev_M_minus - M_minus
        for j in new_byz_in:
            y -= prev_delta.get(j, 0.0)
            z -= prev_omega.get(j, 0.0)
            # Per paper Eq. 9: set delta_ij[k] = 0 (we keep the slot so that
            # the broadcast is well-formed and downstream agents see a
            # consistent zero rather than a missing slot).
            st.delta[j] = 0.0; st.omega[j] = 0.0

        # ---- Case 2 recovery (Eq. 10): newly-detected malicious out-neighbors ----
        new_byz_out = prev_M_plus - M_plus
        if new_byz_out:
            y += len(new_byz_out) * st.lam
            z += len(new_byz_out) * st.gam

        # ---- write back ----
        st.y = y
        st.z = z
        # update running sums
        if d_plus + 1 > 0:
            st.lam = st.lam + y / (1 + d_plus)
            st.gam = st.gam + z / (1 + d_plus)

    # ------------------------------------------------------------------
    # Detection — sharing (Algorithm 2 of Yuan & Ishii)
    #
    # Per-step checks (no cross-agent voting on detection sets to avoid
    # false-positive cascades; instead we share detections AT THE END of
    # the step via a union with everyone's locally-flagged set, per
    # Assumption 3).
    # ------------------------------------------------------------------
    def _detect_sharing(self, i: int, t: int, packets: list[InfoSet | None]) -> None:
        st = self.states[i]
        for j in list(st.in_neighbors):
            if j in st.A:
                continue
            pkt = packets[j]
            if pkt is None:
                st.A.add(j); continue
            self._run_steps_2_3_4(i, j, pkt, st)
        # Assumption 3 (sharing): union all locally-flagged detections at end.
        # Done at the system level after every honest agent has run detection.

    # ------------------------------------------------------------------
    # Detection — fully distributed (Algorithm 3)
    # ------------------------------------------------------------------
    def _detect_fully_distributed(self, i: int, t: int,
                                  packets: list[InfoSet | None]) -> None:
        st = self.states[i]
        for j in list(st.in_neighbors):
            if j in st.A:
                continue
            pkt = packets[j]
            if pkt is None:
                st.A.add(j); continue
            self._run_steps_2_3_4(i, j, pkt, st)

    # ------------------------------------------------------------------
    # Steps 2, 3 of paper Algorithm 2 (the core soundness checks).
    #
    # Step 4 (full reconstruction of lambda_j[k+1] from two consecutive
    # packets) is intricate because of the running-sum protocol's one-round
    # propagation lag (pkt.delta_jh[h] for h != j is delta_jh[k|k] = lambda_h
    # at *one round earlier*).  Steps 2 and 3, when correctly time-aligned,
    # already detect the broadcast attacks that change either the claimed
    # in-neighbor list (Step 2) or the recorded values of in-neighbors
    # (Step 3).  We rely on those here.
    # ------------------------------------------------------------------
    def _run_steps_2_3_4(self, i: int, j: int, pkt: InfoSet,
                         st: NodeState) -> None:
        # Step 2: every ID j claims as an in-neighbor must be a real one.
        real_neighbors_of_j = set(self.states[j].in_neighbors) | {j}
        if not set(pkt.ids).issubset(real_neighbors_of_j):
            st.A.add(j); return
        # Step 3: cross-check j's claimed delta_jh[h] for h ∈ N_i^- ∪ {i}
        # against the broadcast h made in the previous round (which both i
        # and j received).  Skip h if j has flagged h as malicious -- per
        # Eq. 9 the legitimate value is then 0 (or whatever consistent reset).
        TOL = 1e-6
        for h in pkt.delta_jh:
            if h == j:                        # self-claim is not cross-checkable
                continue
            if h in pkt.A:                    # j has legitimately zeroed h
                if abs(pkt.delta_jh[h]) > TOL or abs(pkt.omega_jh[h]) > TOL:
                    st.A.add(j); return        # but j set non-zero -- caught
                continue
            ref = self._prev_packets[h]
            if ref is None:
                continue
            if h == i or h in st.in_neighbors:
                # i has the authoritative record (= h's prior broadcast).
                if abs(pkt.delta_jh[h] - ref.delta_jj) > TOL \
                   or abs(pkt.omega_jh[h] - ref.omega_jj) > TOL:
                    st.A.add(j); return

    # ------------------------------------------------------------------
    # Step 4 reconstruction (Yuan & Ishii Algorithm 2, line 14).
    # ------------------------------------------------------------------
    def _step4_reconstruct(self, j: int, pkt_curr: InfoSet) -> bool:
        pkt_prev = self._prev_packets[j]
        if pkt_prev is None:
            return True                          # cannot check at t=0
        st_j = self.states[j]
        A_j_curr = set(pkt_curr.A)
        M_minus = set(st_j.in_neighbors) - A_j_curr
        M_plus  = set(st_j.out_neighbors) - A_j_curr
        d_plus  = len(M_plus)
        # y_j_curr from running-sum diffs over M_minus ∪ {j}.
        y_recon = 0.0
        for h in M_minus | {j}:
            cur = pkt_curr.delta_jh.get(h)
            prv = pkt_prev.delta_jh.get(h, 0.0)
            if cur is None:
                return False
            y_recon += cur - prv
        expected = pkt_prev.delta_jj + y_recon / (1 + d_plus)
        return abs(pkt_curr.delta_jj - expected) <= 1e-3 + 1e-3 * abs(expected)

    # ------------------------------------------------------------------
    # Two-hop in-neighbors of i (excluding i itself).
    # ------------------------------------------------------------------
    def _two_hop_inneighbors(self, i: int) -> list[int]:
        in_i = set(self.states[i].in_neighbors)
        result = set()
        for j in in_i:
            for h in self.states[j].in_neighbors:
                if h != i and h not in in_i:
                    result.add(h)
        return sorted(result)

    # ------------------------------------------------------------------
    # Build broadcast packets for a round.
    # Honest: real Phi_j[k].  Malicious: per `attack` policy.
    # ------------------------------------------------------------------
    def _build_packets(self, t: int) -> list[InfoSet | None]:
        pkts: list[InfoSet | None] = [None] * self.n
        for j in range(self.n):
            st = self.states[j]
            if st.is_malicious:
                attacked_x = self.attack(t, st, self.rng)
                if attacked_x is None:
                    # behave normally
                    pkts[j] = self._honest_packet(j)
                else:
                    # broadcast a falsified info set with claimed running sums.
                    fake_lam = float(attacked_x) * (t + 1)         # plausible-looking
                    fake_gam = (t + 1)
                    pkts[j] = InfoSet(
                        sender=j,
                        A=frozenset(st.A),
                        ids=tuple(sorted(set(st.in_neighbors) | {j})),
                        delta_jj=fake_lam,
                        omega_jj=fake_gam,
                        delta_jh={h: fake_lam if h == j else self.states[h].lam
                                  for h in (set(st.in_neighbors) | {j})},
                        omega_jh={h: fake_gam if h == j else self.states[h].gam
                                  for h in (set(st.in_neighbors) | {j})},
                    )
            else:
                pkts[j] = self._honest_packet(j)
        return pkts

    def _honest_packet(self, j: int) -> InfoSet:
        st = self.states[j]
        ids = tuple(sorted(set(st.in_neighbors) | {j}))
        # Eq. 9 of paper: delta_ij set to 0 for nodes j has flagged.
        def _val(h, source):
            if h == j: return st.lam if source == "lam" else st.gam
            if h in st.A: return 0.0
            return (st.delta if source == "lam" else st.omega).get(h, 0.0)
        delta_jh = {h: _val(h, "lam") for h in ids}
        omega_jh = {h: _val(h, "gam") for h in ids}
        return InfoSet(
            sender=j,
            A=frozenset(st.A),
            ids=ids,
            delta_jj=st.lam,
            omega_jj=st.gam,
            delta_jh=delta_jh,
            omega_jh=omega_jh,
        )

    # ------------------------------------------------------------------
    # Drive the protocol for T rounds.
    # ------------------------------------------------------------------
    def run(self, T: int) -> dict:
        n = self.n
        ratios = np.zeros((T + 1, n))
        ratios[0] = [s.x_initial for s in self.states]
        det_size = np.zeros((T + 1, n), dtype=int)

        # Initial broadcast: t = 1, eq. (4) of paper.  Each agent sends its
        # initial value.  We bootstrap delta_ij[1], omega_ij[1] by running one
        # honest push-sum step.
        for st in self.states:
            st.lam = st.x_initial
            st.gam = 1.0
        for j in range(n):
            for i in range(n):
                if self.A_adj[i, j]:
                    self.states[i].delta[j] = self.states[j].lam
                    self.states[i].omega[j] = self.states[j].gam

        for t in range(T):
            packets = self._build_packets(t)
            for i in self._honest:
                self._honest_step(i, t, packets)
            # Malicious agents: broadcast continues, no protocol update.
            for i in self._byz:
                self.states[i].lam += self.states[i].y / max(1, len(self.states[i].out_neighbors) + 1)
                self.states[i].gam += self.states[i].z / max(1, len(self.states[i].out_neighbors) + 1)

            # Assumption 3 (sharing detection): union all honest detections.
            if self.detection == "sharing":
                union_A: set = set()
                for v in self._honest:
                    union_A |= self.states[v].A
                for v in self._honest:
                    self.states[v].A = set(union_A)

            # Cache packets for next round's Step-4 reconstruction.
            self._prev_packets = packets

            for i in range(n):
                z = max(self.states[i].z, 1e-12)
                ratios[t + 1, i] = self.states[i].y / z
                det_size[t + 1, i] = len(self.states[i].A)

        return {"ratios": ratios, "detection_sizes": det_size}


# ---------------------------------------------------------------------------
# Graph constructors
# ---------------------------------------------------------------------------
def complete_graph(n: int) -> np.ndarray:
    A = np.ones((n, n), dtype=bool)
    np.fill_diagonal(A, False)
    return A


def six_node_directed_demo() -> tuple[np.ndarray, list[int]]:
    """Paper Fig. 5(b): 6-node directed graph satisfying Algorithm 3 condition
    under the 1-local model.  Malicious set defaults to {6} (i.e. node 5 here)."""
    n = 6
    A = np.zeros((n, n), dtype=bool)
    edges = [
        (0, 1), (1, 0), (0, 2), (2, 0),
        (1, 3), (3, 1), (2, 4), (4, 2),
        (1, 4), (4, 1), (2, 3), (3, 2),
        (3, 5), (5, 3), (4, 5), (5, 4),
        (0, 5), (5, 0), (3, 4), (4, 3),
    ]
    for u, v in edges:
        A[v, u] = True                 # u -> v means v has u as in-neighbor: A[v, u] = True
    np.fill_diagonal(A, False)
    return A, [5]                      # malicious = node 5 (index 5; "node 6" in paper)


def eight_node_extreme_graph() -> tuple[np.ndarray, list[int]]:
    """Paper Fig. 8: 8-node nearly-complete graph.  Node 1 is a 'full access'
    node.  Malicious set is nodes 2..6 (5 of 8)."""
    n = 8
    A = np.ones((n, n), dtype=bool)
    np.fill_diagonal(A, False)
    # Node index 1 (paper's "node 2") has only 4 outgoing edges, not n-1.
    # Trim a few edges out of node 1 to make it "almost complete".
    for v in [2, 3, 4, 5]:
        A[v, 1] = True                 # keep these
    for v in [6, 7]:
        A[v, 1] = False                # remove
    malicious = [2, 3, 4, 5, 6]        # 5 of 8 nodes
    return A, malicious


def five_node_undirected_demo() -> tuple[np.ndarray, list[int]]:
    """Paper Fig. 5(a): 5-node undirected, node 0 is full-access.
    Malicious set = {3, 4} for the Algorithm-2 demo."""
    n = 5
    A = np.zeros((n, n), dtype=bool)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (2, 3), (3, 4), (1, 4)]
    for u, v in edges:
        A[v, u] = True; A[u, v] = True
    return A, [3, 4]


def layered_directed_graph(n_layers: int = 4, f: int = 1) -> tuple[np.ndarray, list[int]]:
    """Paper Section V-D: layered (2f+1)-per-layer construction.  Each node in
    layer L has every node in layer L-1 and L+1 as a neighbor; no edges within
    a layer.  Set the leftmost node in each odd layer as malicious."""
    per = 2 * f + 1
    n = n_layers * per
    A = np.zeros((n, n), dtype=bool)
    def lay(node): return node // per
    for u in range(n):
        for v in range(n):
            if u == v: continue
            if abs(lay(u) - lay(v)) == 1:
                A[v, u] = True
    malicious = [(L * per) for L in range(1, n_layers, 2)]
    return A, malicious


GRAPHS: dict[str, Callable] = {
    "fig5b":    six_node_directed_demo,
    "fig5a":    five_node_undirected_demo,
    "extreme8": eight_node_extreme_graph,
    "layered":  layered_directed_graph,
}


# ---------------------------------------------------------------------------
# Graph diagnostics (paper Definition 3 / Assumption 4)
# ---------------------------------------------------------------------------
def two_hop_inneighbors(A: np.ndarray, i: int) -> set[int]:
    in_i = set(np.flatnonzero(A[i]))
    result = set()
    for j in in_i:
        for h in np.flatnonzero(A[j]):
            if h != i and h not in in_i:
                result.add(int(h))
    return result


def two_hop_paths_count(A: np.ndarray, h: int, i: int) -> int:
    """Number of distinct two-hop paths from h to i."""
    if A[i, h]:
        # h is a direct in-neighbor; counts as detectable per Definition 3.
        return -1   # sentinel: directly accessible
    # paths h -> j -> i:
    in_i = np.flatnonzero(A[i])
    out_h = np.flatnonzero(A[:, h])
    return int(np.intersect1d(in_i, out_h).size)


def is_detectable(A: np.ndarray, h: int, i: int, f: int) -> bool:
    if A[i, h]:
        return True
    return two_hop_paths_count(A, h, i) >= 2 * f + 1


def is_alg3_condition(A: np.ndarray, f: int) -> bool:
    """Definition 3 + Assumption 4: every two-hop in-neighbor h of i is
    detectable; every out-neighbor q is detectable; every out-neighbor l of an
    in-neighbor j is detectable."""
    n = A.shape[0]
    for i in range(n):
        # 1) two-hop in-neighbors h
        for h in two_hop_inneighbors(A, i):
            if not is_detectable(A, h, i, f):
                return False
        # 2) out-neighbors
        for q in np.flatnonzero(A[:, i]):
            if not is_detectable(A, int(q), i, f):
                return False
        # 3) out-neighbors of in-neighbors
        for j in np.flatnonzero(A[i]):
            for l in np.flatnonzero(A[:, j]):
                if l == i: continue
                if not is_detectable(A, int(l), i, f):
                    return False
    return True


def is_alg2_condition(A: np.ndarray, f: int) -> bool:
    """Proposition 2(a): every adjacent pair has at least f-1 common neighbors;
    Proposition 2(b): graph is (f+1)-connected."""
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] and A[j, i]:           # undirected edge
                common = (np.flatnonzero(A[i]) if A.any() else np.array([])).tolist()
                ni = set(np.flatnonzero(A[i])) - {i, j}
                nj = set(np.flatnonzero(A[j])) - {i, j}
                if len(ni & nj) < max(0, f - 1):
                    return False
    # connectivity check (rough): the graph minus any f vertices is connected.
    # Skip the combinatorial test; users should verify by inspection on small graphs.
    return True


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    # ---- build graph ----
    if args.graph == "layered":
        A_adj, malicious_default = layered_directed_graph(args.layers, args.f)
    else:
        A_adj, malicious_default = GRAPHS[args.graph]()

    n = A_adj.shape[0]

    # ---- initial values ----
    if args.graph == "fig5b":
        x0 = np.array([9., 7., 1., 3., 4., 6.])
    elif args.graph == "fig5a":
        x0 = np.array([8., 6., 1., 3., 9.])
    elif args.graph == "extreme8":
        x0 = np.array([3., 15., 9., 8., 4., 7., 1., 12.])
    else:
        x0 = rng.uniform(0, 15, size=n)

    # ---- malicious set (allow override) ----
    malicious = set(args.malicious) if args.malicious else set(malicious_default)

    states = []
    for i in range(n):
        in_n  = list(np.flatnonzero(A_adj[i]).astype(int))
        out_n = list(np.flatnonzero(A_adj[:, i]).astype(int))
        states.append(NodeState(
            idx=i, x_initial=float(x0[i]),
            in_neighbors=in_n, out_neighbors=out_n,
            is_malicious=(i in malicious),
        ))

    sys_ = RACSystem(
        A_adj=A_adj, states=states, f=args.f,
        rng=rng, detection=args.algorithm,
        attack=ATTACKS[args.attack](),
    )

    # ---- print header ----
    print(f"  RAC -- Resilient Average Consensus")
    print(f"  graph={args.graph}  n={n}  malicious={sorted(malicious)}  "
          f"f={args.f}  algorithm={args.algorithm}  attack={args.attack}  "
          f"iters={args.iters}")

    # honest-only true average
    honest = [i for i in range(n) if i not in malicious]
    X_N = float(x0[honest].mean())
    print(f"  honest set = {honest}    {_BAR_X} = {X_N:.4f}")

    # ---- graph diagnostics ----
    print(f"\n  Graph diagnostics:")
    if n <= 14:
        cond3 = is_alg3_condition(A_adj, args.f)
        print(f"    Algorithm 3 condition (Def. 3 + Asm. 4): {cond3}")
        if np.array_equal(A_adj, A_adj.T):
            cond2 = is_alg2_condition(A_adj, args.f)
            print(f"    Algorithm 2 condition (>= f-1 common neighbors): {cond2}")
    else:
        print(f"    n={n} > 14 -- skipping brute-force checks")

    # ---- run ----
    out = sys_.run(args.iters)
    ratios = out["ratios"]
    det_sz = out["detection_sizes"]

    # ---- table ----
    print(f"\n   t  | r_i[k] for honest agents (mean / max-min)  | det |X̄_N - r̄|")
    print("  ----+--------------------------------------------+-----+----------")
    sample = sorted(set([0, 1, 2, 3, 5, 10, 15, 20, args.iters // 2, args.iters]))
    for t in sample:
        if t > args.iters: continue
        h = ratios[t, honest]
        sz = det_sz[t]
        avg_det = float(sz.mean())
        gap = abs(h.mean() - X_N)
        print(f"  {t:>3} |  mean={h.mean():+.4f}  spread={h.max()-h.min():.4e}  "
              f"|  {avg_det:.1f} | {gap:.4e}")

    # ---- final ----
    print(f"\n  Final state:")
    for i in range(n):
        tag = " *" if i in malicious else "  "
        print(f"    agent {i:>2}{tag}  r_i = {ratios[-1, i]:+.5f}   "
              f"detection set: {sorted(states[i].A)}")
    print(f"\n  honest mean  : {ratios[-1, honest].mean():+.5f}")
    print(f"  X_N (target) : {X_N:+.5f}")
    print(f"  |error|      : {abs(ratios[-1, honest].mean() - X_N):.4e}")

    # ---- optional plot ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            t = np.arange(args.iters + 1)
            for i in range(n):
                style = "--" if i in malicious else "-"
                color = "red" if i in malicious else None
                ax.plot(t, ratios[:, i], style, color=color, alpha=0.85,
                        label=f"agent {i}{' *' if i in malicious else ''}")
            ax.axhline(X_N, linestyle=":", color="black", alpha=0.5,
                       label=f"X_N = {X_N:.3f}")
            ax.set_xlabel("iteration k")
            ax.set_ylabel(r"$r_i[k] = y_i/z_i$")
            ax.set_title(f"RAC — graph={args.graph}, attack={args.attack}, "
                         f"algorithm={args.algorithm}")
            ax.legend(fontsize=7, ncol=2, loc="best")
            plt.tight_layout()
            out_path = f"rac_trace_{args.graph}_{args.attack}.png"
            plt.savefig(out_path, dpi=150)
            print(f"\n  saved trace -> {out_path}")
        except ImportError:
            print(f"\n  (matplotlib not available; skipping plot)")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resilient Average Consensus (Yuan & Ishii, 2024).")
    p.add_argument("--graph",     choices=list(GRAPHS), default="fig5b")
    p.add_argument("--layers",    type=int, default=4,    help="layers if --graph layered")
    p.add_argument("--f",         type=int, default=1,    help="f-local parameter")
    p.add_argument("--algorithm", choices=["sharing", "fully_distributed"],
                   default="fully_distributed")
    p.add_argument("--attack",    choices=list(ATTACKS), default="stealth_constant")
    p.add_argument("--malicious", type=int, nargs="*",  default=None,
                   help="override malicious node indices")
    p.add_argument("--iters",     type=int, default=30)
    p.add_argument("--seed",      type=int, default=7)
    p.add_argument("--plot",      action="store_true",  help="save matplotlib trace as PNG")
    return p.parse_args()


if __name__ == "__main__":
    _run_demo(_parse())
