%% RAC -- Resilient Average Consensus with Adversaries via Distributed Detection
%
% Faithful MATLAB port of the algorithm from:
%   Yuan, Ishii, "Resilient Average Consensus with Adversaries via Distributed
%   Detection and Recovery", arXiv 2405.18752v1, 2024.
%
% Algorithm 1 (averaging with running-sum recovery):
%   y_i[k], z_i[k] are computed from running-sum *differences* (paper Eq. 8).
%   When a malicious in-neighbor is first detected, its cumulative contribution
%   is *subtracted* (paper Eq. 9).
%   When a malicious out-neighbor is first detected, the cumulative value sent
%   to it is *added back* to y_i, z_i (paper Eq. 10).
%
% This script implements the 6-node directed example (paper Fig. 5(b)).
% Detection here uses Steps 2 & 3 of paper Algorithm 2 (sharing model;
% Algorithm 3's full majority-voting Step 4 reconstruction is intricate
% and is approximated by the cross-check that delta_jh values reported by
% j match what each authoritative neighbor h broadcast in the prior round).
%
% Run: open in MATLAB and press Run, or `matlab -batch "run('RAC.m')"`.

clear; clc; rng(7);

%% ------- experiment parameters ---------------------------------------------
T            = 30;
attack_name  = "stealth_constant";   % "naive"|"stealth_constant"|"delayed"|"pretend_normal"
algorithm    = "fully_distributed";  % "sharing" | "fully_distributed"
attack_value = 100.0;                % constant value for stealth attack
t_attack     = 3;                    % round at which attack begins

%% ------- 6-node directed graph (paper Fig. 5(b)) ---------------------------
n = 6;
A = false(n, n);                     % A(i,j) = true iff edge (j -> i) ∈ E
edges = [
    1 2; 2 1; 1 3; 3 1;
    2 4; 4 2; 3 5; 5 3;
    2 5; 5 2; 3 4; 4 3;
    4 6; 6 4; 5 6; 6 5;
    1 6; 6 1; 4 5; 5 4
];
for k = 1:size(edges, 1)
    u = edges(k, 1); v = edges(k, 2);
    A(v, u) = true;                  % v has u as in-neighbor
end
A = A & ~eye(n, "logical");

x0 = [9, 7, 1, 3, 4, 6];             % initial values (paper)
malicious = false(n, 1);
malicious(6) = true;                 % node 6 is malicious

%% ------- per-node state ----------------------------------------------------
state = struct( ...
    "y", num2cell(x0), ...
    "z", num2cell(ones(1, n)), ...
    "lam", num2cell(x0), ...         % bootstrap: lam = x_initial
    "gam", num2cell(ones(1, n)), ...
    "delta", cell(1, n), ...
    "omega", cell(1, n), ...
    "A", cell(1, n));
for i = 1:n
    in_n = find(A(i, :));
    delta_i = containers.Map("KeyType", "int32", "ValueType", "double");
    omega_i = containers.Map("KeyType", "int32", "ValueType", "double");
    for j = in_n
        delta_i(int32(j)) = x0(j);    % bootstrap: delta_ij = lam_j_initial
        omega_i(int32(j)) = 1.0;
    end
    state(i).delta = delta_i;
    state(i).omega = omega_i;
    state(i).A     = false(n, 1);
end

% Cache of previous round's packets (initialized to empty).
prev_packets = cell(1, n);
for i = 1:n, prev_packets{i} = []; end

ratios = zeros(T + 1, n);
det_size = zeros(T + 1, n);
ratios(1, :) = x0;

%% ------- main loop ---------------------------------------------------------
for t = 0:T - 1
    % Build per-agent broadcast packets.
    packets = cell(1, n);
    for j = 1:n
        if malicious(j)
            attacked_x = malicious_attack(t, x0(j), attack_name, attack_value, t_attack);
            if isempty(attacked_x)
                packets{j} = honest_packet(j, state(j), A);
            else
                packets{j} = malicious_packet(j, attacked_x, t, state, A);
            end
        else
            packets{j} = honest_packet(j, state(j), A);
        end
    end

    % Honest agents update.
    for i = 1:n
        if malicious(i), continue; end
        state(i) = honest_step(i, t, state(i), packets, A, prev_packets, algorithm, n);
    end

    % Malicious agents continue advancing their lam, gam (broadcast).
    for i = 1:n
        if ~malicious(i), continue; end
        d_plus = max(1, sum(A(:, i)) + 1);
        state(i).lam = state(i).lam + state(i).y / d_plus;
        state(i).gam = state(i).gam + state(i).z / d_plus;
    end

    % Sharing detection: union all locally-flagged sets.
    if algorithm == "sharing"
        union_A = false(n, 1);
        for v = 1:n
            if ~malicious(v), union_A = union_A | state(v).A; end
        end
        for v = 1:n
            if ~malicious(v), state(v).A = union_A; end
        end
    end

    prev_packets = packets;

    for i = 1:n
        ratios(t + 2, i) = state(i).y / max(state(i).z, 1e-12);
        det_size(t + 2, i) = sum(state(i).A);
    end
end

%% ------- diagnostics -------------------------------------------------------
honest_idx = find(~malicious).';
X_N = mean(x0(honest_idx));

fprintf("RAC -- Resilient Average Consensus  (paper Fig. 5(b))\n");
fprintf("  agents=%d  malicious=%s  attack=%s  algorithm=%s  iters=%d\n", ...
        n, mat2str(find(malicious).'), attack_name, algorithm, T);
fprintf("  honest_set = %s    X_N (target) = %.4f\n", ...
        mat2str(honest_idx), X_N);

fprintf("\n   t  |  honest mean  |  honest spread  |  detection sizes\n");
fprintf("  ----+---------------+-----------------+------------------\n");
sample = unique([0, 1, 2, 3, 5, 10, 15, 20, T]);
for tt = sample
    if tt > T, continue; end
    h = ratios(tt + 1, honest_idx);
    fprintf("  %4d |   %8.4f    |   %.4e   |   %s\n", ...
            tt, mean(h), max(h) - min(h), mat2str(det_size(tt + 1, :)));
end

fprintf("\n  Final state:\n");
for i = 1:n
    tag = " ";  if malicious(i), tag = "*"; end
    fprintf("    agent %2d %s  r_i = %+.5f   |A_i| = %d\n", ...
            i, tag, ratios(end, i), sum(state(i).A));
end
fprintf("\n  honest-mean final ratio: %+.5f\n", mean(ratios(end, honest_idx)));
fprintf("  X_N (target)            : %+.5f\n", X_N);
fprintf("  |error|                 : %.4e\n", ...
        abs(mean(ratios(end, honest_idx)) - X_N));


% ===========================================================================
%  Local functions
% ===========================================================================
function pkt = honest_packet(j, st, A)
    in_n = find(A(j, :));
    ids = sort(unique([in_n j]));
    delta_jh = containers.Map("KeyType", "int32", "ValueType", "double");
    omega_jh = containers.Map("KeyType", "int32", "ValueType", "double");
    for h = ids
        if h == j
            delta_jh(int32(h)) = st.lam;
            omega_jh(int32(h)) = st.gam;
        elseif st.A(h)
            delta_jh(int32(h)) = 0.0;     % per Eq. 9
            omega_jh(int32(h)) = 0.0;
        else
            if isKey(st.delta, int32(h))
                delta_jh(int32(h)) = st.delta(int32(h));
                omega_jh(int32(h)) = st.omega(int32(h));
            else
                delta_jh(int32(h)) = 0.0;
                omega_jh(int32(h)) = 0.0;
            end
        end
    end
    pkt = struct("sender", j, "A", st.A, "ids", ids, ...
                 "delta_jj", st.lam, "omega_jj", st.gam, ...
                 "delta_jh", delta_jh, "omega_jh", omega_jh);
end

function pkt = malicious_packet(j, attacked_x, t, state, A)
    in_n = find(A(j, :));
    ids = sort(unique([in_n j]));
    delta_jh = containers.Map("KeyType", "int32", "ValueType", "double");
    omega_jh = containers.Map("KeyType", "int32", "ValueType", "double");
    fake_lam = attacked_x * (t + 1);
    fake_gam = (t + 1);
    for h = ids
        if h == j
            delta_jh(int32(h)) = fake_lam;
            omega_jh(int32(h)) = fake_gam;
        else
            delta_jh(int32(h)) = state(h).lam;
            omega_jh(int32(h)) = state(h).gam;
        end
    end
    n = numel(state);
    pkt = struct("sender", j, "A", false(n, 1), "ids", ids, ...
                 "delta_jj", fake_lam, "omega_jj", fake_gam, ...
                 "delta_jh", delta_jh, "omega_jh", omega_jh);
end

function out = malicious_attack(t, x_initial, name, value, t_attack)
    switch name
        case "naive"
            out = -10 + 20 * rand();
        case "stealth_constant"
            if t < t_attack, out = []; else, out = value; end
        case "delayed"
            if t < t_attack, out = []; else, out = x_initial + 4.0; end
        case "pretend_normal"
            out = [];
        otherwise
            error("unknown attack: %s", name);
    end
end

function st = honest_step(i, t, st, packets, A, prev_packets, algorithm, n)
    % Detection
    in_n = find(A(i, :));
    out_n = find(A(:, i)).';
    prev_M_minus = false(n, 1); prev_M_plus = false(n, 1);
    for j = in_n,  if ~st.A(j), prev_M_minus(j) = true; end, end
    for q = out_n, if ~st.A(q), prev_M_plus(q)  = true; end, end

    for jj = in_n
        if st.A(jj), continue; end
        pkt = packets{jj};
        if isempty(pkt), st.A(jj) = true; continue; end
        st = run_steps_2_3(i, jj, pkt, st, A, prev_packets, n);
    end

    % Updated non-faulty sets.
    M_minus = false(n, 1); M_plus = false(n, 1);
    for j = in_n,  if ~st.A(j), M_minus(j) = true; end, end
    for q = out_n, if ~st.A(q), M_plus(q)  = true; end, end
    d_plus = sum(M_plus);

    % Record received delta_ij, omega_ij from non-faulty in-neighbors.
    prev_delta = containers.Map("KeyType", "int32", "ValueType", "double");
    prev_omega = containers.Map("KeyType", "int32", "ValueType", "double");
    keys_curr = cell2mat(keys(st.delta));
    for k = keys_curr
        prev_delta(int32(k)) = st.delta(k);
        prev_omega(int32(k)) = st.omega(k);
    end
    for jj = in_n
        if st.A(jj), continue; end
        pkt = packets{jj};
        if isempty(pkt), continue; end
        st.delta(int32(jj)) = pkt.delta_jj;
        st.omega(int32(jj)) = pkt.omega_jj;
    end
    st.delta(int32(i)) = st.lam;
    st.omega(int32(i)) = st.gam;
    if ~isKey(prev_delta, int32(i)), prev_delta(int32(i)) = 0.0; end
    if ~isKey(prev_omega, int32(i)), prev_omega(int32(i)) = 0.0; end

    % y_i[k], z_i[k] from running-sum diffs (Eq. 8).
    y = 0; z = 0;
    for h = [find(M_minus.').' i]
        cur_d = 0; cur_o = 0;
        if isKey(st.delta, int32(h)), cur_d = st.delta(int32(h)); cur_o = st.omega(int32(h)); end
        prv_d = 0; prv_o = 0;
        if isKey(prev_delta, int32(h)), prv_d = prev_delta(int32(h)); prv_o = prev_omega(int32(h)); end
        y = y + (cur_d - prv_d);
        z = z + (cur_o - prv_o);
    end

    % Case 1 (Eq. 9): newly-detected malicious in-neighbors.
    for j = find(prev_M_minus & ~M_minus).'
        if isKey(prev_delta, int32(j)), y = y - prev_delta(int32(j)); end
        if isKey(prev_omega, int32(j)), z = z - prev_omega(int32(j)); end
        st.delta(int32(j)) = 0.0;
        st.omega(int32(j)) = 0.0;
    end
    % Case 2 (Eq. 10): newly-detected malicious out-neighbors.
    new_byz_out = sum(prev_M_plus & ~M_plus);
    if new_byz_out > 0
        y = y + new_byz_out * st.lam;
        z = z + new_byz_out * st.gam;
    end

    st.y = y; st.z = z;
    if d_plus + 1 > 0
        st.lam = st.lam + y / (1 + d_plus);
        st.gam = st.gam + z / (1 + d_plus);
    end
end

function st = run_steps_2_3(i, j, pkt, st, A, prev_packets, n)
    % Step 2: every claimed in-neighbor of j must be a real one.
    in_j = find(A(j, :));
    real_neighbors_of_j = unique([in_j j]);
    if any(~ismember(pkt.ids, real_neighbors_of_j))
        st.A(j) = true; return;
    end
    % Step 3: cross-check delta_jh[h] for h ∈ N_i^- ∪ {i}.
    TOL = 1e-6;
    in_i = find(A(i, :));
    keys_pkt = cell2mat(keys(pkt.delta_jh));
    for h = keys_pkt
        if h == j, continue; end
        if pkt.A(h)
            % j has flagged h; legitimate value is 0.
            if abs(pkt.delta_jh(int32(h))) > TOL || abs(pkt.omega_jh(int32(h))) > TOL
                st.A(j) = true; return;
            end
            continue;
        end
        ref = prev_packets{h};
        if isempty(ref), continue; end
        if h == i || ismember(h, in_i)
            if abs(pkt.delta_jh(int32(h)) - ref.delta_jj) > TOL ...
               || abs(pkt.omega_jh(int32(h)) - ref.omega_jj) > TOL
                st.A(j) = true; return;
            end
        end
    end
end
