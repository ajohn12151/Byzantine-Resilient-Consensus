%% ACC23 -- Resilient Distributed Optimization
%
% Faithful implementation of:
%   Zhu, Lin, Velasquez, Liu, "Resilient Distributed Optimization",
%   American Control Conference 2023, pp. 1307-1312.
%
% Algorithm (paper Eqs. 4-6) with the Tverberg-style aggregator implemented
% exactly for:
%     d = 1, any beta : reduces to trimmed mean over (2*beta+1)-subsets
%                       (the paper proves equivalence)
%     d = 2, beta = 1 : Tverberg point of 4 points (unique 2-partition with
%                       crossing segments; closed form via 2x2)
%
% Per-recipient Byzantine messaging supported via "two_faced" attack.
% Step-size schedules:  diminishing alpha(t) = a / (1 + b t)   (Theorem 1)
%                       constant    alpha(t) = c / sqrt(T)     (Theorem 2)
%
% Run: open in MATLAB and press Run, or `matlab -batch "run('ACC23.m')"`.

clear; clc; rng(11);

%% ----- experiment parameters -----------------------------------------------
n           = 8;
beta        = 1;
d           = 2;
attack_name = "max_spread";          % "max_spread"|"two_faced"|"drift"|"uniform"|"gradient_flip"
alpha_rule  = "diminishing";         % "diminishing" | "sqrtT"
alpha       = 0.5;
alpha_decay = 0.02;
T           = 200;
aggregator  = "paper";               % "paper"|"trim"|"cwtm"|"krum"|"geomedian"

if aggregator == "paper" && (d > 2 || (d == 2 && beta > 1))
    error("Paper aggregator only supports (d=1, any beta) or (d=2, beta=1).");
end

%% ----- problem setup -------------------------------------------------------
% Quadratic objectives with shared center -> network is k-redundant for any k.
center = -1 + 2 * rand(1, d);
objectives = repmat(struct("center", center), n, 1);
x_star = center;

states = 0.7 * randn(n, d);
byzantine = false(n, 1);
byz_pick = randperm(n, beta);
byzantine(byz_pick) = true;

A = true(n, n);
A(eye(n) == 1) = false;              % complete graph K_n, no self-loops

H = sum(~byzantine);
fprintf("ACC23 resilient distributed optimization\n");
fprintf("  agents=%d  honest=%d  byzantine=%d (beta=%d)  d=%d  aggregator=%s  attack=%s  iters=%d\n", ...
        n, H, sum(byzantine), beta, d, aggregator, attack_name, T);

%% ----- graph + redundancy diagnostics --------------------------------------
fprintf("\n  Graph diagnostics  G = K_%d  (r = %d, s = d*beta = %d):\n", n, beta, d * beta);
necessary = n >= beta * (1 + d) + 2;
fprintf("    Lemma 2 necessary condition  n >= beta(1+d)+2 = %d  : %s\n", ...
        beta * (1 + d) + 2, ternary(necessary, "true", "false"));
if n <= 6
    [resil, kap] = brute_force_resilience(A, beta, d * beta);
    fprintf("    brute-force (beta, d*beta)-resilient        : %s\n", ternary(resil, "true", "false"));
    fprintf("    kappa_{beta, d*beta}(G)                     : %d\n", kap);
else
    fprintf("    brute-force resilience check skipped (n > 6); rely on empirical convergence.\n");
end
fprintf("    quadratic-objective k-redundancy (k = n-1 = %d): true (shared centers)\n", n - 1);

%% ----- step-size rule ------------------------------------------------------
if alpha_rule == "sqrtT"
    step_rule = @(t) alpha / sqrt(T);
else
    step_rule = @(t) alpha / (1 + alpha_decay * t);
end
fprintf("  step-size: %s   alpha(0)=%.4f   alpha(%d)=%.4f\n", ...
        alpha_rule, step_rule(0), T - 1, step_rule(T - 1));

%% ----- run synchronously ---------------------------------------------------
history = zeros(T + 1, n, d);
history(1, :, :) = states;
diam = zeros(T + 1, 1); diam(1) = honest_diameter(states, byzantine);
err  = zeros(T + 1, 1); err(1)  = norm(mean(states(~byzantine, :), 1) - x_star);

for t = 0:T-1
    buf = build_message_buffer(states, byzantine, attack_name, t);
    snapshot   = states;
    new_states = states;
    a = step_rule(t);
    honest_idx = find(~byzantine).';
    for ii = honest_idx
        nbr = find(A(ii, :));
        received = squeeze(buf(nbr, ii, :));        % (|N_i|, d)
        if isvector(received) && d > 1
            received = reshape(received, [], d);
        elseif d == 1
            received = received(:);
        end
        v_i = aggregate(snapshot(ii, :), received, beta, d, aggregator);
        new_states(ii, :) = v_i - a * (v_i - objectives(ii).center);  % grad of 0.5||x-c||^2
    end
    states = new_states;
    history(t + 2, :, :) = states;
    diam(t + 2) = honest_diameter(states, byzantine);
    err(t + 2)  = norm(mean(states(~byzantine, :), 1) - x_star);
end

%% ----- table ---------------------------------------------------------------
fprintf("\n   t   |  disagreement  |  ||x_bar - x*||\n");
fprintf("  -----+----------------+-----------------\n");
sample = unique([0, 1, 2, 5, 10, 25, 50, floor(T/2), T]);
for tt = sample
    if tt > T, continue; end
    fprintf("  %4d |   %10.4e   |    %10.4e\n", tt, diam(tt + 1), err(tt + 1));
end

fprintf("\n  honest-mean final state  : (%+.5f, %+.5f)\n", ...
        mean(states(~byzantine, 1)), mean(states(~byzantine, 2)));
fprintf("  x*                       : (%+.5f, %+.5f)\n", x_star(1), x_star(2));
fprintf("  final ||x_bar - x*||    : %.4e\n", err(end));
fprintf("  final disagreement      : %.4e\n", diam(end));


% ===========================================================================
%  Local functions
% ===========================================================================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function dmax = honest_diameter(states, byzantine)
    h = states(~byzantine, :);
    if size(h, 1) < 2, dmax = 0; return; end
    diff = reshape(h, [], 1, size(h, 2)) - reshape(h, 1, [], size(h, 2));
    dmax = sqrt(max(sum(diff.^2, 3), [], "all"));
end

function v = aggregate(self_x, neighbors, beta, d, name)
    switch name
        case "paper"
            v = aggregate_paper(self_x, neighbors, beta, d);
        case "trim"
            if size(neighbors, 1) <= beta, v = self_x; return; end
            dists = vecnorm(neighbors - self_x, 2, 2);
            [~, idx] = sort(dists);
            kept = neighbors(idx(1:end - beta), :);
            v = mean(kept, 1);
        case "cwtm"
            n_ = size(neighbors, 1);
            if 2 * beta >= n_
                v = median(neighbors, 1);
            else
                s = sort(neighbors, 1);
                v = mean(s(beta + 1 : n_ - beta, :), 1);
            end
        case "krum"
            n_ = size(neighbors, 1);
            k = n_ - beta - 2;
            if k < 1
                v = mean(neighbors, 1); return;
            end
            sq = zeros(n_, n_);
            for ii = 1:n_
                for jj = 1:n_
                    if ii == jj
                        sq(ii, jj) = inf;
                    else
                        sq(ii, jj) = sum((neighbors(ii, :) - neighbors(jj, :)).^2);
                    end
                end
            end
            sq_sorted = sort(sq, 2);
            sums = sum(sq_sorted(:, 1:k), 2);
            [~, sel] = min(sums);
            v = neighbors(sel, :);
        case "geomedian"
            v = mean(neighbors, 1);
            for it = 1:32
                d_ = vecnorm(neighbors - v, 2, 2);
                if any(d_ < 1e-9)
                    [~, sel] = min(d_); v = neighbors(sel, :); return;
                end
                w = 1 ./ d_;
                v_new = sum(w .* neighbors, 1) / sum(w);
                if norm(v_new - v) < 1e-9, v = v_new; return; end
                v = v_new;
            end
        otherwise
            error("unknown aggregator: %s", name);
    end
end

function v = aggregate_paper(self_x, neighbors, beta, d)
    n_nbr = size(neighbors, 1);
    subset_size = (d + 1) * beta + 1;
    if n_nbr < subset_size
        v = 0.5 * (self_x + mean(neighbors, 1));
        return;
    end
    y_sum = zeros(size(self_x));
    a_i = 0;
    combos = nchoosek(1:n_nbr, subset_size);
    for c = 1:size(combos, 1)
        A_vals = neighbors(combos(c, :), :);
        y_sum = y_sum + y_ij(A_vals, beta, d);
        a_i = a_i + 1;
    end
    v = (self_x + y_sum) / (1 + a_i);
end

function y = y_ij(A_vals, beta, d)
    if d == 1
        s = sort(A_vals, 1);
        y = 0.5 * (s(beta + 1, :) + s(end - beta, :));
        return;
    end
    if d == 2 && beta == 1
        y = tverberg_4_2d(A_vals);
        return;
    end
    error("y_ij only implemented for d=1 (any beta) or d=2 (beta=1).");
end

function y = tverberg_4_2d(p)
    p1 = p(1, :); p2 = p(2, :); p3 = p(3, :); p4 = p(4, :);
    pairs = {{p1, p2, p3, p4}, {p1, p3, p2, p4}, {p1, p4, p2, p3}};
    for k = 1:numel(pairs)
        a = pairs{k}{1}; b = pairs{k}{2}; c = pairs{k}{3}; d_ = pairs{k}{4};
        ix = segment_intersection(a, b, c, d_);
        if ~isempty(ix), y = ix; return; end
    end
    y = mean(p, 1);
end

function ix = segment_intersection(a, b, c, d)
    r = b - a; s = d - c;
    rxs = r(1) * s(2) - r(2) * s(1);
    if abs(rxs) < 1e-12, ix = []; return; end
    qmp = c - a;
    t = (qmp(1) * s(2) - qmp(2) * s(1)) / rxs;
    u = (qmp(1) * r(2) - qmp(2) * r(1)) / rxs;
    if t > 0 && t < 1 && u > 0 && u < 1
        ix = a + t * r;
    else
        ix = [];
    end
end

function buf = build_message_buffer(states, byzantine, attack_name, t)
    [n, d] = size(states);
    buf = zeros(n, n, d);
    for j = 1:n
        for i = 1:n
            buf(j, i, :) = states(j, :);
        end
    end
    byz_idx = find(byzantine).';
    if isempty(byz_idx), return; end
    honest = states(~byzantine, :);
    msgs = zeros(numel(byz_idx), n, d);
    switch attack_name
        case "max_spread"
            c = mean(honest, 1);
            r = max(vecnorm(honest - c, 2, 2)) + 1e-9;
            v = randn(size(c)); v = v / (norm(v) + 1e-12);
            m = c - 5.0 * r * v;
            msgs = repmat(reshape(m, [1 1 d]), numel(byz_idx), n, 1);
        case "drift"
            c = mean(honest, 1) + (t + 1) * 0.05;
            msgs = repmat(reshape(c, [1 1 d]), numel(byz_idx), n, 1);
        case "two_faced"
            c = mean(honest, 1);
            r = max(vecnorm(honest - c, 2, 2)) + 1e-9;
            v = randn(1, n, d); v = v ./ (vecnorm(v, 2, 3) + 1e-12);
            m = c + 5.0 * r * squeeze(v);
            msgs = repmat(reshape(m, [1 n d]), numel(byz_idx), 1, 1);
        case "uniform"
            msgs = -2.5 + 5 * rand(numel(byz_idx), n, d);
        case "gradient_flip"
            c = mean(honest, 1) + 2.0 * (1 + 0.05 * t);
            msgs = repmat(reshape(c, [1 1 d]), numel(byz_idx), n, 1);
        otherwise
            error("unknown attack: %s", attack_name);
    end
    for k = 1:numel(byz_idx)
        buf(byz_idx(k), :, :) = msgs(k, :, :);
    end
end

function [resil, kap] = brute_force_resilience(A, r, s)
    n = size(A, 1);
    resil = true; kap = n;
    if r + s + 1 > n
        resil = false; kap = 0; return;
    end
    subsets = nchoosek(1:n, n - r);
    for k = 1:size(subsets, 1)
        S = subsets(k, :);
        sub = A(S, S);
        [ok, worst] = check_all_removals(sub, s);
        if ~ok
            resil = false; kap = 0; return;
        end
        kap = min(kap, worst);
    end
end

function [ok, worst] = check_all_removals(sub, s)
    n = size(sub, 1);
    incoming = cell(n, 1);
    for v = 1:n
        incoming{v} = find(sub(:, v))';
    end
    ok = true; worst = n + 1;
    [ok, worst] = recurse(1, sub, incoming, s, ok, worst);
    if ~ok || worst > n, worst = 0; end
end

function [ok, worst] = recurse(v, cur, incoming, s, ok, worst)
    n = size(cur, 1);
    if v > n
        if ~is_rooted(cur), ok = false; return; end
        worst = min(worst, count_roots(cur));
        return;
    end
    in_v = incoming{v};
    for r = 0:min(s, numel(in_v))
        if r == 0
            [ok, worst] = recurse(v + 1, cur, incoming, s, ok, worst);
            if ~ok, return; end
        else
            cmb = nchoosek(in_v, r);
            for k = 1:size(cmb, 1)
                new = cur;
                for u = cmb(k, :)
                    new(u, v) = false;
                end
                [ok, worst] = recurse(v + 1, new, incoming, s, ok, worst);
                if ~ok, return; end
            end
        end
    end
end

function tf = is_rooted(A)
    n = size(A, 1);
    tf = false;
    for s = 1:n
        seen = false(1, n); seen(s) = true;
        frontier = s;
        while ~isempty(frontier)
            nxt = [];
            for u = frontier
                vs = find(A(u, :) & ~seen);
                seen(vs) = true;
                nxt = [nxt vs]; %#ok<AGROW>
            end
            frontier = nxt;
        end
        if all(seen), tf = true; return; end
    end
end

function c = count_roots(A)
    n = size(A, 1); c = 0;
    for s = 1:n
        seen = false(1, n); seen(s) = true;
        frontier = s;
        while ~isempty(frontier)
            nxt = [];
            for u = frontier
                vs = find(A(u, :) & ~seen);
                seen(vs) = true;
                nxt = [nxt vs]; %#ok<AGROW>
            end
            frontier = nxt;
        end
        if all(seen), c = c + 1; end
    end
end
