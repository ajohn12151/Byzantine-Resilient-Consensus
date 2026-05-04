%% ACC22 -- Resilient Constrained Consensus over Complete Graphs
%
% Faithful implementation of:
%   Zhu, Lin, Velasquez, Liu, "Resilient Constrained Consensus over Complete
%   Graphs via Feasibility Redundancy", American Control Conference 2022,
%   pp. 3418-3422.
%
% Algorithm (paper Eq. 1):
%   x_i(t+1) = P_{X_i}[ x_i(t) + alpha * sum_{j in M_i(t)} (x_{ji}(t) - x_i(t)) ]
%
% This script:
%   * uses synchronous (atomic) updates from a per-round snapshot,
%   * supports per-recipient Byzantine messaging (paper footnote 1) via an
%     (n, n, d) message buffer -- so the "two_faced" attack class lies
%     differently to different recipients,
%   * uses a closed-form Box / half-space projection (Corollary 1 setting),
%   * tracks the paper's Lyapunov V(t) = sum_{i in H} ||x_i - x*||^2 and
%     fits the empirical contraction rate against the analytic prediction
%     rho = 1 - (mu^2*k - 4f - 2f*mu^2 + mu^2)*alpha + 4|H|^3 alpha^2,
%   * prints PASS / FAIL on each Theorem-2 condition.
%
% Run: open in MATLAB and press Run, or `matlab -batch "run('ACC22.m')"`.

clear; clc; rng(7);

%% ----- experiment parameters -----------------------------------------------
demo        = "polyhedral";          % "polyhedral" | "box"
attack_name = "max_spread";          % "max_spread"|"two_faced"|"drift"|"uniform"|"mimic"
n_per_axis  = 3;
f           = 1;
alpha       = 0.15;
T           = 80;
mu          = 1.0;                   % Hoffman regularity for axis-aligned half-spaces

%% ----- build the demo ------------------------------------------------------
[states, sets, byzantine, x_star, n] = build_demo(demo, n_per_axis, f);
H = sum(~byzantine);

fprintf("ACC22 resilient constrained consensus  --  %s demo\n", demo);
fprintf("  agents=%d  honest=%d  byzantine=%d  alpha=%.3f  attack=%s  T=%d\n", ...
        n, H, f, alpha, attack_name, T);

%% ----- Theorem 2 conditions ------------------------------------------------
k_use = max(0, 2 * f);                % the necessary 2f-redundancy from Theorem 1
[k_lower, alpha_max, rho_pred] = theorem2_bounds(mu, k_use, f, H, alpha);
ok = @(b) ternary(b, "[OK]", "[X ]");
fprintf("\n  Theorem 2 conditions  (mu=%.3f, k=%d, f=%d, |H|=%d, alpha=%.4f)\n", ...
        mu, k_use, f, H, alpha);
fprintf("    %s  k > 4f/mu^2 + 2f - 1 = %.3f      (k = %d)\n", ...
        ok(k_use > k_lower), k_lower, k_use);
fprintf("    %s  alpha < (mu^2 k - 4f - 2f mu^2 + mu^2)/(4|H|^3) = %.4e\n", ...
        ok(alpha < alpha_max), alpha_max);
fprintf("    %s  predicted rho = %.4f  in (0,1)\n", ...
        ok(rho_pred > 0 && rho_pred < 1), rho_pred);
fprintf("    (sufficient k from Theorem 2 is conservative; algorithm typically\n");
fprintf("     converges empirically even when these conditions fail.)\n");

%% ----- run synchronously ---------------------------------------------------
history = zeros(T + 1, n, size(states, 2));
history(1, :, :) = states;
V    = zeros(T + 1, 1); V(1)    = lyapunov_V(states, x_star, byzantine);
diam = zeros(T + 1, 1); diam(1) = honest_diameter(states, byzantine);

for t = 0:T-1
    % (1) Build per-recipient message buffer.
    buf = build_message_buffer(states, byzantine, attack_name, t);
    % (2) Synchronous honest update from the snapshot.
    snapshot   = states;
    new_states = states;
    honest_idx = find(~byzantine).';
    for ii = honest_idx
        received = squeeze(buf(:, ii, :));      % (n, d)
        received(ii, :) = [];                   % drop self
        kept = retain_filter(snapshot(ii, :), received, f);
        update_step = sum(kept - snapshot(ii, :), 1);
        cand = snapshot(ii, :) + alpha * update_step;
        new_states(ii, :) = sets{ii}.project(cand);
    end
    states = new_states;
    history(t + 2, :, :) = states;
    V(t + 2)    = lyapunov_V(states, x_star, byzantine);
    diam(t + 2) = honest_diameter(states, byzantine);
end

%% ----- diagnostics ---------------------------------------------------------
pos = V > 1e-300;
if sum(pos) >= 3
    [rate, r2] = fit_log_linear(V(pos));
    fprintf("\n  V(0)            = %.4e\n", V(1));
    fprintf("  V(%d)           = %.4e\n", T, V(end));
    fprintf("  empirical rho   = %.4f   (R^2=%.3f)\n", exp(rate), r2);
end
fprintf("  honest disagreement diam   t=0:    %.4e\n", diam(1));
fprintf("  honest disagreement diam   t=%d:   %.4e\n", T, diam(end));

fprintf("\n  honest agents (sample of %d) final states:\n", min(H, 6));
honest_idx = find(~byzantine).';
for ii = honest_idx(1:min(H, 6))
    fprintf("    agent %2d: (%+.5f, %+.5f)\n", ii, states(ii, 1), states(ii, 2));
end
if H > 6
    fprintf("    ... (%d more)\n", H - 6);
end


% ===========================================================================
%  Local functions
% ===========================================================================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function [k_lower, alpha_max, rho] = theorem2_bounds(mu, k, f, H, alpha)
    k_lower   = 4 * f / max(mu^2, 1e-12) + 2 * f - 1;
    alpha_max = (mu^2 * k - 4 * f - 2 * f * mu^2 + mu^2) / max(4 * H^3, 1e-12);
    rho       = 1 - (mu^2 * k - 4 * f - 2 * f * mu^2 + mu^2) * alpha ...
                  + 4 * H^3 * alpha^2;
end

function V = lyapunov_V(states, x_star, byzantine)
    h = states(~byzantine, :);
    V = sum(sum((h - x_star).^2));
end

function dmax = honest_diameter(states, byzantine)
    h = states(~byzantine, :);
    if size(h, 1) < 2
        dmax = 0; return;
    end
    diff = reshape(h, [], 1, size(h, 2)) - reshape(h, 1, [], size(h, 2));
    dmax = sqrt(max(sum(diff.^2, 3), [], "all"));
end

function kept = retain_filter(self_state, others, f)
    if f <= 0
        kept = others; return;
    end
    dists = vecnorm(others - self_state, 2, 2);
    [~, idx] = sort(dists);
    kept = others(idx(1:end - f), :);
end

function [rate, r2] = fit_log_linear(y)
    eps0 = 1e-15;
    y = log(max(y(:), eps0));
    t = (0:numel(y) - 1).';
    A = [ones(size(t)), t];
    coef = A \ y;
    rate = coef(2);
    pred = A * coef;
    r2 = 1 - sum((y - pred).^2) / max(sum((y - mean(y)).^2), eps0);
end

function buf = build_message_buffer(states, byzantine, attack_name, t)
    [n, d] = size(states);
    buf = zeros(n, n, d);
    for j = 1:n
        for i = 1:n
            buf(j, i, :) = states(j, :);     % default: honest broadcast
        end
    end
    byz_idx = find(byzantine).';
    if isempty(byz_idx), return; end
    honest = states(~byzantine, :);
    msgs = zeros(numel(byz_idx), n, d);
    switch attack_name
        case "constant"
            v = [2.0, 2.0];
            msgs = repmat(reshape(v, [1 1 d]), numel(byz_idx), n, 1);
        case "drift"
            c = mean(honest, 1) + (t + 1) * [0.05, -0.05];
            msgs = repmat(reshape(c, [1 1 d]), numel(byz_idx), n, 1);
        case "max_spread"
            c = mean(honest, 1);
            r = max(vecnorm(honest - c, 2, 2)) + 1e-9;
            v = randn(size(c)); v = v / (norm(v) + 1e-12);
            m = c - 4.0 * r * v;
            msgs = repmat(reshape(m, [1 1 d]), numel(byz_idx), n, 1);
        case "mimic"
            m = honest(1, :);
            msgs = repmat(reshape(m, [1 1 d]), numel(byz_idx), n, 1);
        case "uniform"
            msgs = -2.5 + 5 * rand(numel(byz_idx), n, d);
        case "two_faced"
            % Per-recipient extremal lies (paper footnote 1).
            c = mean(honest, 1);
            r = max(vecnorm(honest - c, 2, 2)) + 1e-9;
            v = randn(1, n, d); v = v ./ (vecnorm(v, 2, 3) + 1e-12);
            m = c + 4.0 * r * squeeze(v);
            msgs = repmat(reshape(m, [1 n d]), numel(byz_idx), 1, 1);
        otherwise
            error("unknown attack: %s", attack_name);
    end
    for k = 1:numel(byz_idx)
        buf(byz_idx(k), :, :) = msgs(k, :, :);
    end
end

function [states, sets, byzantine, x_star, n] = build_demo(demo, n_per_axis, f)
    n_per_axis = max(n_per_axis, f + 1);
    if demo == "polyhedral"
        axes = [ 1  0;  -1  0;   0  1;   0 -1];     % four cardinal half-space normals
        normals = repelem(axes, n_per_axis, 1);
        n_honest = size(normals, 1);
        n = n_honest + f;
        sets = cell(n, 1);
        byzantine = false(n, 1);
        perm = randperm(n);
        inv = zeros(1, n); inv(perm) = 1:n;
        for k = 1:n_honest
            p = inv(k);
            sets{p} = halfspace_set(normals(k, :), 0.0);
        end
        for k = (n_honest + 1):n
            p = inv(k);
            a = randn(1, 2); a = a / (norm(a) + 1e-12);
            sets{p} = halfspace_set(a, rand() - 0.5);
            byzantine(p) = true;
        end
    else  % "box": four half-boxes whose intersection is {0}
        templates = {
            box_set([ 0 -1], [ 1  1])    % x >= 0
            box_set([-1 -1], [ 0  1])    % x <= 0
            box_set([-1  0], [ 1  1])    % y >= 0
            box_set([-1 -1], [ 1  0])    % y <= 0
        };
        n_honest = 4 * n_per_axis;
        n = n_honest + f;
        sets = cell(n, 1);
        byzantine = false(n, 1);
        perm = randperm(n);
        inv = zeros(1, n); inv(perm) = 1:n;
        for k = 1:n_honest
            p = inv(k);
            sets{p} = templates{floor((k - 1) / n_per_axis) + 1};
        end
        for k = (n_honest + 1):n
            p = inv(k);
            sets{p} = box_set([-1 -1], [1 1]);
            byzantine(p) = true;
        end
    end
    states = 0.6 * randn(n, 2);
    states(byzantine, :) = -2 + 4 * rand(sum(byzantine), 2);
    x_star = [0, 0];
end

function s = box_set(lo, hi)
    s.kind = "box";
    s.lo = lo; s.hi = hi;
    s.project = @(x) min(max(x, lo), hi);
end

function s = halfspace_set(a, b)
    s.kind = "halfspace";
    s.a = a; s.b = b;
    s.project = @(x) project_halfspace(x, a, b);
end

function y = project_halfspace(x, a, b)
    slack = a * x.' - b;
    if slack <= 0
        y = x;
    else
        y = x - (slack / (a * a.')) * a;
    end
end
