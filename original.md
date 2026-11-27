function [X2, success, info] = SARD_SDM_from_data( ...
    V, I, n, Rs, T_K, bounds_lin, opts)
%% 1. Default settings

if nargin < 6 || isempty(bounds_lin)
    % Physical bounds from the SDM parameter settings in the manuscript:
    % [Iph; I0; Rsh]
    bounds_lin = [0, 1;
                  0, 1e-3;
                  0, 100];
end

if nargin < 7 || isempty(opts)
    opts = struct();
end

if ~isfield(opts, 'lambda0')
    opts.lambda0 = 1e-10;
end

if ~isfield(opts, 'alpha')
    opts.alpha = 0.8;
end

if ~isfield(opts, 'tau')
    opts.tau = 1e8;
end

if ~isfield(opts, 'scale_eps')
    opts.scale_eps = eps;
end

if ~isfield(opts, 'feas_tol')
    opts.feas_tol = 1e-10;
end

%% 2. Initialize outputs and diagnostic information

success = false;

X2 = nan(3,1);

info = struct();

info.cond_history = [];
info.lambda_history = [];
info.colnorms = [];
info.solver = 'Not selected';
info.condA_final = NaN;
info.residual = [];
info.residual_norm = NaN;
info.linear_coefficients = nan(3,1);
info.Gsh = NaN;
info.message = '';

%% 3. Validate input data

V = V(:);
I = I(:);

if numel(V) ~= numel(I) || numel(V) < 3
    info.message = 'Invalid voltage/current data dimensions.';
    return;
end

if any(isnan(V)) || any(isinf(V)) || ...
   any(isnan(I)) || any(isinf(I))

    info.message = 'Input data contain NaN or Inf.';
    return;
end

if ~isscalar(n) || ~isscalar(Rs) || ~isscalar(T_K) || ...
   n <= 0 || Rs < 0 || T_K <= 0

    info.message = 'Invalid nonlinear structural parameters.';
    return;
end

if ~isequal(size(bounds_lin), [3,2]) || ...
   any(isnan(bounds_lin(:))) || ...
   any(isinf(bounds_lin(:))) || ...
   any(bounds_lin(:,1) > bounds_lin(:,2))

    info.message = 'Invalid physical parameter bounds.';
    return;
end

if opts.lambda0 <= 0 || opts.alpha < 0 || ...
   opts.tau <= 0 || opts.scale_eps <= 0

    info.message = 'Invalid SARD numerical settings.';
    return;
end

%% 4. Construct the conditional coefficient matrix

% Physical constants
q  = 1.602176634e-19;
kB = 1.380649e-23;

% Voltage across the diode
Vd = V + I .* Rs;

% Exponential basis function
exponent = q .* Vd ./ (n .* kB .* T_K);

% Prevent exponential overflow
exponent = max(min(exponent, 700), -700);

G = -(exp(exponent) - 1);

% Resistive basis function
H = -Vd;

% Conditional linear system:
%
% I = Iph + I0 * G + Gsh * H
%
% M * XL = I
%
% XL = [Iph; I0; Gsh]

N = numel(I);

M = [ones(N,1), G, H];

if any(isnan(M(:))) || any(isinf(M(:)))
    info.message = 'Non-finite coefficient matrix.';
    return;
end

%% 5. Column scaling and matrix equilibration

% Calculate the L2 norm of each column
col_scale = sqrt(sum(M.^2, 1))';

% Avoid division by zero
col_scale = max(col_scale, opts.scale_eps);

% Construct the scale-equilibrated matrix:
%
% M_tilde = M * S^(-1)

M_scaled = bsxfun(@rdivide, M, col_scale');

info.colnorms = col_scale;

%% 6. Evaluate the condition number

% Singular values of the scaled coefficient matrix
sM = svd(M_scaled, 'econ');

if isempty(sM) || sM(1) <= 0
    info.message = 'Invalid singular values of the scaled matrix.';
    return;
end

% Condition number of the scaled matrix
kappa_s = sM(1) / max(sM(end), eps(sM(1)));

info.condA_final = kappa_s;
info.cond_history = kappa_s;

%% 7. Condition-aware adaptive regularization

% Adaptive regularization:
%
% lambda = lambda0 * cond(M_scaled)^alpha

lambda = opts.lambda0 * max(kappa_s, 1)^opts.alpha;

if isnan(lambda) || isinf(lambda) || lambda <= 0
    info.message = 'Invalid adaptive regularization coefficient.';
    return;
end

info.lambda_history = lambda;

% Structure-preserving regularization matrix:
%
% Gamma = diag(0, 1, 1)

Gamma_sqrt = diag([0, 1, 1]);

%% 8. Construct the augmented least-squares system

% B = [M_scaled; sqrt(lambda) * Gamma^(1/2)]
%
% y = [I; 0]

B = [M_scaled;
     sqrt(lambda) .* Gamma_sqrt];

y_aug = [I;
         zeros(3,1)];

%% 9. Select QR or SVD according to matrix conditioning

use_svd = (kappa_s > opts.tau);

z = nan(3,1);

if ~use_svd

    % QR decomposition for a sufficiently well-conditioned matrix
    try

        [Q, R] = qr(B, 0);

        if rcond(R) > eps

            z = R \ (Q' * y_aug);

            info.solver = 'QR';

        else

            use_svd = true;

        end

    catch

        use_svd = true;

    end

end

if use_svd

    % SVD-based solution for an ill-conditioned matrix
    [U, S, V_svd] = svd(B, 'econ');

    singular_values = diag(S);

    if isempty(singular_values) || singular_values(1) <= 0
        info.message = 'SVD failed to produce valid singular values.';
        return;
    end

    % Numerical pseudoinverse tolerance
    svd_tol = max(size(B)) * eps(singular_values(1));

    valid_idx = singular_values > svd_tol;

    if ~any(valid_idx)
        info.message = 'The augmented system is numerically rank deficient.';
        return;
    end

    % Truncated SVD pseudoinverse
    z = V_svd(:,valid_idx) * ...
        ((U(:,valid_idx)' * y_aug) ./ singular_values(valid_idx));

    info.solver = 'SVD';

end

if any(isnan(z)) || any(isinf(z))
    info.message = 'The recovered scaled coefficients are invalid.';
    return;
end

%% 10. Recover the conditionally linear coefficients

% XL = S^(-1) * z

XL = z ./ col_scale;

Iph = XL(1);
I0  = XL(2);
Gsh = XL(3);

info.linear_coefficients = XL;
info.Gsh = Gsh;

%% 11. Reconstruct the physical parameters

% Shunt resistance:
%
% Rsh = 1 / Gsh

if abs(Gsh) > opts.scale_eps

    Rsh = 1 / Gsh;

else

    Rsh = Inf;

end

% Complete physical parameter vector:
%
% X2 = [Iph; I0; Rsh]

X2 = [Iph;
      I0;
      Rsh];

%% 12. Check physical feasibility (without clipping)

% SARD recovery is not followed by coefficient clipping.
% Solutions outside the prescribed physical ranges are rejected.

bound_tol = opts.feas_tol .* max(1, abs(X2));

lower_ok = X2 >= bounds_lin(:,1) - bound_tol;

upper_ok = X2 <= bounds_lin(:,2) + bound_tol;

physical_ok = all(lower_ok) && all(upper_ok) && (Gsh > 0);

finite_ok = ~any(isnan(X2)) && ~any(isinf(X2));

success = physical_ok && finite_ok;

%% 13. Calculate the linear recovery residual

residual = M * XL - I;

info.residual = residual;

info.residual_norm = norm(residual, 2);

%% 14. Return diagnostic information

if success

    info.message = 'SARD coefficient recovery completed successfully.';

else

    info.message = ...
        'Recovered coefficients violate physical constraints or parameter bounds.';

end

end
