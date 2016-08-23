% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bprobit_classification_variational_train(X, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);

    log2pi = log(2 * pi);

    gamma.alpha = (parameters.alpha_gamma + 0.5);
    gamma.beta = parameters.beta_gamma;
    eta.alpha = (parameters.alpha_eta + 0.5) * ones(D, 1);
    eta.beta = parameters.beta_eta * ones(D, 1);
    bw.mu = [0; randn(D, 1)];
    bw.sigma = eye(D + 1, D + 1);
    f.mu = (abs(randn(N, 1)) + parameters.margin) .* sign(y);
    f.sigma = ones(N, 1);

    XXT = X * X';

    lower = -1e40 * ones(N, 1);
    lower(y > 0) = +parameters.margin;
    upper = +1e40 * ones(N, 1);
    upper(y < 0) = -parameters.margin;

    btimesbT.mu = bw.mu(1)^2 + bw.sigma(1, 1);
    wtimeswT.mu = bw.mu(2:D + 1) * bw.mu(2:D + 1)' + bw.sigma(2:D + 1, 2:D + 1);
    wtimesb.mu = bw.mu(2:D + 1) * bw.mu(1) + bw.sigma(2:D + 1, 1);
    iter = 0;
    bounds = [];
    while 1
        iter = iter + 1;
        %%%% update gamma
        gamma.beta = 1 / (1 / parameters.beta_gamma + 0.5 * btimesbT.mu);
        %%%% update eta
        eta.beta = 1 ./ (1 / parameters.beta_eta + 0.5 * diag(wtimeswT.mu));
        %%%% update b and w
        bw.sigma = [gamma.alpha * gamma.beta + N, sum(X, 2)'; sum(X, 2), diag(eta.alpha .* eta.beta) + XXT] \ eye(D + 1, D + 1);
        bw.mu = bw.sigma * ([ones(1, N); X] * f.mu);
        btimesbT.mu = bw.mu(1)^2 + bw.sigma(1, 1);
        wtimeswT.mu = bw.mu(2:D + 1) * bw.mu(2:D + 1)' + bw.sigma(2:D + 1, 2:D + 1);
        wtimesb.mu = bw.mu(2:D + 1) * bw.mu(1) + bw.sigma(2:D + 1, 1);
        %%%% update f
        output = [ones(1, N); X]' * bw.mu;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        f.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        f.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        lb = 0;
        %%%% p(gamma)
        lb = lb + (parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) - gamma.alpha * gamma.beta / parameters.beta_gamma - gammaln(parameters.alpha_gamma) - parameters.alpha_gamma * log(parameters.beta_gamma);
        %%%% p(b | gamma)
        lb = lb - 0.5 * gamma.alpha * gamma.beta * btimesbT.mu - 0.5 * (log2pi - (psi(gamma.alpha) + log(gamma.beta)));
        %%%% p(eta)
        lb = lb + sum((parameters.alpha_eta - 1) * (psi(eta.alpha) + log(eta.beta)) - eta.alpha .* eta.beta / parameters.beta_eta - gammaln(parameters.alpha_eta) - parameters.alpha_eta * log(parameters.beta_eta));
        %%%% p(w | eta)
        lb = lb - 0.5 * sum(sum(eta.alpha .* eta.beta .* diag(wtimeswT.mu))) - 0.5 * (D * log2pi - sum(psi(eta.alpha) + log(eta.beta)));
        %%%% p(f | b, w, X)
        lb = lb - 0.5 * (f.mu' * f.mu + sum(f.sigma)) + f.mu' * (X' * bw.mu(2:D + 1)) + sum(bw.mu(1) * f.mu) - 0.5 * sum(sum(wtimeswT.mu .* XXT)) - sum(X' * wtimesb.mu) - 0.5 * N * btimesbT.mu - 0.5 * N * log2pi;

        %%%% q(gamma)
        lb = lb + gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) * psi(gamma.alpha);
        %%%% q(eta)
        lb = lb + sum(eta.alpha + log(eta.beta) + gammaln(eta.alpha) + (1 - eta.alpha) .* psi(eta.alpha));
        %%%% q(b, w)
        lb = lb + 0.5 * ((D + 1) * (log2pi + 1) + logdet(bw.sigma)); 
        %%%% q(f)
        lb = lb + 0.5 * sum(log2pi + f.sigma) + sum(log(normalization));

        bounds = [bounds; lb]; %#ok<AGROW>
        fprintf(1, 'Iteration: %d Lower bound: %g\n', iter, lb);
        if iter > 1
            if abs((bounds(end) - bounds(end - 1)) ./ bounds(end - 1)) < parameters.threshold
                break;
            end
        end
        if iter == parameters.iteration
            break;
        end
    end

    state.gamma = gamma;
    state.eta = eta;
    state.bw = bw;
    state.bounds = bounds;
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
    ld = 2 * sum(log(diag(U)));
end