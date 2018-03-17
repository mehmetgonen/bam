function prediction = bprobit_classification_variational_test(X, state)
    N = size(X, 2);

    prediction.f.mu = [ones(1, N); X]' * state.bw.mu;
    prediction.f.sigma = 1 + diag([ones(1, N); X]' * state.bw.sigma * [ones(1, N); X]);

    pos = 1 - normcdf((+state.parameters.margin - prediction.f.mu) ./ prediction.f.sigma);
    neg = normcdf((-state.parameters.margin - prediction.f.mu) ./ prediction.f.sigma);
    prediction.p = pos ./ (pos + neg);
end
