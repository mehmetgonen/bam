function prediction = bam_pairwise_classification_variational_test(X, state)
    N = size(X, 2);

    prediction.f.mu = [ones(1, N); X]' * state.bw.mu;
    prediction.f.sigma = 1 + diag([ones(1, N); X]' * state.bw.sigma * [ones(1, N); X]);

    prediction.p = 1 - normcdf(-prediction.f.mu ./ prediction.f.sigma);
end
