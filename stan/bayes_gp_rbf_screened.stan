functions {
  matrix rbf_covariance(matrix X, real signal_scale, real lengthscale, real sigma) {
    int N = rows(X);
    matrix[N, N] K;
    real signal_variance = square(signal_scale);
    real inv_lengthscale_sq = inv_square(lengthscale);
    for (i in 1:N) {
      row_vector[cols(X)] xi = row(X, i);
      K[i, i] = signal_variance + square(sigma) + 1e-8;
      if (i < N) {
        for (j in (i + 1):N) {
          row_vector[cols(X)] xj = row(X, j);
          real value = signal_variance * exp(-0.5 * squared_distance(xi, xj) * inv_lengthscale_sq);
          K[i, j] = value;
          K[j, i] = value;
        }
      }
    }
    return K;
  }
}

data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
  real y_mean;
  real<lower=1e-6> y_scale;
}

parameters {
  real alpha;
  real<lower=1e-6> signal_scale;
  real<lower=1e-6> lengthscale;
  real<lower=1e-6> sigma;
}

model {
  matrix[N, N] K_cov = rbf_covariance(X, signal_scale, lengthscale, sigma);
  matrix[N, N] L_cov = cholesky_decompose(K_cov);

  alpha ~ normal(y_mean, y_scale);
  signal_scale ~ lognormal(log(y_scale), 0.6);
  lengthscale ~ lognormal(log(sqrt(K)), 0.6);
  sigma ~ lognormal(log(0.35 * y_scale), 0.5);

  y ~ multi_normal_cholesky(rep_vector(alpha, N), L_cov);
}
