data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
  int<lower=0> N_eval;
  matrix[N_eval, K] X_eval;
  int<lower=1> G;
  array[K] int<lower=1, upper=G> group_id;
  real<lower=2> nu;
  real y_mean;
  real<lower=1e-6> y_scale;
}

parameters {
  real<lower=-6, upper=6> alpha_raw;
  vector<lower=-6, upper=6>[K] beta_raw;
  real<lower=-6, upper=2> log_sigma_ratio;
}

transformed parameters {
  real alpha;
  vector[K] beta;
  real<lower=1e-6> sigma;

  alpha = y_mean + y_scale * alpha_raw;
  beta = y_scale * beta_raw;
  sigma = y_scale * exp(log_sigma_ratio);
}

model {
  alpha_raw ~ normal(0, 1);
  beta_raw ~ normal(0, 0.75);
  log_sigma_ratio ~ normal(log(0.5), 0.35);
  y ~ student_t(nu, alpha + X * beta, sigma);
}
