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
  vector<lower=-4, upper=4>[K] beta_raw;
  real<lower=-6, upper=2> log_sigma_ratio;
  real<lower=-4, upper=0.5> log_global_scale;
  vector<lower=-4, upper=1>[G] log_group_scale;
}

transformed parameters {
  real alpha;
  real<lower=1e-6> sigma;
  real<lower=0> global_scale;
  vector<lower=0>[G] group_scale;
  vector[K] beta;

  alpha = y_mean + y_scale * alpha_raw;
  sigma = y_scale * exp(log_sigma_ratio);
  global_scale = exp(log_global_scale);
  for (g in 1:G) {
    group_scale[g] = exp(log_group_scale[g]);
  }
  for (k in 1:K) {
    beta[k] = y_scale * beta_raw[k] * global_scale * group_scale[group_id[k]];
  }
}

model {
  alpha_raw ~ normal(0, 1);
  beta_raw ~ normal(0, 1);
  log_sigma_ratio ~ normal(log(0.5), 0.35);
  log_global_scale ~ normal(log(0.35), 0.35);
  log_group_scale ~ normal(0, 0.25);
  y ~ student_t(nu, alpha + X * beta, sigma);
}
