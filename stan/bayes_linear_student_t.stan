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
}

parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ student_t(nu, alpha + X * beta, sigma);
}

generated quantities {
  vector[N] mu_train;
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N_eval] mu_eval;

  mu_train = alpha + X * beta;
  mu_eval = alpha + X_eval * beta;
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu_train[n], sigma);
    y_rep[n] = student_t_rng(nu, mu_train[n], sigma);
  }
}

