data {
  int<lower=1> N;
  vector[N] weight;
  vector[N] polar;
  vector[N] surface;
  vector[N] bond_order;
  vector[N] heavy_log;
  vector[N] halogen_log;
  vector[N] aromatic_ring_log;
  vector[N] aromatic_fraction;
  vector[N] rotatable_log;
  vector[N] y;
}

parameters {
  real intercept;
  real<lower=0> linear_scale;
  real<lower=0> quadratic_scale;
  real<lower=0> descriptor_scale;

  real weight_coeff;
  real polar_coeff;
  real surface_coeff;
  real bond_coeff;

  real heavy_coeff;
  real halogen_coeff;
  real aromatic_ring_coeff;
  real aromatic_fraction_coeff;
  real rotatable_coeff;

  real weight_sq_coeff;
  real polar_sq_coeff;
  real surface_sq_coeff;
  real interaction_wp;
  real interaction_ws;
}

transformed parameters {
  vector[N] mu;
  mu =
      intercept
    + weight_coeff * weight
    + polar_coeff * polar
    + surface_coeff * surface
    + bond_coeff * bond_order
    + heavy_coeff * heavy_log
    + halogen_coeff * halogen_log
    + aromatic_ring_coeff * aromatic_ring_log
    + aromatic_fraction_coeff * aromatic_fraction
    + rotatable_coeff * rotatable_log
    + weight_sq_coeff * square(weight)
    + polar_sq_coeff * square(polar)
    + surface_sq_coeff * square(surface)
    + interaction_wp * elt_multiply(weight, polar)
    + interaction_ws * elt_multiply(weight, surface);
}

model {
  linear_scale ~ gamma(2, 5);
  quadratic_scale ~ gamma(2, 20);
  descriptor_scale ~ gamma(2, 10);

  intercept ~ normal(0, 0.5);

  weight_coeff ~ normal(0, linear_scale);
  polar_coeff ~ normal(0, linear_scale);
  surface_coeff ~ normal(0, linear_scale);
  bond_coeff ~ normal(0, linear_scale);

  heavy_coeff ~ normal(0, descriptor_scale);
  halogen_coeff ~ normal(0, descriptor_scale);
  aromatic_ring_coeff ~ normal(0, descriptor_scale);
  aromatic_fraction_coeff ~ normal(0, descriptor_scale);
  rotatable_coeff ~ normal(0, descriptor_scale);

  weight_sq_coeff ~ normal(0, quadratic_scale);
  polar_sq_coeff ~ normal(0, quadratic_scale);
  surface_sq_coeff ~ normal(0, quadratic_scale);
  interaction_wp ~ normal(0, quadratic_scale);
  interaction_ws ~ normal(0, quadratic_scale);

  y ~ normal(mu, 0.2);
}

