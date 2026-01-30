import distrax
import jax
import jax.numpy as jnp

from blackjax.ns.utils import sample
from common import compute_metrics, plot_comparison_grid, print_results
from nss.ns import run_nested_sampling, run_rw_nested_sampling
from nss.nuts import run_hmc
from nss.smc import (
    run_hmc_sequential_mc,
    run_independent_sequential_mc,
    run_rw_sequential_mc,
    run_ss_sequential_mc,
    sample_smc,
)
from nss.ss import run_slice


rng_key = jax.random.PRNGKey(0)
dim = 2
n_mixes = 40
loc_scaling = 40.0
log_var_scaling = 1.0

rng_key, key = jax.random.split(rng_key)
means = jax.random.uniform(key, (n_mixes, dim), minval=-1, maxval=1) * loc_scaling
tril = jax.nn.softplus(jnp.ones((n_mixes, dim)) * log_var_scaling)[:, None] * jnp.eye(dim)
cat_probs = jnp.ones(n_mixes)

mix = distrax.Categorical(probs=cat_probs)
com = distrax.MultivariateNormalTri(loc=means, scale_tri=tril)
mog = distrax.MixtureSameFamily(mixture_distribution=mix, components_distribution=com)

prior = distrax.Uniform(low=-50 * jnp.ones(dim), high=50 * jnp.ones(dim))
prior = distrax.Independent(prior, reinterpreted_batch_ndims=1)

true_logZ = prior.log_prob(jnp.ones(dim))

rng_key, sampling_key, initial_samples_key = jax.random.split(rng_key, 3)
initial_samples = prior.sample(seed=initial_samples_key, sample_shape=(1000,))

ss_states, ss_res = run_slice(
    sampling_key, mog.log_prob, prior.log_prob, initial_samples, num_steps=10000
)

hmc_states, hmc_res = run_hmc(
    sampling_key, mog.log_prob, prior.log_prob, initial_samples, n_samples=10000
)

rng_key, sampling_key = jax.random.split(rng_key)
smc_ss_states, smc_ss_res = run_ss_sequential_mc(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=2, target_ess=0.9, initial_samples=initial_samples
)

rng_key, sampling_key = jax.random.split(rng_key)
smc_hmc_states, smc_hmc_res = run_hmc_sequential_mc(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=2, hmc_trajectory_length=5, target_ess=0.9, initial_samples=initial_samples
)

rng_key, sampling_key = jax.random.split(rng_key)
ns_states, ns_res = run_nested_sampling(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=2, initial_samples=initial_samples, num_delete=100
)

rng_key, sampling_key = jax.random.split(rng_key)
ns_mc_states, ns_mc_res = run_rw_nested_sampling(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=10, initial_samples=initial_samples, num_delete=100
)

rng_key, sampling_key = jax.random.split(rng_key)
smc_states, smc_res = run_rw_sequential_mc(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=10, target_ess=0.9, initial_samples=initial_samples
)

rng_key, sampling_key = jax.random.split(rng_key)
smc_i_states, smc_i_res = run_independent_sequential_mc(
    sampling_key, mog.log_prob, prior.log_prob, num_mcmc_steps=10, target_ess=0.9, initial_samples=initial_samples
)

# Extract samples
rng_key, ns_sample_key, smc_sample_key = jax.random.split(rng_key, 3)
ns_samples = sample(ns_sample_key, ns_states)
ns_mc_samples = sample(ns_sample_key, ns_mc_states)
smc_samples = sample_smc(smc_sample_key, smc_states)
smc_i_samples = sample_smc(smc_sample_key, smc_i_states)
smc_hmc_samples = sample_smc(smc_sample_key, smc_hmc_states)
smc_ss_samples = sample_smc(smc_sample_key, smc_ss_states)
hmc_samples = hmc_states.position[::10]
ss_samples = ss_states.position[::10]
rng_key, true_samples_key = jax.random.split(rng_key)
true_samples = mog.sample(seed=true_samples_key, sample_shape=(1000,))

# Plot
rng_key, plot_key = jax.random.split(rng_key)
plotting_bounds = (-loc_scaling * 1.4, loc_scaling * 1.4)
plot_comparison_grid(
    true_samples=true_samples,
    algo_samples=[ss_samples, hmc_samples, ns_samples.position, ns_mc_samples.position, smc_ss_samples, smc_samples, smc_i_samples, smc_hmc_samples],
    algo_names=["SS", "NUTS", "NSS", "NS-RW", "SMC-SS", "SMC-RW", "SMC-IRMH", "SMC-HMC"],
    bounds=plotting_bounds,
    output_prefix="plots/mixture_gaussians",
    rng_key=plot_key,
)

# Set names
ss_res.name, hmc_res.name = "SS", "NUTS"
ns_res.name, ns_mc_res.name = "NSS", "NS-RW"
smc_ss_res.name, smc_res.name, smc_i_res.name, smc_hmc_res.name = "SMC-SS", "SMC-RW", "SMC-IRMH", "SMC-HMC"

# Metrics
samples_list = [ss_samples, hmc_samples, ns_samples.position, ns_mc_samples.position, smc_ss_samples, smc_samples, smc_i_samples, smc_hmc_samples]
results_list = [ss_res, hmc_res, ns_res, ns_mc_res, smc_ss_res, smc_res, smc_i_res, smc_hmc_res]

rng_key, metrics_key = jax.random.split(rng_key)
ground_truth_mmd, ground_truth_w2 = compute_metrics(
    metrics_key,
    samples_list,
    results_list,
    target_sampler=lambda k: mog.sample(seed=k, sample_shape=(200,)),
    true_samples=true_samples,
)

print_results(results_list, ground_truth_mmd, ground_truth_w2)
