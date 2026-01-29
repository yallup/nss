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


D = 10
extent = 10.0
theta_prior = distrax.Normal(0.0, 3.0)
z_prior = distrax.Uniform(low=-extent * jnp.ones(D), high=extent * jnp.ones(D))


def prior_logprob(x):
    logp_theta = theta_prior.log_prob(x["theta"])
    logp_z = z_prior.log_prob(x["z"]).sum(axis=-1)
    return logp_theta + logp_z


def simulate_funnel(key, N, M=1000):
    key, ktheta, kz = jax.random.split(key, 3)
    theta = jax.random.normal(ktheta, shape=(M,)) * 3.0
    std_z = jnp.exp(theta / 2.0)
    z = jax.random.normal(kz, shape=(M, N)) * std_z[:, None]
    return key, {"theta": theta, "z": z}


def logp_z_given_theta(z, theta):
    var = jnp.exp(theta)
    return -0.5 * (jnp.sum(z**2, axis=-1) / var + z.shape[-1] * (jnp.log(var) + jnp.log(2.0 * jnp.pi)))


def loglikelihood(x):
    return logp_z_given_theta(x["z"], x["theta"])


def posterior_logprob(x):
    return prior_logprob(x) + loglikelihood(x)


def pack_samples(samples):
    """Flatten dict samples to (n, D+1) array for metrics."""
    if hasattr(samples, "position"):
        samples = samples.position
    if isinstance(samples, dict):
        return jnp.concatenate([samples["theta"][..., None], samples["z"]], axis=-1)
    return samples


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    N = 10
    key, true_samples = simulate_funnel(subkey, N)

    # Generate initial samples from prior
    key, initial_samples_key = jax.random.split(key)
    n_initial = 1000
    initial_theta = jax.random.normal(initial_samples_key, shape=(n_initial,)) * 3.0
    key, z_key = jax.random.split(key)
    initial_z = jax.random.uniform(z_key, shape=(n_initial, D), minval=-extent, maxval=extent)
    initial_samples = {"theta": initial_theta, "z": initial_z}

    # Run algorithms
    key, sampling_key = jax.random.split(key)
    ss_states, ss_res = run_slice(sampling_key, posterior_logprob, prior_logprob, initial_samples, num_steps=10000)

    key, subkey = jax.random.split(key)
    hmc_states, hmc_res = run_hmc(subkey, loglikelihood, prior_logprob, initial_samples, n_samples=10000)

    key, sampling_key = jax.random.split(key)
    ns_states, ns_res = run_nested_sampling(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=D, initial_samples=initial_samples, num_delete=100, termination=-5
    )

    key, sampling_key = jax.random.split(key)
    ns_mc_states, ns_mc_res = run_rw_nested_sampling(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=5 * D, initial_samples=initial_samples, num_delete=100, termination=-5
    )

    key, sampling_key = jax.random.split(key)
    smc_ss_states, smc_ss_res = run_ss_sequential_mc(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=D, target_ess=0.95, initial_samples=initial_samples
    )

    key, sampling_key = jax.random.split(key)
    smc_states, smc_res = run_rw_sequential_mc(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=5 * D, target_ess=0.95, initial_samples=initial_samples
    )

    key, sampling_key = jax.random.split(key)
    smc_i_states, smc_i_res = run_independent_sequential_mc(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=5 * D, target_ess=0.95, initial_samples=initial_samples
    )

    key, sampling_key = jax.random.split(key)
    smc_hmc_states, smc_hmc_res = run_hmc_sequential_mc(
        sampling_key, posterior_logprob, prior_logprob, num_mcmc_steps=2, hmc_trajectory_length=5, target_ess=0.95, initial_samples=initial_samples
    )

    # Extract samples
    key, ns_sample_key, smc_sample_key = jax.random.split(key, 3)
    ns_samples = sample(ns_sample_key, ns_states).position
    ns_mc_samples = sample(ns_sample_key, ns_mc_states).position
    smc_samples = sample_smc(smc_sample_key, smc_states)
    smc_i_samples = sample_smc(smc_sample_key, smc_i_states)
    smc_hmc_samples = sample_smc(smc_sample_key, smc_hmc_states)
    smc_ss_samples = sample_smc(smc_sample_key, smc_ss_states)
    ss_samples = {"theta": ss_states.position["theta"][::10], "z": ss_states.position["z"][::10]}
    hmc_samples = {"theta": hmc_states.position["theta"][::10], "z": hmc_states.position["z"][::10]}

    # Plot
    plotting_bounds = (-extent, extent)
    plot_comparison_grid(
        true_samples=true_samples,
        algo_samples=[ss_samples, hmc_samples, ns_samples, ns_mc_samples, smc_ss_samples, smc_samples, smc_i_samples, smc_hmc_samples],
        algo_names=["SS", "NUTS", "NSS", "NS-RW", "SMC-SS", "SMC-RW", "SMC-IRMH", "SMC-HMC"],
        bounds=plotting_bounds,
        output_prefix="plots/funnel",
        dim_y="z",
    )

    # Set names
    ss_res.name, hmc_res.name = "SS", "NUTS"
    ns_res.name, ns_mc_res.name = "NSS", "NS-RW"
    smc_ss_res.name, smc_res.name, smc_i_res.name, smc_hmc_res.name = "SMC-SS", "SMC-RW", "SMC-IRMH", "SMC-HMC"

    # Metrics
    samples_list = [ss_samples, hmc_samples, ns_samples, ns_mc_samples, smc_ss_samples, smc_samples, smc_i_samples, smc_hmc_samples]
    results_list = [ss_res, hmc_res, ns_res, ns_mc_res, smc_ss_res, smc_res, smc_i_res, smc_hmc_res]

    key, metrics_key = jax.random.split(key)
    ground_truth_mmd, ground_truth_w2 = compute_metrics(
        metrics_key,
        samples_list,
        results_list,
        target_sampler=lambda k: pack_samples(simulate_funnel(k, N, M=200)[1]),
        true_samples=true_samples,
        pack_fn=pack_samples,
    )

    print_results(results_list, ground_truth_mmd, ground_truth_w2)
