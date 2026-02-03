from timeit import default_timer as timer
from typing import Any, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import tqdm
from blackjax.ns.utils import get_first_row
from blackjax.smc.ess import ess as smc_ess
from blackjax.smc.pretuning import esjd
from blackjax.smc.resampling import systematic

from nss.utils import Results, concat_particles


class SMCState(NamedTuple):
    """State containing particles and weights from all SMC rungs."""

    particles: Any  # pytree with shape (n_total, ...)
    weights: jax.Array  # normalized weights, shape (n_total,)


def smc_weights(all_log_weights, cumulative_log_zs):
    """Compute normalized weights for particles from all SMC rungs.

    Each rung's log-weights are shifted by the negative cumulative log evidence
    at that rung, so that later rungs (closer to the posterior)
    contribute proportionally more.

    Parameters
    ----------
    all_log_weights : list of arrays
        Log-weights at each rung, each of shape (n_particles,).
    cumulative_log_zs : array
        Cumulative log evidence at each rung, shape (n_rungs,).

    Returns
    -------
    array
        Normalized weights across all particles from all rungs, shape (n_total,).
    """
    adjusted = [lw - clz for lw, clz in zip(all_log_weights, cumulative_log_zs)]
    all_log_w = jnp.concatenate(adjusted)
    log_sum = jax.scipy.special.logsumexp(all_log_w)
    return jnp.exp(all_log_w - log_sum)


def run_independent_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    target_ess=0.9,
):
    kernel = blackjax.irmh.build_kernel()

    def step(key, state, logdensity, means, cov):
        def proposal_distribution(key):
            _, ravel_fn = jax.flatten_util.ravel_pytree(state.position)
            return ravel_fn(jax.random.multivariate_normal(key, means, cov))

        def proposal_logdensity_fn(proposal, state):
            x, _ = jax.flatten_util.ravel_pytree(state.position)
            return jax.scipy.stats.multivariate_normal.logpdf(x, mean=means, cov=cov)

        return kernel(
            key,
            state,
            logdensity,
            proposal_distribution,
            proposal_logdensity_fn,
        )

    mean = blackjax.smc.tuning.from_particles.particles_means(initial_samples)
    cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
        initial_samples
    )
    init_params = {"means": mean, "cov": cov}

    def irmh_update_fn(key, state, info):
        mean = blackjax.smc.tuning.from_particles.particles_means(state.particles)
        cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
            state.particles
        )
        return blackjax.smc.extend_params({"means": mean, "cov": cov})

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.rmh.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=irmh_update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    # Warmup JIT and block until compiled
    (_, _), _ = jax.block_until_ready(one_step((state, sample_key), None))

    steps = 0
    log_zs = []
    all_log_weights = []
    all_particles = []
    t0 = timer()
    with tqdm.tqdm(desc="SMC-IRMH", unit=" step") as pbar:
        while state[0].tempering_param < 1:
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            all_log_weights.append(jnp.log(state.sampler_state.weights + 1e-16))
            all_particles.append(state.sampler_state.particles)
            pbar.set_postfix({"λ": f"{state[0].tempering_param:.3f}", "logZ": f"{sum(log_zs):.2f}"})
            pbar.update(1)

    t1 = timer()
    cumulative_log_zs = jnp.cumsum(jnp.array(log_zs))
    weights = smc_weights(all_log_weights, cumulative_log_zs)
    ess = smc_ess(jnp.log(weights + 1e-16))
    logzs = jnp.array(log_zs).sum()
    particles = concat_particles(all_particles)
    smc_state = SMCState(particles=particles, weights=weights)
    res = Results(
        name="SMC IRMH",
        time=t1 - t0,
        evals=jnp.prod(jnp.array(smc_info.update_info.is_accepted.shape)) * steps,
        ess=float(ess),
        logZs=logzs,
    )
    return smc_state, res


def sample_smc(rng_key, smc_state, n=1000):
    """Sample from SMC state using importance weights.

    Parameters
    ----------
    rng_key : PRNGKey
        Random key for sampling.
    smc_state : SMCState
        State containing particles and weights from all SMC rungs.
    n : int
        Number of samples to draw.

    Returns
    -------
    pytree
        Resampled particles with shape (n, ...).
    """
    indices = jax.random.choice(
        rng_key,
        smc_state.weights.shape[0],
        p=smc_state.weights,
        shape=(n,),
        replace=True,
    )
    return jax.tree_util.tree_map(lambda leaf: leaf[indices], smc_state.particles)


def run_rw_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    target_ess=0.9,
    max_steps=None,
):
    kernel = blackjax.mcmc.random_walk.build_additive_step()

    def step(key, state, logdensity, cov):
        def proposal_distribution(key, position):
            x, ravel_fn = jax.flatten_util.ravel_pytree(position)
            return ravel_fn(jax.random.multivariate_normal(key, jnp.zeros_like(x), cov))

        return kernel(
            key,
            state,
            logdensity,
            proposal_distribution,
        )

    cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
        initial_samples
    )
    init_params = {"cov": cov}

    def update_fn(key, state, info):
        cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
            state.particles
        )
        return blackjax.smc.extend_params({"cov": cov})

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.rmh.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    # Warmup JIT and block until compiled
    (_, _), _ = jax.block_until_ready(one_step((state, sample_key), None))

    steps = 0
    log_zs = []
    all_log_weights = []
    all_particles = []
    t0 = timer()
    with tqdm.tqdm(desc="SMC-RW", unit=" step") as pbar:
        while state[0].tempering_param < 1:
            if max_steps is not None and steps >= max_steps:
                break
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            all_log_weights.append(jnp.log(state.sampler_state.weights + 1e-16))
            all_particles.append(state.sampler_state.particles)
            pbar.set_postfix({"λ": f"{state[0].tempering_param:.3f}", "logZ": f"{sum(log_zs):.2f}"})
            pbar.update(1)
    t1 = timer()
    cumulative_log_zs = jnp.cumsum(jnp.array(log_zs))
    weights = smc_weights(all_log_weights, cumulative_log_zs)
    ess = smc_ess(jnp.log(weights + 1e-16))
    logzs = jnp.array(log_zs).sum()
    particles = concat_particles(all_particles)
    smc_state = SMCState(particles=particles, weights=weights)
    res = Results(
        name="SMC RW",
        time=t1 - t0,
        evals=jnp.prod(jnp.array(smc_info.update_info.is_accepted.shape)) * steps,
        ess=float(ess),
        logZs=logzs,
    )
    return smc_state, res


def run_hmc_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    hmc_trajectory_length=5,
    target_ess=0.9,
    target_acceptance_rate=0.8,
    warmup_steps=200,
    max_steps=None,
):
    # Joint logdensity
    def logdensity_fn(x):
        return loglikelihood_fn(x) + prior_logprob(x)

    class TransitionInfo(NamedTuple):
        start: jax.Array
        end: jax.Array
        acceptence: jax.Array

    kernel = blackjax.hmc.build_kernel()

    def step(key, state, logdensity, step_size, inverse_mass_matrix):
        new_state, transition_info = kernel(
            key,
            state,
            logdensity,
            num_integration_steps=hmc_trajectory_length,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        tinfo = TransitionInfo(
            start=state.position,
            end=new_state.position,
            acceptence=transition_info.acceptance_rate,
        )
        return new_state, tinfo

    # Window adaptation warmup for step size + mass matrix
    adapt = blackjax.window_adaptation(
        blackjax.hmc,
        logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
        num_integration_steps=hmc_trajectory_length,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    (last_state, parameters), adapt_info = adapt.run(
        warmup_key, blackjax.ns.utils.get_first_row(initial_samples), warmup_steps
    )

    # Compute static shapes for ESJD subsampling
    first_sample, _ = jax.flatten_util.ravel_pytree(get_first_row(initial_samples))
    ndim = first_sample.shape[0]
    n_particles = jax.tree_util.tree_leaves(initial_samples)[0].shape[0]
    n_total = n_particles * num_mcmc_steps
    stride = max(n_total // 100, 1)
    identity = jnp.eye(ndim)

    @jax.jit
    def update_fn(key, state, info):
        # Inverse mass matrix from particle variances (diagonal matrix)
        imm = blackjax.smc.tuning.from_particles.inverse_mass_matrix_from_particles(
            state.particles
        )

        # ESJD-based step size tuning
        start, _ = jax.flatten_util.ravel_pytree(info.update_info.start)
        end, _ = jax.flatten_util.ravel_pytree(info.update_info.end)
        acc, _ = jax.flatten_util.ravel_pytree(info.update_info.acceptence)

        # Reshape: (n_particles * n_steps * dim,) -> (n_particles * n_steps, dim)
        start = start.reshape(n_total, ndim)
        end = end.reshape(n_total, ndim)

        # Use identity matrix: ESJD in position space (units: distance^2)
        measure = esjd(identity)
        esjd_vals = measure(start[::stride], end[::stride], acc[::stride])
        # Step size from sqrt(ESJD) / trajectory_length (units: distance)
        tuned_step = jnp.maximum(
            jnp.sqrt(jnp.mean(esjd_vals)) / jnp.maximum(hmc_trajectory_length, 1), 1e-6
        )

        return blackjax.smc.extend_params(
            {
                "inverse_mass_matrix": imm.diagonal(),
                "step_size": tuned_step,
            }
        )

    init_params = {
        "inverse_mass_matrix": parameters["inverse_mass_matrix"],
        "step_size": parameters["step_size"],
    }

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.hmc.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    # Warmup JIT and block until compiled
    (_, _), _ = jax.block_until_ready(one_step((state, sample_key), None))

    steps = 0
    log_zs = []
    all_log_weights = []
    all_particles = []
    t0 = timer()
    with tqdm.tqdm(desc="SMC-HMC", unit=" step") as pbar:
        while state[0].tempering_param < 1:
            if max_steps is not None and steps >= max_steps:
                print(f"SMC-HMC reached max steps ({max_steps}). Terminating at lambda={state[0].tempering_param:.4f}")
                break
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            all_log_weights.append(jnp.log(state.sampler_state.weights + 1e-16))
            all_particles.append(state.sampler_state.particles)
            pbar.set_postfix({"λ": f"{state[0].tempering_param:.3f}", "logZ": f"{sum(log_zs):.2f}"})
            pbar.update(1)
    t1 = timer()
    cumulative_log_zs = jnp.cumsum(jnp.array(log_zs))
    weights = smc_weights(all_log_weights, cumulative_log_zs)
    ess = smc_ess(jnp.log(weights + 1e-16))
    logzs = jnp.array(log_zs).sum()
    particles = concat_particles(all_particles)
    smc_state = SMCState(particles=particles, weights=weights)
    # Evals: particles * mcmc_steps * trajectory_length * annealing_steps + warmup
    n_particles, n_mcmc_steps = smc_info.update_info.acceptence.shape
    res = Results(
        name="SMC HMC",
        time=t1 - t0,
        evals=n_particles * n_mcmc_steps * hmc_trajectory_length * steps
        + warmup_steps * hmc_trajectory_length,
        ess=float(ess),
        logZs=logzs,
    )
    return smc_state, res


def run_ss_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    target_ess=0.9,
):
    def step(key, state, logdensity, cov):
        kernel = blackjax.mcmc.ss.build_hrss_kernel(cov)
        return kernel(
            key,
            state,
            logdensity,
        )

    cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
        initial_samples
    )
    init_params = {"cov": cov}

    def update_fn(key, state, info):
        cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
            state.particles
        )
        return blackjax.smc.extend_params({"cov": cov})

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.hrss.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    # Warmup JIT and block until compiled
    (_, _), _ = jax.block_until_ready(one_step((state, sample_key), None))

    steps = 0
    log_zs = []
    all_log_weights = []
    all_particles = []
    evals = 0
    t0 = timer()
    with tqdm.tqdm(desc="SMC-SS", unit=" step") as pbar:
        while state[0].tempering_param < 1:
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            all_log_weights.append(jnp.log(state.sampler_state.weights + 1e-16))
            all_particles.append(state.sampler_state.particles)
            evals += (
                smc_info.update_info.num_steps.sum()
                + smc_info.update_info.num_shrink.sum()
            )
            pbar.set_postfix({"λ": f"{state[0].tempering_param:.3f}", "logZ": f"{sum(log_zs):.2f}"})
            pbar.update(1)
    t1 = timer()
    cumulative_log_zs = jnp.cumsum(jnp.array(log_zs))
    weights = smc_weights(all_log_weights, cumulative_log_zs)
    ess = smc_ess(jnp.log(weights + 1e-16))
    logzs = jnp.array(log_zs).sum()
    particles = concat_particles(all_particles)
    smc_state = SMCState(particles=particles, weights=weights)
    res = Results(
        name="SMC SS",
        time=t1 - t0,
        evals=evals,
        ess=float(ess),
        logZs=logzs,
    )
    return smc_state, res


def run_nuts_sequential_mc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    target_ess=0.9,
    target_acceptance_rate=0.8,
    warmup_steps=50,
):
    """SMC with NUTS kernel, using window adaptation at each rung to tune parameters."""

    kernel = blackjax.nuts.build_kernel()

    def step(key, state, logdensity, step_size, inverse_mass_matrix):
        return kernel(
            key,
            state,
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )

    # Initial parameters from particle covariance
    imm = blackjax.smc.tuning.from_particles.inverse_mass_matrix_from_particles(
        initial_samples
    ).diagonal()
    init_params = {
        "inverse_mass_matrix": imm,
        "step_size": 0.1,
    }

    def update_fn(key, state, info):
        """Run window adaptation on a random particle to tune step size and mass matrix."""
        lmbda = state.tempering_param

        def tempered_logdensity(x):
            return prior_logprob(x) + lmbda * loglikelihood_fn(x)

        # Pick a random particle to run adaptation from
        n_particles = jax.tree_util.tree_leaves(state.particles)[0].shape[0]
        key, idx_key, adapt_key = jax.random.split(key, 3)
        idx = jax.random.randint(idx_key, (), 0, n_particles)
        start_position = jax.tree_util.tree_map(lambda x: x[idx], state.particles)

        # Run window adaptation
        adapt = blackjax.window_adaptation(
            blackjax.nuts,
            tempered_logdensity,
            target_acceptance_rate=target_acceptance_rate,
        )
        (_, parameters), _ = adapt.run(adapt_key, start_position, warmup_steps)

        return blackjax.smc.extend_params(
            {
                "inverse_mass_matrix": parameters["inverse_mass_matrix"],
                "step_size": parameters["step_size"],
            }
        )

    smc_alg = blackjax.inner_kernel_tuning(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=step,
        mcmc_init_fn=blackjax.hmc.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=update_fn,
        initial_parameter_value=blackjax.smc.extend_params(init_params),
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc_alg.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_alg.step(subk, state)
        return (state, k), info

    rng_key, sample_key = jax.random.split(rng_key)
    # Warmup JIT and block until compiled
    (_, _), _ = jax.block_until_ready(one_step((state, sample_key), None))

    steps = 0
    log_zs = []
    all_log_weights = []
    all_particles = []
    total_integration_steps = 0
    t0 = timer()
    with tqdm.tqdm(desc="SMC-NUTS", unit=" step") as pbar:
        while state[0].tempering_param < 1:
            (state, rng_key), smc_info = one_step((state, rng_key), None)
            steps += 1
            log_zs.append(smc_info[1])
            all_log_weights.append(jnp.log(state.sampler_state.weights + 1e-16))
            all_particles.append(state.sampler_state.particles)
            total_integration_steps += int(jnp.sum(smc_info.update_info.num_integration_steps))
            pbar.set_postfix({"λ": f"{state[0].tempering_param:.3f}", "logZ": f"{sum(log_zs):.2f}"})
            pbar.update(1)
    t1 = timer()
    cumulative_log_zs = jnp.cumsum(jnp.array(log_zs))
    weights = smc_weights(all_log_weights, cumulative_log_zs)
    ess = smc_ess(jnp.log(weights + 1e-16))
    logzs = jnp.array(log_zs).sum()
    particles = concat_particles(all_particles)
    smc_state = SMCState(particles=particles, weights=weights)
    # Evals: MCMC integration steps + adaptation steps per rung
    res = Results(
        name="SMC NUTS",
        time=t1 - t0,
        evals=total_integration_steps + steps * warmup_steps,
        ess=float(ess),
        logZs=logzs,
    )
    return smc_state, res
