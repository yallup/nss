from functools import partial
from timeit import default_timer as timer

import blackjax
import jax
import jax.flatten_util
import jax.numpy as jnp
import tqdm
from blackjax import SamplingAlgorithm
from blackjax.mcmc.random_walk import RWState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import delete_fn, init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.ns.utils import finalise, log_weights

from nss.utils import Results, safe_ess


def run_nested_sampling(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    num_delete=1,
    termination=-3,
):
    """Run the Nested Sampling algorithm."""

    algo = blackjax.nss(
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_mcmc_steps,
    )
    rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
    state = algo.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    dead = []

    # Warmup JIT and block until compiled
    (_, rng_key), _ = jax.block_until_ready(one_step((state, sample_key), None))

    t0 = timer()
    with tqdm.tqdm(desc="NSS", unit=" dead") as pbar:
        while not state.integrator.logZ_live - state.integrator.logZ < termination:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.set_postfix({"logZ": f"{state.integrator.logZ:.2f}"})
            pbar.update(num_delete)
    t1 = timer()
    final_state = finalise(state, dead)
    logw = log_weights(rng_key, final_state)
    minimum = jnp.nan_to_num(logw).min()
    logzs = jax.scipy.special.logsumexp(jnp.nan_to_num(logw, nan=minimum), axis=0)
    res = Results(
        name="NSS",
        time=t1 - t0,
        evals=final_state.update_info.num_steps.sum()
        + final_state.update_info.num_shrink.sum(),
        ess=int(safe_ess(logw.mean(axis=-1))),
        logZs=logzs,
    )
    return final_state, res


def run_rw_nested_sampling(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    num_mcmc_steps,
    initial_samples,
    num_delete=1,
    termination=-3,
):
    """Run the Nested Sampling algorithm."""
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
    )

    add_kernel = blackjax.mcmc.random_walk.build_additive_step()

    def build_nsrw_kernel(num_delete, num_inner_steps, update_inner_kernel_params_fn):
        def constrained_rw(rng_key, state, loglikelihood_0, **params):
            def proposal_distribution(key, position):
                x, ravel_fn = jax.flatten_util.ravel_pytree(position)
                return ravel_fn(
                    jax.random.multivariate_normal(
                        key, jnp.zeros_like(x), params["cov"]
                    )
                )

            rng_key, step_key = jax.random.split(rng_key)
            mcmc_state = RWState(state.position, state.logdensity)

            new_mcmc_state, mcmc_info = add_kernel(
                step_key, mcmc_state, prior_logprob, proposal_distribution
            )
            proposed_state = init_state_fn(
                new_mcmc_state.position, loglikelihood_birth=loglikelihood_0
            )
            within_contour = proposed_state.loglikelihood > loglikelihood_0
            is_accepted = within_contour
            new_state = jax.lax.cond(
                is_accepted,
                lambda _: proposed_state,
                lambda _: state,
                operand=None,
            )

            return new_state, mcmc_info

        inner_kernel = update_with_mcmc_take_last(constrained_rw, num_inner_steps)

        _delete_fn = partial(delete_fn, num_delete=num_delete)

        kernel = build_adaptive_kernel(
            _delete_fn,
            inner_kernel,
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )
        return kernel

    def update_fn(rng_key, state, info, _):
        cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
            state.particles.position
        )
        dimension = cov.shape[0]
        scale = 2.38 / jnp.sqrt(dimension)
        cov *= scale**2
        return {"cov": cov}

    def ns_mcmc_kernel(
        logprior_fn,
        loglikelihood_fn,
        num_delete=10,
        num_inner_steps=10,
    ):
        step_fn = build_nsrw_kernel(num_delete, num_inner_steps, update_fn)

        def init_fn(position, rng_key=None):
            # Vectorize the functions for parallel evaluation over particles
            # vmap maps over positional args, keyword args (like loglikelihood_birth) are broadcast
            return init(
                position,
                init_state_fn=jax.vmap(init_state_fn),
                update_inner_kernel_params_fn=update_fn,
            )

        return SamplingAlgorithm(init_fn, step_fn)

    algo = ns_mcmc_kernel(
        logprior_fn=prior_logprob,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_mcmc_steps,
    )
    rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
    state = algo.init(initial_samples)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    dead = []

    # Warmup JIT and block until compiled
    (_, rng_key), _ = jax.block_until_ready(one_step((state, sample_key), None))

    t0 = timer()
    with tqdm.tqdm(desc="NS-RW", unit=" dead") as pbar:
        while not state.integrator.logZ_live - state.integrator.logZ < termination:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.set_postfix({"logZ": f"{state.integrator.logZ:.2f}"})
            pbar.update(num_delete)
    t1 = timer()
    final_state = finalise(state, dead)
    logw = log_weights(rng_key, final_state)
    minimum = jnp.nan_to_num(logw).min()
    logzs = jax.scipy.special.logsumexp(jnp.nan_to_num(logw, nan=minimum), axis=0)
    res = Results(
        name="NS-RW",
        time=t1 - t0,
        evals=jnp.prod(jnp.array([final_state.update_info.acceptance_rate.shape])),
        ess=int(safe_ess(logw.mean(axis=-1))),
        logZs=logzs,
    )
    return final_state, res
