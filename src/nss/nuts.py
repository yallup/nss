from timeit import default_timer as timer

import blackjax
import jax
import jax.numpy as jnp

from nss.utils import Results


def run_hmc(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    initial_samples,
    n_warmup=1000,
    n_samples=1000,
    target_acceptance_rate=0.8,
):
    rng_key, sample_key = jax.random.split(rng_key)

    def target(x):
        return loglikelihood_fn(x) + prior_logprob(x)

    # Check that starting point is valid
    init_point = blackjax.ns.utils.get_first_row(initial_samples)
    init_logprob = target(init_point)
    if not jnp.isfinite(init_logprob):
        raise ValueError(
            f"NUTS starting point has non-finite log probability: {init_logprob}. "
            "Filter initial_samples to exclude points with NaN/inf log probability."
        )

    # Adaptation (includes its own compilation)
    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        target,
        target_acceptance_rate=target_acceptance_rate,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    effective_warmup = min(n_warmup, 100)
    (last_state, parameters), adapt_info = adapt.run(
        warmup_key, blackjax.ns.utils.get_first_row(initial_samples), effective_warmup
    )

    kernel = blackjax.nuts(target, **parameters).step

    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    def inference_loop(rng_key, initial_state, num_samples):
        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    # Warmup JIT and block until compiled
    _, loop_key = jax.random.split(sample_key)
    jax.block_until_ready(one_step(last_state, loop_key))

    print("Running NUTS")
    t0 = timer()
    hmc_states, hmc_infos = inference_loop(loop_key, last_state, n_samples)
    jax.block_until_ready(hmc_states)
    t1 = timer()

    res = Results(
        name="NUTS",
        time=t1 - t0,
        evals=hmc_infos.num_integration_steps.sum() + adapt_info[1].num_integration_steps.sum(),
        ess=n_samples,
        logZs=jnp.asarray([0.0]),
    )
    return hmc_states, res
