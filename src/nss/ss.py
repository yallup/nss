from timeit import default_timer as timer

import blackjax
import jax

from nss.utils import Results


def run_slice(
    rng_key,
    loglikelihood_fn,
    prior_logprob,
    initial_samples,
    num_steps=1000,
):
    rng_key, sample_key = jax.random.split(rng_key)

    def target(x):
        return loglikelihood_fn(x) + prior_logprob(x)

    cov = blackjax.smc.tuning.from_particles.particles_covariance_matrix(
        initial_samples
    )
    slice_alg = blackjax.hrss(target, cov=cov)
    slice_state = slice_alg.init(blackjax.ns.utils.get_first_row(initial_samples))
    kernel = slice_alg.step

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
    jax.block_until_ready(one_step(slice_state, loop_key))

    print("Running SS")
    t0 = timer()
    slice_states, slice_infos = inference_loop(loop_key, slice_state, num_steps)
    jax.block_until_ready(slice_states)
    t1 = timer()

    res = Results(
        name="SS",
        time=t1 - t0,
        evals=slice_infos.num_steps.sum() + slice_infos.num_shrink.sum(),
        ess=num_steps,
        logZs=jax.numpy.asarray([0.0]),
    )
    return slice_states, res
