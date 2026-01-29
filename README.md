# nss

Nested Slice Sampling in JAX. Code accompanying paper

## Installation

```bash
uv sync
```

## Usage

All `run_*` functions follow the same pattern:

```python
states, results = run_*(
    rng_key,           # JAX PRNGKey
    loglikelihood_fn,  # Callable: position -> log-likelihood
    prior_logprob,     # Callable: position -> log-prior
    initial_samples,   # Initial particles (pytree with leading batch dimension)
    **kwargs           # Algorithm-specific parameters
)
```

**Returns:**
- `states` - Final sampler states (algorithm-specific)
- `results` - `Results` dataclass with fields:
  - `name`: Algorithm name
  - `time`: Wall-clock time (seconds)
  - `evals`: Number of likelihood evaluations
  - `ess`: Effective sample size
  - `logZs`: Evidence estimates (array for multiple runs)

**Example:**

```python
import jax
import distrax
from nss.ns import run_nested_sampling

# Define target
prior = distrax.Uniform(low=-10.0, high=10.0)
likelihood = distrax.Normal(loc=0.0, scale=1.0)

# Run
key = jax.random.PRNGKey(0)
initial = prior.sample(seed=key, sample_shape=(1000,))

states, res = run_nested_sampling(
    key,
    likelihood.log_prob,
    prior.log_prob,
    num_mcmc_steps=5,
    initial_samples=initial,
    num_delete=100,
)

print(f"logZ = {res.logZs.mean():.2f} ± {res.logZs.std():.2f}")
```

**Available algorithms:**

| Module | Function | Description |
|--------|----------|-------------|
| `nss.ns` | `run_nested_sampling` | Nested sampling with slice sampling |
| `nss.ns` | `run_rw_nested_sampling` | Nested sampling with random walk |
| `nss.smc` | `run_ss_sequential_mc` | SMC with slice sampling |
| `nss.smc` | `run_rw_sequential_mc` | SMC with random walk |
| `nss.smc` | `run_hmc_sequential_mc` | SMC with HMC |
| `nss.smc` | `run_independent_sequential_mc` | SMC with independent proposals |
| `nss.ss` | `run_slice` | Standalone slice sampling |
| `nss.nuts` | `run_hmc` | NUTS/HMC |

## Experiments

```bash
uv run python experiments/mog_40.py    # 40-component mixture of Gaussians (2D)
uv run python experiments/mog_10d.py   # Mixture of Gaussians (10D)
uv run python experiments/funnel.py    # Neal's funnel (10D)
```

Outputs are saved to `plots/`.
