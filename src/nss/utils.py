import jax
import jax.numpy as jnp
from dataclasses import dataclass
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def concat_particles(particles_list):
    """Concatenate particles from multiple rungs along the batch dimension."""
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.concatenate(leaves, axis=0), *particles_list
    )


def safe_ess(log_weights):
    """Compute ESS from log-weights, handling NaNs.

    Parameters
    ----------
    log_weights : array
        Log-weights, shape (n,).

    Returns
    -------
    float
        Effective sample size.
    """
    minimum = jnp.nan_to_num(log_weights).min()
    logw = jnp.nan_to_num(log_weights, nan=minimum)
    logw = logw - logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    return jnp.exp(2 * l_sum_w - l_sum_w_sq)


class calculate_mmd:
    def __init__(self, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def __call__(self, source, target):
        batch_size = source.shape[0]
        kernels = self.gaussian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = jnp.mean(XX + YY - XY - YX)
        return loss

    def gaussian_kernel(self, source, target):
        n_samples = source.shape[0] + target.shape[0]
        total = jnp.concatenate([source, target], axis=0)

        total0 = jnp.expand_dims(total, 0)
        total1 = jnp.expand_dims(total, 1)
        L2_distance = jnp.sum((total0 - total1) ** 2, axis=2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = jnp.sum(L2_distance) / (n_samples**2 - n_samples)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]

        kernel_val = [
            jnp.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)


def sliced_wasserstein2(samples_p, samples_q, n_projections=200, key=None):
    """
    Calculates the sliced 2-Wasserstein distance between two sets of samples.

    Projects samples onto random 1D directions, computes exact 1D W2, and averages.
    More robust in high dimensions and has no regularization bias.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n_p, d = samples_p.shape
    n_q = samples_q.shape[0]

    # Random projection directions (unit vectors)
    directions = jax.random.normal(key, (n_projections, d))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    # Project samples onto directions: (n_samples, n_projections)
    proj_p = samples_p @ directions.T
    proj_q = samples_q @ directions.T

    # Sort projections along sample axis for 1D W2
    proj_p = jnp.sort(proj_p, axis=0)
    proj_q = jnp.sort(proj_q, axis=0)

    # If sample sizes differ, interpolate the smaller one
    if n_p != n_q:
        n_max = max(n_p, n_q)
        if n_p < n_max:
            indices = jnp.linspace(0, n_p - 1, n_max)
            proj_p = jax.vmap(lambda col: jnp.interp(indices, jnp.arange(n_p), col), in_axes=1, out_axes=1)(proj_p)
        if n_q < n_max:
            indices = jnp.linspace(0, n_q - 1, n_max)
            proj_q = jax.vmap(lambda col: jnp.interp(indices, jnp.arange(n_q), col), in_axes=1, out_axes=1)(proj_q)

    # 1D W2 is just mean squared difference of sorted samples
    return jnp.sqrt(jnp.mean((proj_p - proj_q) ** 2))


def entropic_wasserstein2(samples_p, samples_q, epsilon=1e-3):
    """
    Calculates the entropic-regularized 2-Wasserstein distance between two sets of samples.

    Uses Sinkhorn algorithm from ott-jax with squared Euclidean cost.
    Suitable for lower-dimensional problems where regularization bias is less problematic.
    """
    geom = pointcloud.PointCloud(samples_p, samples_q, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    return jnp.sqrt(out.reg_ot_cost)


@dataclass
class Results:
    name: str
    time: float
    evals: int
    ess: float
    logZs: jnp.array
    mmd: float = 0.0
    mmd_std: float = 0.0
    w2: float = 0.0
    w2_std: float = 0.0

    def __repr__(self) -> str:
        logz_str = f"logZ={self.logZs.mean():.2f}+/-{self.logZs.std():.2f}" if not jnp.isnan(self.logZs).all() else "logZ=N/A"
        return f"Results({self.name}, evals={self.evals}, time={self.time:.2f}s, ess={self.ess:.0f}, {logz_str})"
