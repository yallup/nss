"""Common utilities for experiment scripts."""

import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

from nss.utils import calculate_mmd, sliced_wasserstein2

os.makedirs("plots", exist_ok=True)


def warmup_jit(one_step, state, rng_key, name="Algorithm"):
    """Warmup JIT compilation with a dummy step.

    Parameters
    ----------
    one_step : callable
        The JIT-compiled step function.
    state : any
        Initial state for the algorithm.
    rng_key : PRNGKey
        Random key.
    name : str
        Algorithm name for progress message.

    Returns
    -------
    rng_key : PRNGKey
        Updated random key.
    """
    print(f"Compiling {name}...", end=" ", flush=True)
    _ = one_step((state, rng_key), None)
    print("done")
    return jax.random.split(rng_key)[0]


def progress_bar(desc, unit, total=None):
    """Create a unified progress bar.

    Parameters
    ----------
    desc : str
        Description for the progress bar.
    unit : str
        Unit label (e.g., "step", "sample").
    total : int, optional
        Total number of iterations if known.

    Returns
    -------
    tqdm.tqdm
        Progress bar context manager.
    """
    return tqdm.tqdm(desc=desc, unit=f" {unit}", total=total)


def create_plot_grid(figsize=(6, 5)):
    """Create a 3x3 subplot grid for algorithm comparison.

    Parameters
    ----------
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        3x3 array of axes.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=figsize)
    return fig, axes


def scatter_panel(ax, x, y, title, color="C0", alpha=0.7, s=8):
    """Add a scatter plot to an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    x : array
        X coordinates.
    y : array
        Y coordinates.
    title : str
        Panel title.
    color : str
        Marker color.
    alpha : float
        Marker transparency.
    s : int
        Marker size.
    """
    ax.scatter(x, y, s=s, c=color, alpha=alpha, rasterized=True)
    ax.set_title(title)


def plot_comparison_grid(
    true_samples,
    algo_samples,
    algo_names,
    bounds,
    output_prefix,
    dim_x=0,
    dim_y=1,
    n_plot=1000,
):
    """Create a 3x3 comparison grid with truth in top-right.

    Parameters
    ----------
    true_samples : array
        Ground truth samples, shape (n, d) or dict with position arrays.
    algo_samples : list of arrays
        List of 8 sample arrays from algorithms.
    algo_names : list of str
        List of 8 algorithm names.
    bounds : tuple
        (min, max) for plot axes.
    output_prefix : str
        Prefix for output files (e.g., "plots/mixture_gaussians").
    dim_x : int
        Dimension index for x-axis.
    dim_y : int or str
        Dimension index for y-axis, or key for dict samples.
    n_plot : int
        Number of samples to plot per panel.
    """
    fig, axes = create_plot_grid()

    def get_xy(samples, n=None):
        if isinstance(samples, dict):
            # For funnel-style dict samples
            x = samples["theta"]
            y = samples["z"][:, 0] if dim_y == "z" else samples[dim_y]
        else:
            x = samples[:, dim_x]
            y = samples[:, dim_y]
        if n is not None and len(x) > n:
            x, y = x[:n], y[:n]
        return x, y

    true_x, true_y = get_xy(true_samples, n_plot)

    # Layout: SS, NUTS, Truth (top row)
    #         NSS, NS-RW, SMC-SS (middle row)
    #         SMC-RW, SMC-IRMH, SMC-HMC (bottom row)
    positions = [
        (0, 0), (0, 1),  # SS, NUTS
        (1, 0), (1, 1), (1, 2),  # NSS, NS-RW, SMC-SS
        (2, 0), (2, 1), (2, 2),  # SMC-RW, SMC-IRMH, SMC-HMC
    ]

    # Plot truth in top-right
    scatter_panel(axes[0, 2], true_x, true_y, "Truth", color="C1")

    # Plot algorithms
    for (row, col), samples, name in zip(positions, algo_samples, algo_names):
        ax = axes[row, col]
        # Plot truth as background
        ax.scatter(true_x, true_y, s=8, c="C1", alpha=0.3, rasterized=True)
        # Plot algorithm samples
        x, y = get_xy(samples, n_plot)
        scatter_panel(ax, x, y, name, color="C0")

    # Set bounds
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)

    fig.savefig(f"{output_prefix}.png", dpi=300)
    plt.close(fig)


def subsample(x, key, n):
    """Subsample array to n samples."""
    if x.shape[0] <= n:
        return x
    idx = jax.random.choice(key, x.shape[0], shape=(n,), replace=False)
    return x[idx]


def compute_metrics(
    rng_key,
    samples_list,
    results_list,
    target_sampler,
    true_samples,
    metric_sample_size=200,
    num_reps=5,
    pack_fn=None,
):
    """Compute MMD and W2 metrics for all algorithms.

    Parameters
    ----------
    rng_key : PRNGKey
        Random key.
    samples_list : list
        List of sample arrays from algorithms.
    results_list : list
        List of Results objects to update with metrics.
    target_sampler : callable
        Function (key) -> samples to generate target samples.
    true_samples : array
        Ground truth samples for baseline comparison.
    metric_sample_size : int
        Number of samples for metric computation.
    num_reps : int
        Number of repetitions for metric statistics.
    pack_fn : callable, optional
        Function to pack dict samples to array (for funnel).

    Returns
    -------
    ground_truth_mmd : tuple
        (mean, std) of MMD for ground truth.
    ground_truth_w2 : tuple
        (mean, std) of W2 for ground truth.
    """
    if pack_fn is None:
        pack_fn = lambda x: x

    # Generate metric targets
    rng_key, targets_key = jax.random.split(rng_key)
    target_keys = jax.random.split(targets_key, num_reps)
    metric_targets = jnp.stack([target_sampler(k) for k in target_keys])

    mmd_loss = calculate_mmd(kernel_mul=2.0, kernel_num=5)

    def mmd_fn(ref):
        return partial(mmd_loss, target=ref)

    def w2_fn(ref):
        return partial(sliced_wasserstein2, ref)

    def metric_stats(metric_fn):
        vals = jnp.array([metric_fn(target) for target in metric_targets])
        return float(vals.mean()), float(vals.std())

    # Subsample all
    rng_key, sub_key = jax.random.split(rng_key)
    sub_keys = jax.random.split(sub_key, len(samples_list) + 1)
    true_ref = subsample(pack_fn(true_samples), sub_keys[0], metric_sample_size)

    # Ground truth metrics
    ground_truth_mmd = metric_stats(mmd_fn(true_ref))
    ground_truth_w2 = metric_stats(w2_fn(true_ref))

    # Algorithm metrics
    for i, (samples, res) in enumerate(zip(samples_list, results_list)):
        ref = subsample(pack_fn(samples), sub_keys[i + 1], metric_sample_size)
        res.mmd, res.mmd_std = metric_stats(mmd_fn(ref))
        res.w2, res.w2_std = metric_stats(w2_fn(ref))

    return ground_truth_mmd, ground_truth_w2


def print_results(results_list, ground_truth_mmd, ground_truth_w2):
    """Print results table.

    Parameters
    ----------
    results_list : list
        List of Results objects.
    ground_truth_mmd : tuple
        (mean, std) of MMD for ground truth.
    ground_truth_w2 : tuple
        (mean, std) of W2 for ground truth.
    """
    # Header
    print()
    print("=" * 80)
    print(f"{'Method':<10} {'MMD':<16} {'W2':<16} {'logZ':<16} {'ESS/s':>8}")
    print("-" * 80)

    # Ground truth row
    mmd_str = f"{ground_truth_mmd[0]:.3f} ± {ground_truth_mmd[1]:.3f}"
    w2_str = f"{ground_truth_w2[0]:.2f} ± {ground_truth_w2[1]:.2f}"
    print(f"{'Truth':<10} {mmd_str:<16} {w2_str:<16} {'-':<16} {'-':>8}")

    # Algorithm rows
    for res in results_list:
        ess_per_s = res.ess / res.time if res.time > 0 else float("inf")
        mmd_str = f"{res.mmd:.3f} ± {res.mmd_std:.3f}"
        w2_str = f"{res.w2:.2f} ± {res.w2_std:.2f}"
        if not jnp.isnan(res.logZs).all():
            logz_str = f"{res.logZs.mean():.1f} ± {res.logZs.std():.1f}"
        else:
            logz_str = "-"
        ess_str = f"{ess_per_s:.0f}" if ess_per_s < float("inf") else "-"
        print(f"{res.name:<10} {mmd_str:<16} {w2_str:<16} {logz_str:<16} {ess_str:>8}")

    print("=" * 80)
    print()
