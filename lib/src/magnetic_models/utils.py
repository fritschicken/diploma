"""
Post-processing and plotting of exchange-interaction sweeps.

A sweep is a set of ``data_mu_{mu}.npz`` files in one directory, one file per
chemical potential. Each file holds a full (c_inter, b) grid, row-major in
c_inter, with arrays:

- ``c_inter``, ``b_val``     : (n_rows,) parameter values of each row
- ``distances``              : (n_rows, n_pairs) pair distances (same in every row)
- ``J_iso_meV``, ``D_meV``   : (n_rows, n_pairs[, 3]) computed quantities
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator

__all__ = [
    "Sweep",
    "load_sweep",
    "interactive_dual_heatmap",
    "plot_interactive_mu_c",
    "plot_interactive_mu_dist",
    "plot_heatmap_mu_cinter",
]

QUANTITY_LABELS = {"J_iso_meV": "J_iso [meV]", "D_meV": "D [meV]"}
B_LABEL = "Zeeman splitting (b) [eV]"
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
GRID_STYLE = dict(showgrid=True, gridwidth=1, gridcolor="LightGray")

# Rounding used to compare floating-point parameters and distances
PARAM_DECIMALS = 5
DIST_DECIMALS = 4


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
@dataclass
class Sweep:
    """
    A (mu, c_inter, b) sweep of one quantity.

    Attributes
    ----------
    quantity : str
        Name of the loaded array, e.g. 'J_iso_meV'.
    c_inter, b : np.ndarray
        Sorted grid axes.
    distances : np.ndarray
        Distance of every pair, shape (n_pairs,).
    data : dict
        Maps each loaded mu to an array of shape (n_c, n_b, n_pairs).
    """
    quantity: str
    c_inter: np.ndarray
    b: np.ndarray
    distances: np.ndarray
    data: Dict[float, np.ndarray] = field(default_factory=dict)

    @property
    def mu_values(self) -> List[float]:
        return list(self.data)

    @property
    def label(self) -> str:
        return QUANTITY_LABELS.get(self.quantity, self.quantity)

    def shells(self, num_neighbors: Optional[int] = None) -> np.ndarray:
        """Distinct neighbour distances, nearest first."""
        return np.unique(np.round(self.distances, DIST_DECIMALS))[:num_neighbors]

    def pairs_at(self, shell: float) -> np.ndarray:
        """Indices of the pairs whose distance equals `shell`."""
        return np.flatnonzero(np.isclose(np.round(self.distances, DIST_DECIMALS), shell))


def load_sweep(save_dir: str, mu_values, quantity: str = "J_iso_meV") -> Sweep:
    """
    Loads ``data_mu_{mu}.npz`` for every mu in `mu_values` from `save_dir`.

    Missing files are skipped with a warning. Raises if no file is found, if a
    file does not contain a full (c_inter, b) grid, or if the grids differ
    between files.
    """
    sweep = None
    for mu in mu_values:
        path = os.path.join(save_dir, f"data_mu_{mu}.npz")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping mu={mu}.")
            continue

        with np.load(path) as f:
            c = np.round(f["c_inter"], PARAM_DECIMALS)
            b = np.round(f["b_val"], PARAM_DECIMALS)
            values = f[quantity]
            distances = f["distances"][0]

        c_axis, b_axis = np.unique(c), np.unique(b)
        n_c, n_b = len(c_axis), len(b_axis)
        if not (
            len(c) == n_c * n_b
            and np.array_equal(c, np.repeat(c_axis, n_b))
            and np.array_equal(b, np.tile(b_axis, n_c))
        ):
            raise ValueError(f"{path} does not contain a full (c_inter, b) grid.")

        if sweep is None:
            sweep = Sweep(quantity, c_axis, b_axis, distances)
        elif not (np.array_equal(c_axis, sweep.c_inter) and np.array_equal(b_axis, sweep.b)):
            raise ValueError(f"{path} uses a different (c_inter, b) grid than the other files.")

        sweep.data[mu] = values.reshape(n_c, n_b, *values.shape[1:])

    if sweep is None:
        raise FileNotFoundError(f"No data_mu_*.npz files found in {save_dir} for mu={list(mu_values)}.")
    if next(iter(sweep.data.values())).ndim != 3:
        raise ValueError(f"'{quantity}' is not a scalar per pair; the plots need one value per pair.")
    return sweep


# -----------------------------------------------------------------------------
# Plotly helpers
# -----------------------------------------------------------------------------
def _add_curve(fig, x, y, color, name, showlegend, visible, **subplot):
    """Adds the data points and a PCHIP interpolation line through them."""
    xs = np.linspace(x.min(), x.max(), 400)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(symbol="diamond", size=8, color=color),
        name=name, legendgroup=name, showlegend=False, visible=visible,
    ), **subplot)
    fig.add_trace(go.Scatter(
        x=xs, y=PchipInterpolator(x, y)(xs), mode="lines",
        line=dict(color=color, width=2),
        name=name, legendgroup=name, showlegend=showlegend,
        hoverinfo="x+y+name", visible=visible,
    ), **subplot)


def _add_mu_controls(fig, groups, mu_values, slider_labels, slider_prefix, title):
    """
    Adds a mu dropdown and, for each mu, a slider over `slider_labels`.

    `groups[m][s]` lists the trace indices shown for mu index m at slider step s.
    `title(mu, label)` gives the figure title for each state.
    """
    n_traces = len(fig.data)

    def visibility(trace_ids):
        mask = [False] * n_traces
        for i in trace_ids:
            mask[i] = True
        return mask

    sliders = []
    buttons = []
    for m, mu in enumerate(mu_values):
        steps = [
            dict(
                method="update",
                args=[{"visible": visibility(groups[m][s])}, {"title": title(mu, label)}],
                label=label,
            )
            for s, label in enumerate(slider_labels)
        ]
        sliders.append(dict(
            active=0, currentvalue={"prefix": f"{slider_prefix} (μ={mu}): "},
            pad={"t": 50}, steps=steps,
        ))
        buttons.append(dict(
            label=f"μ = {mu}",
            method="update",
            args=[
                {"visible": visibility(groups[m][0])},
                {"sliders": [sliders[m]], "title": title(mu, slider_labels[0])},
            ],
        ))

    fig.update_layout(
        title=dict(text=title(mu_values[0], slider_labels[0]), x=0.5, xanchor="center"),
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=120),
        sliders=[sliders[0]],
        updatemenus=[dict(
            buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
            x=0.01, xanchor="left", y=1.20, yanchor="top",
        )],
    )
    fig.add_annotation(
        text="Chemical Potential:", x=0.01, y=1.27,
        xref="paper", yref="paper", showarrow=False, xanchor="left",
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def interactive_dual_heatmap(save_dir, mu_values, quantity="J_iso_meV", num_neighbors=5):
    """
    Two (b, c_inter) heatmaps with widgets for mu and neighbour distance: the mean
    of `quantity` over all pairs at that distance, and its spread (max - min),
    which shows how much the pairs' symmetry is broken.
    """
    sweep = load_sweep(save_dir, mu_values, quantity)
    shells = sweep.shells(num_neighbors)
    B, C = np.meshgrid(sweep.b, sweep.c_inter)

    def plot(mu, shell):
        values = sweep.data[mu][:, :, sweep.pairs_at(shell)]
        mean = values.mean(axis=2)
        spread = np.ptp(values, axis=2)
        subtitle = rf"$\mu$ = {mu}, dist $\approx$ {shell:.3f}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        vmax = np.max(np.abs(mean)) or 1e-6
        cp1 = ax1.pcolormesh(B, C, mean, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(cp1, ax=ax1).set_label(f"Mean {sweep.label}")
        ax1.set_title(f"Mean interaction strength\n{subtitle}")

        cp2 = ax2.pcolormesh(B, C, spread, shading="auto", cmap="viridis")
        fig.colorbar(cp2, ax=ax2).set_label("Spread (max - min) [meV]")
        ax2.set_title(f"Symmetry breaking (max - min)\n{subtitle}")

        for ax in (ax1, ax2):
            ax.set_xlabel(B_LABEL)
            ax.set_ylabel("$c_{inter}$ [eV]")

        fig.tight_layout()
        plt.show()

    widgets.interact(
        plot,
        mu=widgets.SelectionSlider(options=sweep.mu_values, description=r"$\mu$:"),
        shell=widgets.SelectionSlider(options=[float(s) for s in shells], description="Distance:"),
    )


def plot_interactive_mu_c(save_dir, mu_values, quantity="J_iso_meV", num_neighbors=3):
    """
    `quantity` against b for every pair, coloured by neighbour distance.
    A dropdown picks mu and a slider picks c_inter.
    """
    sweep = load_sweep(save_dir, mu_values, quantity)
    shells = sweep.shells(num_neighbors)
    fig = go.Figure()

    groups = []
    for m, mu in enumerate(sweep.mu_values):
        groups.append([])
        for k in range(len(sweep.c_inter)):
            start = len(fig.data)
            visible = m == 0 and k == 0
            for n, shell in enumerate(shells):
                for j, pair in enumerate(sweep.pairs_at(shell)):
                    _add_curve(
                        fig, sweep.b, sweep.data[mu][k, :, pair],
                        color=COLORS[n % len(COLORS)], name=f"dist={shell:.2f}",
                        showlegend=j == 0, visible=visible,
                    )
            groups[m].append(range(start, len(fig.data)))

    _add_mu_controls(
        fig, groups, sweep.mu_values,
        slider_labels=[f"{c:.3f}" for c in sweep.c_inter],
        slider_prefix="c_inter",
        title=lambda mu, c: f"Neighbours for μ = {mu}, c_inter = {c}",
    )
    fig.update_layout(
        xaxis_title=B_LABEL, yaxis_title=sweep.label,
        legend=dict(title="Distances", yanchor="top", y=1, xanchor="left", x=1.05),
        width=950, height=700,
    )
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(**GRID_STYLE)
    fig.show()
    return fig


def plot_interactive_mu_dist(save_dir, mu_values, quantity="J_iso_meV", num_neighbors=5):
    """
    `quantity` against b, with one subplot per c_inter.
    A dropdown picks mu and a slider picks the neighbour distance.
    """
    sweep = load_sweep(save_dir, mu_values, quantity)
    shells = sweep.shells(num_neighbors)
    num_c = len(sweep.c_inter)
    fig = make_subplots(
        rows=num_c, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[f"c_inter = {c:.3f}" for c in sweep.c_inter],
    )

    groups = []
    for m, mu in enumerate(sweep.mu_values):
        groups.append([])
        for n, shell in enumerate(shells):
            start = len(fig.data)
            visible = m == 0 and n == 0
            pairs = sweep.pairs_at(shell)
            for k, c in enumerate(sweep.c_inter):
                for j, pair in enumerate(pairs):
                    _add_curve(
                        fig, sweep.b, sweep.data[mu][k, :, pair],
                        color=COLORS[k % len(COLORS)], name=f"c={c:.3f}",
                        showlegend=j == 0, visible=visible, row=k + 1, col=1,
                    )
            groups[m].append(range(start, len(fig.data)))

    _add_mu_controls(
        fig, groups, sweep.mu_values,
        slider_labels=[f"{d:.3f}" for d in shells],
        slider_prefix="Dist",
        title=lambda mu, d: f"Data for μ = {mu}, dist = {d}",
    )
    fig.update_layout(
        legend=dict(title="c_inter values", yanchor="top", y=1, xanchor="left", x=1.05),
        width=950, height=max(700, 200 * num_c),
    )
    fig.update_yaxes(title_text=sweep.label, **GRID_STYLE)
    fig.update_xaxes(**GRID_STYLE)
    fig.update_xaxes(title_text=B_LABEL, row=num_c, col=1)
    fig.show()
    return fig


def plot_heatmap_mu_cinter(save_dir, mu_values, quantity="J_iso_meV", b_val_target=0.0):
    """
    Heatmap of `quantity` over mu (x) and c_inter (y), for the b closest to
    `b_val_target` and the first pair at the nearest-neighbour distance.
    """
    sweep = load_sweep(save_dir, mu_values, quantity)
    b_idx = np.argmin(np.abs(sweep.b - b_val_target))
    shell = sweep.shells(1)[0]
    pair = sweep.pairs_at(shell)[0]

    Z = np.full((len(sweep.c_inter), len(mu_values)), np.nan)
    for j, mu in enumerate(mu_values):
        if mu in sweep.data:
            Z[:, j] = sweep.data[mu][:, b_idx, pair]

    fig = go.Figure(go.Heatmap(
        z=Z, x=mu_values, y=sweep.c_inter,
        colorscale="RdBu_r", zmid=0, colorbar=dict(title=sweep.label),
    ))
    fig.update_layout(
        title=(f"{sweep.label} as a function of μ and c_inter"
               f"<br><sup>(b ≈ {sweep.b[b_idx]:.4f}, dist ≈ {shell:.4f})</sup>"),
        xaxis_title="Chemical potential (μ) [eV]",
        yaxis_title="c_inter [eV]",
        width=800, height=600,
        template="plotly_white",
    )
    fig.show()
    return fig
