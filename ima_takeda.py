"""
ima_takeda.py

Agent-Based Simulation of Effector-Tumor Cell Dynamics with Stochastic Binding, Growth, and Monte Carlo Analysis.

This module implements a 2D spatial simulation of immune effector cells (E), tumor cells (T),
and their interactions through complex formation (C). It supports single-run dynamics,
Monte Carlo simulations, grid sweeps over initial conditions, parameter optimization, and visualizations.

Main Components:
---------------
1. Simulation Core (Random Walk & Interaction Logic)
    - `SquareWell`: Bounded spatial environment with reflective or wrapping edges.
    - `WellSimulation`: Main class to simulate movement, binding/unbinding, cell growth, and logging.
    - `combine_behavior_jit`: Fast JIT-compiled function for E-T binding events.

2. Monte Carlo Simulations
    - `single_scenario_monte_carlo`: Repeats simulation with fixed initial conditions.
    - `grid_scenario_monte_carlo`: Sweeps across a grid of (E0, T0) values and records results.

3. Visualization Utilities
    - `plot_history`, `plot_snapshot`, `to_gif`: Within-class methods to visualize simulation dynamics.
    - `plot_result_trend`, `plot_result_distribution`: Plot aggregate Monte Carlo behavior.
    - `plot_F_surface_*`: 3D visualizations and animations across E0-T0 parameter space.
    - `plot_F_diagonal`: Extracts slices of the F surface along fixed E:T ratios.

4. Parameter Fitting
    - `fit_params`, `fit_params_weighted_T_ONLY`: Estimate simulation parameters from observed data.
    - Uses stochastic objective functions and differential evolution for fitting.

Notes:
------
- All core simulation logic is vectorized or JIT-accelerated via NumPy/Numba.
- `WellSimulationV2` is an unused exploratory section and is intentionally left undocumented.

Author: Zedan Liu
Project: Takeda Capstone
"""

import os
import json
import warnings
import random
import numpy as np
import shutil
import imageio.v2 as imageio
import ipywidgets as widgets
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from numba import njit
from typing import Literal
from collections.abc import Callable
from scipy.optimize import differential_evolution
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm
from IPython.display import display

plt.style.use("default")

# Agent-Based Cell Simulation with Random Walk and Binding Dynamics


@njit
def combine_behavior_jit(
    e_cells: np.ndarray, t_cells: np.ndarray, r: float, k_on: float, rng_vals: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, int, int):
    """
    Simulates E-T cell binding behavior with spatial constraints and stochastic activation.

    Parameters:
        e_cells (np.ndarray): Array of effector cell positions, shape (N, 2).
        t_cells (np.ndarray): Array of tumor cell positions, shape (M, 2).
        r (float): Binding interaction radius.
        k_on (float): Probability of successful binding upon contact.
        rng_vals (np.ndarray): Array of uniform random numbers (0,1) for binding attempts, shape (N,).

    Returns:
        remaining_e (np.ndarray): E cells that did not bind.
        kept_t (np.ndarray): T cells that were not bound.
        new_c (np.ndarray): Newly formed complex cell positions (midpoints), shape (K, 2).
        touched_count (int): Number of E-T pairs within binding radius (regardless of outcome).
        combined_count (int): Number of successful E-T bindings (complexes formed).
    """
    max_e: int = e_cells.shape[0]
    max_t: int = t_cells.shape[0]
    r2: float = (2 * r) ** 2  # Squared interaction radius (distance threshold)

    touched_count = 0  # Counts how many E cells found any T cell within range
    combined_count = 0  # Counts how many successful bindings occurred

    remaining_e: np.ndarray = np.empty_like(e_cells)  # E cells that failed to bind
    new_c: np.ndarray = np.empty((max_e, 2))  # Complexes formed via successful bindings
    e_idx = 0
    c_idx = 0

    taken_t: np.ndarray = np.zeros(max_t, dtype=np.uint8)  # Mask of used T cells

    for i in range(max_e):
        ex, ey = e_cells[i]
        found = False

        for j in range(max_t):
            if taken_t[j] == 1:
                continue
            tx, ty = t_cells[j]
            dx = ex - tx
            dy = ey - ty
            dist2 = dx * dx + dy * dy

            if dist2 <= r2:
                touched_count += 1
                if rng_vals[i] < k_on:
                    # Successful binding; create a new complex at midpoint
                    new_c[c_idx, 0] = (ex + tx) / 2
                    new_c[c_idx, 1] = (ey + ty) / 2
                    c_idx += 1
                    taken_t[j] = 1
                    combined_count += 1
                    found = True
                    break

        if not found:
            # E cell did not find a T cell to bind with
            remaining_e[e_idx] = e_cells[i]
            e_idx += 1

    # Slice arrays to actual filled sizes
    remaining_e = remaining_e[:e_idx]
    new_c = new_c[:c_idx]
    kept_t: np.ndarray = t_cells[taken_t == 0]

    return remaining_e, kept_t, new_c, touched_count, combined_count


class SquareWell:
    def __init__(self, size: float, mode: Literal["reflect", "wrap"] = "reflect") -> None:
        """
        Represents a 2D square domain with configurable boundary behavior.

        Parameters:
            size (float): The side length of the square well.
            mode (str): Boundary condition mode, either:
                - "reflect": Points are reflected back when hitting the boundary (hard wall).
                - "wrap": Points wrap around to the opposite side (like a torus).
        """
        self.L: float = size
        self.mode: Literal["reflect", "wrap"] = mode

    def random_position(self) -> tuple[np.float64, np.float64]:
        """
        Generates a random position (x, y) uniformly distributed within the square.

        Returns:
            tuple[float, float]: Random coordinates in [0, L) × [0, L).
        """
        return tuple(np.random.uniform(0, self.L, size=2))

    def apply_boundary_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Applies boundary behavior to an array of coordinates.

        Parameters:
            coords (np.ndarray): Array of (x, y) coordinates, shape (N, 2).

        Returns:
            np.ndarray: Coordinates after applying boundary rules.
        """
        if self.mode == "reflect":
            return np.clip(coords, 0, self.L)  # Clamp to boundary
        else:
            return coords % self.L  # Wrap around


class WellSimulation:
    def __init__(
        self,
        E_0: int,
        T_0: int,
        r: float,
        m: float,
        N: int,
        L: float,
        k_on: float,
        k_off: float,
        k_kill: float,
        g_E: float,
        g_T: float,
        early_stop: tuple[bool, int, float],
        log_all: bool = False,
        use_tqdm: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initializes a 2D agent-based simulation of interacting immune and tumor cells.

        Parameters:
            E_0 (int): Initial number of effector (E) cells.
            T_0 (int): Initial number of tumor (T) cells.
            r (float): Binding radius for interactions.
            m (float): Movement scale (diffusion rate).
            N (int): Total number of simulation steps.
            L (float): Side length of the square domain.
            k_on (float): Binding rate (probability of E-T complex formation).
            k_off (float): Unbinding rate for complexes.
            k_kill (float): Rate at which bound complexes result in tumor cell death.
            g_E (float): Growth rate of effector cells.
            g_T (float): Growth rate of tumor cells.
            early_stop (tuple[bool, int, float]):
                - Flag for early stopping,
                - Step to begin checking,
                - Threshold on total tumor cells.
            log_all (bool): Whether to record detailed step-by-step data.
            use_tqdm (bool): Whether to display a progress bar.
            seed (int | None): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if E_0 == 0 and T_0 == 0:
            raise ValueError("Initializing both E and T cell to be 0 is not allowed.")

        # Simulation parameters
        self.E_0: int = E_0
        self.T_0: int = T_0
        self.r: float = r
        self.m: float = m
        self.N: int = N
        self.L: float = L
        self.k_on: float = k_on
        self.k_off: float = k_off
        self.k_kill: float = k_kill
        self.g_E: float = g_E
        self.g_T: float = g_T
        self.early_stop: tuple[bool, int, float] = early_stop
        self.early_stop_step: int | None = None
        self.log_all: bool = log_all
        self.use_tqdm: bool = use_tqdm

        self.simulated = False  # Flag to track if simulation has been run

        # Create simulation domain
        self.well = SquareWell(L, mode="reflect")

        # Initialize cell populations
        self.e_cells: np.ndarray = self._initialize_cells(E_0)
        self.t_cells: np.ndarray = self._initialize_cells(T_0)
        self.existing_c_cells: np.ndarray = np.empty((0, 2))  # C cells carried over between steps
        self.new_c_cells: np.ndarray = np.empty((0, 2))  # C cells formed in current step

        # Track E, T, C counts over time
        self.history: dict[str, list] = {"E": [], "T": [], "C": []}

        if log_all:
            self.full_history: list[dict] = []  # Store full cell states each step
            self.step_stats: list[dict[str, int]] = []  # Store intermediate event counts

        # Log initial state
        self._log()

    def _initialize_cells(self, count: int) -> np.ndarray:
        """
        Initialize `count` random (x, y) positions within the square well.

        Parameters:
            count (int): Number of cells to initialize.

        Returns:
            np.ndarray: Array of shape (count, 2) with random positions.
        """
        return np.random.uniform(0, self.L, size=(count, 2))

    def _combine_behavior(self) -> None:
        """
        Handles E-T cell binding behavior:
            - For each E cell, checks if a T cell is within range.
            - With probability `k_on`, the pair forms a new C (complex) cell.
            - Updates E, T, and new C cell arrays accordingly.
        """
        if self.e_cells.shape[0] == 0 or self.t_cells.shape[0] == 0:
            # No interactions possible
            if self.log_all:
                self.step_stats.append({"touched": 0, "combined": 0, "split": 0, "killed": 0})
            return

        rng_vals: np.ndarray = np.random.rand(self.e_cells.shape[0])
        self.e_cells, self.t_cells, self.new_c_cells, touched, combined = combine_behavior_jit(
            self.e_cells, self.t_cells, self.r, self.k_on, rng_vals
        )

        if self.log_all:
            self.step_stats.append({"touched": touched, "combined": combined, "split": 0, "killed": 0})

    def _c_cell_behavior(self) -> None:
        """
        Handles the fate of C (complex) cells:
            - With probability `k_off`: C cell splits into E + T at the same position.
            - With probability `k_kill`: T cell is killed, only E remains.
            - Remaining C cells stay bound.
        """
        n_c: int = self.existing_c_cells.shape[0]
        if n_c == 0:
            if self.log_all:
                if len(self.step_stats) < len(self.history["E"]):
                    self.step_stats.append({"touched": 0, "combined": 0, "split": 0, "killed": 0})
            return

        rnd: np.ndarray = np.random.rand(n_c)
        is_split = rnd < self.k_off
        is_kill = (rnd >= self.k_off) & (rnd < self.k_off + self.k_kill)
        is_stay = ~(is_split | is_kill)

        # Split: create new E and T cells from split C cells
        split_coords: np.ndarray = self.existing_c_cells[is_split]
        self.e_cells = np.vstack([self.e_cells, split_coords]) if split_coords.size > 0 else self.e_cells
        self.t_cells = np.vstack([self.t_cells, split_coords]) if split_coords.size > 0 else self.t_cells

        # Kill: create only E cells
        kill_coords: np.ndarray = self.existing_c_cells[is_kill]
        self.e_cells = np.vstack([self.e_cells, kill_coords]) if kill_coords.size > 0 else self.e_cells

        # Keep the rest of the C cells
        self.existing_c_cells = self.existing_c_cells[is_stay]

        # Log results
        if self.log_all:
            if len(self.step_stats) < len(self.history["E"]):
                self.step_stats.append({"touched": 0, "combined": 0, "split": np.count_nonzero(is_split), "killed": np.count_nonzero(is_kill)})
            else:
                self.step_stats[-1]["split"] += np.count_nonzero(is_split)
                self.step_stats[-1]["killed"] += np.count_nonzero(is_kill)

    def _cell_move(self) -> None:
        """
        Applies Brownian motion to E cells and enforces boundary conditions.
        """
        if self.e_cells.shape[0] == 0:
            return

        # Generate Gaussian displacement (mean=0, std=m)
        noise: np.ndarray = np.random.normal(0, self.m, size=self.e_cells.shape)

        # Move E cells and apply boundary behavior
        moved_coords: np.ndarray = self.e_cells + noise
        self.e_cells = self.well.apply_boundary_batch(moved_coords)

    def _cell_grow(self) -> None:
        """
        Randomly duplicates E and T cells based on their respective growth rates.
        Offspring are placed near parent cells within [-r, r] square window.
        """
        # --- E cell growth ---
        n_E: int = self.e_cells.shape[0]
        if n_E > 0:
            grow_mask_E = np.random.rand(n_E) < self.g_E
            parent_E: np.ndarray = self.e_cells[grow_mask_E]
            displacements_E: np.ndarray = np.random.uniform(-self.r, self.r, size=parent_E.shape)
            new_E: np.ndarray = self.well.apply_boundary_batch(parent_E + displacements_E)
            self.e_cells = np.vstack([self.e_cells, new_E])

        # --- T cell growth ---
        n_T: int = self.t_cells.shape[0]
        if n_T > 0:
            grow_mask_T = np.random.rand(n_T) < self.g_T
            parent_T: np.ndarray = self.t_cells[grow_mask_T]
            displacements_T: np.ndarray = np.random.uniform(-self.r, self.r, size=parent_T.shape)
            new_T: np.ndarray = self.well.apply_boundary_batch(parent_T + displacements_T)
            self.t_cells = np.vstack([self.t_cells, new_T])

    def _log(self) -> None:
        """
        Logs population sizes and optionally full state at each time step.
        Merges new C cells into the existing pool for next step.
        """
        # Record current counts
        self.history["E"].append(self.e_cells.shape[0])
        self.history["T"].append(self.t_cells.shape[0])
        self.history["C"].append(self.existing_c_cells.shape[0] + self.new_c_cells.shape[0])

        if self.log_all:
            frame: dict = {
                "E": self.e_cells.copy(),
                "T": self.t_cells.copy(),
                "C": np.vstack([self.existing_c_cells, self.new_c_cells]) if self.new_c_cells.size > 0 else self.existing_c_cells.copy(),
            }
            self.full_history.append(frame)

        # Prepare C cells for next step
        if self.new_c_cells.size > 0:
            self.existing_c_cells = np.vstack([self.existing_c_cells, self.new_c_cells]) if self.existing_c_cells.size > 0 else self.new_c_cells
        self.new_c_cells = np.empty((0, 2))

    def _step(self) -> None:
        """
        Performs one full simulation step:
            - Move E cells
            - Attempt E-T binding
            - Process C cell outcomes (split/kill)
            - Grow E and T cells
            - Log all relevant states
        """
        self._cell_move()
        self._combine_behavior()
        self._c_cell_behavior()
        self._cell_grow()
        self._log()

    def run(self) -> None:
        """
        Runs the simulation over N steps or until early stopping conditions are met.

        Early stopping can occur if:
            - All T and C cells are eliminated
            - All E and C cells are eliminated
            - Tumor population stabilizes for 100 steps
            - Tumor grows beyond a specified E-to-T ratio

        Pads history and logs appropriately if early stopping is triggered.
        """
        # Edge case: no E cells — purely exponential T growth
        if self.E_0 == 0:
            self.history["E"] = [0] * (self.N + 1)
            self.history["C"] = [0] * (self.N + 1)
            self.history["T"] = [round(self.T_0 * (1 + self.g_T) ** t) for t in range(self.N + 1)]
            self.simulated = True
            return

        unchanged_count = 0
        prev_T: int = self.history["T"][-1]

        iterator: tqdm | range = tqdm(range(self.N), desc="Simulating") if self.use_tqdm else range(self.N)

        for step in iterator:
            self._step()

            curr_E: int = self.history["E"][-1]
            curr_T: int = self.history["T"][-1]
            curr_C: int = self.history["C"][-1]

            if self.early_stop[0]:
                # Detect relative stability in T-cell population
                if prev_T * (1 - self.early_stop[2]) <= curr_T <= prev_T * (1 + self.early_stop[2]):
                    unchanged_count += 1
                else:
                    unchanged_count = 0
                prev_T = curr_T

                # Check early termination conditions
                if curr_T == 0 and curr_C == 0:
                    break  # All T cells eliminated
                if curr_E == 0 and curr_C == 0:
                    break  # All E cells eliminated
                if unchanged_count >= 100:
                    break  # T cell count stable for 100 steps
                if curr_E + curr_C > 0 and curr_T / (curr_E + curr_C) >= self.early_stop[1]:
                    # Tumor has overwhelmed E+C population
                    self.early_stop_step = step
                    remaining = self.N + 1 - len(self.history["E"])

                    if remaining > 0:
                        # Pad history with extrapolated or constant values
                        self.history["E"].extend([curr_E] * remaining)
                        self.history["C"].extend([curr_C] * remaining)
                        self.history["T"].extend([int(round(curr_T * ((self.g_T + 1) ** i))) for i in range(1, remaining + 1)])

                        # Pad detailed logs if necessary
                        if self.log_all and self.full_history is not None:
                            last_frame = self.full_history[-1]
                            self.full_history.extend([last_frame] * remaining)

                        if self.log_all and self.step_stats is not None:
                            last_stats = self.step_stats[-1]
                            self.step_stats.extend([last_stats] * remaining)

                    break

        self.simulated = True

    def plot_history(self) -> None:
        """
        Plot population dynamics over time in a 2×2 grid:

        - Top-left: T-cell and C-cell counts
        - Top-right: E-cell count
        - Bottom-left: Normalized tumor count F(t) = T(t) / T_baseline(t)
        - Bottom-right: Tumor burden ratio G(t) = T / (E + C)

        Requires the simulation to have been run.
        """
        if not self.simulated:
            warnings.warn("No simulation has been run yet. Run the run() method first.", UserWarning)
            return

        T: np.ndarray = np.array(self.history["T"])
        E: np.ndarray = np.array(self.history["E"])
        C: np.ndarray = np.array(self.history["C"])

        # Run baseline (E_0 = 0) for computing F(t)
        baseline = WellSimulation(
            E_0=0,
            T_0=self.T_0,
            r=self.r,
            m=self.m,
            N=self.N,
            L=self.L,
            k_on=self.k_on,
            k_off=self.k_off,
            k_kill=self.k_kill,
            g_E=self.g_E,
            g_T=self.g_T,
            early_stop=(False, 0, 0.0),
            log_all=False,
            use_tqdm=False,
        )
        baseline.run()
        baseline_T: list[int] = baseline.history["T"]

        time_steps: np.ndarray = np.arange(len(T))
        min_len: int = min(len(T), len(baseline_T))
        baseline_T_array: np.ndarray = np.array(baseline_T[:min_len])
        T_crop: np.ndarray = T[:min_len]

        # Derived metrics
        F: np.ndarray = np.where(baseline_T_array > 0, T_crop / baseline_T_array, 0.0)
        G: np.ndarray = np.where((E + C) > 0, T / (E + C), 0.0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        stopped_early: bool = self.early_stop_step is not None

        # Plot T and C
        axes[0, 0].plot(time_steps, T, label="T-cells", color="red")
        axes[0, 0].plot(time_steps, C, label="C-cells", color="green")
        axes[0, 0].set_ylabel("T / C Count")
        axes[0, 0].set_title("T and C Cells")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot E
        axes[0, 1].plot(time_steps, E, label="E-cells", color="blue")
        axes[0, 1].set_ylabel("E Count")
        axes[0, 1].set_title("E Cells")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot F(t)
        axes[1, 0].plot(np.arange(len(F)), F, label="F(t)", color="purple")
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("F(t)")
        axes[1, 0].set_title("Normalized T-Cell Count")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim(0, max(1.1, np.max(F) * 1.1))

        # Plot G(t)
        axes[1, 1].plot(time_steps, G, label="G = T / (E + C)", color="orange")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("G(t)")
        axes[1, 1].set_title("T-to-E Ratio")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim(0, max((self.T_0 / max(1, self.E_0)) * 1.1, np.max(G) * 1.1))

        if stopped_early:
            for ax in axes.flatten():
                ax.axvline(self.early_stop_step + 1, color="gray", linestyle="--", alpha=0.7)  # pyright: ignore[reportOptionalOperand]

        for ax in axes[1]:
            ax.set_xlim(0, self.N)

        plt.tight_layout()
        plt.show()

    def plot_snapshot(self, figsize: int = 6) -> None:
        """
        Launch an interactive widget to view cell positions at each time step.

        Requires:
            - `log_all=True` at initialization
            - Simulation must be completed via run()

        Includes +/- buttons and a slider to select the time step.
        """
        if self.E_0 == 0:
            warnings.warn("This is a baseline simulation, no snapshot info is recorded.", UserWarning)
            return
        if not self.simulated:
            warnings.warn("No simulation has been run yet. Run the run() method first.", UserWarning)
            return
        if self.full_history is None or len(self.full_history) == 0:
            print("No full history to explore. Set log_all=True when initializing.")
            return

        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.full_history) - 1 if self.early_stop_step is None else self.early_stop_step + 1,
            step=1,
            description="Time step:",
            continuous_update=False,
        )
        button_prev = widgets.Button(description="−", layout=widgets.Layout(width="40px"))
        button_next = widgets.Button(description="+", layout=widgets.Layout(width="40px"))
        figsize_widget = widgets.IntSlider(value=figsize, min=4, max=10)

        def decrease(_) -> None:
            if slider.value > slider.min:
                slider.value -= 1

        def increase(_) -> None:
            if slider.value < slider.max:
                slider.value += 1

        button_prev.on_click(decrease)
        button_next.on_click(increase)

        controls = widgets.HBox([button_prev, slider, button_next])

        def _plot_snapshot_at_time(t: int, figsize: int = 6) -> None:
            frame: dict = self.full_history[t]
            fig, ax = plt.subplots(figsize=(figsize + 2, figsize))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, self.L)
            ax.set_title(f"Cell Positions at t = {t}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("A Square Petri Dish", fontsize=12, labelpad=10)

            r_marker_size: float = (self.r / (self.L / figsize)) * 72
            marker_area: float = r_marker_size**2

            for key, color, label, marker, alpha, size_mult in [
                ("E", "blue", "E", "o", 0.5, 1.0),
                ("T", "red", "T", "o", 0.5, 1.0),
                ("C", "green", "C", "D", 0.75, 2.0),
            ]:
                if len(frame[key]) > 0:
                    coords: np.ndarray = np.array(frame[key])
                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color=color,
                        label=label,
                        alpha=alpha,
                        marker=marker,
                        s=marker_area * size_mult,
                    )

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)
            ax.grid(True)
            plt.show()

        out = widgets.interactive_output(
            _plot_snapshot_at_time,
            {"t": slider, "figsize": figsize_widget},
        )

        display(controls, out)

    def to_gif(self, filename: str = "simulation.gif", fps: int = 10, dpi: int = 100, figsize: int = 6) -> None:
        """
        Save the simulation as a GIF animation of cell positions.

        Parameters:
            filename (str): Output filename (should end in .gif).
            fps (int): Frames per second.
            dpi (int): Dots per inch (resolution).
            figsize (int): Plot size.

        Requires:
            - `log_all=True` at initialization
            - Simulation must be completed via run()
        """
        if self.E_0 == 0 or not self.simulated or self.full_history is None or len(self.full_history) == 0:
            raise RuntimeError("Simulation history is empty. Ensure `log_all=True` and `run()` has been called.")

        fig, ax = plt.subplots(figsize=(figsize + 2, figsize))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_title("Cell Positions")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("A Square Petri Dish", fontsize=12, labelpad=10)
        ax.grid(True)

        scatter_E = ax.scatter([], [], color="blue", alpha=0.5, label="E", marker="o")
        scatter_T = ax.scatter([], [], color="red", alpha=0.5, label="T", marker="o")
        scatter_C = ax.scatter([], [], color="green", alpha=0.75, label="C", marker="D")

        r_marker_size = (self.r / (self.L / figsize)) * 72
        marker_area = r_marker_size**2

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)

        def init():
            # Empty initial frame
            empty = np.empty((0, 2))
            scatter_E.set_offsets(empty)
            scatter_T.set_offsets(empty)
            scatter_C.set_offsets(empty)
            scatter_E.set_sizes([])
            scatter_T.set_sizes([])
            scatter_C.set_sizes([])
            return scatter_E, scatter_T, scatter_C

        def update(frame):
            # Update scatter plot with new positions
            data = self.full_history[frame]
            for scatter, key, size_mult in [
                (scatter_E, "E", 1.0),
                (scatter_T, "T", 1.0),
                (scatter_C, "C", 2.0),
            ]:
                coords = np.array(data[key])
                if coords.shape[0] == 0:
                    coords = np.empty((0, 2))
                scatter.set_offsets(coords)
                scatter.set_sizes([marker_area * size_mult] * len(coords))

            ax.set_title(f"Cell Positions at t = {frame}")
            return scatter_E, scatter_T, scatter_C

        anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(self.full_history), interval=1000 / fps, blit=True)

        anim.save(filename, writer="pillow", dpi=dpi)
        plt.close()
        print(f"✅ GIF saved to {os.path.abspath(filename)}")


# Monte Carlo Simulation Across Repeated Runs


def single_scenario_monte_carlo(n_runs: int, sim_kwargs: dict, seed_offset: int = 0, use_tqdm: bool = True) -> (dict[str, np.ndarray], np.ndarray):
    """
    Run multiple stochastic simulations of a single parameter scenario.

    Parameters:
        n_runs (int): Number of Monte Carlo simulation runs.
        sim_kwargs (dict): Keyword arguments to initialize WellSimulation.
        seed_offset (int): Offset for RNG seeds to ensure reproducibility.
        use_tqdm (bool): Whether to show a progress bar.

    Returns:
        simulation (dict[str, np.ndarray]): Arrays of E, T, C histories and computed F values.
        T_baseline (np.ndarray): Tumor trajectory for the baseline run (E_0 = 0).
    """
    simulation: dict = {"E": [], "T": [], "C": []}
    target_len: int = sim_kwargs["N"] + 1  # Expected number of time steps

    # Remove fixed flags to override later per run
    for fixed_arg in ["log_all", "use_tqdm", "seed"]:
        if fixed_arg in sim_kwargs:
            del sim_kwargs[fixed_arg]

    iterator: tqdm | range = tqdm(range(n_runs), desc="Simulating (E>0)") if use_tqdm else range(n_runs)
    for i in iterator:
        sim = WellSimulation(**sim_kwargs, log_all=False, use_tqdm=False, seed=seed_offset + i)
        sim.run()

        # Pad simulation to full length if early stopping occurred
        for k in ["E", "T", "C"]:
            history: list = sim.history[k]
            if len(history) < target_len:
                pad_value: int = history[-1]
                history += [pad_value] * (target_len - len(history))
            simulation[k].append(history)

    # Convert to array for each quantity
    for k in ["E", "T", "C"]:
        simulation[k] = np.array(simulation[k])

    # Run one baseline simulation with E_0 = 0
    baseline_kwargs: dict = sim_kwargs.copy()
    baseline_kwargs["E_0"] = 0
    baseline_kwargs["early_stop"] = (False, 0, 0.0)
    baseline = WellSimulation(**baseline_kwargs, log_all=False, use_tqdm=False, seed=99999)
    baseline.run()
    T_baseline: np.ndarray = np.array(baseline.history["T"])

    # Compute F = T(t; E>0) / T(t; E=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        simulation["F"] = simulation["T"] / T_baseline

    return simulation, T_baseline


def plot_result_trend(results: dict[str, np.ndarray], use_median: bool = False) -> None:
    """
    Plot mean (or median) trends with ±1 standard deviation bands for E, T, C, and F.

    Parameters:
        results (dict): Dictionary with simulation results (output from `single_scenario_monte_carlo`).
        use_median (bool): Whether to use median instead of mean for central tendency.
    """
    average_func: Callable = np.nanmedian if use_median else np.nanmean
    time: np.ndarray = np.arange(results["E"].shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    def summarize(values: np.ndarray) -> str:
        final: np.ndarray = values[:, -1]
        return f"(final: {average_func(final):.4f} ± {np.nanstd(final):.4f})"

    def plot_panel(ax, key: str, color: str, label: str):
        avg: np.ndarray = average_func(results[key], axis=0)
        std: np.ndarray = np.nanstd(results[key], axis=0)
        ax.plot(time, avg, color=color, label=label)
        ax.fill_between(time, avg - std, avg + std, color=color, alpha=0.3, label="±1 SD")
        ax.set_title(f"{label} {summarize(results[key])}")
        ax.grid(True)
        ax.legend()

    plot_panel(axes[0, 0], "E", "blue", "E-cells")
    plot_panel(axes[0, 1], "T", "red", "T-cells")
    plot_panel(axes[1, 0], "C", "green", "C-cells")
    plot_panel(axes[1, 1], "F", "purple", "F = T(t; E>0) / T(t; E=0)")

    for ax in axes[1]:
        ax.set_xlabel("Time step")

    plt.tight_layout()
    plt.show()


def plot_result_distribution(results: dict[str, np.ndarray], step: int = 20, use_violin: bool = False, use_logscale: bool = False) -> None:
    """
    Plot distribution (box or violin) of E, T, C, and F across selected time steps.

    Parameters:
        results (dict): Dictionary with simulation results.
        step (int): Step interval for plotting.
        use_violin (bool): Use violin plots instead of box plots.
        use_logscale (bool): Use log scale on y-axis for skewed distributions.
    """
    max_t: int = results["E"].shape[1]
    time_points: np.ndarray = np.unique(np.concatenate(([1], np.arange(step, max_t, step), [max_t - 1])))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    def plot_panel(ax, key: str, color: str, title: str) -> None:
        raw_data: list[np.ndarray] = [results[key][:, t] for t in time_points]

        if use_logscale:
            # Replace zeros or negatives with a small positive value for log scale
            all_non_zero: np.ndarray = np.concatenate([d[d > 0] for d in raw_data if np.any(d > 0)])
            smallest_non_zero_value: float = np.min(all_non_zero) if len(all_non_zero) > 0 else 1e-4
            replacement_value: float = smallest_non_zero_value / 10
            print(f"=== Log scale active: replacing {key} values ≤ 0 with {replacement_value:.1e} ===")
            data: list[np.ndarray] = [np.where(d <= 0, replacement_value, d) for d in raw_data]
        else:
            data = raw_data

        if use_violin:
            parts = ax.violinplot(data, positions=time_points, widths=step * 0.6)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
        else:
            ax.boxplot(
                data,
                positions=time_points,
                widths=step * 0.6,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.4),
                medianprops=dict(color="black", linewidth=2),
            )

        ax.set_title(title)
        if use_logscale:
            ax.set_yscale("log")
        ax.grid(True)

    plot_panel(axes[0, 0], "E", "blue", "E-cell Distribution")
    plot_panel(axes[0, 1], "T", "red", "T-cell Distribution")
    plot_panel(axes[1, 0], "C", "green", "C-cell Distribution")
    plot_panel(axes[1, 1], "F", "purple", "F = T(t; E>0) / T(t; E=0)")

    for ax in axes[1]:
        ax.set_xlabel("Time step")

    plt.tight_layout()
    plt.show()


# Grid-Based Monte Carlo over (E₀, T₀) Parameter Space


def grid_scenario_monte_carlo(E0_range: np.ndarray, T0_range: np.ndarray, base_kwargs: dict, n_runs: int = 100, seed_offset: int = 0) -> np.ndarray:
    """
    Run Monte Carlo simulations across a grid of E₀ and T₀ values.

    For each (E₀, T₀) pair:
        - Computes F_final = T_final(E > 0) / T_final(E = 0)

    Parameters:
        E0_range (np.ndarray): Array of effector cell initial counts.
        T0_range (np.ndarray): Array of tumor cell initial counts.
        base_kwargs (dict): Base keyword arguments for WellSimulation.
        n_runs (int): Number of simulations per grid point.
        seed_offset (int): Offset for reproducibility.

    Returns:
        np.ndarray: 3D array of shape (len(E₀), len(T₀), n_runs) containing F_final values.
    """
    shape = (len(E0_range), len(T0_range), n_runs)
    grid_F = np.full(shape, np.nan)

    iterator = tqdm([(i, j) for i in range(len(E0_range)) for j in range(len(T0_range))], desc="Grid Monte Carlo")
    for i, j in iterator:
        E_0 = E0_range[i]
        T_0 = T0_range[j]

        kwargs = base_kwargs.copy()
        kwargs.update({"E_0": E_0, "T_0": T_0})
        sim_result, T_baseline = single_scenario_monte_carlo(n_runs=n_runs, sim_kwargs=kwargs, seed_offset=seed_offset, use_tqdm=False)

        T_E_final = sim_result["T"][:, -1]
        T_0_final = T_baseline[-1]
        F_final = T_E_final / T_0_final
        grid_F[i, j, :] = F_final

    return grid_F


def plot_F_surface_interactive(grid_F: np.ndarray, E0_range: np.ndarray, T0_range: np.ndarray, use_median: bool = True) -> None:
    """
    Interactive 3D surface plot of F(t) across E₀ and T₀ using Plotly.

    Parameters:
        grid_F (np.ndarray): Output from grid_scenario_monte_carlo.
        E0_range (np.ndarray): Range of E₀ values used.
        T0_range (np.ndarray): Range of T₀ values used.
        use_median (bool): Aggregate with median (True) or mean (False).
    """
    x = np.arange(T0_range[0], T0_range[-1] + 1)
    y = np.arange(E0_range[0], E0_range[-1] + 1)

    stat_func = np.nanmedian if use_median else np.nanmean
    z = stat_func(grid_F, axis=2)

    scale_factor = (len(y) / len(E0_range), len(x) / len(T0_range))
    z = zoom(z, zoom=scale_factor, order=3)
    z = gaussian_filter(z, sigma=10)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="F(t)"))])
    fig.update_layout(title="F(t) Across (E₀, T₀)", scene=dict(xaxis_title="T₀", yaxis_title="E₀", zaxis_title="F(t)"), width=900, height=700)
    fig.show()


def plot_F_surface_gif(
    grid_F: np.ndarray,
    E0_range: np.ndarray,
    T0_range: np.ndarray,
    use_median: bool = True,
    n_frames: int = 60,
    fps: int = 10,
    save_path="./rotation.gif",
) -> None:
    """
    Create a rotating 3D surface GIF of F(t) over (E₀, T₀) using Plotly.

    Saves the rotating animation to disk and removes temporary files.
    """
    x: np.ndarray = np.arange(T0_range[0], T0_range[-1] + 1)
    y: np.ndarray = np.arange(E0_range[0], E0_range[-1] + 1)

    stat_func: Callable = np.nanmedian if use_median else np.nanmean
    z: np.ndarray = stat_func(grid_F, axis=2)

    scale_factor: tuple[float, float] = (
        len(y) / len(E0_range),
        len(x) / len(T0_range),
    )
    z: np.ndarray = zoom(z, zoom=scale_factor, order=3)  # pyright: ignore[reportAssignmentType]
    z: np.ndarray = gaussian_filter(z, sigma=10)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="F(t)"))])
    fig.update_layout(
        scene=dict(
            xaxis_title="T_0",
            yaxis_title="E_0",
            zaxis_title="F(t)",
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        width=900,
        height=700,
    )

    temp_folder = "frames"
    os.makedirs(temp_folder, exist_ok=True)
    frames: list = []

    for i in tqdm(range(n_frames)):
        angle: float = i * 360 / n_frames
        camera: dict[str, dict] = dict(eye=dict(x=np.cos(np.radians(angle)) * 2, y=np.sin(np.radians(angle)) * 2, z=0.8))
        fig.update_layout(scene_camera=camera)
        filename: str = f"{temp_folder}/frame_{i:03d}.png"
        fig.write_image(filename)
        frames.append(imageio.imread(filename))

    imageio.mimsave(save_path, frames, fps=fps)

    for f in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, f))
    os.rmdir(temp_folder)


def plot_F_surface_quick(
    grid_F: np.ndarray,
    E0_range: np.ndarray,
    T0_range: np.ndarray,
    use_median: bool = True,
) -> None:
    """
    Fast 3D surface plot of F(t) using Matplotlib with fixed camera.

    Parameters:
        grid_F (np.ndarray): F values from grid_scenario_monte_carlo.
        E0_range (np.ndarray): Effector cell range.
        T0_range (np.ndarray): Tumor cell range.
        use_median (bool): Use median (True) or mean (False) for Z values.
    """
    z = np.nanmedian(grid_F, axis=2) if use_median else np.nanmean(grid_F, axis=2)
    X, Y = np.meshgrid(T0_range, E0_range)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)
    surf = ax.plot_surface(X, Y, z, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]

    ax.set_xlabel("T₀")
    ax.set_ylabel("E₀")
    ax.set_zlabel("Median F")  # pyright: ignore[reportAttributeAccessIssue]
    ax.view_init(elev=30, azim=135)  # pyright: ignore[reportAttributeAccessIssue]

    plt.tight_layout()
    plt.show()


def plot_F_surface_gif_quick(
    grid_F: np.ndarray,
    E0_range: np.ndarray,
    T0_range: np.ndarray,
    use_median: bool = True,
    n_frames: int = 240,
    fps: int = 10,
    save_path: str = "rotating_surface.gif",
) -> None:
    """
    Generate a rotating 3D surface plot of F(t) using Matplotlib and save as a GIF.

    Faster alternative to Plotly-based surface GIF.
    """
    z = np.nanmedian(grid_F, axis=2) if use_median else np.nanmean(grid_F, axis=2)
    X, Y = np.meshgrid(T0_range, E0_range)

    temp_folder = "frames"
    os.makedirs(temp_folder, exist_ok=True)

    for i in range(n_frames):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, z, cmap="viridis", edgecolor="none")  # pyright: ignore[reportAttributeAccessIssue]

        ax.set_xlabel("T₀")
        ax.set_ylabel("E₀")
        ax.set_zlabel("Median F")  # pyright: ignore[reportAttributeAccessIssue]

        angle = i * 720 / n_frames
        ax.view_init(elev=30, azim=angle)  # pyright: ignore[reportAttributeAccessIssue]

        fname = f"{temp_folder}/frame_{i:03d}.png"
        plt.savefig(fname, dpi=100)
        plt.close(fig)

    # Compile GIF
    with imageio.get_writer(save_path, mode="I", duration=1 / fps) as writer:
        for i in range(n_frames):
            image = imageio.imread(f"{temp_folder}/frame_{i:03d}.png")
            writer.append_data(image)  # pyright: ignore[reportAttributeAccessIssue]

    # Clean up
    shutil.rmtree(temp_folder)
    print(f"GIF saved as {save_path}")


def plot_F_surface_with_ratio_lines(
    grid_F: np.ndarray,
    E0_range: np.ndarray,
    T0_range: np.ndarray,
    use_median: bool = True,
    ratios_to_plot: list[float] = [0.1, 1, 10],
) -> None:
    """
    Fast 3D surface plot of F(t) using Matplotlib with fixed camera, including ratio lines.

    Parameters:
        grid_F (np.ndarray): F values from grid_scenario_monte_carlo.
        E0_range (np.ndarray): Effector cell range.
        T0_range (np.ndarray): Tumor cell range.
        use_median (bool): Use median (True) or mean (False) for Z values.
        ratios_to_plot (list of float): E:T ratios to draw as red lines on the surface.
    """
    z = np.nanmedian(grid_F, axis=2) if use_median else np.nanmean(grid_F, axis=2)
    X, Y = np.meshgrid(T0_range, E0_range)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)
    surf = ax.plot_surface(X, Y, z, cmap="viridis", edgecolor="none", alpha=0.6)  # pyright: ignore[reportAttributeAccessIssue]

    # Add red ratio lines
    for ratio in ratios_to_plot:
        T_vals = np.linspace(T0_range[0], T0_range[-1], 100)
        E_vals = ratio * T_vals
        # Only keep values within range
        mask = (E_vals >= E0_range[0]) & (E_vals <= E0_range[-1])
        T_vals = T_vals[mask]
        E_vals = E_vals[mask]

        # Interpolate z values from surface
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((E0_range, T0_range), z)
        points = np.vstack([E_vals, T_vals]).T
        z_vals = interp(points)

        ax.plot(T_vals, E_vals, z_vals + 0.1, color="red", linewidth=2, linestyle="-", label=f"E:T = {ratio:.1f}")

    ax.set_xlabel("T₀", fontsize=12)
    ax.set_ylabel("E₀", fontsize=12)
    ax.set_zlabel("Median F", fontsize=12)  # pyright: ignore[reportAttributeAccessIssue]
    ax.view_init(elev=30, azim=135)  # pyright: ignore[reportAttributeAccessIssue]
    plt.tight_layout()
    plt.show()


def plot_F_diagonal(grid_F: np.ndarray, E0_range: np.ndarray, T0_range: np.ndarray, ratio_list: np.ndarray, use_median: bool = True) -> None:
    """
    Plot smoothed F(t) curves along constant E:T ratio diagonals.

    Parameters:
        grid_F (np.ndarray): F values from grid_scenario_monte_carlo.
        E0_range (np.ndarray): Effector cell range.
        T0_range (np.ndarray): Tumor cell range.
        ratio_list (np.ndarray): List of E:T ratios to trace.
        use_median (bool): Whether to use median (True) or mean (False) for F.
    """
    x: np.ndarray = np.arange(T0_range[0], T0_range[-1] + 1)
    y: np.ndarray = np.arange(E0_range[0], E0_range[-1] + 1)

    stat_func: Callable = np.nanmedian if use_median else np.nanmean
    z: np.ndarray = stat_func(grid_F, axis=2)

    scale_factor: tuple[float, float] = (
        len(y) / len(E0_range),
        len(x) / len(T0_range),
    )
    z: np.ndarray = zoom(z, zoom=scale_factor, order=3)  # pyright: ignore[reportAssignmentType]
    z: np.ndarray = gaussian_filter(z, sigma=40)

    curves: dict = {}

    for ratio in ratio_list:
        y_vals = ratio * x

        # Only keep values within bounds
        valid: bool = (y_vals >= np.min(y)) & (y_vals <= np.max(y))
        x_vals: np.ndarray = x[valid]
        y_vals: np.ndarray = y_vals[valid]

        xi: np.ndarray = (x_vals - np.min(x)).astype(int)
        yi: np.ndarray = (y_vals - np.min(y)).astype(int)

        if len(xi) == 0:
            continue

        z_vals: np.ndarray = z[yi, xi]
        gamma: np.ndarray = np.linspace(0, 1, len(z_vals))
        curves[ratio] = (gamma, z_vals)

    norm = mcolors.Normalize(vmin=min(ratio_list), vmax=max(ratio_list))
    cmap = plt.get_cmap("plasma")  # or try "viridis", "cividis", "magma", "turbo"

    plt.figure(figsize=(10, 6))
    for ratio, (gamma, z_vals) in curves.items():
        color = cmap(norm(ratio))
        plt.plot(gamma, z_vals, label=f"E:T = {ratio:.2f}", color=color, linewidth=2, alpha=0.9)

    plt.xlabel("γ (normalized curve position)\nHigher γ corresponds to larger E₀ and T₀ values", fontsize=14)
    plt.ylabel("F-Metric (Lower = More Effective)", fontsize=14)
    plt.title("Normalized Slices Along Constant E:T Ratio", fontsize=16)
    plt.grid(True)

    # Legend to the right
    plt.legend(title="E:T Ratio", loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=12, title_fontsize=14, frameon=False)

    plt.tight_layout()
    plt.show()


# Parameter Fitting via Stochastic Optimization


def build_optimizer_components(param_spec: dict, obs_E, obs_T, obs_C, n_runs=30) -> (Callable, list[tuple], list[str], dict):
    """
    Prepares the components for fitting model parameters using full E, T, C trajectories.

    Parameters:
        param_spec (dict): Dictionary specifying parameters.
            - Keys mapped to tuples are treated as variables to optimize (with bounds).
            - Others are treated as known constants.
        obs_E, obs_T, obs_C (np.ndarray): Observed population time series.
        n_runs (int): Number of Monte Carlo runs per evaluation.

    Returns:
        objective (Callable): Objective function mapping theta → loss.
        bounds (list[tuple]): Optimization bounds for each parameter.
        variable_names (list[str]): Names of parameters to optimize.
        known_variables (dict): Parameters that are fixed (not optimized).
    """
    variable_names = []
    bounds = []
    known_variables = {}

    for key, val in param_spec.items():
        if isinstance(val, tuple) and key != "early_stop":
            variable_names.append(key)
            bounds.append(val)
        else:
            known_variables[key] = val

    def objective(theta: np.ndarray) -> np.float64:
        param_values = dict(zip(variable_names, theta))
        sim_kwargs = {**known_variables, **param_values}

        sim_dict, _ = single_scenario_monte_carlo(n_runs=n_runs, sim_kwargs=sim_kwargs, use_tqdm=False)

        sim_E = np.median(sim_dict["E"], axis=0)
        sim_T = np.median(sim_dict["T"], axis=0)
        sim_C = np.median(sim_dict["C"], axis=0)

        loss = np.mean((sim_E - obs_E) ** 2 + (sim_T - obs_T) ** 2 + (sim_C - obs_C) ** 2)

        record = {
            "params": param_values,
            "loss": loss,
        }
        with open("opt_progress.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        return loss

    return objective, bounds, variable_names, known_variables


def fit_params(param_spec: dict, obs_E, obs_T, obs_C, n_runs=30) -> dict:
    """
    Run parameter optimization using differential evolution over full E, T, C trajectories.

    Parameters:
        param_spec (dict): See `build_optimizer_components`.
        obs_E, obs_T, obs_C (np.ndarray): Observed time series.
        n_runs (int): Monte Carlo runs per simulation.

    Returns:
        dict: Fitted parameters (merged with fixed values).
    """
    loss_fn, bounds, var_names, known_vars = build_optimizer_components(param_spec, obs_E, obs_T, obs_C, n_runs)
    result = differential_evolution(func=loss_fn, bounds=bounds, strategy="best1bin", workers=1, updating="deferred", maxiter=20, disp=False)

    theta_best = result.x
    fitted_params = dict(zip(var_names, theta_best))
    return {**known_vars, **fitted_params}


def build_weighted_optimizer_components_T_ONLY(
    param_spec: dict,
    obs_T: np.ndarray,
    obs_E_final: float,
    n_runs: int = 30,
    w_T: float = 1.0,
    w_E_final: float = 1.0,
) -> (Callable, list[tuple], list[str], dict):
    """
    Build an objective for fitting only T(t) and final E(t_final), with weighted loss terms.

    Parameters:
        param_spec (dict): Same format as above.
        obs_T (np.ndarray): Observed tumor trajectory.
        obs_E_final (float): Observed final E value.
        n_runs (int): Number of Monte Carlo runs.
        w_T (float): Weight for trajectory loss.
        w_E_final (float): Weight for final E-cell loss.

    Returns:
        Tuple of objective function, bounds, variable names, and known variables.
    """
    variable_names = []
    bounds = []
    known_variables = {}

    for key, val in param_spec.items():
        if isinstance(val, tuple) and key != "early_stop":
            variable_names.append(key)
            bounds.append(val)
        else:
            known_variables[key] = val

    def objective(theta: np.ndarray) -> float:
        param_values = dict(zip(variable_names, theta))
        sim_kwargs = {**known_variables, **param_values}

        sim_dict, _ = single_scenario_monte_carlo(n_runs=n_runs, sim_kwargs=sim_kwargs, use_tqdm=False)

        sim_T = np.median(sim_dict["T"], axis=0)
        sim_E = np.median(sim_dict["E"], axis=0)

        loss_T = np.mean((sim_T - obs_T) ** 2)
        loss_Ef = (sim_E[-1] - obs_E_final) ** 2

        loss = w_T * loss_T + w_E_final * loss_Ef

        record = {
            "params": param_values,
            "loss": loss,
        }
        with open("opt_progress.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        return loss

    return objective, bounds, variable_names, known_variables


def fit_params_weighted_T_ONLY(
    param_spec: dict,
    obs_T: np.ndarray,
    obs_E_final: float,
    n_runs: int = 30,
    w_T: float = 1.0,
    w_E_final: float = 1.0,
) -> dict:
    """
    Fit parameters using only tumor dynamics and final E count with weighted loss.

    Parameters:
        param_spec (dict): See earlier functions.
        obs_T (np.ndarray): Observed T trajectory.
        obs_E_final (float): Final value of E(t).
        n_runs (int): Monte Carlo runs per simulation.
        w_T (float): Weight for T loss.
        w_E_final (float): Weight for E_final loss.

    Returns:
        dict: Fitted parameters merged with known values.
    """
    loss_fn, bounds, var_names, known_vars = build_weighted_optimizer_components_T_ONLY(param_spec, obs_T, obs_E_final, n_runs, w_T, w_E_final)

    result = differential_evolution(
        func=loss_fn,
        bounds=bounds,
        strategy="best1bin",
        workers=1,
        updating="deferred",
        maxiter=20,
        disp=False,
    )

    theta_best = result.x
    fitted_params = dict(zip(var_names, theta_best))
    return {**known_vars, **fitted_params}


# ========================================================
# Experimental Section: WellSimulationV2 (Not Maintained)
# This section was an alternate attempt and is
# not actively used or documented. Retained for reference.
# ========================================================


@njit
def combine_behavior_jitV2(
    e_cells: np.ndarray, t_cells: np.ndarray, r: float, k_on: float, rng_vals: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, int, int):

    max_e = e_cells.shape[0]
    max_t = t_cells.shape[0]
    r2 = (2 * r) ** 2  # squared interaction radius

    touched_count = 0
    combined_count = 0

    taken_e = np.zeros(max_e, dtype=np.uint8)
    taken_t = np.zeros(max_t, dtype=np.uint8)

    new_c = np.empty((max_t, 2))  # at most one C per T cell
    c_idx = 0

    for j in range(max_t):
        tx, ty = t_cells[j]

        nearby_e_indices = []
        for i in range(max_e):
            if taken_e[i]:
                continue
            ex, ey = e_cells[i]
            dx = ex - tx
            dy = ey - ty
            if dx * dx + dy * dy <= r2:
                nearby_e_indices.append(i)

        e_count = len(nearby_e_indices)
        if e_count > 0:
            touched_count += 1
            chosen_idx = nearby_e_indices[0]  # could randomize among them if you want

            effective_k_on = k_on * e_count
            if rng_vals[j] < effective_k_on:
                ex, ey = e_cells[chosen_idx]
                cx, cy = (ex + tx) / 2, (ey + ty) / 2
                new_c[c_idx, 0] = cx
                new_c[c_idx, 1] = cy
                c_idx += 1

                taken_e[chosen_idx] = 1
                taken_t[j] = 1
                combined_count += 1

    # Extract survivors
    remaining_e = e_cells[taken_e == 0]
    remaining_t = t_cells[taken_t == 0]
    new_c = new_c[:c_idx]

    return remaining_e, remaining_t, new_c, touched_count, combined_count


class WellSimulationV2:
    def __init__(
        self,
        E_0: int,  # Initial number of effector (E) cells
        T_0: int,  # Initial number of tumor (T) cells
        r: float,  # Interaction radius for binding, and local dispersion for growth
        m: float,  # Standard deviation of E-cell random movement (diffusion strength)
        N: int,  # Maximum number of simulation steps
        L: float,  # Side length of the square well (simulation domain size)
        k_on: float,  # Probability of binding when E is within distance r of T
        k_off: float,  # Probability a bound C cell splits back into E + T per step
        k_kill: float,  # Probability a bound C cell kills the T cell and releases an E
        g_E: float,  # Baseline growth rate of E cells (before logistic/activation modifiers)
        g_T: float,  # Baseline growth rate of T cells (before Allee modifier)
        K_T: int,  # Carrying capacity for tumor (T) cells (logistic ceiling for T)
        A_T: float,  # Allee threshold scale for tumor cells (boosts growth when T moderately dense)
        K_E: int,  # Carrying capacity for effector (E) cells (logistic ceiling for E)
        activation_boost_per_T: float,  # Coefficient for T-driven activation of E cell growth (positive stimulation)
        activation_saturation_per_T2: float,  # Coefficient for T-driven suppression of E cell growth at high T (negative feedback)
        early_stop: tuple[bool, int, float],  # Tuple: (use_early_stop: bool, runaway_ratio_threshold: float, stability_tolerance: float)
        log_all: bool = False,  # Whether to log full cell positions (for visualization/snapshots)
        use_tqdm: bool = True,  # Whether to show progress bar using tqdm
        seed: int | None = None,  # Random seed for reproducibility
    ) -> None:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if E_0 == 0 and T_0 == 0:
            raise ValueError("Initializing both E and T cell to be 0 is not allowed.")

        self.E_0: int = E_0
        self.T_0: int = T_0
        self.r: float = r
        self.m: float = m
        self.N: int = N
        self.L: float = L
        self.k_on: float = k_on
        self.k_off: float = k_off
        self.k_kill: float = k_kill
        self.g_E: float = g_E
        self.g_T: float = g_T
        self.K_T = K_T
        self.A_T = A_T
        self.K_E = K_E
        self.activation_boost_per_T = activation_boost_per_T
        self.activation_saturation_per_T2 = activation_saturation_per_T2
        self.early_stop: tuple[bool, int, float] = early_stop
        self.early_stop_step: int | None = None
        self.log_all: bool = log_all
        self.use_tqdm: bool = use_tqdm

        self.simulated = False

        # Well
        self.well = SquareWell(L, mode="reflect")

        # Initialize cells as (N, 2) arrays
        self.e_cells: np.ndarray = self._initialize_cells(E_0)
        self.t_cells: np.ndarray = self._initialize_cells(T_0)
        self.existing_c_cells: np.ndarray = np.empty((0, 2))
        self.new_c_cells: np.ndarray = np.empty((0, 2))

        # History
        self.history: dict[str, list] = {"E": [], "T": [], "C": []}
        if log_all:
            self.full_history: list[dict] = []
            self.step_stats: list[dict[str, int]] = []

        # Initial logging
        self._log()

    def _initialize_cells(self, count: int) -> np.ndarray:
        # Initialize count random positions in the well as an (N, 2) NumPy array.
        return np.random.uniform(0, self.L, size=(count, 2))

    def _combine_behavior(self) -> None:
        if self.e_cells.shape[0] == 0 or self.t_cells.shape[0] == 0:
            if self.log_all:
                self.step_stats.append({"touched": 0, "combined": 0, "split": 0, "killed": 0})
            return

        rng_vals: np.ndarray = np.random.rand(self.e_cells.shape[0])
        self.e_cells, self.t_cells, self.new_c_cells, touched, combined = combine_behavior_jitV2(
            self.e_cells, self.t_cells, self.r, self.k_on, rng_vals
        )
        if self.log_all:
            self.step_stats.append({"touched": touched, "combined": combined, "split": 0, "killed": 0})

    def _c_cell_behavior(self) -> None:
        n_c: int = self.existing_c_cells.shape[0]
        if n_c == 0:
            if self.log_all:
                if len(self.step_stats) < len(self.history["E"]):
                    self.step_stats.append({"touched": 0, "combined": 0, "split": 0, "killed": 0})
            return

        rnd: np.ndarray = np.random.rand(n_c)
        is_split = rnd < self.k_off
        is_kill = (rnd >= self.k_off) & (rnd < self.k_off + self.k_kill)
        is_stay = ~(is_split | is_kill)

        # Split: create E and T from the same location
        split_coords: np.ndarray = self.existing_c_cells[is_split]
        self.e_cells = np.vstack([self.e_cells, split_coords]) if split_coords.size > 0 else self.e_cells
        self.t_cells = np.vstack([self.t_cells, split_coords]) if split_coords.size > 0 else self.t_cells

        # Kill: create E only
        kill_coords: np.ndarray = self.existing_c_cells[is_kill]
        self.e_cells = np.vstack([self.e_cells, kill_coords]) if kill_coords.size > 0 else self.e_cells

        # Keep the rest
        self.existing_c_cells = self.existing_c_cells[is_stay]

        # Log stats
        if self.log_all:
            if len(self.step_stats) < len(self.history["E"]):
                self.step_stats.append({"touched": 0, "combined": 0, "split": np.count_nonzero(is_split), "killed": np.count_nonzero(is_kill)})
            else:
                self.step_stats[-1]["split"] += np.count_nonzero(is_split)
                self.step_stats[-1]["killed"] += np.count_nonzero(is_kill)

    def _cell_move(self) -> None:
        if self.e_cells.shape[0] == 0:
            return

        # Generate 2D Gaussian noise for all E-cells
        noise: np.ndarray = np.random.normal(0, self.m, size=self.e_cells.shape)

        # Apply movement and boundary conditions
        moved_coords: np.ndarray = self.e_cells + noise
        self.e_cells = self.well.apply_boundary_batch(moved_coords)

    def _cell_grow(self) -> None:
        n_E = self.e_cells.shape[0]
        n_T = self.t_cells.shape[0]

        # Normalize to Cell Index (CI) space
        n_T_CI = n_T / self.T_0 if self.T_0 > 0 else 0
        n_E_CI = n_E / self.T_0 if self.T_0 > 0 else 0

        # Effector (E) cell growth: logistic + Type II activation
        if n_E > 0:
            activation_effect = 1 + self.activation_boost_per_T * n_T_CI - self.activation_saturation_per_T2 * n_T_CI**2
            activation_effect = max(activation_effect, 0.0)

            prob_E = self.g_E * activation_effect * (1 - n_E_CI / self.K_E)
            prob_E = max(prob_E, 0.0)

            grow_mask_E = np.random.rand(n_E) < prob_E
            parent_E = self.e_cells[grow_mask_E]
            displacements_E = np.random.uniform(-self.r, self.r, size=parent_E.shape)
            new_E = self.well.apply_boundary_batch(parent_E + displacements_E)
            self.e_cells = np.vstack([self.e_cells, new_E])

        # Tumor (T) cell growth: weak Allee effect
        if n_T > 0:
            prob_T = self.g_T * (1 - n_T_CI / self.K_T) * (n_T_CI / self.A_T + 1)
            prob_T = max(prob_T, 0.0)

            grow_mask_T = np.random.rand(n_T) < prob_T
            parent_T = self.t_cells[grow_mask_T]
            displacements_T = np.random.uniform(-self.r, self.r, size=parent_T.shape)
            new_T = self.well.apply_boundary_batch(parent_T + displacements_T)
            self.t_cells = np.vstack([self.t_cells, new_T])

    def _log(self) -> None:
        # Record counts
        self.history["E"].append(self.e_cells.shape[0])
        self.history["T"].append(self.t_cells.shape[0])
        self.history["C"].append(self.existing_c_cells.shape[0] + self.new_c_cells.shape[0])

        if self.log_all:
            frame: dict = {
                "E": self.e_cells.copy(),
                "T": self.t_cells.copy(),
                "C": np.vstack([self.existing_c_cells, self.new_c_cells]) if self.new_c_cells.size > 0 else self.existing_c_cells.copy(),
            }
            self.full_history.append(frame)

        # Merge in new C-cells for the next step
        if self.new_c_cells.size > 0:
            self.existing_c_cells = np.vstack([self.existing_c_cells, self.new_c_cells]) if self.existing_c_cells.size > 0 else self.new_c_cells
        self.new_c_cells = np.empty((0, 2))

    def _step(self) -> None:
        """
        Perform one simulation step and log:

        - Move E-cells
        - Handle E–T binding
        - Resolve C-cell actions (split, kill, stay)
        - Grow E and T cells
        """

        self._cell_move()
        self._cell_grow()
        self._combine_behavior()
        self._c_cell_behavior()
        self._log()

    def run(self) -> None:
        # Fast path: if no E-cells, nothing happens
        if self.E_0 == 0 and self.T_0 != 0:
            self.history["E"] = [0] * (self.N + 1)
            self.history["C"] = [0] * (self.N + 1)

            T_vals = [self.T_0]
            for _ in range(self.N):
                last_T = T_vals[-1]
                last_T_CI = last_T / self.T_0
                delta = self.g_T * last_T * (1 - last_T_CI / self.K_T) * (last_T_CI / self.A_T + 1)
                next_T = int(round(last_T + delta))
                T_vals.append(max(next_T, 0))

            self.history["T"] = T_vals
            self.simulated = True
            return

        unchanged_count = 0
        prev_T: int = self.history["T"][-1]

        iterator: tqdm | range = tqdm(range(self.N), desc="Simulating") if self.use_tqdm else range(self.N)

        for step in iterator:
            self._step()

            curr_E: int = self.history["E"][-1]
            curr_T: int = self.history["T"][-1]
            curr_C: int = self.history["C"][-1]

            if self.early_stop[0]:
                # Track T stability
                if prev_T * (1 - self.early_stop[2]) <= curr_T <= prev_T * (1 + self.early_stop[2]):
                    unchanged_count += 1
                else:
                    unchanged_count = 0
                prev_T = curr_T

                # Tumor extinct
                if curr_T == 0 and curr_C == 0:
                    break

                # Effector extinct
                if curr_E == 0 and curr_C == 0:
                    break

                # Tumor stabilized for too long
                if unchanged_count >= 100:
                    break

                # Tumor outpaced E (T is too strong)
                if curr_E + curr_C > 0 and curr_T / (curr_E + curr_C) >= self.early_stop[1]:
                    self.early_stop_step = step
                    remaining = self.N + 1 - len(self.history["E"])

                    # Pad T using weak Allee + logistic growth
                    T_fill = [curr_T]
                    for _ in range(remaining - 1):
                        last_T = T_fill[-1]
                        last_T_CI = last_T / self.T_0
                        delta = self.g_T * last_T * (1 - last_T_CI / self.K_T) * (last_T_CI / self.A_T + 1)
                        next_T = int(round(last_T + delta))
                        T_fill.append(max(next_T, 0))

                    self.history["T"].extend(T_fill)
                    self.history["E"].extend([curr_E] * remaining)
                    self.history["C"].extend([curr_C] * remaining)

                    if self.log_all and self.full_history is not None:
                        last_frame = self.full_history[-1]
                        self.full_history.extend([last_frame] * remaining)

                    if self.log_all and self.step_stats is not None:
                        last_stats = self.step_stats[-1]
                        self.step_stats.extend([last_stats] * remaining)

                    break

                # Effector outpaced T (E is too strong)
                if curr_T > 0 and (curr_E + curr_C) / curr_T >= self.early_stop[1]:
                    self.early_stop_step = step
                    remaining = self.N + 1 - len(self.history["E"])

                    self.history["T"].extend([curr_T] * remaining)
                    self.history["E"].extend([curr_E] * remaining)
                    self.history["C"].extend([curr_C] * remaining)

                    if self.log_all and self.full_history is not None:
                        last_frame = self.full_history[-1]
                        self.full_history.extend([last_frame] * remaining)

                    if self.log_all and self.step_stats is not None:
                        last_stats = self.step_stats[-1]
                        self.step_stats.extend([last_stats] * remaining)

                    break

        self.simulated = True

    def plot_history(self) -> None:
        """
        Plot population dynamics over time in a 2×2 layout.

        - Top-left: T-cell and C-cell counts
        - Top-right: E-cell count
        - Bottom-left: F(t) = T(t) / T_baseline(t)
        - Bottom-right: G(t) = T / (E + C)
        """
        if not self.simulated:
            warnings.warn("No simulation has been run yet. Run the run() method first.", UserWarning)
            return

        T: np.ndarray = np.array(self.history["T"])[:]
        E: np.ndarray = np.array(self.history["E"])
        C: np.ndarray = np.array(self.history["C"])

        if self.T_0 == 0:
            self.T_0 = 1

        baseline = WellSimulationV2(
            E_0=0,
            T_0=self.T_0,
            r=self.r,
            m=self.m,
            N=self.N,
            L=self.L,
            k_on=self.k_on,
            k_off=self.k_off,
            k_kill=self.k_kill,
            g_E=self.g_E,
            g_T=self.g_T,
            K_T=self.K_T,
            A_T=self.A_T,
            K_E=self.K_E,
            activation_boost_per_T=self.activation_boost_per_T,
            activation_saturation_per_T2=self.activation_saturation_per_T2,
            early_stop=(False, 0, 0.0),
            log_all=False,
            use_tqdm=False,
        )
        baseline.run()
        baseline_T: list[int] = baseline.history["T"]

        time_steps: np.ndarray = np.arange(len(T))
        min_len: int = min(len(T), len(baseline_T))
        baseline_T_array: np.ndarray = np.array(baseline_T[:min_len])
        T_crop: np.ndarray = T[:min_len]

        # Avoid division by zero
        F: np.ndarray = np.where(baseline_T_array > 0, T_crop / baseline_T_array, 0.0)
        G: np.ndarray = np.where((E + C) > 0, T / (E + C), 0.0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

        stopped_early: bool = self.early_stop_step is not None

        # --- T and C ---
        axes[0, 0].plot(time_steps, T, label="T-cells", color="red")
        axes[0, 0].plot(time_steps, C, label="C-cells", color="green")
        axes[0, 0].set_ylabel("T / C Count")
        axes[0, 0].set_title("T and C Cells")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # --- E ---
        axes[0, 1].plot(time_steps, E, label="E-cells", color="blue")
        axes[0, 1].set_ylabel("E Count")
        axes[0, 1].set_title("E Cells")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # --- F ---
        axes[1, 0].plot(np.arange(len(F)), F, label="F(t)", color="purple")
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("F(t)")
        axes[1, 0].set_title("Normalized T-Cell Count")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim(0, max(1.1, np.max(F) * 1.1))

        # --- G ---
        axes[1, 1].plot(time_steps, G, label="G = T / (E + C)", color="orange")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("G(t)")
        axes[1, 1].set_title("T-to-E Ratio")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim(0, max((self.T_0 / max(1, self.E_0)) * 1.1, np.max(G) * 1.1))

        if stopped_early:
            for ax in axes.flatten():
                ax.axvline(self.early_stop_step + 1, color="gray", linestyle="--", alpha=0.7)  # pyright: ignore[reportOptionalOperand]

        for ax in axes[1]:
            ax.set_xlim(0, self.N)

        plt.tight_layout()
        plt.show()

    def plot_snapshot(self, figsize: int = 6) -> None:
        """
        Display an interactive slider to visualize snapshots of cell positions at different time steps.
        Requires log_all=True to be set when initializing.
        """
        if self.E_0 == 0:
            warnings.warn("This is a baseline simulation, no snapshot info is recorded.", UserWarning)
            return
        if not self.simulated:
            warnings.warn("No simulation has been run yet. Run the run() method first.", UserWarning)
            return
        if self.full_history is None or len(self.full_history) == 0:
            print("No full history to explore. Set log_all=True when initializing.")
            return

        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.full_history) - 1 if self.early_stop_step is None else self.early_stop_step + 1,
            step=1,
            description="Time step:",
            continuous_update=False,
        )
        button_prev = widgets.Button(description="−", layout=widgets.Layout(width="40px"))
        button_next = widgets.Button(description="+", layout=widgets.Layout(width="40px"))
        figsize_widget = widgets.IntSlider(value=figsize, min=4, max=10)

        def decrease(_) -> None:
            if slider.value > slider.min:
                slider.value -= 1

        def increase(_) -> None:
            if slider.value < slider.max:
                slider.value += 1

        button_prev.on_click(decrease)
        button_next.on_click(increase)

        controls = widgets.HBox([button_prev, slider, button_next])

        def _plot_snapshot_at_time(t: int, figsize: int = 6) -> None:
            frame: dict = self.full_history[t]
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, self.L)
            ax.set_title(f"Cell Positions at t = {t}")

            r_marker_size: float = (self.r / (self.L / figsize)) * 72
            marker_area: float = r_marker_size**2

            for key, color, label, marker, alpha, size_mult in [
                ("E", "blue", "E", "o", 0.5, 1.0),
                ("T", "red", "T", "o", 0.5, 1.0),
                ("C", "green", "C", "D", 0.75, 2.0),
            ]:
                if len(frame[key]) > 0:
                    coords: np.ndarray = np.array(frame[key])
                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        color=color,
                        label=label,
                        alpha=alpha,
                        marker=marker,
                        s=marker_area * size_mult,
                    )

            ax.legend(loc="upper right")
            ax.grid(True)
            plt.show()

        out = widgets.interactive_output(
            _plot_snapshot_at_time,
            {"t": slider, "figsize": figsize_widget},
        )

        display(controls, out)

    def to_gif(self, filename: str = "simulation.gif", fps: int = 10, dpi: int = 100, figsize: int = 6) -> None:
        """
        Save the full simulation as a GIF animation.
        Requires log_all=True and simulation already run.
        """
        if self.E_0 == 0 or not self.simulated or self.full_history is None or len(self.full_history) == 0:
            raise RuntimeError("Simulation history is empty. Ensure log_all=True and run() has been called.")

        fig, ax = plt.subplots(figsize=(figsize + 2, figsize))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_title("Cell Positions")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("A Square Petri Dish", fontsize=12, labelpad=10)
        ax.grid(True)

        scatter_E = ax.scatter([], [], color="blue", alpha=0.5, label="E", marker="o")
        scatter_T = ax.scatter([], [], color="red", alpha=0.5, label="T", marker="o")
        scatter_C = ax.scatter([], [], color="green", alpha=0.75, label="C", marker="D")

        r_marker_size = (self.r / (self.L / figsize)) * 72
        marker_area = r_marker_size**2

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)

        def init():
            empty = np.empty((0, 2))
            scatter_E.set_offsets(empty)
            scatter_T.set_offsets(empty)
            scatter_C.set_offsets(empty)
            scatter_E.set_sizes([])
            scatter_T.set_sizes([])
            scatter_C.set_sizes([])
            return scatter_E, scatter_T, scatter_C

        def update(frame):
            data = self.full_history[frame]

            for scatter, key, size_mult in [
                (scatter_E, "E", 1.0),
                (scatter_T, "T", 1.0),
                (scatter_C, "C", 2.0),
            ]:
                coords = np.array(data[key])
                if coords.shape[0] == 0:
                    coords = np.empty((0, 2))
                scatter.set_offsets(coords)
                scatter.set_sizes([marker_area * size_mult] * len(coords))

            ax.set_title(f"Cell Positions at t = {frame}")
            return scatter_E, scatter_T, scatter_C

        anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(self.full_history), interval=1000 / fps, blit=True)

        anim.save(filename, writer="pillow", dpi=dpi)
        plt.close()
        print(f"✅ GIF saved to {os.path.abspath(filename)}")
