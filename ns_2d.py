import os
import sys
from typing import List, Tuple

import imageio
import numpy as np
from matplotlib.pylab import plt

# Initialize the random number generator
rng = np.random.default_rng(742022)

# Constants
IMAGE_DIR = './images'
GIF_NAME = 'walkers.gif'
ENERGY_LIMIT_NAME = 'energy_limit.png'
WALKER_IMG_TEMPLATE = 'walkers_{}.png'
NUM_RANDOM_STEPS = 1600
ENERGY_PLOT_INTERVAL = 100
GIF_DURATION = 0.5
ENERGY_Y_LIM = (-4.5, 1.5)
ADJUSTMENT_INTERVAL = 100
STEP_SIZE_INCREASE_FACTOR = 1.1
STEP_SIZE_DECREASE_FACTOR = 0.9
INITIAL_STEP_SIZE = 0.1
MIN_ACCEPTANCE_RATIO = 0.25
MAX_ACCEPTANCE_RATIO = 0.75


def potential_energy(walker: np.ndarray, cutoff: float = 2.5) -> float:
    distances = np.linalg.norm(walker[:, None, :] - walker[None, :, :], axis=-1)
    energy_contributions = np.where(
        distances < cutoff,
        4 * (distances ** -12 - distances ** -6),
        0
    )
    return np.sum(np.triu(energy_contributions, k=1))


def perform_random_walk(walker: np.ndarray, a: float, nsteps: int = NUM_RANDOM_STEPS,
                        step_size: float = INITIAL_STEP_SIZE) -> np.ndarray:
    accepted_moves = 0
    for i in range(nsteps):
        old_energy = potential_energy(walker)
        particle = rng.integers(0, walker.shape[0])
        move = rng.normal(0, step_size, size=2)
        proposed_walker = walker.copy()
        proposed_walker[particle] = (proposed_walker[particle] + move) % a
        new_energy = potential_energy(proposed_walker)

        # If the move decreases the potential energy, accept the move
        if new_energy < old_energy:
            walker = proposed_walker
            accepted_moves += 1

        # Adjust step size at regular intervals
        if (i + 1) % ADJUSTMENT_INTERVAL == 0:
            acceptance_ratio = accepted_moves / ADJUSTMENT_INTERVAL
            if acceptance_ratio > MAX_ACCEPTANCE_RATIO:
                step_size *= STEP_SIZE_INCREASE_FACTOR
            elif acceptance_ratio < MIN_ACCEPTANCE_RATIO:
                step_size *= STEP_SIZE_DECREASE_FACTOR
            accepted_moves = 0  # Reset the counter for the next interval

    return walker


def update_walker(walkers: np.ndarray, energies: List[float], a: float) -> Tuple[np.ndarray, List[float]]:
    max_energy_index = np.argmax(energies)
    old_walker = walkers[max_energy_index].copy()
    new_walker = perform_random_walk(old_walker.copy(), a)
    if potential_energy(new_walker) < potential_energy(old_walker):
        walkers[max_energy_index] = new_walker
        energies[max_energy_index] = potential_energy(new_walker)
    return walkers, energies


def plot_walkers(nrows: int, ncols: int, walkers: np.ndarray, energies: List[float], a: float, iteration: int,
                 red: bool = True) -> None:
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.67, 4.76))
    max_energy = max(energies)

    # Define colors for the walkers using a colorblind-friendly palette
    max_energy_color = "#e41a1c"  # red
    other_walkers_color = "#377eb8"  # blue

    for i, walker in enumerate(walkers):
        row, col = divmod(i, ncols)

        # Use different colors for max energy walker and others
        color = max_energy_color if energies[i] == max_energy else other_walkers_color
        ax[row, col].scatter(walker[:, 0], walker[:, 1], c=color, edgecolors='grey', linewidth=0.5,
                             s=80)  # Increased size for visibility

        # Update the title format with a more readable font size
        ax[row, col].set_title(f"$E_{{\mathrm{{pot}}}} = {energies[i]:.2f}$", fontsize=12, pad=15)
        ax[row, col].set_aspect('equal')

        # Add gridlines with minor ticks (reduced linewidth for less obtrusive gridlines)
        ax[row, col].grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
        ax[row, col].minorticks_on()
        ax[row, col].grid(which='minor', linestyle=':', linewidth=0.15, alpha=0.5)

        # Remove x and y labels and tick labels
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_xticklabels([])
        ax[row, col].set_yticklabels([])

    # Set x and y axis limits and ensure all subplots share the same axes
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_xlim(0, a)
            ax[i, j].set_ylim(0, a)

    plt.tight_layout()
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    plt.savefig(os.path.join(IMAGE_DIR, WALKER_IMG_TEMPLATE.format(iteration)), dpi=300, bbox_inches="tight")


def main():
    nrows, ncols, N, a = 3, 3, 5, 10
    nwalkers = nrows * ncols
    walkers = rng.random((nwalkers, N, 2)) * a
    energies = [potential_energy(walker) for walker in walkers]

    plot_walkers(nrows, ncols, walkers, energies, a, iteration=0)

    walkers, energies = update_walker(walkers, energies, a)
    plot_walkers(nrows, ncols, walkers, energies, a, iteration=1, red=False)
    sys.exit()

    nsteps, energy_limit = 5000, []
    for i in range(nsteps):
        walkers, energies = update_walker(walkers, energies, a)
        energy_limit.append(max(energies))
        if i % ENERGY_PLOT_INTERVAL == 0:
            plot_walkers(nrows, ncols, walkers, energies, a, iteration=i + 2, red=False)

    images = [imageio.imread(os.path.join(IMAGE_DIR, WALKER_IMG_TEMPLATE.format(i + 2))) for i in
              range(0, nsteps, ENERGY_PLOT_INTERVAL)]
    imageio.mimsave(GIF_NAME, images, duration=GIF_DURATION)

    fig, ax = plt.subplots(figsize=(5.67, 4.76))
    ax.plot(energy_limit)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy limit")
    ax.set_ylim(*ENERGY_Y_LIM)
    plt.tight_layout()
    plt.savefig(ENERGY_LIMIT_NAME, dpi=300)


# Run main if the script is executed as a standalone file
if __name__ == "__main__":
    main()
