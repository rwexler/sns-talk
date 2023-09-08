import os
import time
from copy import deepcopy
from typing import List

import imageio
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib.pylab import plt

# Initialize the random number generator
rng = np.random.default_rng(7422)

# Constants
OUTPUT_DIR = './output'
WALKER_IMG_TEMPLATE = 'walkers_{:03d}.png'
WALKER_IMG_RED_TEMPLATE = 'walkers_red_{:03d}.png'


def calculate_potential_energy(positions, lattice_constant, cutoff=2.5):
    potential_energy = 0
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i < j:
                dx = positions[i, 0] - positions[j, 0]
                dx = dx - np.rint(dx / lattice_constant) * lattice_constant
                dy = positions[i, 1] - positions[j, 1]
                dy = dy - np.rint(dy / lattice_constant) * lattice_constant
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < cutoff:
                    potential_energy += 4 * (distance ** -12 - distance ** -6)
    return potential_energy + rng.uniform() * 1.0e-12


def walk(positions, energy_limit, lattice_constant, step_size, number_of_steps):
    number_of_accepted_steps = 0
    for _ in range(number_of_steps):
        i = rng.integers(positions.shape[0])
        old_position = deepcopy(positions[i])

        positions[i] += rng.uniform(-step_size, step_size, size=2)
        positions[i] = np.mod(positions[i], lattice_constant)
        new_energy = calculate_potential_energy(positions, lattice_constant)
        if new_energy < energy_limit:
            number_of_accepted_steps += 1
        else:
            positions[i] = old_position
    acceptance_ratio = number_of_accepted_steps / number_of_steps
    return positions, acceptance_ratio


def decorrelate(positions, energy_limit, lattice_constant, number_of_steps=100):
    # Find a suitable step size
    step_size = 0.1
    acceptance_ratio = 0
    while acceptance_ratio < 0.25 or acceptance_ratio > 0.75:
        _, acceptance_ratio = walk(positions, energy_limit, lattice_constant, step_size, number_of_steps)
        if acceptance_ratio < 0.25:
            step_size *= 0.9
        elif acceptance_ratio > 0.75:
            step_size *= 1.1
            if step_size > lattice_constant / 2:
                step_size = lattice_constant / 2
                break

    # Decorrelate the walker
    new_positions, _ = walk(positions, energy_limit, lattice_constant, step_size, number_of_steps)
    new_energy = calculate_potential_energy(new_positions, lattice_constant)
    return new_positions, new_energy


def plot_walkers(
        walkers: np.ndarray,
        a: float,
        energies: List[float] = None,
        iteration: int = 0,
        red: bool = False,
):
    nrows = int(np.sqrt(walkers.shape[0]))
    ncols = int(np.ceil(walkers.shape[0] / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.67, 4.76))
    for i, ax in enumerate(axs.flat):
        if red and i == np.argmax(energies):
            sns.scatterplot(x=walkers[i, :, 0], y=walkers[i, :, 1], ax=ax, color='red')
        else:
            sns.scatterplot(x=walkers[i, :, 0], y=walkers[i, :, 1], ax=ax)

        # Set the axis limits
        ax.set_xlim(0, a)
        ax.set_ylim(0, a)

        # Set equal aspect ratio
        ax.set_aspect("equal", "box")

        # Remove the axis labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Add the energy as a title
        if energies is not None:
            ax.set_title(f"$E = {energies[i]:.2f}$")

    plt.tight_layout()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if red:
        plt.savefig(os.path.join(OUTPUT_DIR, WALKER_IMG_RED_TEMPLATE.format(iteration)), dpi=300)
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, WALKER_IMG_TEMPLATE.format(iteration)), dpi=300)
    plt.close()


# plot lennard jones potential from 0 to 10 sigma
def plot_lj_potential():
    # set the figure size
    plt.figure(figsize=(5.67, 2.9))

    x = np.linspace(0.5, 3, 1000)
    y = 4 * (x ** -12 - x ** -6)
    plt.plot(x, y)
    plt.xlabel(r"$r \ \left( \sigma \right)$")
    plt.ylabel(r"$E_{LJ} \ \left( \epsilon \right)$")
    plt.ylim(-1.5, 3.5)

    # plot horizontal dashed line at 0 and -epsilon
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axhline(y=-1, color='black', linestyle='--')

    # plot vertical dashed line at the minimum of the potential and add a label
    plt.axvline(x=2 ** (1 / 6), color='black', linestyle='--')
    plt.text(2 ** (1 / 6) + 0.05, 3, "$r_{min}$", va="center")

    # add the lennard jones potential as the title
    plt.title(
        r"$E_{LJ} \left( r \right) = 4 \epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right]$")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lj_potential.png"), dpi=300)
    plt.close()


def ns():
    K = 9  # Number of walkers
    N = 3  # Number of particles
    a = 5  # Box size
    rN = rng.random((K, N, 2)) * a  # Random walkers
    E = [calculate_potential_energy(r, a) for r in rN]  # Energies of the walkers

    # Plot the Lennard-Jones potential
    plot_lj_potential()

    # Run nested sampling
    M = 200  # Number of steps
    rN_lim = []  # List of walker positions
    E_lim = []  # List of energy limits
    plot_walkers(rN, a, E, iteration=0)
    plot_walkers(rN, a, E, iteration=0, red=True)
    for i in range(M):
        initial_time = time.time()
        i_max = np.argmax(E)
        rN_max = deepcopy(rN[i_max])
        E_max = E[i_max]
        if i == 0:
            rN_lim.append(deepcopy(rN_max))
            E_lim.append(E_max)
            print(f"Iteration {0:03d}: E = {max(E):.2f}")
        rN_new, E_new = decorrelate(rN_max, E_max, a)
        rN[i_max] = rN_new
        E[i_max] = E_new
        rN_lim.append(deepcopy(rN_max))
        E_lim.append(E_max)
        plot_walkers(rN, a, E, iteration=i + 1, red=True)
        final_time = time.time()
        print(f"Iteration {i + 1:03d}: E = {max(E):.2f} (took {final_time - initial_time:.2f} seconds)")

    # Save rN_lim and E_lim
    np.save("rN_lim.npy", rN_lim)
    np.save("E_lim.npy", E_lim)


# make a movie of the evolution of the walkers
def make_movie():
    # load every tenth image
    images = []
    for i in range(0, 201, 10):
        filename = os.path.join(OUTPUT_DIR, WALKER_IMG_RED_TEMPLATE.format(i))
        images.append(Image.open(filename))

    # save the images as a gif
    duration = 1000  # milliseconds
    imageio.mimsave(os.path.join(OUTPUT_DIR, "walkers.gif"), images, duration=duration)


def plot_energy():
    # load the energy values
    E_lim = np.load("E_lim.npy")

    # plot the energy values
    plt.figure(figsize=(5.67, 4.76))
    plt.plot(np.log10(E_lim - min(E_lim)), color='black')
    plt.xlabel("Iteration")
    plt.ylabel(r"$\log_{10} \left( E - E_{min} \right)$")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy.png"), dpi=300)
    plt.close()


# Run main if the script is executed as a standalone file
if __name__ == "__main__":
    # make a directory for the output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ns()
    make_movie()
    plot_energy()
