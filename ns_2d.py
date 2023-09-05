""" NVT nested sampling for a 2D LJ cluster """
from copy import deepcopy

import imageio
import numpy as np
from matplotlib.pylab import plt

rng = np.random.default_rng(742022)


def potential_energy(walker, cutoff=2.5):
    """ Calculate the potential energy of a walker where its particles interact through a LJ potential """
    # Get the number of particles
    N = walker.shape[0]

    # Calculate the distances between all particles
    distances = np.linalg.norm(walker[:, None, :] - walker[None, :, :], axis=-1)

    # Calculate the potential energy
    energy = 0
    for i in range(N):
        for j in range(i + 1, N):
            if distances[i, j] < cutoff:
                energy += 4 * (distances[i, j] ** -12 - distances[i, j] ** -6)

    return energy


def plot_walkers(nrows, ncols, walkers, energies, a, name, red=True, figsize=(5.67, 4.76)):
    """
    Plot the walkers
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)

    # Plot the walkers
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].plot(walkers[i * ncols + j, :, 0], walkers[i * ncols + j, :, 1], 'o')

            # Set the title of the subplot to the potential energy in scientific notation
            ax[i, j].set_title(f"{energies[i * ncols + j]:.2e}")

            if red:
                # Color the highest energy walker red
                if energies[i * ncols + j] == max(energies):
                    ax[i, j].plot(walkers[i * ncols + j, :, 0], walkers[i * ncols + j, :, 1], 'ro')

    # Remove the x and y ticks from all subplots
    ax[-1, 0].set_xticks([])
    ax[-1, 0].set_yticks([])

    # Make sure all subplots have the same x and y limits
    ax[-1, 0].set_xlim(0, a)
    ax[-1, 0].set_ylim(0, a)

    # Make sure all subplots render as squares
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_aspect('equal')

    # Add energy of the highest energy walker
    # and the iteration number
    # to the figure as a title
    fig.suptitle(f"Energy: {max(energies):.2e} | Iteration: {name[8:-4]}")

    plt.tight_layout()
    plt.savefig(name, dpi=300)
    pass


def perform_random_walk(walker, a, nsteps=1600):
    """ Perform a random walk on a walker for nsteps """
    # Get the number of particles
    N = walker.shape[0]

    # Perform a random walk on the new walker for 1600 steps
    for i in range(nsteps):
        # Pick a random particle
        particle = rng.integers(0, N)

        # Generate a random move
        move = rng.normal(0, 0.1, size=2)

        # Given the shape of a walker is (N, 2), the first index is the particle and the second index is the x or y
        # Apply periodic boundary conditions along the x-axis and y-axis
        if walker[particle, 0] + move[0] < 0 or walker[particle, 0] + move[0] > a:
            move[0] *= -1
        if walker[particle, 1] + move[1] < 0 or walker[particle, 1] + move[1] > a:
            move[1] *= -1

        # Update the walker
        walker[particle] += move

    return walker


def main():
    """ Main function """
    # Initialize walkers
    nrows = 3
    ncols = 3
    N = 5
    nwalkers = nrows * ncols
    a = 10

    # Generate random walkers with positions in [0, 5]
    walkers = rng.random((nwalkers, N, 2)) * a

    # Calculate potential energies of the walkers
    energies = [potential_energy(walker) for walker in walkers]

    # Plot the walkers
    plot_walkers(nrows, ncols, walkers, energies, a, name="walkers_0.png")

    # Get the index of the highest energy walker
    max_energy_index = np.argmax(energies)
    old_walker = deepcopy(walkers[max_energy_index])
    new_walker = deepcopy(walkers[max_energy_index])

    # Perform a random walk on the new walker for 1600 steps
    new_walker = perform_random_walk(new_walker, a, nsteps=1600)

    # Calculate the potential energy of the new walker
    old_energy = potential_energy(old_walker)
    new_energy = potential_energy(new_walker)

    # Update the walker if the new energy is lower
    if new_energy < old_energy:
        walkers[max_energy_index] = new_walker
        energies[max_energy_index] = new_energy

    # Plot the walkers
    plot_walkers(nrows, ncols, walkers, energies, a, name="walkers_1.png", red=False)

    # Perform nested sampling for 1000 steps
    nsteps = 5000
    energy_limit = []
    for i in range(nsteps):
        # Get the index of the highest energy walker
        max_energy_index = np.argmax(energies)
        energy_limit.append(max(energies))
        old_walker = deepcopy(walkers[max_energy_index])
        new_walker = deepcopy(walkers[max_energy_index])

        # Perform a random walk on the new walker for 1600 steps
        new_walker = perform_random_walk(new_walker, a, nsteps=1600)

        # Calculate the potential energy of the new walker
        old_energy = potential_energy(old_walker)
        new_energy = potential_energy(new_walker)

        # Update the walker if the new energy is lower
        if new_energy < old_energy:
            walkers[max_energy_index] = new_walker
            energies[max_energy_index] = new_energy

        # Plot the walkers every 100 steps
        if i % 100 == 0:
            plot_walkers(nrows, ncols, walkers, energies, a, name=f"walkers_{i + 2}.png", red=False)

    # Generate a gif of the walkers
    images = []
    for i in range(0, nsteps, 100):
        images.append(imageio.imread(f"walkers_{i + 2}.png"))
    imageio.mimsave('walkers.gif', images, duration=0.5)

    # Plot the energy limit
    fig, ax = plt.subplots(figsize=(5.67, 4.76))
    ax.plot(energy_limit)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy limit")
    ax.set_ylim(-4.5, 1.5)
    plt.tight_layout()
    plt.savefig("energy_limit.png", dpi=300)


if __name__ == "__main__":
    main()
