"""
2D Channel Network Generation Module

Generates 2D channel networks using a probabilistic headward-growth algorithm.
Uses any watershed boundary as input with specified location of outlet.

Reference:
    Borse, D. & Biswal, B. (2023). A novel probabilistic model to explain drainage
    network evolution. Advances in Water Resources, 171, 104342.
    https://doi.org/10.1016/j.advwatres.2022.104342
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import random
import time


def FD_coordinates(xy, FD):
    """
    Return downstream pixel coordinates based on flow direction value.

    Parameters
    ----------
    xy : tuple
        (i, j) coordinates of current pixel.
    FD : numpy.ndarray
        Flow direction matrix using D8 encoding:
        1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE.

    Returns
    -------
    tuple
        (i, j) coordinates of downstream pixel.
    """
    directions = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }
    di, dj = directions[FD[xy[0]][xy[1]]]
    return (xy[0] + di, xy[1] + dj)


def get_coordinates(ix, grid_size):
    """Convert linear index to (i, j) 2D coordinates."""
    return (int(ix / grid_size[1]), ix % grid_size[1])


def get_id(i, j, grid_size):
    """Convert (i, j) 2D coordinates to linear index."""
    return grid_size[1] * i + j


def generate_2d_network(watershed_boundary, outlet, alpha=1, beta=1,
                        save_data=True, output_prefix="network_2d"):
    """
    Generate a 2D channel network using a probabilistic headward-growth algorithm.

    Parameters
    ----------
    watershed_boundary : numpy.ndarray
        2D binary array (1 = inside watershed, 0 = outside).
    outlet : tuple
        (row, col) coordinates of the watershed outlet.
    alpha : float, default=1
        Exponent controlling length-based pixel selection probability.
    beta : float, default=1
        Exponent controlling flow-accumulation-based direction probability.
    save_data : bool, default=True
        If True, saves results to an .npz file.
    output_prefix : str, default="network_2d"
        Prefix for the output filename.

    Returns
    -------
    dict
        Dictionary containing:
        - arr_shp : watershed boundary array (padded)
        - FD      : flow direction matrix
        - Facc    : flow accumulation matrix
        - FD_connect : list of [from_id, to_id] connectivity pairs
        - Down_length : downstream length matrix (in pixels)
        - grid_size : (rows, cols) tuple
        - outlet  : outlet coordinates
        - Area    : per-pixel accumulation (1D, linear indexing)
        - Length  : upstream length from each pixel (1D, linear indexing)
        - true_area : boolean mask of watershed pixels
    """
    print("Starting 2D network generation...")
    tic = time.time()

    arr_shp = watershed_boundary.copy()
    if arr_shp.ndim == 2:
        arr_shp = np.vstack((arr_shp, np.zeros((1, arr_shp.shape[1]))))

    grid_size = arr_shp.shape

    FD = np.zeros((grid_size[0], grid_size[1]))
    Facc = np.zeros((grid_size[0], grid_size[1]))
    Area = np.zeros(grid_size[0] * grid_size[1])
    Length = np.zeros(grid_size[0] * grid_size[1])
    Down_length = np.zeros((grid_size[0], grid_size[1]))

    Pot_ids = []
    FD_connect = []
    Label = {}

    true_area = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if arr_shp[i, j] > 0:
                Label[(i, j)] = "n"
                true_area[i, j] = True

    # 8-connected neighbor offsets with D8 direction codes
    ij = ((0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
          (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128))

    # Seed from outlet neighbors
    pi = outlet
    for y in ij:
        ni, nj = pi[0] + y[0], pi[1] + y[1]
        if (0 <= ni < grid_size[0] and 0 <= nj < grid_size[1] and
                arr_shp[ni, nj] > 0 and Label[(ni, nj)] == 'n'):
            iid = get_id(ni, nj, grid_size)
            Pot_ids.append(iid)
            Label[(ni, nj)] = "p"
            Length[iid] = 1
            Down_length[ni, nj] = 1

    Label[outlet] = "o"

    while len(Pot_ids) > 0:
        # Select pixel based on cumulative length probability
        pot_len = Length[Pot_ids]
        len_alpha = pot_len ** alpha
        cum_len = np.cumsum(len_alpha)

        rand_n = random.randint(1, int(cum_len[-1]))
        pi_id = -1
        for j in range(len(Pot_ids)):
            if rand_n <= cum_len[j]:
                pi_id = Pot_ids[j]
                break

        pi = get_coordinates(pi_id, grid_size)

        # Choose flow direction proportional to downstream accumulation
        Values = []
        Facc_cum = 0
        for y in ij:
            ni, nj = pi[0] + y[0], pi[1] + y[1]
            if (0 <= ni < grid_size[0] and 0 <= nj < grid_size[1] and
                    true_area[ni, nj]):
                if Label[(ni, nj)] in ('y', 'o'):
                    Facc_cum += int((1 + Facc[ni][nj]) ** beta)
                    Values.append((y[2], Facc[ni][nj], Facc_cum))

        if Values:
            value_rand = random.randint(1, Values[-1][2])
            for i in range(len(Values)):
                if value_rand <= Values[i][2]:
                    FD[pi[0]][pi[1]] = Values[i][0]
                    break

        next_pi = FD_coordinates(pi, FD)
        next_pi_id = get_id(next_pi[0], next_pi[1], grid_size)
        FD_connect.append([pi_id, next_pi_id])

        Pot_ids.remove(pi_id)
        Label[pi] = "y"

        # Update downstream length
        coordinates = FD_coordinates(pi, FD)
        if FD[pi[0]][pi[1]] in [2, 8, 32, 128]:
            Down_length[pi[0]][pi[1]] = Down_length[coordinates[0]][coordinates[1]] + np.sqrt(2)
        else:
            Down_length[pi[0]][pi[1]] = Down_length[coordinates[0]][coordinates[1]] + 1

        # Activate new neighbor pixels
        for x in ij:
            ni, nj = pi[0] + x[0], pi[1] + x[1]
            if (0 <= ni < grid_size[0] and 0 <= nj < grid_size[1] and
                    true_area[ni, nj] and Label[(ni, nj)] == 'n'):
                Label[(ni, nj)] = 'p'
                new_id = get_id(ni, nj, grid_size)
                Pot_ids.append(new_id)
                Down_length[ni][nj] = 1 + Down_length[pi[0]][pi[1]]
                Length[new_id] = 1 + Length[get_id(pi[0], pi[1], grid_size)]

        # Propagate flow accumulation downstream
        Facc[pi[0]][pi[1]] = 1
        Area[get_id(pi[0], pi[1], grid_size)] = 1
        current_pi = pi
        while Label[(current_pi[0], current_pi[1])] != "o":
            current_pi = FD_coordinates(current_pi, FD)
            Facc[current_pi[0], current_pi[1]] += 1
            Area[get_id(current_pi[0], current_pi[1], grid_size)] += 1

    results = {
        'arr_shp': arr_shp,
        'FD': FD,
        'Facc': Facc,
        'FD_connect': np.array(FD_connect),
        'Down_length': Down_length,
        'grid_size': np.array(grid_size),
        'outlet': np.array(outlet),
        'Area': Area,
        'Length': Length,
        'true_area': true_area,
        'alpha': alpha,
        'beta': beta
    }

    if save_data:
        filepath = f"{output_prefix}_network_data.npz"
        np.savez(filepath, **results)
        print(f"Network data saved: {filepath}")
        print("Load with: data = np.load(filepath, allow_pickle=True)")

    toc = time.time()
    print(f"Network generation completed in {toc - tic:.2f} seconds")
    print(f"Total pixels processed: {int(np.sum(Facc > 0))}")

    return results


def load_network_data(filepath):
    """
    Load network data from a .npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file saved by generate_2d_network.

    Returns
    -------
    dict
        Dictionary of network arrays and metadata.
    """
    data = np.load(filepath, allow_pickle=True)
    results = {key: data[key] for key in data.files}

    # Restore Python types for scalar/tuple fields
    results['grid_size'] = tuple(results['grid_size'])
    results['outlet'] = tuple(results['outlet'])
    results['alpha'] = float(results['alpha'])
    results['beta'] = float(results['beta'])
    results['FD_connect'] = results['FD_connect'].tolist()

    print(f"Network data loaded from: {filepath}")
    print(f"Available keys: {list(results.keys())}")
    return results


def plot_network(Facc, threshold=100, title="Stream Network",
                save_fig=True, filename="stream_network.png"):
    """
    Plot binary stream network from flow accumulation matrix.

    Parameters
    ----------
    Facc : numpy.ndarray
        Flow accumulation matrix.
    threshold : int, default=100
        Minimum accumulation value to display as stream.
    title : str
        Plot title.
    save_fig : bool, default=True
        Whether to save the figure.
    filename : str
        Output filename.
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"])
    W = np.where(Facc > threshold, 1, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(W, cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar(label="Stream Network")

    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Network plot saved: {filename}")

    plt.show()


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    watershed_file = "watershed_sb.npy"
    outlet_coords = (381, 153)   # (row, col) outlet location
    alpha = 1                    # Length exponent
    beta = 1                     # Flow accumulation direction exponent
    output_prefix = "network_2d"
    save_data = True
    plot_thresholds = [50]

    # =========================================================================
    # RUN
    # =========================================================================
    print("=" * 60)
    print("2D CHANNEL NETWORK GENERATION")
    print("=" * 60)

    watershed_boundary = np.load(watershed_file)
    print(f"Watershed shape: {watershed_boundary.shape}, "
          f"pixels: {int(np.sum(watershed_boundary > 0))}")

    network_data = generate_2d_network(
        watershed_boundary=watershed_boundary,
        outlet=outlet_coords,
        alpha=alpha,
        beta=beta,
        save_data=save_data,
        output_prefix=output_prefix
    )

    Facc = network_data['Facc']
    print(f"\nNetwork pixels: {int(np.sum(Facc > 0))}")
    print(f"Max flow accumulation: {int(np.max(Facc))}")
    print(f"Connections: {len(network_data['FD_connect'])}")

    for threshold in plot_thresholds:
        if np.sum(Facc >= threshold) > 0:
            plot_network(
                Facc=Facc,
                threshold=threshold,
                title=f"Stream Network (threshold={threshold})",
                save_fig=True,
                filename=f"{output_prefix}_threshold_{threshold}.png"
            )

    plt.figure(figsize=(10, 8))
    plt.imshow(Facc, cmap='viridis')
    plt.colorbar(label="Flow Accumulation")
    plt.title("Flow Accumulation Map")
    plt.savefig(f"{output_prefix}_flow_accumulation.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(network_data['Down_length'], cmap='plasma')
    plt.colorbar(label="Distance to Outlet (pixels)")
    plt.title("Downstream Length Map")
    plt.savefig(f"{output_prefix}_downstream_length.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
