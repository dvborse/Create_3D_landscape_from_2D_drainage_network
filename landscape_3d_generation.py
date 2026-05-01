"""
3D Landscape Generation Module

Converts 2D channel networks into 3D landscapes using hydraulic geometry
equations (Parker et al., 2007) and hillslope-channel coupling.

Takes 2D drainage network data as input.
Use available 2D network data/ run 2D_network_generation before this module. 

Reference:
    Parker, G., Wilcock, P. R., Paola, C., Dietrich, W. E., & Pitlick, J. (2007).
    Physical basis for quasi-universal relations describing bankfull hydraulic
    geometry of single-thread gravel bed rivers. Journal of Geophysical Research,
    112, F04005. https://doi.org/10.1029/2006JF000549
"""

import numpy as np
import pandas as pd
import networkx as nx


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
    return (int(ix / grid_size[1]), int(ix % grid_size[1]))


def get_id(i, j, grid_size):
    """Convert (i, j) 2D coordinates to linear index."""
    return grid_size[1] * i + j


def calculate_hydraulic_geometry(Q, D_size=0.059, g=9.8):
    """
    Calculate bankfull hydraulic geometry using Parker et al. (2007) relations.

    Parameters
    ----------
    Q : numpy.ndarray
        Bankfull discharge (m³/s).
    D_size : float, default=0.059
        Characteristic bed material size (m). Default is 59 mm.
    g : float, default=9.8
        Gravitational acceleration (m/s²).

    Returns
    -------
    B : numpy.ndarray
        Channel width (m).
    H : numpy.ndarray
        Channel depth (m).
    S : numpy.ndarray
        Channel slope (dimensionless).
    """
    Q_ = Q / ((D_size**2) * np.sqrt(g * D_size))
    Q_masked = np.where(Q_ == 0, np.nan, Q_)

    B_ = 4.63 * Q_masked**0.0667
    H_ = 0.382 * Q_masked**0.0004
    S  = 0.101 * Q_masked**-0.344

    B = B_ * Q**0.4 / g**0.2
    H = H_ * Q**0.4 / g**0.2

    return B, H, S


def assign_elevations_recursive(node, FD_network, FD, slope, cell_length, Z, grid_size):
    """
    Recursively assign elevations to all upstream nodes from a given node.

    Elevation is computed by integrating slope upstream from the outlet.
    Diagonal steps use sqrt(2) * cell_length; cardinal steps use cell_length.

    Parameters
    ----------
    node : int
        Linear node ID to process.
    FD_network : networkx.DiGraph
        Flow direction network.
    FD : numpy.ndarray
        Flow direction matrix.
    slope : numpy.ndarray
        Slope matrix.
    cell_length : float
        Physical cell size (m).
    Z : numpy.ndarray
        Elevation matrix (modified in place).
    grid_size : tuple
        (rows, cols) grid dimensions.
    """
    for predecessor in FD_network.predecessors(node):
        i, j = get_coordinates(predecessor, grid_size)
        ni, nj = get_coordinates(node, grid_size)
        if FD[i][j] in [1, 4, 16, 64]:
            Z[i, j] = Z[ni, nj] + slope[i, j] * cell_length
        else:
            Z[i, j] = Z[ni, nj] + slope[i, j] * cell_length * np.sqrt(2)
        assign_elevations_recursive(predecessor, FD_network, FD, slope, cell_length, Z, grid_size)


def extract_channel_profiles(FD_network, FD, Z, Down_length, B, H,
                              slope, cell_length, grid_size, stream_grid):
    """
    Extract longitudinal elevation and hydraulic profiles along the main channel.

    Returns two profiles:
    - 'full'    : from the topographic head to outlet (includes hillslopes)
    - 'channel' : from the channel head to outlet (stream pixels only)

    Parameters
    ----------
    FD_network : networkx.DiGraph
    FD : numpy.ndarray
    Z : numpy.ndarray
    Down_length : numpy.ndarray
        Downstream length in pixels.
    B, H : numpy.ndarray
        Channel width and depth (m).
    slope : numpy.ndarray
    cell_length : float
    grid_size : tuple
    stream_grid : numpy.ndarray
        Boolean mask of stream pixels.

    Returns
    -------
    dict
        'full'    : {'Z', 'L', 'slope'}
        'channel' : {'Z', 'L', 'B', 'H'}
    """
    profiles = {}
    Length_from_out = Down_length * cell_length * 1e-3  # pixels -> km

    # Full profile: topographic head to outlet
    head_node = np.where(Down_length == np.max(Down_length))
    head_node_id = get_id(head_node[0][0], head_node[1][0], grid_size)

    Z_stream, L_stream, slope_stream = [], [], []
    node_id = head_node_id
    i, j = get_coordinates(node_id, grid_size)
    Z_stream.append(Z[i, j])
    L_stream.append(Length_from_out[i, j])
    slope_stream.append(slope[i, j])

    while True:
        successor = list(FD_network.successors(node_id))
        if successor:
            i, j = get_coordinates(successor[0], grid_size)
            Z_stream.append(Z[i, j])
            L_stream.append(Length_from_out[i, j])
            slope_stream.append(slope[i, j])
            node_id = successor[0]
        else:
            break

    profiles['full'] = {'Z': Z_stream, 'L': L_stream, 'slope': slope_stream}

    # Channel-only profile
    Length_channel = np.where(stream_grid, Length_from_out, np.nan)
    if np.any(~np.isnan(Length_channel)):
        ch_head = np.where(Length_channel == np.nanmax(Length_channel))
        ch_head_id = get_id(ch_head[0][0], ch_head[1][0], grid_size)

        Zs, Ls, Bs, Hs = [], [], [], []
        current = ch_head_id
        i, j = get_coordinates(current, grid_size)
        Zs.append(Z[i, j]); Ls.append(Length_from_out[i, j])
        Bs.append(B[i, j]); Hs.append(H[i, j])

        while True:
            successor = list(FD_network.successors(current))
            if successor:
                i, j = get_coordinates(successor[0], grid_size)
                Zs.append(Z[i, j]); Ls.append(Length_from_out[i, j])
                Bs.append(B[i, j]); Hs.append(H[i, j])
                current = successor[0]
            else:
                break

        profiles['channel'] = {'Z': Zs, 'L': Ls, 'B': Bs, 'H': Hs}

    return profiles


def create_3d_landscape(network_data, scale_factors, th_values, D_size=0.059,
                        hillslope_slope=0.1, save_data=True, output_prefix="landscape_3d"):
    """
    Create 3D landscapes from 2D network data at multiple spatial scales.

    Each scale is defined by a (scale_factor, threshold) pair. The scale_factor
    sets the physical cell size (m/pixel), and the threshold is the minimum flow
    accumulation to classify a pixel as a stream channel.

    Discharge is estimated from the empirical Britain II relation:
        Q_bf = 3.11 * A^0.61  (A in km²)

    Parameters
    ----------
    network_data : dict
        Output of generate_2d_network().
    scale_factors : list of float
        Physical cell lengths (m/pixel) for each landscape scale.
    th_values : list of int
        Flow accumulation thresholds corresponding to each scale.
    D_size : float, default=0.059
        Characteristic bed material size (m).
    hillslope_slope : float, default=0.1
        Constant slope assigned to hillslope pixels.
    save_data : bool, default=True
        If True, saves landscape data to an .npz file.
    output_prefix : str, default="landscape_3d"
        Prefix for output filename.

    Returns
    -------
    dict
        Keys 'landscape_1', 'landscape_2', ... each containing:
        scale_factor, threshold, cell_length, Area_basin, A, Q, B, H, S,
        slope, Z, stream_grid, profiles, max_Z, max_L, max_B, min_B,
        max_H, min_H, min_slope.
    """
    print("Creating 3D landscapes from 2D network...")

    FD          = network_data['FD']
    Facc        = network_data['Facc']
    FD_connect  = network_data['FD_connect']
    Down_length = network_data['Down_length']
    grid_size   = tuple(network_data['grid_size'])
    outlet      = tuple(network_data['outlet'])

    FD_network = nx.DiGraph()
    FD_network.add_edges_from(FD_connect)

    landscapes = {}
    g = 9.8

    for ix, (scale_factor, th) in enumerate(zip(scale_factors, th_values)):
        print(f"\nLandscape {ix+1}: cell_length={scale_factor:.2f} m, threshold={th}")

        cell_length  = scale_factor
        Area_basin   = np.sum(Facc > 0) * cell_length**2
        A            = Facc * cell_length**2               # drainage area (m²)
        Q            = 0.518*(A*10**-6)          # discharge (m³/s) A in km2
        B, H, S      = calculate_hydraulic_geometry(Q, D_size, g)

        stream_grid = Facc >= th

        slope = np.full_like(S, np.nan, dtype=np.float64)
        slope[stream_grid] = S[stream_grid]
        slope[(~np.isnan(S)) & (~stream_grid)] = hillslope_slope

        Z = np.where(Q == 0, np.nan, 0.0)

        outlet_id = get_id(outlet[0], outlet[1], grid_size)
        assign_elevations_recursive(
            outlet_id, FD_network, FD, slope, cell_length, Z, grid_size)

        profiles = extract_channel_profiles(
            FD_network, FD, Z, Down_length, B, H, slope, cell_length, grid_size, stream_grid)

        Length_channel = np.where(stream_grid, Down_length * cell_length * 1e-3, np.nan)
        max_L = float(np.nanmax(Length_channel)) if np.any(~np.isnan(Length_channel)) else 0.0

        landscape = {
            'scale_factor': scale_factor,
            'threshold':    th,
            'cell_length':  cell_length,
            'Area_basin':   Area_basin,
            'A': A, 'Q': Q, 'B': B, 'H': H, 'S': S,
            'slope': slope, 'Z': Z,
            'stream_grid': stream_grid,
            'profiles': profiles,
            'max_Z':    float(np.nanmax(Z)),
            'max_L':    max_L,
            'max_B':    float(np.nanmax(B[stream_grid])),
            'min_B':    float(np.nanmin(B[stream_grid])),
            'max_H':    float(np.nanmax(H[stream_grid])),
            'min_H':    float(np.nanmin(H[stream_grid])),
            'min_slope': float(np.nanmin(slope[stream_grid]))
        }

        landscapes[f'landscape_{ix+1}'] = landscape

        print(f"  Max elevation: {landscape['max_Z']:.1f} m")
        print(f"  Max channel length: {landscape['max_L']:.1f} km")
        print(f"  Width range: {landscape['min_B']:.1f} - {landscape['max_B']:.1f} m")
        print(f"  Depth range: {landscape['min_H']:.2f} - {landscape['max_H']:.2f} m")

    if save_data:
        filepath = f"{output_prefix}_data.npz"
        np.savez(filepath, **{k: np.array(v, dtype=object) for k, v in landscapes.items()})
        print(f"\nLandscape data saved: {filepath}")
        print("Load with: data = np.load(filepath, allow_pickle=True)")

    print(f"\n3D landscape generation complete for {len(scale_factors)} scales.")
    return landscapes


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    network_file    = "network_2d_network_data.npz"
    scale_factors   = [33.2604, 49.3429, 76.6634, 103.571, 138.7]
    th_values       = [1000, 500, 200, 100, 50]
    D_size          = 0.059   # Bed material size (m)
    hillslope_slope = 0.1     # Effective hillslope slope

    # =========================================================================
    # RUN
    # =========================================================================
    print("=" * 60)
    print("3D LANDSCAPE GENERATION")
    print("=" * 60)

    from network_2d_generation import load_network_data
    network_data = load_network_data(network_file)

    landscapes = create_3d_landscape(
        network_data=network_data,
        scale_factors=scale_factors,
        th_values=th_values,
        D_size=D_size,
        hillslope_slope=hillslope_slope,
        save_data=True,
        output_prefix="landscape_3d"
    )

    rows = []
    for key, lnd in landscapes.items():
        rows.append([key,
                     f"{lnd['max_Z']:.1f}", f"{lnd['max_L']:.1f}",
                     f"{lnd['max_B']:.1f}", f"{lnd['min_B']:.1f}",
                     f"{lnd['max_H']:.2f}", f"{lnd['min_H']:.2f}",
                     f"{lnd['min_slope']:.4f}"])

    df = pd.DataFrame(rows, columns=[
        "Landscape", "Max_Z(m)", "Max_L(km)", "Max_B(m)",
        "Min_B(m)", "Max_H(m)", "Min_H(m)", "Min_Slope"])

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
