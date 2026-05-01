"""
Plotting Functions Module

Visualization functions for 2D networks and 3D landscapes.
Includes elevation grids, longitudinal profiles, hypsometric curves,
slope-area relationships, and combined multi-landscape comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import matplotlib.colors
import os


def setup_plot_style():
    """Set global matplotlib style for publication-quality figures."""
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.bf'] = 'cm:bold'
    rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'


def create_output_directory(output_dir):
    """Create output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


def apply_axis_style(ax, ws=28, ws1=28, xlw=4):
    """Apply consistent axis styling (font sizes, spine widths)."""
    ax.set_xlabel(ax.get_xlabel(), fontsize=ws, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=ws, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=ws1,
                   direction='out', pad=10, width=xlw, length=15)
    for spine in ax.spines.values():
        spine.set_color('k')
        spine.set_linewidth(xlw)


def plot_elevation_grid(Z, stream_grid, th, landscape_idx, save_fig=True,
                        output_dir="./plots", filename_prefix="elevation_grid"):
    """
    Plot elevation map with stream network overlay.

    Parameters
    ----------
    Z : numpy.ndarray
        Elevation matrix (m).
    stream_grid : numpy.ndarray
        Boolean mask of stream pixels.
    th : int
        Flow accumulation threshold used to define stream pixels.
    landscape_idx : int
        Index label for this landscape (used in filename).
    save_fig : bool, default=True
    output_dir : str, default="./plots"
    filename_prefix : str, default="elevation_grid"
    """
    setup_plot_style()
    if save_fig:
        create_output_directory(output_dir)

    ws1, xlw, ws = 45, 5, 40

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z, cmap='turbo', interpolation='nearest')

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Elevation (m)", fontsize=ws, fontweight='bold')
    cb.ax.tick_params(labelsize=ws1, direction='in', width=xlw, length=15)

    ax.imshow(stream_grid, cmap='Blues', alpha=0.7, interpolation='nearest')
    ax.set_xlabel("Distance (km)", fontsize=ws, fontweight='bold', labelpad=10)
    ax.set_ylabel("Distance (km)", fontsize=ws, fontweight='bold', labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=ws1,
                    direction='in', pad=10, width=xlw, length=15,
                    top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('k')
        spine.set_linewidth(xlw)

    plt.tight_layout()

    if save_fig:
        fname = f"{output_dir}/{filename_prefix}_landscape_{landscape_idx}.png"
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Elevation grid saved: {fname}")

    plt.show()
    plt.close()


def plot_individual_profiles(landscape, landscape_idx, save_fig=True, output_dir="./plots"):
    """
    Plot individual longitudinal profiles for a single landscape:
    elevation, width, depth (channel only) and full elevation + slope.

    Parameters
    ----------
    landscape : dict
        Landscape data dictionary (output of create_3d_landscape).
    landscape_idx : int
        Index label for this landscape.
    save_fig : bool, default=True
    output_dir : str, default="./plots"
    """
    profiles = landscape['profiles']
    if save_fig:
        create_output_directory(output_dir)

    def _save_show(filename):
        if save_fig:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        plt.show()

    if 'channel' in profiles:
        ch = profiles['channel']

        plt.figure(figsize=(4, 3))
        plt.plot(ch['L'], ch['Z'], 'b-', linewidth=2)
        plt.xlabel("Distance from Outlet (km)", fontsize=10)
        plt.ylabel("Elevation (m)", fontsize=10)
        plt.title(f"Elevation Profile - Landscape {landscape_idx}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        _save_show(f"{output_dir}/elevation_profile_landscape_{landscape_idx}.png")

        plt.figure(figsize=(4, 3))
        plt.plot(ch['L'], ch['B'], 'g-', linewidth=2)
        plt.xlabel("Distance from Outlet (km)", fontsize=10)
        plt.ylabel("Channel Width B (m)", fontsize=10)
        plt.title(f"Channel Width Profile - Landscape {landscape_idx}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        _save_show(f"{output_dir}/width_profile_landscape_{landscape_idx}.png")

        plt.figure(figsize=(4, 3))
        plt.plot(ch['L'], ch['H'], 'r-', linewidth=2)
        plt.xlabel("Distance from Outlet (km)", fontsize=10)
        plt.ylabel("Channel Depth H (m)", fontsize=10)
        plt.title(f"Channel Depth Profile - Landscape {landscape_idx}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        _save_show(f"{output_dir}/depth_profile_landscape_{landscape_idx}.png")

    if 'full' in profiles:
        fl = profiles['full']

        plt.figure(figsize=(4, 3))
        plt.plot(fl['L'], fl['Z'], 'purple', linewidth=2)
        plt.xlabel("Distance from Outlet (km)", fontsize=10)
        plt.ylabel("Elevation (m)", fontsize=10)
        plt.title(f"Full Elevation Profile - Landscape {landscape_idx}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        _save_show(f"{output_dir}/full_elevation_profile_landscape_{landscape_idx}.png")

        

def calculate_hypsometry(Z):
    """
    Compute the hypsometric curve for a landscape.

    Parameters
    ----------
    Z : numpy.ndarray
        Elevation matrix (NaN for non-watershed pixels).

    Returns
    -------
    area_fraction : numpy.ndarray
        Cumulative area fraction (0 to 1).
    normalized_elevation : numpy.ndarray
        Elevation normalized to [0, 1].
    """
    valid = Z[~np.isnan(Z)]
    if len(valid) == 0:
        return np.array([]), np.array([])

    sorted_elev = np.sort(valid)[::-1]
    area_fraction = np.arange(1, len(sorted_elev) + 1) / len(sorted_elev)

    z_min, z_max = np.min(sorted_elev), np.max(sorted_elev)
    if z_max == z_min:
        return area_fraction, np.ones_like(sorted_elev)

    normalized_elevation = (sorted_elev - z_min) / (z_max - z_min)
    return area_fraction, normalized_elevation


def calculate_slope_area_relationship(Facc, slope, cell_length, num_bins=50):
    """
    Compute binned slope-area relationship.

    Parameters
    ----------
    Facc : numpy.ndarray
        Flow accumulation matrix.
    slope : numpy.ndarray
        Slope matrix.
    cell_length : float
        Physical cell size (m).
    num_bins : int, default=50
        Number of logarithmic bins.

    Returns
    -------
    binned_area : numpy.ndarray
        Mean drainage area per bin (km²).
    binned_slope : numpy.ndarray
        Mean slope per bin.
    """
    x = Facc.flatten() * cell_length**2 * 1e-6
    y = slope.flatten()

    valid = (x > 0) & (~np.isnan(y)) & (y > 0)
    x, y = x[valid], y[valid]

    if len(x) == 0:
        return np.array([]), np.array([])

    bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins + 1)
    bin_idx = np.digitize(x, bin_edges) - 1

    binned_x = np.full(num_bins, np.nan)
    binned_y = np.full(num_bins, np.nan)

    for i in range(num_bins):
        mask = bin_idx == i
        if np.any(mask):
            binned_x[i] = np.mean(x[mask])
            binned_y[i] = np.mean(y[mask])

    valid = ~np.isnan(binned_x) & ~np.isnan(binned_y)
    return binned_x[valid], binned_y[valid]


def plot_combined_profiles(landscapes_data, save_fig=True, output_dir="./plots"):
    """
    Generate combined multi-landscape comparison plots:
    elevation, width, depth, full elevation, slope, hypsometry, slope-area.

    Parameters
    ----------
    landscapes_data : dict
        Dictionary of landscape dicts (output of create_3d_landscape).
    save_fig : bool, default=True
    output_dir : str, default="./plots"
    """
    setup_plot_style()
    if save_fig:
        create_output_directory(output_dir)

    colors  = ['b', 'g', 'r', 'c', 'm']
    labels  = [f'Landscape {i+1}' for i in range(len(landscapes_data))]
    line_w  = 4
    ws, ws1, xlw = 28, 28, 4

    elevation_profiles, width_profiles, depth_profiles = [], [], []
    full_elevation_profiles, slope_profiles = [], []
    hypsometry_data, slope_area_data = [], []

    for key, landscape in landscapes_data.items():
        profiles = landscape['profiles']

        if 'channel' in profiles:
            elevation_profiles.append(profiles['channel'])
            width_profiles.append(profiles['channel'])
            depth_profiles.append(profiles['channel'])

        if 'full' in profiles:
            full_elevation_profiles.append(profiles['full'])
            
        hypsometry_data.append(calculate_hypsometry(landscape['Z']))
        slope_area_data.append(calculate_slope_area_relationship(
            landscape['A'] / landscape['cell_length']**2,
            landscape['slope'],
            landscape['cell_length']
        ))

    def _save(fig, ax, xlabel, ylabel, fname_base):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=24)
        apply_axis_style(ax, ws=ws, ws1=ws1, xlw=xlw)
        ax.invert_xaxis()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{output_dir}/{fname_base}.png", dpi=300)
            print(f"Saved: {output_dir}/{fname_base}.png")
        plt.show()

    if elevation_profiles:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, p in enumerate(elevation_profiles):
            ax.plot(p['L'], p['Z'], label=labels[i], color=colors[i % len(colors)], linewidth=line_w)
        _save(fig, ax, "Distance (km)", "Elevation (m)", "combined_elevation_vs_length")

    if width_profiles:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, p in enumerate(width_profiles):
            ax.plot(p['L'], p['B'], label=labels[i], color=colors[i % len(colors)], linewidth=line_w)
        _save(fig, ax, "Distance (km)", "Channel Width B (m)", "combined_width_vs_length")

    if depth_profiles:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, p in enumerate(depth_profiles):
            ax.plot(p['L'], p['H'], label=labels[i], color=colors[i % len(colors)], linewidth=line_w)
        ax.set_xlabel("Distance (km)"); ax.set_ylabel("Channel Depth H (m)")
        ax.legend(fontsize=20)
        apply_axis_style(ax, ws=24, ws1=24, xlw=xlw)
        ax.invert_xaxis(); plt.tight_layout()
        if save_fig:
            plt.savefig(f"{output_dir}/combined_depth_vs_length.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/combined_depth_vs_length.png")
        plt.show()

    if full_elevation_profiles:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, p in enumerate(full_elevation_profiles):
            ax.plot(p['L'], p['Z'], label=labels[i], color=colors[i % len(colors)], linewidth=line_w)
        _save(fig, ax, "Distance (km)", "Elevation (m)", "combined_full_elevation_vs_length")

    if hypsometry_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, (af, zn) in enumerate(hypsometry_data):
            if len(af) > 0:
                ax.plot(af, zn, label=labels[i], color=colors[i % len(colors)], linewidth=line_w)

        if len(hypsometry_data) > 1:
            common_x = np.linspace(0, 1, 1000)
            curves = [np.interp(common_x, af, zn)
                      for af, zn in hypsometry_data if len(af) > 0]
            if curves:
                ax.plot(common_x, np.mean(curves, axis=0), 'k--', linewidth=2.5, label='Mean')

        ax.set_xlabel("Cumulative Area Fraction"); ax.set_ylabel("Normalized Elevation")
        ax.legend(fontsize=24)
        apply_axis_style(ax, ws=ws, ws1=ws1, xlw=xlw)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{output_dir}/combined_hypsometry.png", dpi=300)
            print(f"Saved: {output_dir}/combined_hypsometry.png")
        plt.show()

    if slope_area_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, (ba, bs) in enumerate(slope_area_data):
            if len(ba) > 0:
                ax.loglog(ba, bs, 's-', label=labels[i],
                          color=colors[i % len(colors)],
                          markerfacecolor='none',
                          markeredgecolor=colors[i % len(colors)],
                          linewidth=line_w)
        ax.set_xlabel("Area (km²)"); ax.set_ylabel("Slope")
        ax.legend(fontsize=24)
        ax.grid(True, which="both", linestyle="--")
        apply_axis_style(ax, ws=ws, ws1=ws1, xlw=xlw)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{output_dir}/combined_slope_area_relationship.png", dpi=300)
            print(f"Saved: {output_dir}/combined_slope_area_relationship.png")
        plt.show()


def plot_all_visualizations(landscapes_data, individual_plots=False, combined_plots=True,
                            elevation_grids=True, save_fig=True, output_dir="./plots"):
    """
    Run all visualization routines for a landscapes dictionary.

    Parameters
    ----------
    landscapes_data : dict
        Output of create_3d_landscape().
    individual_plots : bool, default=True
    combined_plots : bool, default=True
    elevation_grids : bool, default=True
    save_fig : bool, default=True
    output_dir : str, default="./plots"
    """
    print("Generating visualization plots...")

    if elevation_grids:
        print("\nGenerating elevation grids...")
        for i, (key, landscape) in enumerate(landscapes_data.items()):
            plot_elevation_grid(
                Z=landscape['Z'], stream_grid=landscape['stream_grid'],
                th=landscape['threshold'], landscape_idx=i + 1,
                save_fig=save_fig, output_dir=output_dir)

    if individual_plots:
        print("\nGenerating individual landscape plots...")
        for i, (key, landscape) in enumerate(landscapes_data.items()):
            print(f"  {key}...")
            plot_individual_profiles(landscape=landscape, landscape_idx=i + 1,
                                     save_fig=save_fig, output_dir=output_dir)

    if combined_plots:
        print("\nGenerating combined comparison plots...")
        plot_combined_profiles(landscapes_data=landscapes_data,
                               save_fig=save_fig, output_dir=output_dir)

    print(f"\nAll plots complete. Output directory: {output_dir}")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    landscape_file   = "landscape_3d_data.npz"
    output_directory = "./plots"
    plot_individual  = False
    plot_combined    = True
    plot_grids       = True
    save_figures     = True

    # =========================================================================
    # RUN
    # =========================================================================
    print("=" * 60)
    print("LANDSCAPE VISUALIZATION")
    print("=" * 60)

    raw = np.load(landscape_file, allow_pickle=True)
    landscapes_data = {key: raw[key].item() for key in raw.files}
    print(f"Loaded {len(landscapes_data)} landscapes")

    plot_all_visualizations(
        landscapes_data=landscapes_data,
        individual_plots=plot_individual,
        combined_plots=plot_combined,
        elevation_grids=plot_grids,
        save_fig=save_figures,
        output_dir=output_directory
    )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
