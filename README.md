# Fluvial Landscape Generation

Code for reverse-engineering 3D fluvial landscapes from 2D channel networks using
hydraulic geometry relationships.

**Associated paper:**  
Zhang, L., Borse, D., Singh, A., Pizzuto, J. E., Fu, X., & Parker, G. (2025).  
*Hydraulic geometry hypothesis allows reverse engineering of 3D quasi-equilibrium
landscapes from 2D channel networks.* PNAS.

---

## Contents

| File | Description |
|------|-------------|
| `network_generation.py` | Probabilistic 2D channel network generation |
| `landscape_3d_generation.py` | 3D landscape construction via hydraulic geometry |
| `plotting_functions.py` | Visualization: profiles, hypsometry, slope-area |
| Data:
  `watershed_sb.npy` | Input: watershed boundary array (350 × 300 pixels) |
| `Original_2D_network_data.npz` | Pre-computed network used to produce the paper results |

---

## Requirements

```
numpy
matplotlib
pandas
networkx
```

Install with:
```bash
pip install numpy matplotlib pandas networkx
```

---

## Quick Start

### Step 1 — Generate a 2D network

```python
import numpy as np
from network_generation import generate_2d_network

watershed = np.load("watershed_sb.npy")
network_data = generate_2d_network(
    watershed_boundary=watershed,
    outlet=(381, 153),
    alpha=1,
    beta=1,
    save_data=True,
    output_prefix="my_network"
)
```

### Step 2 — Build 3D landscapes

```python
from landscape_3d_generation import create_3d_landscape

scale_factors = [33.2604, 49.3429, 76.6634, 103.571, 138.7]  # m/pixel
th_values     = [1000, 500, 200, 100, 50]                     # accumulation thresholds

landscapes = create_3d_landscape(
    network_data=network_data,
    scale_factors=scale_factors,
    th_values=th_values,
    D_size=0.059,           # bed material size (m)
    hillslope_slope=0.1,    # effective hillslope slope
    save_data=True,
    output_prefix="my_landscape"
)
```

### Step 3 — Visualize

```python
from plotting_functions import plot_all_visualizations

plot_all_visualizations(
    landscapes_data=landscapes,
    individual_plots=True,
    combined_plots=True,
    elevation_grids=True,
    save_fig=True,
    output_dir="./plots"
)
```

---

## Example Data

### 2D network (generate from scratch)

The network generation algorithm is **stochastic** — each run produces a statistically
similar but not identical network. To reproduce a run with fixed randomness:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# then call generate_2d_network(...)
```

### Pre-computed network from the paper

`Original_2D_network_data.npz` contains the exact 2D network used to produce the results
in the paper. Use this as input to `create_3d_landscape` to reproduce the paper figures.

```python
from network_2d_generation import load_network_data

network_data = load_network_data("Original_2D_network_data.npz")
```

---

## Data Format

All data is saved as NumPy `.npz` files, readable in any Python environment:

```python
data = np.load("example_network_data_network_data.npz", allow_pickle=True)
print(list(data.files))
# ['arr_shp', 'FD', 'Facc', 'FD_connect', 'Down_length', 'grid_size',
#  'outlet', 'Area', 'Length', 'true_area', 'alpha', 'beta']
```

### Network data keys

| Key | Shape | Description |
|-----|-------|-------------|
| `FD` | (rows, cols) | Flow direction matrix (D8 encoding) |
| `Facc` | (rows, cols) | Flow accumulation (upstream pixel count) |
| `FD_connect` | (N, 2) | Connectivity list: [from_id, to_id] |
| `Down_length` | (rows, cols) | Downstream distance to outlet (pixels) |
| `arr_shp` | (rows+1, cols) | Watershed boundary (padded) |
| `grid_size` | (2,) | Grid dimensions |
| `outlet` | (2,) | Outlet (row, col) |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1 | Exponent for length-based pixel selection probability |
| `beta` | 1 | Exponent for flow-accumulation-based direction probability |
| `D_size` | 0.059 m | Characteristic bed material size (Britain II catchments) |
| `hillslope_slope` | 0.1 | Constant effective hillslope slope |
| `scale_factors` | see paper | Physical cell size (m) per landscape scale |
| `th_values` | see paper | Flow accumulation thresholds per landscape scale |

---

## References

- **Network algorithm:** Borse, D. & Biswal, B. (2023). A novel probabilistic model to explain
  drainage network evolution. *Advances in Water Resources*, 171, 104342.
  https://doi.org/10.1016/j.advwatres.2022.104342

- **Hydraulic geometry:** Parker, G., Wilcock, P. R., Paola, C., Dietrich, W. E., & Pitlick, J.
  (2007). Physical basis for quasi-universal relations describing bankfull hydraulic geometry of
  single-thread gravel bed rivers. *Journal of Geophysical Research*, 112, F04005.
  https://doi.org/10.1029/2006JF000549
