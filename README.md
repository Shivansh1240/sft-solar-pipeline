# SFT Solar Pipeline

> **A modular, numerically validated Surface Flux Transport solver for solar magnetic field evolution.**  
> Developed at the School of Physical Sciences, NISER Bhubneshwar, under the guidance Prof. B. B. Karak.(IIT Varanasi)

## Table of Contents

1. [Overview](#overview)
2. [Physical Model](#physical-model)
3. [Numerical Scheme](#numerical-scheme)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Modules](#modules)
8. [Validation](#validation)
9. [Known Bugs Fixed](#known-bugs-fixed)
10. [Citation](#citation)
11. [Contact](#contact)

---

## Overview

The **SFT Solar Pipeline** implements the Surface Flux Transport (SFT) model, which simulates the advection and diffusion of the Sun's large-scale radial magnetic field on the photosphere. Starting from a sequence of bipolar magnetic region (BMR) emergences, the solver reproduces the buildup of polar fields over a solar cycle — a critical observable for predicting the amplitude of the *next* cycle via the Babcock–Leighton mechanism.

This code is developed as part of ongoing research into solar cycle prediction and hemispheric asymmetry, associated with the following publications:

- **Karak, Mishra & Sreedevi (2026)** — *MNRAS* — Sunspot area statistics and hemispheric asymmetry.


## Physical Model

The SFT equation governing the radial magnetic field $B_r(\lambda, \phi, t)$ on the solar surface is:

$$
\frac{\partial B_r}{\partial t}
= -\frac{1}{R_\odot \cos\lambda} \frac{\partial}{\partial \phi}(u_\phi B_r)
- \frac{1}{R_\odot \cos\lambda} \frac{\partial}{\partial \lambda}(u_\lambda \cos\lambda \, B_r)
+ \frac{\eta}{R_\odot^2 \cos\lambda} \frac{\partial}{\partial \lambda}\!\left(\cos\lambda \frac{\partial B_r}{\partial \lambda}\right)
+ S(\lambda, \phi, t)
$$

where:

| Symbol | Description |
|---|---|
| $u_\phi$ | Differential rotation (Snodgrass 1983 profile) |
| $u_\lambda$ | Meridional flow (van Ballegooijen & Choudhuri 1988 profile) |
| $\eta$ | Supergranular diffusivity (~250–450 km² s⁻¹) |
| $S(\lambda, \phi, t)$ | BMR source term (Hale's Law + Joy's Law) |

### Meridional Flow Profile

The van Ballegooijen–Choudhuri profile is used:

$$
u_\lambda = u_0 \sin(2\lambda) \cos^p(\lambda)
$$

where the flow vanishes naturally at the poles ($\lambda = \pm 90°$), consistent with Neumann boundary conditions.

### BMR Source Term

Each BMR injection follows:

- **Hale's Law**: Leading polarity sign reverses with hemisphere and cycle parity.
- **Joy's Law**: Tilt angle $\gamma = \gamma_0 \sin(\lambda)$ (replication of Jiao et al. 2021, Table 8, within ~5%).
- **Theoretically correct dipole contribution**: $X_\text{SFT} = A \times \sin\gamma \times \sin(2\lambda)/2$

A stochastic ensemble mode (`StochasticJoyEnsemble`) draws tilt angles from a Joy's Law distribution with observed scatter, enabling Monte Carlo solar cycle forecasting.

---

## Numerical Scheme

### Spatial Discretisation

The domain is discretised on a uniform latitude–longitude grid:

$$
\lambda_i = -90° + (i + \tfrac{1}{2})\,\Delta\lambda, \quad i = 0, \dots, N_\lambda - 1
$$

Grid cells are **cell-centred**, with ghost cells at the poles enforcing Neumann zero-flux boundary conditions:

$$
\left.\frac{\partial B_r}{\partial \lambda}\right|_{\lambda = \pm 90°} = 0
$$

### IMEX Time Integration — SSP2(2,2,2)

The solver uses the second-order **Implicit–Explicit Runge–Kutta** scheme SSP2(2,2,2) (Pareschi & Russo 2005), splitting the right-hand side into:

- **Explicit**: Advection (differential rotation + meridional flow) via MUSCL scheme with van Leer limiter.
- **Implicit**: Diffusion, treated implicitly to relax the CFL constraint.

The correct tableau coefficients are:

$$
\gamma = 1 - \frac{1}{\sqrt{2}}
$$

**Stage 1:**
$$
U^{(1)} = B^n + \Delta t \left[\mathcal{L}_E(B^n) + \gamma \mathcal{L}_I(U^{(1)})\right]
$$

**Stage 2:**
$$
U^{(2)} = B^n + \Delta t \left[(1-2\gamma)\mathcal{L}_E(B^n) + 2\gamma\mathcal{L}_E(U^{(1)}) + (1-\gamma)\mathcal{L}_I(U^{(1)}) + \gamma \mathcal{L}_I(U^{(2)})\right]
$$

**Update:**
$$
B^{n+1} = \tfrac{1}{2}\left(B^n + U^{(2)}\right)
$$

> ⚠️ **Critical**: Stage 2 evaluates the stiff implicit operator at $U^{(1)}$, **not** at $B^n$.

### Advection — MUSCL with van Leer Limiter

Meridional advection uses a second-order MUSCL scheme. The van Leer slope limiter ensures TVD (Total Variation Diminishing) compliance, preventing non-physical sign flipping in unipolar pulses. Grid indexing is cell-centred throughout (no off-by-one shift between advection and diffusion stencils).

---

## Repository Structure

```
sft-solar-pipeline/
├── src/
│   └── sft/
│       ├── __init__.py
│       ├── grid.py                  # Latitude–longitude grid, cell-centred spacing
│       ├── advection.py             # MUSCL + van Leer limiter (differential rotation & meridional flow)
│       ├── diffusion.py             # Implicit diffusion operator (Neumann BCs)
│       ├── imex.py                  # SSP2(2,2,2) IMEX-RK integrator
│       ├── source.py                # SolarCycleSource: BMR injection (Hale's + Joy's Law)
│       ├── stochastic.py            # StochasticJoyEnsemble: Monte Carlo tilt scatter
│       └── pipeline.py             # Top-level SFT run orchestration
├── scripts/
│   ├── run_sft.py                   # CLI entry point for a single deterministic run
│   └── run_ensemble.py             # Ensemble runner for GPR predictor generation
├── tests/
│   ├── test_flux_conservation.py    # Signed flux integral under pure diffusion
│   ├── test_diffusion_gaussian.py   # FWHM growth, amplitude decay diagnostics
│   ├── test_boundary.py             # Neumann BC verification at poles
│   ├── test_advection.py            # TVD compliance, profile translation integrity
│   └── test_source.py              # Hale's Law parity, Joy's Law tilt, dipole moment
└── README.md
```

---

## Installation

Clone the repository and install as an editable package:

```bash
git clone https://github.com/Shivansh1240/sft-solar-pipeline.git
cd sft-solar-pipeline
pip install -e .
```

**Dependencies**: `numpy`, `scipy`, `matplotlib`, `astropy` (optional, for time utilities).

On Windows (PowerShell):

```powershell
git clone https://github.com/Shivansh1240/sft-solar-pipeline.git
cd sft-solar-pipeline
pip install -e .
```

---

## Quick Start

### Deterministic single-cycle run

```python
from sft.grid import SFTGrid
from sft.pipeline import SFTPipeline

grid = SFTGrid(n_lat=180, n_lon=360)
pipe = SFTPipeline(grid, eta=300.0, dt_days=1.0)

# Load BMR catalogue (e.g., Mandal et al. or DPD)
pipe.load_catalogue("path/to/bmr_catalogue.csv")

# Run for one solar cycle (~11 years)
B_final = pipe.run(n_years=11, output_interval_days=27)

# Compute axial dipole moment
dipole = pipe.axial_dipole_moment(B_final)
print(f"Cycle-end axial dipole: {dipole:.3f} G")
```

### Stochastic ensemble run (for GPR Stage 1 predictor)

```python
from sft.stochastic import StochasticJoyEnsemble

ensemble = StochasticJoyEnsemble(grid, n_members=500, seed=42)
polar_fields = ensemble.run_cycle(cycle_number=23, catalogue="dpd")

# polar_fields shape: (500, 2)  →  (north polar field, south polar field)
```

---

## Modules

### `sft.grid` — `SFTGrid`

Constructs the cell-centred latitude–longitude grid. Computes cell-edge positions, cosine-latitude weights for area integrals, and ghost-cell indices.

```python
grid = SFTGrid(n_lat=180, n_lon=360)
grid.lat   # shape (180,) in radians, cell centres
grid.dlat  # uniform spacing π/180
```

### `sft.advection` — MUSCL Advection

Implements the van Leer slope-limited MUSCL scheme for both differential rotation (longitudinal) and meridional flow (latitudinal). The van Ballegooijen meridional flow profile vanishes smoothly at the poles — no artificial cutoff needed.

### `sft.diffusion` — Implicit Diffusion

Solves the implicit diffusion step as a tridiagonal system using `scipy.linalg.solve_banded`. Neumann BCs are enforced by zero-gradient ghost cells, not Dirichlet $B=0$.

### `sft.imex` — SSP2(2,2,2) Integrator

Implements the full two-stage IMEX-RK step. Both tableau weights and stage evaluation points are correct (see [Known Bugs Fixed](#known-bugs-fixed)).

### `sft.source` — `SolarCycleSource`

Injects BMRs according to the input catalogue. Hale's Law parity is applied correctly per hemisphere and cycle:

```
Odd cycle  →  Northern leading = positive
Even cycle →  Northern leading = negative
```

Dipole moment $\Delta D$ is non-zero after each injection, confirming correct polarity orientation.

### `sft.stochastic` — `StochasticJoyEnsemble`

Draws tilt angles $\gamma_i \sim \mathcal{N}(\gamma_0 \sin\lambda_i,\, \sigma_\gamma^2)$ per BMR and runs an ensemble of SFT simulations. Outputs are used as Stage 1 inputs for the GPR solar cycle forecasting pipeline (Paper 3).

---

## Validation

Six validation suites are included in `tests/`:

| Test | Diagnostic | Expected behaviour | Pass criterion |
|---|---|---|---|
| Flux conservation | $\int B_r \cos\lambda\, d\lambda\, d\phi$ vs $t$ | Constant under pure diffusion (no sources) | Drift < 0.1% |
| Gaussian diffusion | FWHM$(t)$, amplitude$(t)$ | FWHM grows as $\sqrt{2\eta t + \sigma_0^2}$; amplitude decays | Monotonic, symmetric |
| Boundary conditions | $B_r$ at $\lambda = \pm 89°$ | Smooth approach, $\partial B/\partial\lambda \approx 0$ | No Dirichlet sink |
| Advection TVD | Unipolar pulse translation | No sign flip, no spurious diffusion | TVD compliant |
| Source Hale's Law | $\Delta D$ after BMR injection | Non-zero axial dipole | $\Delta D \neq 0$ |
| IMEX convergence | Norm decay under refinement | 2nd-order convergence | Slope $\approx 2.0$ |

---

## Known Bugs Fixed

Four critical bugs were identified and corrected during development. These are documented in the forensic analysis report (`docs/SFT_Forensic_Analysis.md`).

| ID | Module | Bug | Fix |
|---|---|---|---|
| BUG-A | `imex.py` | SSP2(2,2,2) tableau weights summed to $\gamma$ instead of 1, causing ~$(\gamma - 1)$ fractional flux loss per step | Corrected update coefficients: $(1, 0)$ → $(\frac{1}{2}, \frac{1}{2})$ |
| BUG-B | `imex.py` | Stage 2 stiff operator evaluated at $B^n$ instead of $U^{(1)}$, breaking 2nd-order accuracy | Changed evaluation point to $U^{(1)}$ |
| BUG-C | `diffusion.py` | Dirichlet $B=0$ boundary conditions at poles, acting as artificial flux sinks | Replaced with Neumann zero-flux ghost-cell BCs |
| BUG-D | `diffusion.py` | Boundary cells included inconsistently in explicit diffusion stencil while also being handled implicitly | Unified treatment; boundary cells handled exclusively in implicit solve |
| BUG-E | `advection.py` | MUSCL advection off-by-one in cell-edge indexing causing ~40% signed-flux drift over 11-year runs | Corrected stencil to cell-centred indexing throughout |
| BUG-F | `source.py` | Hale's Law sign inversion in `SolarCycleSource.source_term`: leading/following polarities swapped, producing quadrupolar rather than dipolar injection | Corrected polarity assignment per hemisphere and cycle parity |
| BUG-G | `stochastic.py` | Same Hale's Law inversion in `StochasticJoyEnsemble._source_term_stochastic` | Corrected consistently with BUG-F fix |
| BUG-H | `imex.py` | Diffusion decay double-counted between explicit and implicit stages in the ARS prototype | Removed explicit diffusion term; diffusion handled exclusively implicitly |



## Contact

**Shivansh Mishra**  
Integrated MSc Physical Sciences, School of Physical Sciences  
National Institute of Science Education and Research (NISER), Bhubneshwar  
DAE-DISHA Fellow  
 
*GitHub*: [@Shivansh1240](https://github.com/Shivansh1240)

---

<p align="center">
<i>Built for solar physics. Validated for science runs.</i>
</p>
