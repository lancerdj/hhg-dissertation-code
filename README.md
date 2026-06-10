HHG Dissertation Code

This repository contains Python scripts and processed TDSE-LUT data used for
the numerical simulations in Jian Dong's Ph.D. dissertation at Texas A&M
University.

The repository focuses on mask-dependent high-harmonic generation (HHG)
simulations discussed in Chapter 5 of the dissertation. It includes code for
the random-phase-screen (RPS) beam model, the Hermite-Gaussian (HG) beam model,
and the TDSE-LUT-based macroscopic HHG yield calculation.

Repository Contents

```text
hhg-dissertation-code/
|-- README.md
|-- requirements.txt
|-- fitting and yield HG.py
|-- fitting and yield random phase screen.py
|-- yield random phase screen TDSE.py
`-- tdse_lut_8cycle/
    |-- code/
    |-- input_tdse_npz/
    |-- input_tdse_npz_manifest.csv
    `-- lut_outputs/
```

Description

The code in this repository was developed to study how spatial aperturing of
the driving laser beam affects the detected HHG yield. The main goal is to
compare the full beam, round aperture, two-sided aperture, and diagonal aperture
under the same macroscopic propagation framework.

The repository includes three main simulation approaches:

1. RPS model with empirical dipole fitting

   This model represents the experimental driving beam using a random phase
   screen and amplitude modulation. The microscopic harmonic response is
   described using an empirical intensity-dependent dipole function. The model
   is used to calculate near-field and far-field HHG yields for different
   aperture geometries.

2. HG-mode model with empirical dipole fitting

   This model represents the finite beam quality using an incoherent mixture of
   Hermite-Gaussian modes. It provides an independent beam-model comparison to
   test whether the aperture-dependent enhancement is specific to the
   random-phase-screen representation.

3. TDSE-LUT-based model

   This model replaces the empirical dipole response with a lookup table derived
   from time-dependent Schrodinger equation (TDSE) calculations. The TDSE-LUT
   data provide the microscopic harmonic amplitude and phase as functions of
   laser intensity, which are then used in the macroscopic HHG propagation
   calculation.

Data

The `tdse_lut_8cycle/` folder contains processed TDSE-LUT data used by the
TDSE-based HHG yield calculation.

Large raw experimental data files, raw TDSE wavefunction outputs, and other
intermediate simulation outputs are not included in this repository because of
file-size limitations and laboratory data-management considerations.

Requirements

Python 3.10 or later is recommended.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Usage

Clone the repository:

```bash
git clone https://github.com/lancerdj/hhg-dissertation-code.git
cd hhg-dissertation-code
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the desired simulation script from the repository root. For example:

```bash
python "fitting and yield random phase screen.py"
python "fitting and yield HG.py"
python "yield random phase screen TDSE.py"
```

To rebuild or inspect the TDSE lookup-table workflow:

```bash
python tdse_lut_8cycle/code/build_tdse_lut.py
```

The scripts generate representative mask-dependent HHG yield results, including
near-field yield, far-field collection, angular lineouts, and normalized
enhancement factors.

Notes

The code is intended to document the numerical methods used in the dissertation
and to reproduce representative calculations. It is not intended as a fully
general HHG simulation package.

Some file paths or input filenames may need to be adjusted depending on the
local directory structure.

Citation

If using or referring to this repository, please cite:

Jian Dong, Ph.D. Dissertation, Texas A&M University, 2026.

Contact

Jian Dong  
Texas A&M University  
jiandong@tamu.edu
