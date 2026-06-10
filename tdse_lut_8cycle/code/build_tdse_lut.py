"""
Build TDSE-based HHG Dipole LUT
================================
Reads all Argon TDSE npz files, extracts complex d_q(I) for each harmonic
via 4th-order Simpson FFT, and saves a LUT cache file compatible with
the existing fitting/yield pipeline.

Default output is tagged by the selected pulse, Gabor window, extraction
method, and intensity-use range, plus _lut_cache_tdse_argon.npz as a
compatibility copy for existing yield scripts.
"""

import numpy as np
from scipy import signal
import os
import argparse

# =====================================================================
# 4th-order Simpson FFT (from extract_hhg_data.py)
# =====================================================================
def compute_hhg_spectrum_simpson_fft(time_grid, signal_arr, omega_laser_val,
                                     window="flattop", remove_dc=True):
    """
    4th-order full-spectrum Fourier transform using composite Simpson rule.

    Returns
    -------
    harm_order : 1-D array, frequency axis in units of omega_laser
    F          : 1-D complex array, Fourier coefficients
    """
    t = np.asarray(time_grid, dtype=np.float64)
    a = np.asarray(signal_arr, dtype=np.float64)

    N = len(t) - 1
    if N < 2:
        raise ValueError("Need at least 3 points")
    if N % 2 != 0:
        N -= 1
        t = t[:N + 1]
        a = a[:N + 1]

    dt_val = t[1] - t[0]
    T = N * dt_val
    t0 = t[0]

    if window is None:
        win = np.ones(N + 1, dtype=np.float64)
    else:
        win_func = getattr(signal.windows, window)
        win = win_func(N + 1, sym=True).astype(np.float64)

    f = a.copy()

    if remove_dc:
        c_dc = np.ones(N + 1, dtype=np.float64)
        c_dc[1:N:2] = 4.0
        c_dc[2:N:2] = 2.0
        mean_simpson = np.dot(c_dc, f) * (dt_val / 3.0) / T
        f = f - mean_simpson

    f = f * win

    y = np.zeros(N, dtype=np.float64)
    y[0] = f[0]
    y[1:N:2] = 4.0 * f[1:N:2]
    y[2:N:2] = 2.0 * f[2:N:2]

    Y = np.fft.rfft(y, n=N)

    omega_arr = 2.0 * np.pi * np.fft.rfftfreq(N, d=dt_val)

    phase0 = np.exp(-1j * omega_arr * t0)
    F = phase0 * (dt_val / 3.0) * (Y + f[N])

    harm_order = omega_arr / omega_laser_val
    return harm_order, F


# =====================================================================
# Gabor window extraction (isolate short-trajectory response)
# =====================================================================
def gabor_extract_at_peak(time_arr, E_field, signal_arr,
                          omega, n_win_cycles=2.5, center_method="field"):
    """
    Slice (time_arr, signal_arr) to a Gabor window centred on the pulse peak.
    Returns arrays ready to feed into the Simpson FFT.

    center_method="envelope" uses the Hilbert-envelope maximum. center_method
    ="field" first finds the envelope maximum, then snaps to the strongest
    |E(t)| peak within +/- one optical cycle. The field-centered option is
    usually more stable for few-cycle TDSE files.
    """
    t = np.asarray(time_arr, dtype=np.float64)
    E = np.asarray(E_field,  dtype=np.float64)
    a = np.asarray(signal_arr, dtype=np.float64)

    T_cycle = 2.0 * np.pi / omega
    env = np.abs(signal.hilbert(E))
    i_env_peak = int(np.argmax(env))
    if center_method == "field":
        local = np.abs(t - t[i_env_peak]) <= T_cycle
        if np.any(local):
            local_idx = np.flatnonzero(local)
            i_peak = int(local_idx[np.argmax(np.abs(E[local]))])
        else:
            i_peak = i_env_peak
    elif center_method == "envelope":
        i_peak = i_env_peak
    else:
        raise ValueError(f"Unknown Gabor center method: {center_method}")
    t_peak = t[i_peak]

    half = n_win_cycles * T_cycle
    mask = (t >= t_peak - half) & (t <= t_peak + half)
    if mask.sum() < 8:
        raise ValueError(f"Gabor window too narrow ({mask.sum()} points)")
    return t[mask], a[mask], t_peak


def extract_harmonic_response(harm_order, F_acc, q, method="fixed", fixed_sigma=0.20):
    """Extract complex harmonic response at q.

    method="fixed" uses fixed Gaussian frequency weights centered at q, avoiding
    sub-peak hopping between adjacent TDSE intensity points.
    method="peak" reproduces the old local-peak +/-2-bin weighted extraction.
    """
    mask_q = (harm_order >= q - 0.5) & (harm_order <= q + 0.5)
    if not np.any(mask_q):
        idx_q = int(np.argmin(np.abs(harm_order - q)))
        return F_acc[idx_q]

    h_local = harm_order[mask_q]
    F_local = F_acc[mask_q]

    if method == "fixed":
        sigma = max(float(fixed_sigma), 1e-6)
        w = np.exp(-0.5 * ((h_local - q) / sigma)**2)
        w_sum = w.sum()
        return np.sum(w * F_local) / w_sum if w_sum > 0 else F_local[np.argmin(np.abs(h_local - q))]

    if method == "peak":
        mag_local = np.abs(F_local)
        idx_peak = int(np.argmax(mag_local))
        n_local = len(F_local)
        lo = max(0, idx_peak - 2)
        hi = min(n_local, idx_peak + 3)
        F_near = F_local[lo:hi]
        w = np.abs(F_near)**2
        w_sum = w.sum()
        return np.sum(w * F_near) / w_sum if w_sum > 0 else F_local[idx_peak]

    raise ValueError(f"Unknown harmonic extraction method: {method}")


# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TDSE_ROOT = os.path.join(SCRIPT_DIR,
                         "..", "blocked beam", "TDSE", "Argon",
                         "HHG", "SIN2", "0p057")
COMPAT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "_lut_cache_tdse_argon.npz")

HARMONICS = [11, 13, 15, 17, 19, 21]
I_TO_E0 = 3.5094e16  # I(W/cm^2) = E0^2 * 3.5094e16
DEFAULT_CYC_SELECT = 8  # use --cyc 58 explicitly for 55 fs data

# Gabor extraction: window the dipole around the pulse envelope peak
USE_GABOR = True
N_WIN_CYCLES = 2.0   # window half-width in optical cycles (pulse-peak ± N_WIN_CYCLES*T)
GABOR_WINDOW = "hann"  # FFT window inside the Gabor slice
DEFAULT_GABOR_CENTER = "field"
DEFAULT_EXTRACT_METHOD = "fixed"
DEFAULT_FIXED_SIGMA = 0.20
DEFAULT_I_USE_MIN = 5.0e13
DEFAULT_I_USE_MAX = 7.0e14


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TDSE-based HHG dipole LUT from Cyc8 or Cyc58 TDSE scans."
    )
    parser.add_argument(
        "--cyc",
        type=int,
        default=DEFAULT_CYC_SELECT,
        help="TDSE file cycle tag to use, e.g. 8 for 8 fs data or 58 for 55 fs data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .npz path. Default includes Cyc, window, center, and extraction tags.",
    )
    parser.add_argument(
        "--no-compat-copy",
        action="store_true",
        help="Do not also write _lut_cache_tdse_argon.npz for existing yield scripts.",
    )
    parser.add_argument(
        "--win-cycles",
        type=float,
        default=N_WIN_CYCLES,
        help="Gabor half-window width in optical cycles.",
    )
    parser.add_argument(
        "--center",
        choices=("field", "envelope"),
        default=DEFAULT_GABOR_CENTER,
        help="Center Gabor slice on strongest local field peak or envelope peak.",
    )
    parser.add_argument(
        "--extract",
        choices=("fixed", "peak"),
        default=DEFAULT_EXTRACT_METHOD,
        help="Harmonic extraction method: fixed Gaussian weights or old local peak picking.",
    )
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=DEFAULT_FIXED_SIGMA,
        help="Gaussian width in harmonic-order units for --extract fixed.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Savitzky-Golay smoothing window. It is forced odd and clipped to the LUT length.",
    )
    parser.add_argument(
        "--i-use-min",
        type=float,
        default=DEFAULT_I_USE_MIN,
        help="Optional downstream lower intensity bound in W/cm^2. Below this, yield scripts zero the response. Default: 5e13.",
    )
    parser.add_argument(
        "--i-use-max",
        type=float,
        default=DEFAULT_I_USE_MAX,
        help="Optional downstream upper intensity bound in W/cm^2. Above this, yield scripts clip the response. Default: 7e14.",
    )
    return parser.parse_args()


args = parse_args()
CYC_SELECT = args.cyc
N_WIN_CYCLES = args.win_cycles
GABOR_CENTER = args.center
EXTRACT_METHOD = args.extract
FIXED_SIGMA = args.fixed_sigma
SMOOTH_WINDOW_ARG = args.smooth_window
I_USE_MIN_ARG = args.i_use_min
I_USE_MAX_ARG = args.i_use_max
range_tag = ""
if I_USE_MIN_ARG is not None or I_USE_MAX_ARG is not None:
    min_tag = "auto" if I_USE_MIN_ARG is None else f"{I_USE_MIN_ARG/1e14:g}"
    max_tag = "auto" if I_USE_MAX_ARG is None else f"{I_USE_MAX_ARG/1e14:g}"
    range_tag = f"_use{min_tag}-{max_tag}e14"
run_tag = (
    f"Cyc{CYC_SELECT}_win{N_WIN_CYCLES:g}cyc_"
    f"{GABOR_CENTER}_{EXTRACT_METHOD}{range_tag}"
)
OUTPUT_FILE = (
    os.path.abspath(args.output)
    if args.output is not None
    else os.path.join(SCRIPT_DIR, f"_lut_cache_tdse_argon_{run_tag}.npz")
)
WRITE_COMPAT_COPY = not args.no_compat_copy
PLOT_TAG = run_tag
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 15
SUBPLOT_TITLE_SIZE = 18
SUPTITLE_SIZE = 22

print("=" * 60)
print("BUILD TDSE-BASED HHG DIPOLE LUT")
print(f"Selected data tag: Cyc{CYC_SELECT}")
print(f"Gabor: center={GABOR_CENTER}, half-window={N_WIN_CYCLES:g} cycles, FFT window={GABOR_WINDOW}")
print(f"Harmonic extraction: {EXTRACT_METHOD}" +
      (f" (sigma={FIXED_SIGMA:g})" if EXTRACT_METHOD == "fixed" else ""))
print(f"Tagged output: {OUTPUT_FILE}")
if WRITE_COMPAT_COPY:
    print(f"Compatibility output: {COMPAT_OUTPUT_FILE}")
print("=" * 60)

# =====================================================================
# Step 1: Find all npz files
# =====================================================================
cyc_tag = f"Cyc{CYC_SELECT}_"
npz_files = []
for root, dirs, files in os.walk(TDSE_ROOT):
    for f in files:
        if f.endswith(".npz") and "velocity" in f and cyc_tag in f:
            npz_files.append(os.path.join(root, f))
npz_files.sort()

print(f"\nFound {len(npz_files)} TDSE velocity-gauge files (Cyc{CYC_SELECT} only)")
if not npz_files:
    raise RuntimeError(
        f"No TDSE velocity .npz files found for Cyc{CYC_SELECT} under:\n{TDSE_ROOT}"
    )

# =====================================================================
# Step 2: Extract d_q(I) for each file
# =====================================================================
results = []  # list of (I_Wcm2, E0, {q: complex_dq})

for i, fpath in enumerate(npz_files):
    fname = os.path.basename(fpath)
    d = np.load(fpath, allow_pickle=True)

    E0 = float(d["E0"])
    omega = float(d["omega"])
    I_Wcm2 = E0**2 * I_TO_E0

    time_arr = d["time_arr"]
    E_field = d["E_field"]
    dipole_acc = d["dipole_acc"]

    # Align lengths: dipole_acc has one extra point (initial condition)
    Nt = len(time_arr)
    if len(dipole_acc) == Nt + 1:
        dipole_acc = dipole_acc[1:]
    elif len(dipole_acc) != Nt:
        print(f"  WARNING: length mismatch for {fname}, skipping")
        continue

    # Optionally isolate short-trajectory response with a Gabor window
    # centred on the pulse envelope peak.
    if USE_GABOR:
        t_fft, a_fft, t_peak = gabor_extract_at_peak(
            time_arr, E_field, dipole_acc, omega,
            n_win_cycles=N_WIN_CYCLES,
            center_method=GABOR_CENTER
        )
        fft_window = GABOR_WINDOW
    else:
        t_fft, a_fft = time_arr, dipole_acc
        fft_window = "flattop"

    # Simpson FFT (acceleration form)
    harm_order, F_acc = compute_hhg_spectrum_simpson_fft(
        t_fft, a_fft, omega, window=fft_window
    )

    # Extract complex d_q at each harmonic — local peak + weighted average
    dq_dict = {}
    for q in HARMONICS:
        dq_dict[q] = extract_harmonic_response(
            harm_order, F_acc, q,
            method=EXTRACT_METHOD,
            fixed_sigma=FIXED_SIGMA
        )
    """
    for q in HARMONICS:
        # Find bins within q ± 0.5
        mask_q = (harm_order >= q - 0.5) & (harm_order <= q + 0.5)
        if not np.any(mask_q):
            # Fallback to nearest bin
            idx_q = np.argmin(np.abs(harm_order - q))
            dq_dict[q] = F_acc[idx_q]
            continue

        F_local = F_acc[mask_q]
        mag_local = np.abs(F_local)

        # Find local peak
        idx_peak = np.argmax(mag_local)

        # Weighted average of peak ± 2 bins (or available range)
        n_local = len(F_local)
        lo = max(0, idx_peak - 2)
        hi = min(n_local, idx_peak + 3)
        F_near = F_local[lo:hi]
        w = np.abs(F_near)**2  # |F|^2 weighting
        w_sum = w.sum()
        if w_sum > 0:
            dq_dict[q] = np.sum(w * F_near) / w_sum
        else:
            dq_dict[q] = F_local[idx_peak]
    """

    results.append((I_Wcm2, E0, dq_dict))

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(npz_files)}] E0={E0:.4f}, I={I_Wcm2:.2e} W/cm^2")

print(f"\nProcessed {len(results)} intensity points")
if not results:
    raise RuntimeError(
        f"No valid TDSE intensity points were processed for Cyc{CYC_SELECT}."
    )

# =====================================================================
# Step 3: Sort by intensity and build arrays
# =====================================================================
results.sort(key=lambda x: x[0])

I_lut = np.array([r[0] for r in results])
E0_lut = np.array([r[1] for r in results])

print(f"\nIntensity range: {I_lut[0]:.2e} - {I_lut[-1]:.2e} W/cm^2")
print(f"E0 range: {E0_lut[0]:.4f} - {E0_lut[-1]:.4f} a.u.")

I_lut_actual_min = float(I_lut[0])
I_lut_actual_max = float(I_lut[-1])
I_lut_use_min = float(I_USE_MIN_ARG) if I_USE_MIN_ARG is not None else I_lut_actual_min
I_lut_use_max = float(I_USE_MAX_ARG) if I_USE_MAX_ARG is not None else I_lut_actual_max
if I_lut_use_min < I_lut_actual_min or I_lut_use_max > I_lut_actual_max:
    raise ValueError(
        f"Requested use range {I_lut_use_min:.2e}-{I_lut_use_max:.2e} W/cm^2 "
        f"exceeds LUT data range {I_lut_actual_min:.2e}-{I_lut_actual_max:.2e} W/cm^2"
    )
if I_lut_use_min >= I_lut_use_max:
    raise ValueError("Intensity use range must satisfy --i-use-min < --i-use-max")
print(f"Downstream use range: {I_lut_use_min:.2e} - {I_lut_use_max:.2e} W/cm^2")
use_mask = (I_lut >= I_lut_use_min) & (I_lut <= I_lut_use_max)
use_indices = np.flatnonzero(use_mask)
if use_indices.size < 3:
    raise ValueError(
        f"Only {use_indices.size} LUT points fall inside the requested use range; "
        "choose a wider --i-use-min/--i-use-max range."
    )
print(f"Smoothing/peak/plots use {use_indices.size}/{len(I_lut)} points inside downstream range")

# Build per-harmonic magnitude and phase arrays
# Strategy: smooth log|d_q| and unwrap(phase) directly along E0, each in its
# own natural representation.  Re/Im smoothing averaged the multi-burst
# interference but destroyed the magnitude trend when |d_q| varied strongly.
from scipy.signal import savgol_filter

SMOOTH_WINDOW = SMOOTH_WINDOW_ARG  # Savitzky-Golay window (odd); ~20-25% of the 90-point E0 grid
SMOOTH_ORDER = 3    # polynomial order
MAG_VALID_FRAC = 0.01  # points with |dq| < this fraction of peak are "low signal"

multi_lut = {}
for q in HARMONICS:
    dq_complex = np.array([r[2][q] for r in results])

    mag_raw = np.abs(dq_complex)
    phase_raw = np.unwrap(np.angle(dq_complex))
    dq_complex_use = dq_complex[use_mask]
    I_use = I_lut[use_mask]
    mag_raw_use = mag_raw[use_mask]
    # Keep the same 2*pi branch as the full raw phase. Re-unwrapping only the
    # use range can shift the smoothed curve by an arbitrary multiple of 2*pi.
    phase_raw_use = phase_raw[use_mask]
    mag_peak = mag_raw_use.max()
    valid_use = mag_raw_use >= MAG_VALID_FRAC * mag_peak

    n_pts = len(dq_complex)
    n_use = len(dq_complex_use)
    win_requested = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW - 1
    win = min(win_requested, n_use if n_use % 2 == 1 else n_use - 1)
    if win >= SMOOTH_ORDER + 2:
        # Smooth log|d_q| to track magnitude trend across orders of magnitude
        log_mag_raw = np.log(np.maximum(mag_raw_use, mag_peak * 1e-6))
        log_mag_smooth = savgol_filter(log_mag_raw, win, SMOOTH_ORDER)
        mag_use = np.exp(log_mag_smooth)
        # Replace low-signal phase samples before smoothing; their phase is
        # usually noise and can bend the Savitzky-Golay fit.
        if np.count_nonzero(valid_use) >= SMOOTH_ORDER + 2:
            phase_for_smooth = phase_raw_use.copy()
            phase_for_smooth[~valid_use] = np.interp(
                I_use[~valid_use],
                I_use[valid_use],
                phase_raw_use[valid_use],
            )
        else:
            phase_for_smooth = phase_raw_use
        phase_use = savgol_filter(phase_for_smooth, win, SMOOTH_ORDER)
    else:
        mag_use = mag_raw_use
        phase_use = phase_raw_use

    # Save arrays on the full TDSE grid for compatibility, but make the
    # out-of-use region a constant edge extension so it cannot bend interpolation
    # or diagnostic peak detection.
    mag = np.interp(I_lut, I_use, mag_use, left=mag_use[0], right=mag_use[-1])
    phase = np.interp(I_lut, I_use, phase_use, left=phase_use[0], right=phase_use[-1])

    n_invalid = np.sum(~valid_use)

    multi_lut[q] = {
        'mag': mag,
        'phase': phase,
        'mag_raw': mag_raw,
        'phase_raw': phase_raw,
    }

    peak_local_idx = int(np.argmax(mag_use))
    peak_idx = int(use_indices[peak_local_idx])
    print(f"  H{q}: peak |d_q| at I={I_lut[peak_idx]:.2e} W/cm^2 "
          f"(E0={E0_lut[peak_idx]:.4f}), |d_q|={mag[peak_idx]:.4e}, "
          f"low-signal pts: {n_invalid}/{n_use} used ({n_pts} total)")

# =====================================================================
# Step 4: Save in compatible format
# =====================================================================
# Use H21 as default (backward compatibility with single-harmonic code)
q_ref = 21

# Strip diagnostic arrays before saving (keep only mag + phase)
multi_lut_save = {}
for q in HARMONICS:
    multi_lut_save[q] = {
        'mag': multi_lut[q]['mag'],
        'phase': multi_lut[q]['phase'],
    }

save_dict = {
    'I_lut': I_lut,
    'I_lut_min': I_lut_use_min,
    'I_lut_max': I_lut_use_max,
    'I_lut_actual_min': I_lut_actual_min,
    'I_lut_actual_max': I_lut_actual_max,
    'dq_mag': multi_lut[q_ref]['mag'],
    'dq_phase': multi_lut[q_ref]['phase'],
    'multi_lut': multi_lut_save,
    'sfa_omega': 0.057,
    'sfa_Ip_au': 0.579,
    'sfa_I_to_E0': I_TO_E0,
    'source': 'TDSE',
    'cyc_select': CYC_SELECT,
    'pulse_tag': f'Cyc{CYC_SELECT}',
    'gabor_center': GABOR_CENTER,
    'gabor_win_cycles': N_WIN_CYCLES,
    'gabor_fft_window': GABOR_WINDOW,
    'extract_method': EXTRACT_METHOD,
    'fixed_sigma_harmonic': FIXED_SIGMA,
    'smooth_window': SMOOTH_WINDOW,
    'smooth_order': SMOOTH_ORDER,
    'mag_valid_frac': MAG_VALID_FRAC,
    'I_lut_use_min': I_lut_use_min,
    'I_lut_use_max': I_lut_use_max,
    'n_points': len(I_lut),
    'n_points_use': int(use_indices.size),
    'harmonics': np.array(HARMONICS),
    'E0_lut': E0_lut,
}

np.savez_compressed(OUTPUT_FILE, **save_dict)
file_size = os.path.getsize(OUTPUT_FILE) / 1024
print(f"\nSaved: {OUTPUT_FILE} ({file_size:.1f} KB)")

if WRITE_COMPAT_COPY and os.path.abspath(OUTPUT_FILE) != os.path.abspath(COMPAT_OUTPUT_FILE):
    np.savez_compressed(COMPAT_OUTPUT_FILE, **save_dict)
    compat_size = os.path.getsize(COMPAT_OUTPUT_FILE) / 1024
    print(f"Saved compatibility LUT: {COMPAT_OUTPUT_FILE} ({compat_size:.1f} KB)")

# =====================================================================
# Step 5: Diagnostic plots
# =====================================================================
try:
    import matplotlib.pyplot as plt

    # Magnitude: raw vs smoothed
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x_use_min = I_lut_use_min / 1e14
    x_use_max = I_lut_use_max / 1e14
    for iq, q in enumerate(HARMONICS):
        ax = axes[0, iq] if iq < 3 else axes[1, iq - 3]
        mag_raw = multi_lut[q]['mag_raw']
        mag_smooth = multi_lut[q]['mag']
        ax.semilogy(I_lut[use_mask] / 1e14, mag_raw[use_mask], 'b.',
                    markersize=4, alpha=0.50, label='raw')
        ax.semilogy(I_lut[use_mask] / 1e14, mag_smooth[use_mask], 'r-',
                    linewidth=2.2, label='smoothed')
        peak_idx = int(use_indices[np.argmax(mag_smooth[use_mask])])
        ax.axvline(I_lut[peak_idx] / 1e14, color='orange', ls=':', alpha=0.5,
                   label=f'peak at {I_lut[peak_idx]/1e14:.2f}')
        ax.set_xlim(x_use_min, x_use_max)
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(r'$|d_q|$', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f'H{q} magnitude (TDSE)', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'TDSE Dipole Response |d_q(I)| - Argon, 800nm (use-range smoothing)',
        fontsize=SUPTITLE_SIZE,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(OUTPUT_FILE),
                             f'tdse_lut_magnitude_{PLOT_TAG}.png'), dpi=200)
    print(f"Saved: tdse_lut_magnitude_{PLOT_TAG}.png")

    # Phase: raw vs smoothed
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for iq, q in enumerate(HARMONICS):
        ax = axes2[0, iq] if iq < 3 else axes2[1, iq - 3]
        phase_raw = multi_lut[q]['phase_raw']
        phase_smooth = multi_lut[q]['phase']
        ax.plot(I_lut[use_mask] / 1e14, phase_raw[use_mask], 'b.',
                markersize=4, alpha=0.50, label='raw')
        ax.plot(I_lut[use_mask] / 1e14, phase_smooth[use_mask], 'r-',
                linewidth=2.2, label='smoothed')
        ax.set_xlim(x_use_min, x_use_max)
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Phase (rad)', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f'H{q} phase (TDSE)', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(
        'TDSE Dipole Phase - Argon, 800nm (use-range smoothing)',
        fontsize=SUPTITLE_SIZE,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(OUTPUT_FILE),
                             f'tdse_lut_phase_{PLOT_TAG}.png'), dpi=200)
    print(f"Saved: tdse_lut_phase_{PLOT_TAG}.png")

    plt.show()

except ImportError:
    print("matplotlib not available, skipping plots")

print("\nDone.")
