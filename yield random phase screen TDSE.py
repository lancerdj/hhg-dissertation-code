"""
HHG Yield: Random Phase Screen (RPS) Beam Model — TDSE Dipole LUT
====================================================================
Beam propagation + HHG yield + multi-mask comparison.
Uses TDSE-computed d_q(I) lookup table (magnitude + phase) instead of
empirical alpha/Is parametrization or SFA Lewenstein model.

All units normalized to mm for beam propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import time
import gc  # For memory management
from math import factorial

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


def configure_paper_matplotlib():
    """Use consistent dissertation-style typography and axes for generated figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.75',
        'axes.linewidth': 1.25,
        'lines.linewidth': 1.9,
        'grid.color': '0.5',
        'grid.alpha': 0.18,
        'grid.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5.5,
        'ytick.major.size': 5.5,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'xtick.minor.width': 0.85,
        'ytick.minor.width': 0.85,
        'figure.dpi': 160,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def style_paper_axis(ax, grid=False):
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_size(), 13))
    ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_size(), 13))
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.title.set_fontsize(max(ax.title.get_size(), 13))
    ax.title.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    if grid:
        ax.grid(True, color='0.5', alpha=0.18, linewidth=0.8)


def finalize_paper_figure(fig):
    for ax in fig.axes:
        style_paper_axis(ax)


def save_tdse_mask_comparison_npz(mask_results, output_file):
    """Save TDSE mask-comparison results in the plot_from_npz format."""
    mask_names = np.array(['none', 'circular', 'twosided', 'diagonal'])
    mask_labels = np.array(['No mask', 'Circular', 'Two-side', 'Diagonal'])
    q_arr = np.array(multi_q_list, dtype=int)
    center = N_hhg_2d // 2
    x_um = x_hhg_2d * 1e3
    extent_um = np.array([x_um[0], x_um[-1], x_um[0], x_um[-1]], dtype=float)
    z_ref = np.asarray(mask_results['none']['z_gas_mm'], dtype=float)

    n_q = len(q_arr)
    n_m = len(mask_names)
    n_x = len(x_um)
    n_z = len(z_ref)

    yield_slit = np.zeros((n_q, n_m), dtype=float)
    yield_circ = np.zeros((n_q, n_m), dtype=float)
    yield_nf = np.zeros((n_q, n_m), dtype=float)
    nearfield_intensity_norm = np.zeros((n_q, n_m, n_x, n_x), dtype=float)
    nearfield_peak_rel = np.zeros((n_q, n_m), dtype=float)
    nearfield_lineout_x_norm = np.zeros((n_q, n_m, n_x), dtype=float)
    nearfield_lineout_y_norm = np.zeros((n_q, n_m, n_x), dtype=float)
    ff_theta_mrad = np.zeros((n_q, n_x), dtype=float)
    ff_lineout_x_norm = np.zeros((n_q, n_m, n_x), dtype=float)
    z_gas_mm = np.zeros((n_m, n_z), dtype=float)
    gouy_grad = np.zeros((n_m, n_z), dtype=float)
    yield_vs_z = np.full((n_q, n_m, n_z), np.nan, dtype=float)
    yield_vs_z_slit = np.full((n_q, n_m, n_z), np.nan, dtype=float)
    yield_vs_z_circ = np.full((n_q, n_m, n_z), np.nan, dtype=float)

    def interp_to_ref(z_src, values):
        if values is None:
            return np.full(n_z, np.nan, dtype=float)
        z_src = np.asarray(z_src, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(z_src) == n_z and np.allclose(z_src, z_ref):
            return values.copy()
        return np.interp(z_ref, z_src, values)

    for im, mn in enumerate(mask_names):
        z_src = np.asarray(mask_results[mn]['z_gas_mm'], dtype=float)
        z_gas_mm[im] = z_ref
        gouy_grad[im] = interp_to_ref(z_src, mask_results[mn]['gouy_grad'])

    for iq, q in enumerate(q_arr):
        nf_global = max(
            np.nanmax(np.abs(mask_results[mn]['per_q'][int(q)]['E_q'])**2)
            for mn in mask_names
        )
        nf_global = max(float(nf_global), 1e-30)
        ff_theta_mrad[iq] = np.asarray(mask_results['none']['per_q'][int(q)]['theta_axis'], dtype=float) * 1e3

        for im, mn in enumerate(mask_names):
            rq = mask_results[mn]['per_q'][int(q)]
            yield_slit[iq, im] = rq['yield_slit']
            yield_circ[iq, im] = rq['yield_circ']
            yield_nf[iq, im] = rq['yield_nf']

            I_nf = np.abs(rq['E_q'])**2
            nearfield_intensity_norm[iq, im] = I_nf / nf_global
            nearfield_peak_rel[iq, im] = np.nanmax(I_nf) / nf_global
            nearfield_lineout_x_norm[iq, im] = I_nf[center, :] / nf_global
            nearfield_lineout_y_norm[iq, im] = I_nf[:, center] / nf_global

            line = np.asarray(rq['I_ff'], dtype=float)[center, :]
            ff_lineout_x_norm[iq, im] = line / max(float(np.nanmax(line)), 1e-30)

            z_src = np.asarray(mask_results[mn]['z_gas_mm'], dtype=float)
            yield_vs_z[iq, im] = interp_to_ref(z_src, rq.get('yield_vs_z'))
            yield_vs_z_slit[iq, im] = interp_to_ref(z_src, rq.get('yield_vs_z_slit'))
            yield_vs_z_circ[iq, im] = interp_to_ref(z_src, rq.get('yield_vs_z_circ'))

    np.savez_compressed(
        output_file,
        harmonic_orders=q_arr,
        mask_names=mask_names,
        mask_labels=mask_labels,
        yield_slit=yield_slit,
        yield_circ=yield_circ,
        yield_nf=yield_nf,
        nearfield_intensity_norm=nearfield_intensity_norm,
        nearfield_peak_rel=nearfield_peak_rel,
        nearfield_lineout_x_um=x_um,
        nearfield_x_um=x_um,
        nearfield_lineout_x_norm=nearfield_lineout_x_norm,
        nearfield_lineout_y_norm=nearfield_lineout_y_norm,
        nearfield_extent_um=extent_um,
        ff_theta_mrad=ff_theta_mrad,
        ff_lineout_x_norm=ff_lineout_x_norm,
        z_gas_mm=z_gas_mm,
        gouy_grad=gouy_grad,
        yield_vs_z=yield_vs_z,
        yield_vs_z_slit=yield_vs_z_slit,
        yield_vs_z_circ=yield_vs_z_circ,
        slit_half_angle_mrad=np.array(slit_half_angle_x * 1e3),
        aperture_half_angle_mrad=np.array(aperture_half_angle * 1e3),
        hhg_gas_type=np.array(hhg_gas_type),
        hhg_gas_pressure=np.array(hhg_gas_pressure),
        M2x=np.array(M2x),
        M2y=np.array(M2y),
        lavg_tag=np.array('tdse_rps'),
        pressure_tag=np.array(f'P{hhg_gas_pressure:.0f}mbar'),
    )
    print(f"  Saved TDSE mask-comparison NPZ: {output_file}")


configure_paper_matplotlib()

# Try to use scipy.fft (faster) if available, fallback to numpy
try:
    from scipy import fft as scipy_fft
    from scipy.signal import czt as scipy_czt
    USE_SCIPY_FFT = True
except ImportError:
    USE_SCIPY_FFT = False
    scipy_czt = None

# Try to use Numba for JIT compilation (significant speedup for matrix operations)
try:
    from numba import njit, prange
    USE_NUMBA = True
    print("Numba available - using JIT-compiled functions for speedup")
except ImportError:
    USE_NUMBA = False
    print("Numba not available - using pure NumPy (install numba for 2-10x speedup)")


# =============================================================================
# TIMING UTILITIES
# =============================================================================
class Timer:
    """Simple timer class to track execution time of different parts of the code."""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.section_times = {}
        self.current_section = None
        self.section_start = None

    def start_section(self, name):
        """Start timing a new section."""
        if self.current_section is not None:
            self.end_section()
        self.current_section = name
        self.section_start = time.perf_counter()
        print(f"\n>>> Starting: {name}...")

    def end_section(self):
        """End timing the current section."""
        if self.current_section is not None and self.section_start is not None:
            elapsed = time.perf_counter() - self.section_start
            self.section_times[self.current_section] = elapsed
            print(f"    Completed in {elapsed:.2f} s")
            self.current_section = None
            self.section_start = None

    def total_elapsed(self):
        """Get total elapsed time since timer creation."""
        return time.perf_counter() - self.start_time

    def summary(self):
        """Print timing summary."""
        self.end_section()  # End any ongoing section
        total = self.total_elapsed()
        print(f"\n{'='*60}")
        print("EXECUTION TIME SUMMARY")
        print(f"{'='*60}")

        # Sort sections by time (descending)
        sorted_sections = sorted(self.section_times.items(), key=lambda x: x[1], reverse=True)

        for name, elapsed in sorted_sections:
            percentage = (elapsed / total) * 100
            bar_len = int(percentage / 2)
            bar = '█' * bar_len + '░' * (50 - bar_len)
            print(f"  {name:<35} {elapsed:>8.2f}s ({percentage:>5.1f}%) {bar[:20]}")

        print(f"{'─'*60}")
        print(f"  {'TOTAL':<35} {total:>8.2f}s (100.0%)")
        print(f"{'='*60}")
        return total

# Initialize global timer
TIMER = Timer()
print(f"Program started at {time.strftime('%H:%M:%S')}")

# Set number of threads for parallel processing
NUM_WORKERS = os.cpu_count() or 4

# Parameters (all in mm)
wavelength = 800e-6     # wavelength (mm), 800 nm
k = 2 * np.pi / wavelength  # wave number (1/mm)
laser_omega_au = 0.057  # 800 nm angular frequency in a.u.
laser_wavelength_nm = wavelength * 1e6
lut_wavelength_tag = f'lam{laser_wavelength_nm:.0f}nm'

# Beam quality factors (M² = 1 for ideal Gaussian)
M2x = 1.6  # M² in x direction
M2y = 1.6  # M² in y direction
_m2_tag = f'M2x{M2x:.2f}_M2y{M2y:.2f}'
PHASE_SCREEN_SEED = 42  # Random seed for reproducible phase screen

# Amplitude screen parameters (models intensity non-uniformity)
APPLY_AMPLITUDE_SCREEN = True      # True = apply amplitude modulation
AMP_SCREEN_SIGMA = 0.10            # σ_a: amplitude fluctuation strength (~20% RMS intensity variation)
AMP_SCREEN_LC_FACTOR = 0.5         # correlation length = factor × w0_ref
AMP_SCREEN_SEED = 137              # separate seed from phase screen for independence

# --- Control flags ---
PLOT_OPTICAL_DIAGNOSTICS = False

# MEASURED beam waist (D4σ method, corresponds to the real M²>1 beam)
w0x_measured = 6.0  # measured beam waist in x direction (mm)
w0y_measured = 6.0  # measured beam waist in y direction (mm)

# Embedded Gaussian waist: w0_fund = w0_measured / sqrt(M²)
# Phase screen will increase M² from 1 to target, restoring measured waist
w0x = w0x_measured / np.sqrt(M2x)  # fundamental (embedded Gaussian) waist
w0y = w0y_measured / np.sqrt(M2y)
print(f"Measured beam waist: {w0x_measured:.2f} x {w0y_measured:.2f} mm")
print(f"Embedded Gaussian waist (w0/sqrt(M²)): {w0x:.3f} x {w0y:.3f} mm")

# Rayleigh ranges (for analytical estimates and diagnostics)
zRx = np.pi * w0x**2 / (M2x * wavelength)
zRy = np.pi * w0y**2 / (M2y * wavelength)

# Lens parameters
focal_length = 400.0    # focal length (mm)
lens_position = 5000.0    # distance from beam waist to lens (mm)

# Propagation method: 'asm', 'fresnel', or 'auto'
# ASM: Angular Spectrum Method (exact, best for short distances)
# Fresnel: Paraxial approximation (stable for large distances)
# Auto: Selects based on Fresnel number
# Aperture parameters
aperture_distance_before_lens = 2000.0  # aperture is 2000mm before the lens
aperture_radius = 7.0  # aperture radius (mm)
aperture_position = lens_position - aperture_distance_before_lens  # distance from beam waist to aperture

# Mask configuration for beam blocking comparison
mask_type = 'circular'  # 'circular', 'twosided', 'diagonal', 'none'
# Two-side slit parameters (blocks left and right, keeps central strip)
twosided_halfwidth = 4.3   # mm, half-width of clear slit (auto-calibrated below)
# Diagonal mask parameters (two 30x30mm blocks at diagonal corners)
diag_x_offset = 3.8        # mm, inner edge distance from beam center (auto-calibrated below)
diag_block_size = 30.0     # mm, block rectangle size
diag_y_shift = 13.8        # mm, vertical shift magnitude for each block
# Soft-edge transition width for masks (mm) — models finite edge roughness
mask_edge_width = 0.5      # mm, typical for printed/cardboard masks (0 = hard edge)

# Auto-calibration of mask geometry to match experimental transmission
AUTO_CALIBRATE_MASK = False  # True = bisection search for mask size; False = use values above directly

# Experimental transmission targets (used only when AUTO_CALIBRATE_MASK = True)
circular_target_transmission = 0.93   # 93%
twosided_target_transmission = 0.85   # 85%
diagonal_target_transmission = 0.86   # 86%

# Effective beam radius for Fresnel number calculation (use measured waist)
# Estimated focus spot size (for plot range calculation)
# w_focus ≈ M² * λ * f / (π * w_input) where w_input is the MEASURED beam size at lens
# At lens position, beam radius ≈ w0_measured * sqrt(1 + (lens_position/zR_real)^2)
w_at_lens_est = w0x_measured * np.sqrt(1 + (lens_position / zRx)**2)
focus_spot_size = M2x * wavelength * focal_length / (np.pi * w_at_lens_est)
print(f"Estimated focus spot size: {focus_spot_size*1e3:.2f} um")

def get_plot_range(z_from_lens, base_range=None):
    """
    Determine appropriate plot range based on distance from lens.
    Near focus: small range to see the focal spot
    Far from focus: larger range to see the full beam

    Returns: (xlim_range, ylim_range) in mm
    """
    if base_range is None:
        base_range = focus_spot_size * 10  # 10x focus spot size as minimum

    # Distance from focus
    dist_from_focus = abs(z_from_lens - focal_length)

    # Near focus (within 10% of focal length): use small range
    if dist_from_focus < 0.1 * focal_length:
        # Scale with distance from exact focus
        scale = max(1, dist_from_focus / (0.01 * focal_length))
        return min(base_range * scale, 2.0)  # max 2 mm near focus
    else:
        # Far from focus: use larger range based on beam divergence
        # Beam expands roughly linearly with distance from focus
        expansion = 1 + dist_from_focus / focal_length
        return min(base_range * expansion * 5, 15.0)  # max 15 mm

# Spatial grid - reduced from 4096 to 2048 for memory efficiency
# Memory usage scales as N² - reducing from 4096 to 2048 saves 4x memory
N = 4096
L = 40.0  # grid size (mm)
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2  # Pre-compute R squared

# Pre-compute frequency grid for FFT
fx = np.fft.fftfreq(N, dx)
fy = np.fft.fftfreq(N, dx)
FX, FY = np.meshgrid(fx, fy)
freq_term = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
propagating_mask = freq_term >= 0
evanescent_mask = ~propagating_mask
sqrt_freq_term_prop = np.sqrt(np.where(propagating_mask, freq_term, 0))
sqrt_freq_term_evan = np.sqrt(np.where(evanescent_mask, -freq_term, 0))


# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS FOR SPEEDUP
# =============================================================================

if USE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _compute_quadratic_phase_numba(X, Y, k_over_2z, N):
        """Compute quadratic phase: exp(ik(x²+y²)/(2z))"""
        result = np.zeros((N, N), dtype=np.complex128)
        for i in prange(N):
            for j in range(N):
                r2 = X[i, j]**2 + Y[i, j]**2
                phase = k_over_2z * r2
                result[i, j] = np.cos(phase) + 1j * np.sin(phase)
        return result

    @njit(parallel=True, fastmath=True, cache=True)
    def _thin_lens_phase_numba(field_real, field_imag, R2, k_over_2f, N):
        """Apply thin lens phase: field * exp(-ik*R2/(2f))"""
        result_real = np.zeros((N, N), dtype=np.float64)
        result_imag = np.zeros((N, N), dtype=np.float64)
        for i in prange(N):
            for j in range(N):
                phase = -k_over_2f * R2[i, j]
                cos_p = np.cos(phase)
                sin_p = np.sin(phase)
                result_real[i, j] = field_real[i, j] * cos_p - field_imag[i, j] * sin_p
                result_imag[i, j] = field_real[i, j] * sin_p + field_imag[i, j] * cos_p
        return result_real + 1j * result_imag

    @njit(parallel=True, fastmath=True, cache=True)
    def _asm_transfer_function_numba(sqrt_prop, sqrt_evan, mask_prop, k_z, N):
        """Compute ASM transfer function H"""
        result = np.zeros((N, N), dtype=np.complex128)
        for i in prange(N):
            for j in range(N):
                if mask_prop[i, j]:
                    phase = k_z * sqrt_prop[i, j]
                    result[i, j] = np.cos(phase) + 1j * np.sin(phase)
                else:
                    result[i, j] = np.exp(-k_z * sqrt_evan[i, j])
        return result

    print("Numba JIT functions compiled successfully")

else:
    def _compute_quadratic_phase_numba(X, Y, k_over_2z, N):
        """NumPy fallback for quadratic phase"""
        return np.exp(1j * k_over_2z * (X**2 + Y**2))

    def _thin_lens_phase_numba(field_real, field_imag, R2, k_over_2f, N):
        """NumPy fallback for thin lens"""
        field = field_real + 1j * field_imag
        return field * np.exp(-1j * k_over_2f * R2)

    def _asm_transfer_function_numba(sqrt_prop, sqrt_evan, mask_prop, k_z, N):
        """NumPy fallback for ASM transfer function"""
        return np.where(mask_prop,
                       np.exp(1j * k_z * sqrt_prop),
                       np.exp(-k_z * sqrt_evan))


# =============================================================================
# CACHING UTILITIES FOR SPEEDUP
# =============================================================================
from collections import OrderedDict

# Global caches for CZT parameters and quadratic phases (with LRU eviction)
_CZT_CACHE_MAX = 5      # CZT params are tiny (4 scalars + 1 small array)
_QUAD_CACHE_MAX = 3      # Each quad phase ~256MB for N=4096

_czt_cache = OrderedDict()
_quad_phase_cache = OrderedDict()
_cache_stats = {'czt_hits': 0, 'czt_misses': 0, 'quad_hits': 0, 'quad_misses': 0}


def _get_cached_czt_params(L_in, L_out, N_in, N_out, scale):
    """
    Get or compute CZT parameters with caching.
    Returns (W, A, const, phase_k) for the Chirp-Z Transform.

    The matrix DFT kernel K[k,n] = exp(-2πi * x_out[k] * x_in[n] / scale)
    decomposes into CZT form: sum_n x[n] * A^{-n} * W^{nk} with phase corrections.
    """
    global _cache_stats
    cache_key = (L_in, L_out, N_in, N_out, round(scale, 10))

    if cache_key in _czt_cache:
        _cache_stats['czt_hits'] += 1
        _czt_cache.move_to_end(cache_key)
        return _czt_cache[cache_key]

    _cache_stats['czt_misses'] += 1

    while len(_czt_cache) >= _CZT_CACHE_MAX:
        _czt_cache.popitem(last=False)

    # Grid spacings (matching np.linspace convention)
    dx_in = L_in / (N_in - 1)
    dx_out = L_out / (N_out - 1)
    x_in_0 = -L_in / 2.0
    x_out_0 = -L_out / 2.0

    # CZT parameters: K[k,n] = const * phase_k[k] * A^{-n} * W^{nk}
    W = np.exp(-2j * np.pi * dx_out * dx_in / scale)
    A = np.exp(2j * np.pi * x_out_0 * dx_in / scale)
    const = np.exp(-2j * np.pi * x_out_0 * x_in_0 / scale)
    phase_k = np.exp(-2j * np.pi * np.arange(N_out) * dx_out * x_in_0 / scale)

    params = (W, A, const, phase_k)
    _czt_cache[cache_key] = params
    return params


def _get_cached_quad_phase(L, N, k_over_2z, cache_id):
    """
    Get or compute quadratic phase with caching and LRU eviction.
    Uses cache_id to distinguish input vs output grids.

    Parameters:
    -----------
    L : float - Grid size
    N : int - Grid points
    k_over_2z : float - k/(2z) factor
    cache_id : str - 'in' or 'out' to distinguish grids

    Returns:
    --------
    quad_phase : 2D array
    """
    global _cache_stats
    cache_key = (L, N, round(k_over_2z, 10), cache_id)

    if cache_key in _quad_phase_cache:
        _cache_stats['quad_hits'] += 1
        # Move to end for LRU
        _quad_phase_cache.move_to_end(cache_key)
        return _quad_phase_cache[cache_key]

    _cache_stats['quad_misses'] += 1

    # Evict oldest if cache is full
    while len(_quad_phase_cache) >= _QUAD_CACHE_MAX:
        _quad_phase_cache.popitem(last=False)

    # Compute quadratic phase
    coords = np.linspace(-L/2, L/2, N)
    X_grid, Y_grid = np.meshgrid(coords, coords)

    if USE_NUMBA:
        quad = _compute_quadratic_phase_numba(X_grid, Y_grid, k_over_2z, N)
    else:
        quad = np.exp(1j * k_over_2z * (X_grid**2 + Y_grid**2))

    _quad_phase_cache[cache_key] = quad
    return quad


def gaussian_beam_field_astigmatic(X, Y, z, w0x, w0y, wavelength, k, zRx, zRy, M2x, M2y):
    """
    Generate complex Gaussian beam field at position z with M² beam quality
    Handles astigmatic beams with different parameters in x and y directions.

    For a real beam with M² > 1:
    - Beam size evolves as: w(z) = w0 * sqrt(1 + (z/zR)²) where zR = π*w0²/(M²*λ)
    - The beam diverges faster than an ideal Gaussian
    - At focus, spot size is the same, but divergence is M² times larger

    Parameters:
    -----------
    X, Y : 2D arrays - spatial coordinates
    z : float - propagation distance from waist
    w0x, w0y : float - beam waist in x and y directions
    wavelength : float - wavelength
    k : float - wave number
    zRx, zRy : float - Rayleigh ranges in x and y (already include M² effect)
    M2x, M2y : float - beam quality factors
    """
    # Beam sizes at z
    wx_z = w0x * np.sqrt(1 + (z / zRx)**2)
    wy_z = w0y * np.sqrt(1 + (z / zRy)**2)

    # Radii of curvature (handle z=0 case)
    if z == 0:
        curvature_phase_x = 0
        curvature_phase_y = 0
    else:
        Rx_z = z * (1 + (zRx / z)**2)
        Ry_z = z * (1 + (zRy / z)**2)
        curvature_phase_x = -k * X**2 / (2 * Rx_z)
        curvature_phase_y = -k * Y**2 / (2 * Ry_z)

    # Gouy phases (separate for x and y in astigmatic case)
    gouy_phase_x = 0.5 * np.arctan(z / zRx)
    gouy_phase_y = 0.5 * np.arctan(z / zRy)
    gouy_phase = gouy_phase_x + gouy_phase_y

    # Propagation phase
    prop_phase = -k * z

    # Amplitude (separable in x and y)
    amplitude = np.sqrt(w0x * w0y / (wx_z * wy_z)) * np.exp(-X**2 / wx_z**2 - Y**2 / wy_z**2)

    # Total phase
    phase = prop_phase + curvature_phase_x + curvature_phase_y + gouy_phase

    E = amplitude * np.exp(1j * phase)

    return E


def compute_beam_m2(field, x_1d, dx, verbose=False):
    """Compute M²_x and M²_y using ISO 11146 second-moment method."""
    intensity = np.abs(field)**2
    Ix = intensity.sum(axis=0)
    Iy = intensity.sum(axis=1)
    total_x = Ix.sum()
    x_bar = (x_1d * Ix).sum() / total_x
    var_x = ((x_1d - x_bar)**2 * Ix).sum() / total_x
    total_y = Iy.sum()
    y_bar = (x_1d * Iy).sum() / total_y
    var_y = ((x_1d - y_bar)**2 * Iy).sum() / total_y
    field_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    int_ft = np.abs(field_ft)**2
    Ifx = int_ft.sum(axis=0)
    Ify = int_ft.sum(axis=1)
    del field_ft, int_ft
    fx = np.fft.fftshift(np.fft.fftfreq(len(x_1d), dx))
    total_fx = Ifx.sum()
    fx_bar = (fx * Ifx).sum() / total_fx
    var_fx = ((fx - fx_bar)**2 * Ifx).sum() / total_fx
    total_fy = Ify.sum()
    fy_bar = (fx * Ify).sum() / total_fy
    var_fy = ((fx - fy_bar)**2 * Ify).sum() / total_fy
    M2x_val = 4 * np.pi * np.sqrt(var_x * var_fx)
    M2y_val = 4 * np.pi * np.sqrt(var_y * var_fy)
    if verbose:
        print(f"    Spatial:   sigma_x={np.sqrt(var_x):.4f} mm, sigma_y={np.sqrt(var_y):.4f} mm")
        print(f"    Frequency: sigma_fx={np.sqrt(var_fx):.4f} mm^-1, sigma_fy={np.sqrt(var_fy):.4f} mm^-1")
        print(f"    M²x={M2x_val:.4f}, M²y={M2y_val:.4f}")
    return M2x_val, M2y_val


def apply_m2_phase_screen(field, x_1d, dx, M2x_target, M2y_target,
                          w0_ref, seed=42, n_iter=25, tol=0.01):
    """Apply random phase screen to achieve target M² (preserves near-field intensity)."""
    M2_target = 0.5 * (M2x_target + M2y_target)
    if M2_target <= 1.02:
        print(f"  M² target ({M2_target:.2f}) <= 1.02, no phase screen applied.")
        return field
    N = len(x_1d)
    rng = np.random.default_rng(seed)
    l_c = w0_ref * 0.4
    anisotropic = abs(M2x_target - M2y_target) / M2_target > 0.05
    power_before = np.sum(np.abs(field)**2)
    m2x0, m2y0 = compute_beam_m2(field, x_1d, dx, verbose=True)
    print(f"  Initial beam M²: x={m2x0:.3f}, y={m2y0:.3f}")
    print(f"  Correlation length: {l_c:.3f} mm ({l_c/w0_ref*100:.0f}% of w0)")
    if anisotropic:
        print(f"  Anisotropic mode: M²x_target={M2x_target:.2f}, M²y_target={M2y_target:.2f}")
        noise_x_1d = rng.standard_normal(N)
        fx_ps = np.fft.fftfreq(N, dx)
        filt_1d = np.exp(-2 * np.pi**2 * l_c**2 * fx_ps**2)
        phase_x_1d = np.real(np.fft.ifft(np.fft.fft(noise_x_1d) * filt_1d))
        phase_x_1d /= phase_x_1d.std()
        phase_x_2d = np.ones((N, 1)) * phase_x_1d[np.newaxis, :]
        noise_y_1d = rng.standard_normal(N)
        phase_y_1d = np.real(np.fft.ifft(np.fft.fft(noise_y_1d) * filt_1d))
        phase_y_1d /= phase_y_1d.std()
        phase_y_2d = phase_y_1d[:, np.newaxis] * np.ones((1, N))
        del noise_x_1d, noise_y_1d, filt_1d
        amp_x, amp_y = 0.0, 0.0
        for alt_round in range(3):
            lo, hi = 0.0, 100.0
            for i in range(n_iter):
                amp_x = 0.5 * (lo + hi)
                trial = field * np.exp(1j * (amp_x * phase_x_2d + amp_y * phase_y_2d))
                m2x, m2y = compute_beam_m2(trial, x_1d, dx)
                if abs(m2x - M2x_target) / M2x_target < tol:
                    break
                if m2x < M2x_target:
                    lo = amp_x
                else:
                    hi = amp_x
            lo, hi = 0.0, 100.0
            for i in range(n_iter):
                amp_y = 0.5 * (lo + hi)
                trial = field * np.exp(1j * (amp_x * phase_x_2d + amp_y * phase_y_2d))
                m2x, m2y = compute_beam_m2(trial, x_1d, dx)
                if abs(m2y - M2y_target) / M2y_target < tol:
                    break
                if m2y < M2y_target:
                    lo = amp_y
                else:
                    hi = amp_y
        final_phase = amp_x * phase_x_2d + amp_y * phase_y_2d
        result = field * np.exp(1j * final_phase)
        m2x_f, m2y_f = compute_beam_m2(result, x_1d, dx)
        del phase_x_2d, phase_y_2d
    else:
        noise = rng.standard_normal((N, N))
        fx_ps = np.fft.fftfreq(N, dx)
        FX_ps, FY_ps = np.meshgrid(fx_ps, fx_ps)
        filt = np.exp(-2 * np.pi**2 * l_c**2 * (FX_ps**2 + FY_ps**2))
        phase_base = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
        phase_base /= phase_base.std()
        del noise, FX_ps, FY_ps, filt
        lo, hi = 0.0, 100.0
        best_amp = 0.0
        for i in range(n_iter):
            amp = 0.5 * (lo + hi)
            trial = field * np.exp(1j * amp * phase_base)
            m2x, m2y = compute_beam_m2(trial, x_1d, dx)
            m2_avg = 0.5 * (m2x + m2y)
            if abs(m2_avg - M2_target) / M2_target < tol:
                best_amp = amp
                break
            if m2_avg < M2_target:
                lo = amp
            else:
                hi = amp
            best_amp = amp
        final_phase = best_amp * phase_base
        result = field * np.exp(1j * final_phase)
        m2x_f, m2y_f = compute_beam_m2(result, x_1d, dx)
        amp_x = best_amp
        del phase_base
    power_after = np.sum(np.abs(result)**2)
    power_ratio = power_after / power_before
    if abs(1.0 - power_ratio) > 1e-10:
        print(f"  WARNING: Power not conserved! P_out/P_in = {power_ratio:.15f}")
    else:
        print(f"  Power conservation: P_out/P_in = {power_ratio:.15f} (OK)")
    err_x = abs(m2x_f - M2x_target) / M2x_target * 100
    err_y = abs(m2y_f - M2y_target) / M2y_target * 100
    print(f"  Final M²:  x = {m2x_f:.4f} (target {M2x_target:.2f}, err {err_x:.2f}%)")
    print(f"             y = {m2y_f:.4f} (target {M2y_target:.2f}, err {err_y:.2f}%)")
    if err_x > 2 * tol * 100:
        print(f"  WARNING: M²x error ({err_x:.2f}%) exceeds 2x tolerance ({2*tol*100:.1f}%)")
    if err_y > 2 * tol * 100:
        print(f"  WARNING: M²y error ({err_y:.2f}%) exceeds 2x tolerance ({2*tol*100:.1f}%)")
    phase_rms = np.std(final_phase)
    phase_pv = np.max(final_phase) - np.min(final_phase)
    print(f"  Phase screen: RMS = {phase_rms:.3f} rad, PV = {phase_pv:.3f} rad")
    print(f"  Seed = {seed}")
    del final_phase
    return result


def apply_amplitude_screen(field, x_1d, dx, w0_ref, sigma_a=0.15,
                           lc_factor=0.5, seed=137):
    """Apply smooth amplitude screen to model intensity non-uniformity.

    A(x,y) = exp(σ_a * g(x,y)), normalized to preserve total power.
    g(x,y) is a zero-mean, unit-variance Gaussian random field with
    spatial correlation length l_c = lc_factor * w0_ref.

    Parameters:
    -----------
    field : 2D array - input complex field
    x_1d : 1D array - spatial coordinates
    dx : float - grid spacing
    w0_ref : float - reference beam waist for correlation length
    sigma_a : float - amplitude fluctuation strength (RMS of log-amplitude)
    lc_factor : float - correlation length as fraction of w0_ref
    seed : int - random seed (separate from phase screen)

    Returns:
    --------
    field_out : 2D array - field with amplitude modulation applied
    """
    if sigma_a <= 0:
        print("  Amplitude screen: σ_a <= 0, skipped.")
        return field

    N = len(x_1d)
    l_c = w0_ref * lc_factor
    rng = np.random.default_rng(seed)

    # Generate spatially correlated random field
    noise = rng.standard_normal((N, N))
    fx_ps = np.fft.fftfreq(N, dx)
    FX_ps, FY_ps = np.meshgrid(fx_ps, fx_ps)
    filt = np.exp(-2 * np.pi**2 * l_c**2 * (FX_ps**2 + FY_ps**2))
    g = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
    g /= g.std()  # normalize to unit variance
    del noise, FX_ps, FY_ps, filt

    # Amplitude screen: A = exp(σ_a * g), always > 0
    A = np.exp(sigma_a * g)

    # Normalize to preserve total power: <|A|²> = 1
    A /= np.sqrt(np.mean(A**2))

    # Apply to field
    power_before = np.sum(np.abs(field)**2)
    result = field * A
    power_after = np.sum(np.abs(result)**2)

    # Diagnostics
    intensity_screen = A**2
    print(f"  Amplitude screen applied:")
    print(f"    σ_a = {sigma_a:.3f}, l_c = {l_c:.3f} mm ({lc_factor*100:.0f}% of w0)")
    print(f"    Intensity fluctuation: min={intensity_screen.min():.3f}, max={intensity_screen.max():.3f}")
    print(f"    RMS intensity variation: {np.std(intensity_screen):.3f}")
    print(f"    Power conservation: {power_after/power_before:.10f}")
    print(f"    Seed = {seed}")

    del g, A
    return result


def angular_spectrum_propagate(field, z, k):
    """
    Propagate optical field using Angular Spectrum Method (ASM).
    Exact method, no paraxial approximation.
    Optimized with Numba JIT compilation when available.

    Transfer function: H = exp(ikz * sqrt(1 - λ²fx² - λ²fy²))
    """
    if z == 0:
        return field.copy()

    # Angular spectrum transfer function
    if USE_NUMBA:
        H = _asm_transfer_function_numba(sqrt_freq_term_prop, sqrt_freq_term_evan,
                                         propagating_mask, k * z, N)
    else:
        H = np.where(propagating_mask,
                     np.exp(1j * k * z * sqrt_freq_term_prop),
                     np.exp(-k * z * sqrt_freq_term_evan))

    # Propagate: FFT -> multiply by H -> IFFT
    if USE_SCIPY_FFT:
        field_fft = scipy_fft.fft2(field, workers=-1)
        propagated_field = scipy_fft.ifft2(field_fft * H, workers=-1)
    else:
        field_fft = np.fft.fft2(field)
        propagated_field = np.fft.ifft2(field_fft * H)

    return propagated_field


def fresnel_propagate_zoom(field_in, z, k, L_in, L_out, N_out=None):
    """
    Fresnel propagation with different input/output grid sizes.
    Allows zooming into focus region with higher resolution.

    Uses the Fresnel diffraction integral with scaled coordinates:
    E(x',y') = exp(ikz)/(iλz) * exp(ik(x'²+y'²)/(2z))
               * FFT{ E(x,y) * exp(ik(x²+y²)/(2z)) }

    The key is that input and output grid spacings are related by:
    dx_out = λz / (N_in * dx_in)  for standard FFT

    To achieve arbitrary output grid size, we use matrix DFT.
    Optimized with Numba JIT compilation when available.

    Parameters:
    -----------
    field_in : 2D array - Input complex field (N_in x N_in)
    z : float - Propagation distance (mm)
    k : float - Wave number
    L_in : float - Input grid size (mm)
    L_out : float - Output grid size (mm), can be different from L_in
    N_out : int - Output grid points (default: same as input)

    Returns:
    --------
    field_out : 2D array - Output field at z (N_out x N_out)
    x_out, y_out : 1D arrays - Output grid coordinates
    """
    N_in = field_in.shape[0]
    if N_out is None:
        N_out = N_in

    dx_in = L_in / N_in

    # Output grid coordinates (needed for return values)
    x_out = np.linspace(-L_out/2, L_out/2, N_out)
    y_out = np.linspace(-L_out/2, L_out/2, N_out)

    # Fresnel propagation using CZT (Chirp-Z Transform) for O(N log N) zoom
    k_over_2z = k / (2 * z)
    scale = wavelength * z

    # Get CACHED CZT parameters (tiny memory footprint)
    W, A, const, phase_k = _get_cached_czt_params(L_in, L_out, N_in, N_out, scale)

    # Get CACHED quadratic phases
    quad_in = _get_cached_quad_phase(L_in, N_in, k_over_2z, 'in')
    quad_out = _get_cached_quad_phase(L_out, N_out, k_over_2z, 'out')

    # Apply input quadratic phase
    field_quad = field_in * quad_in

    # 2D zoom DFT via separable CZT — O(N log N) instead of O(N²)
    # Step 1: CZT along x (last axis), all rows at once
    temp = scipy_czt(field_quad, m=N_out, w=W, a=A)      # (N_in, N_out)
    temp *= const * phase_k[np.newaxis, :]
    # Step 2: CZT along y (transpose, CZT along last axis, transpose back)
    field_out = scipy_czt(temp.T, m=N_out, w=W, a=A).T   # (N_out, N_out)
    field_out *= const * phase_k[:, np.newaxis]

    # Prefactor
    prefactor = np.exp(1j * k * z) / (1j * wavelength * z)

    # Apply output phase and scaling
    field_out = prefactor * quad_out * field_out * (dx_in ** 2)

    return field_out, x_out, y_out


def propagate_field(field, z, k, method=None):
    """
    Propagate optical field using Angular Spectrum Method (ASM).

    Parameters:
    -----------
    field : 2D array - Complex optical field
    z : float - Propagation distance (mm)
    k : float - Wave number
    method : str - ignored, always uses ASM

    Returns:
    --------
    propagated_field : 2D array - Complex field after propagation
    """
    if z == 0:
        return field.copy()

    return angular_spectrum_propagate(field, z, k)


def thin_lens(field, R2, f, k):
    """
    Apply thin lens phase transformation.
    Optimized with Numba JIT compilation when available.
    """
    if USE_NUMBA:
        N_field = field.shape[0]
        k_over_2f = k / (2 * f)
        return _thin_lens_phase_numba(field.real, field.imag, R2, k_over_2f, N_field)
    else:
        lens_phase = np.exp(-1j * k * R2 / (2 * f))
        return field * lens_phase



# =============================================================================
# Generate Gaussian beam at z=0 (beam waist) with random phase screen for M²
# =============================================================================
TIMER.start_section("Grid and beam initialization")

# Generate ideal Gaussian beam at z=0
gaussian_field = gaussian_beam_field_astigmatic(X, Y, 0, w0x, w0y, wavelength, k, zRx, zRy, M2x, M2y)
center_idx = N // 2

# Apply M² phase screen to model real beam quality
if M2x > 1.02 or M2y > 1.02:
    print(f"Applying M² phase screen (target M²x={M2x:.2f}, M²y={M2y:.2f})...")
    gaussian_field = apply_m2_phase_screen(
        gaussian_field, x, dx, M2x, M2y,
        w0_ref=min(w0x, w0y), seed=PHASE_SCREEN_SEED
    )
else:
    print("M² <= 1.02, using ideal Gaussian beam.")

# Apply amplitude screen to model intensity non-uniformity
if APPLY_AMPLITUDE_SCREEN and AMP_SCREEN_SIGMA > 0:
    print(f"\nApplying amplitude screen (σ_a={AMP_SCREEN_SIGMA:.2f})...")
    gaussian_field = apply_amplitude_screen(
        gaussian_field, x, dx,
        w0_ref=min(w0x, w0y),
        sigma_a=AMP_SCREEN_SIGMA,
        lc_factor=AMP_SCREEN_LC_FACTOR,
        seed=AMP_SCREEN_SEED
    )
    # Re-check M² after amplitude screen (it may change slightly)
    m2x_after, m2y_after = compute_beam_m2(gaussian_field, x, dx, verbose=True)
    print(f"  M² after amplitude screen: x={m2x_after:.3f}, y={m2y_after:.3f}")

# =============================================================================
# FOCUSING THROUGH A LENS WITH APERTURE
# =============================================================================
TIMER.start_section("Beam propagation (waist -> lens)")

# Build mask (configurable shape)
def build_mask(X, Y, mtype, params):
    """Build a 2D mask array for the given mask type.
    Supports soft edges via 'mask_edge_width' parameter (tanh transition)."""
    edge_w = params.get('mask_edge_width', 0.0)

    if mtype == 'circular':
        r = np.sqrt(X**2 + Y**2)
        R = params['aperture_radius']
        if edge_w > 0:
            return 0.5 * (1.0 - np.tanh((r - R) / edge_w))
        return np.where(r <= R, 1.0, 0.0)
    elif mtype == 'twosided':
        hw = params['twosided_halfwidth']
        if edge_w > 0:
            return 0.5 * (1.0 - np.tanh((np.abs(X) - hw) / edge_w))
        return np.where(np.abs(X) <= hw, 1.0, 0.0)
    elif mtype == 'diagonal':
        xo = params['diag_x_offset']
        bs = params['diag_block_size']
        ys = params['diag_y_shift']
        if edge_w > 0:
            def smooth_rect(X, Y, x_lo, x_hi, y_lo, y_hi, ew):
                sx = 0.5 * (np.tanh((X - x_lo)/ew) - np.tanh((X - x_hi)/ew))
                sy = 0.5 * (np.tanh((Y - y_lo)/ew) - np.tanh((Y - y_hi)/ew))
                return sx * sy
            block1 = smooth_rect(X, Y, -xo-bs, -xo, ys-bs/2, ys+bs/2, edge_w)
            block2 = smooth_rect(X, Y, xo, xo+bs, -ys-bs/2, -ys+bs/2, edge_w)
            return 1.0 - np.clip(block1 + block2, 0, 1)
        # Hard-edge fallback
        block1 = (X >= -xo - bs) & (X <= -xo) & (Y >= ys - bs/2) & (Y <= ys + bs/2)
        block2 = (X >= xo) & (X <= xo + bs) & (Y >= -ys - bs/2) & (Y <= -ys + bs/2)
        return np.where(block1 | block2, 0.0, 1.0)
    else:  # 'none'
        return np.ones_like(X)

mask_params = {
    'aperture_radius': aperture_radius,
    'twosided_halfwidth': twosided_halfwidth,
    'diag_x_offset': diag_x_offset,
    'diag_block_size': diag_block_size,
    'diag_y_shift': diag_y_shift,
    'mask_edge_width': mask_edge_width,
}
aperture_mask = build_mask(X, Y, mask_type, mask_params)

# =============================================================================
# SINGLE-FIELD PROPAGATION: waist -> aperture -> mask -> lens (blocked & unblocked)
# =============================================================================
print("Propagating beam through optical system...")

# --- Blocked path: waist -> aperture -> mask -> lens ---
print(f"  Blocked: waist -> aperture (z={aperture_position:.0f} mm)...")
field_at_aperture = propagate_field(gaussian_field, aperture_position, k)
f_after_ap = field_at_aperture * aperture_mask

pwr_before = np.sum(np.abs(field_at_aperture)**2)
pwr_after = np.sum(np.abs(f_after_ap)**2)
aperture_transmission = pwr_after / pwr_before * 100
print(f"  Mask type: {mask_type}, transmission: {aperture_transmission:.2f}%")

print(f"  Blocked: aperture -> lens ({aperture_distance_before_lens:.0f} mm)...")
f_at_lens = propagate_field(f_after_ap, aperture_distance_before_lens, k)
field_after_lens = thin_lens(f_at_lens, R2, focal_length, k)

# --- Unblocked path: waist -> lens (no aperture) ---
print(f"  Unblocked: waist -> lens ({lens_position:.0f} mm)...")
f_at_lens_ub = propagate_field(gaussian_field, lens_position, k)
field_after_lens_unblocked = thin_lens(f_at_lens_ub, R2, focal_length, k)

# --- Auto-calibrate mask parameters to match experimental transmission ---
def compute_mask_transmission(mtype, params, field_at_ap, X_grid, Y_grid):
    """Compute power transmission through a mask for single field."""
    msk = build_mask(X_grid, Y_grid, mtype, params)
    pwr_before = np.sum(np.abs(field_at_ap)**2)
    pwr_after = np.sum(np.abs(field_at_ap * msk)**2)
    return pwr_after / pwr_before

def calibrate_mask_param(mtype, param_name, target_trans,
                         lo, hi, field_at_ap, X_grid, Y_grid, base_params, tol=1e-4):
    """Bisection search for mask parameter matching target transmission."""
    for _ in range(50):
        mid = (lo + hi) / 2
        params = dict(base_params)
        params[param_name] = mid
        trans = compute_mask_transmission(mtype, params, field_at_ap, X_grid, Y_grid)
        if abs(trans - target_trans) < tol:
            break
        if trans > target_trans:
            hi = mid
        else:
            lo = mid
    return mid, trans

if AUTO_CALIBRATE_MASK:
    print("\n--- Auto-calibrating mask parameters to match experimental transmission ---")

    aperture_radius, circ_trans = calibrate_mask_param(
        'circular', 'aperture_radius', circular_target_transmission,
        0.1, 20.0, field_at_aperture, X, Y, mask_params)
    mask_params['aperture_radius'] = aperture_radius
    print(f"  Circular radius: {aperture_radius:.3f} mm → transmission: {circ_trans*100:.1f}%")

    twosided_halfwidth, tw_trans = calibrate_mask_param(
        'twosided', 'twosided_halfwidth', twosided_target_transmission,
        0.1, aperture_radius, field_at_aperture, X, Y, mask_params)
    mask_params['twosided_halfwidth'] = twosided_halfwidth
    print(f"  Two-side halfwidth: {twosided_halfwidth:.3f} mm → transmission: {tw_trans*100:.1f}%")

    diag_x_offset, dg_trans = calibrate_mask_param(
        'diagonal', 'diag_x_offset', diagonal_target_transmission,
        0.1, aperture_radius, field_at_aperture, X, Y, mask_params)
    mask_params['diag_x_offset'] = diag_x_offset
    print(f"  Diagonal x_offset: {diag_x_offset:.3f} mm → transmission: {dg_trans*100:.1f}%")

    # Re-propagate main blocked beam with calibrated mask (aperture_radius may have changed)
    aperture_mask = build_mask(X, Y, mask_type, mask_params)
    f_after_ap = field_at_aperture * aperture_mask
    f_at_lens = propagate_field(f_after_ap, aperture_distance_before_lens, k)
    field_after_lens = thin_lens(f_at_lens, R2, focal_length, k)
    # Report calibrated transmission
    aperture_transmission = compute_mask_transmission(mask_type, mask_params, field_at_aperture, X, Y) * 100
    print(f"  Main blocked beam re-propagated with {mask_type} mask (transmission: {aperture_transmission:.1f}%)")
else:
    print("\n--- Using fixed mask parameters (auto-calibration OFF) ---")
    aperture_transmission = compute_mask_transmission(mask_type, mask_params, field_at_aperture, X, Y) * 100
    print(f"  {mask_type} mask transmission: {aperture_transmission:.1f}%")

# field_after_lens and field_after_lens_unblocked are now single fields (set above)

# =============================================================================
# Find the TRUE focus position (where phase is flattest)
# =============================================================================
def find_true_focus(field_after_lens, z_search, k, center_idx, dx, verbose=False):
    """
    Find the z position where peak intensity is highest (true focus).
    This is robust for all beam types including astigmatic beams.
    """
    best_z = z_search[0]
    max_peak_I = 0.0
    peak_x_list = []
    peak_y_list = []

    for z in z_search:
        field_z = propagate_field(field_after_lens, z, k)
        peak_I = np.max(np.abs(field_z)**2)

        if verbose:
            peak_x_list.append(np.max(np.abs(field_z[center_idx, :])**2))
            peak_y_list.append(np.max(np.abs(field_z[:, center_idx])**2))

        if peak_I > max_peak_I:
            max_peak_I = peak_I
            best_z = z

    if verbose:
        best_z_x = z_search[np.argmax(peak_x_list)]
        best_z_y = z_search[np.argmax(peak_y_list)]
        print(f"    Focus diagnostics:")
        print(f"      x cross-section peak at: z={best_z_x:.3f} mm")
        print(f"      y cross-section peak at: z={best_z_y:.3f} mm")
        print(f"      2D peak intensity at:    z={best_z:.3f} mm")
        print(f"      astigmatism: {abs(best_z_x - best_z_y)*1e3:.0f} um between x,y foci")

    return best_z, max_peak_I

print("Searching for true focus position...")
z_search = np.linspace(focal_length * 0.95, focal_length * 1.05, 100)
true_focus_z = find_true_focus(field_after_lens, z_search, k, center_idx, dx, verbose=True)[0]
print(f"Geometric focus (f): {focal_length:.2f} mm")
print(f"True focus found at: {true_focus_z:.2f} mm")
print(f"Difference: {true_focus_z - focal_length:.3f} mm")

# Find focus for unblocked beam
print("Searching for unblocked beam focus position...")
true_focus_z_unblocked, _ = find_true_focus(field_after_lens_unblocked, z_search, k, center_idx, dx, verbose=True)
print(f"Unblocked beam focus at: {true_focus_z_unblocked:.2f} mm")

# =============================================================================
# CALCULATE RAYLEIGH RANGE FOR BLOCKED AND UNBLOCKED BEAMS
# =============================================================================
TIMER.start_section("Rayleigh range calculation")
print("Calculating Rayleigh ranges...")

def calculate_rayleigh_range(field_after_lens, focus_z, k, L, N_calc=512, L_calc=1.0, z_range=10.0, n_z=50):
    """
    Calculate Rayleigh range by finding where beam area doubles from focus.
    Rayleigh range z_R is where w(z_R) = sqrt(2) * w0, or intensity drops to 50%.
    """
    dx_calc = L_calc / N_calc

    # Compute intensity at focus
    field_focus, x_focus, y_focus = fresnel_propagate_zoom(field_after_lens, focus_z, k, L, L_calc, N_calc)
    I_focus = np.abs(field_focus)**2
    I_focus_norm = I_focus / I_focus.max()

    # Calculate 1/e² radius at focus
    center_calc = N_calc // 2
    I_x_focus = I_focus_norm[center_calc, :]
    above_e2 = I_x_focus > 1/np.e**2
    w0_focus = np.sum(above_e2) * dx_calc / 2  # radius

    # Peak intensity at focus (for normalization)
    I_peak_focus = I_focus.max()

    # Search for Rayleigh range (where w = sqrt(2) * w0, i.e., area doubles)
    z_test_array = np.linspace(focus_z, focus_z + z_range, n_z)

    rayleigh_z = None
    for z_test in z_test_array[1:]:  # Skip focus itself
        field_test, _, _ = fresnel_propagate_zoom(field_after_lens, z_test, k, L, L_calc, N_calc)
        I_test = np.abs(field_test)**2
        I_test_norm = I_test / I_test.max()

        I_x_test = I_test_norm[center_calc, :]
        above_e2_test = I_x_test > 1/np.e**2
        w_test = np.sum(above_e2_test) * dx_calc / 2

        # Rayleigh range: w(z_R) = sqrt(2) * w0
        if w_test >= np.sqrt(2) * w0_focus:
            rayleigh_z = z_test - focus_z
            break

    return w0_focus, rayleigh_z, I_peak_focus

# Calculate for blocked beam
print("  Computing blocked beam Rayleigh range...")
w0_blocked, zR_blocked, I_peak_blocked = calculate_rayleigh_range(
    field_after_lens, true_focus_z, k, L, N_calc=512, L_calc=1.0, z_range=20.0, n_z=100
)

# Calculate for unblocked beam
print("  Computing unblocked beam Rayleigh range...")
w0_unblocked, zR_unblocked, I_peak_unblocked = calculate_rayleigh_range(
    field_after_lens_unblocked, true_focus_z_unblocked, k, L, N_calc=512, L_calc=1.0, z_range=20.0, n_z=100
)

print(f"\n{'='*50}")
print("RAYLEIGH RANGE COMPARISON")
print(f"{'='*50}")
print(f"Blocked beam (with aperture r={aperture_radius}mm):")
print(f"  Focus position: z = {true_focus_z:.2f} mm")
print(f"  Spot size w0: {w0_blocked*1e3:.2f} μm")
print(f"  Rayleigh range zR: {zR_blocked:.2f} mm" if zR_blocked else "  Rayleigh range: > 20 mm")
print(f"  Peak intensity (arb): {I_peak_blocked:.2e}")
print(f"\nUnblocked beam (no aperture):")
print(f"  Focus position: z = {true_focus_z_unblocked:.2f} mm")
print(f"  Spot size w0: {w0_unblocked*1e3:.2f} μm")
print(f"  Rayleigh range zR: {zR_unblocked:.2f} mm" if zR_unblocked else "  Rayleigh range: > 20 mm")
print(f"  Peak intensity (arb): {I_peak_unblocked:.2e}")
print(f"\nIntensity ratio at focus (blocked/unblocked): {I_peak_blocked/I_peak_unblocked:.4f}")
print(f"{'='*50}\n")

# Distances from lens to observe (around focal plane)
z_focus_distances = [0, focal_length*0.5, focal_length*0.9, true_focus_z,
                     focal_length*1.1, focal_length*1.5]

# Pre-compute fields at different propagation distances
def propagate_full_field(args):
    field, z, k = args
    return propagate_field(field, z, k)

TIMER.start_section("Field propagation for figures")
print("Computing Gaussian beam through lens...")

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    args_list = [(field_after_lens, z, k) for z in z_focus_distances]
    focus_list = list(executor.map(propagate_full_field, args_list))

focus_propagated_fields = {z: field for z, field in zip(z_focus_distances, focus_list)}

# Also compute unblocked beam at focus for comparison
print("Computing unblocked beam fields...")
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    args_list_unblocked = [(field_after_lens_unblocked, z, k) for z in z_focus_distances]
    focus_list_unblocked = list(executor.map(propagate_full_field, args_list_unblocked))


TIMER.start_section("High-res propagation + gas region")
# --- Figure 2: High-Resolution x-z Propagation Near Focus ---
print("Computing high-resolution x-z propagation near focus...")

# High-resolution parameters for x-z plane
n_z_steps_hr = 200
z_focus_prop = np.linspace(focal_length - 2.0, focal_length + 2.0, n_z_steps_hr)

# High-resolution output grid
L_xz_focus = 0.5  # mm - zoomed view for focus region
N_xz_focus = 1024
dx_xz_focus = L_xz_focus / N_xz_focus
x_xz_hires = np.linspace(-L_xz_focus/2, L_xz_focus/2, N_xz_focus)

print(f"  High-res x-z grid: L={L_xz_focus}mm, N={N_xz_focus}, dx={dx_xz_focus*1e3:.2f}μm")

# --- 2D HHG data storage setup (for macroscopic HHG yield with non-round apertures) ---
N_hhg_2d = 512
hhg_gas_length_prop = 1.0   # mm, gas length for HHG (centered at focal_length)
hhg_crop = slice(N_xz_focus // 2 - N_hhg_2d // 2, N_xz_focus // 2 + N_hhg_2d // 2)
x_hhg_2d = x_xz_hires[hhg_crop]   # coordinates for cropped 2D HHG grid
gas_z_start_prop = focal_length - hhg_gas_length_prop / 2.0
gas_z_end_prop = focal_length + hhg_gas_length_prop / 2.0
print(f"  2D HHG grid: {N_hhg_2d}x{N_hhg_2d} crop, gas region {gas_z_start_prop:.1f}-{gas_z_end_prop:.1f} mm")

# Gas region storage for HHG calculation (single field)
gas_I_list_b = []
gas_phase_list_b = []
gas_z_list_b = []

# Compute high-resolution x-z and y-z planes using fresnel_propagate_zoom (BLOCKED beam)
xz_intensity_hires = np.zeros((n_z_steps_hr, N_xz_focus))
xz_phase_hires = np.zeros((n_z_steps_hr, N_xz_focus))
xz_gouy_phase_hires = np.zeros((n_z_steps_hr, N_xz_focus))
yz_intensity_hires = np.zeros((n_z_steps_hr, N_xz_focus))
yz_phase_hires = np.zeros((n_z_steps_hr, N_xz_focus))
yz_gouy_phase_hires = np.zeros((n_z_steps_hr, N_xz_focus))

center_hr = N_xz_focus // 2
print(f"  BLOCKED beam: propagating at {n_z_steps_hr} z-steps...")
for i, z in enumerate(z_focus_prop):
    if i % 20 == 0:
        print(f"    z = {z:.2f} mm ({i+1}/{n_z_steps_hr})...")
    field_z, x_out, y_out = fresnel_propagate_zoom(field_after_lens, z, k, L, L_xz_focus, N_xz_focus)
    field_no_pw = field_z * np.exp(-1j * k * z)

    xz_intensity_hires[i, :] = np.abs(field_z[center_hr, :])**2
    yz_intensity_hires[i, :] = np.abs(field_z[:, center_hr])**2

    xz_phase_hires[i, :] = np.angle(field_z[center_hr, :])
    xz_gouy_phase_hires[i, :] = np.angle(field_no_pw[center_hr, :])
    yz_phase_hires[i, :] = np.angle(field_z[:, center_hr])
    yz_gouy_phase_hires[i, :] = np.angle(field_no_pw[:, center_hr])

    if gas_z_start_prop <= z <= gas_z_end_prop:
        gas_I_list_b.append(np.abs(field_z[hhg_crop, hhg_crop])**2)
        gas_phase_list_b.append(np.angle(field_no_pw[hhg_crop, hhg_crop]))
        gas_z_list_b.append(z)
# Convert gas data to arrays
z_gas_2d_b = np.array(gas_z_list_b)
I_2d_gas_b = np.array(gas_I_list_b)
phase_geom_2d_gas_b = np.array(gas_phase_list_b)
del gas_I_list_b, gas_phase_list_b, gas_z_list_b
print(f"  Stored 2D blocked beam data: {I_2d_gas_b.shape} ({I_2d_gas_b.nbytes/1e6:.0f} MB)")

# Compute high-resolution x-z and y-z planes for UNBLOCKED beam
z_focus_prop_ub = np.linspace(focal_length - 2.0, focal_length + 2.0, n_z_steps_hr)
xz_intensity_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
xz_phase_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
xz_gouy_phase_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
yz_intensity_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
yz_phase_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
yz_gouy_phase_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))

gas_I_list_ub = []
gas_phase_list_ub = []
gas_z_list_ub = []
center_hr = N_xz_focus // 2
print(f"  UNBLOCKED beam: propagating at {n_z_steps_hr} z-steps...")
for i, z in enumerate(z_focus_prop_ub):
    if i % 20 == 0:
        print(f"    z = {z:.2f} mm ({i+1}/{n_z_steps_hr})...")
    field_z, x_out, y_out = fresnel_propagate_zoom(field_after_lens_unblocked, z, k, L, L_xz_focus, N_xz_focus)
    field_no_pw = field_z * np.exp(-1j * k * z)

    xz_intensity_unblocked[i, :] = np.abs(field_z[center_hr, :])**2
    yz_intensity_unblocked[i, :] = np.abs(field_z[:, center_hr])**2

    xz_phase_unblocked[i, :] = np.angle(field_z[center_hr, :])
    xz_gouy_phase_unblocked[i, :] = np.angle(field_no_pw[center_hr, :])
    yz_phase_unblocked[i, :] = np.angle(field_z[:, center_hr])
    yz_gouy_phase_unblocked[i, :] = np.angle(field_no_pw[:, center_hr])

    if gas_z_start_prop <= z <= gas_z_end_prop:
        gas_I_list_ub.append(np.abs(field_z[hhg_crop, hhg_crop])**2)
        gas_phase_list_ub.append(np.angle(field_no_pw[hhg_crop, hhg_crop]))
        gas_z_list_ub.append(z)
# Convert gas data to arrays
z_gas_2d_ub = np.array(gas_z_list_ub)
I_2d_gas_ub = np.array(gas_I_list_ub)
phase_geom_2d_gas_ub = np.array(gas_phase_list_ub)
del gas_I_list_ub, gas_phase_list_ub, gas_z_list_ub
print(f"  Stored 2D unblocked beam data: {I_2d_gas_ub.shape} ({I_2d_gas_ub.nbytes/1e6:.0f} MB)")

print("  High-resolution x-z and y-z computation complete (blocked + unblocked).")

# Focus indices (used by HHG sections downstream)
focus_idx = np.argmin(np.abs(z_focus_prop - true_focus_z))
focus_idx_ub = np.argmin(np.abs(z_focus_prop_ub - true_focus_z_unblocked))

# Mask display config (used by HHG mask comparison figures)
mask_disp = {
    'none':     ('Unblocked', '#6E6E6E'),
    'circular': ('Circular',  '#2F6F9F'),
    'twosided': ('Two-side',  '#B3262E'),
    'diagonal': ('Diagonal',  '#218C4A'),
}

if PLOT_OPTICAL_DIAGNOSTICS:
    TIMER.start_section("Figure 1 - Propagation overview")
    # --- Figure 1: 2D Intensity and Phase after lens at different z ---
    # Use high-resolution for focus position
    L_fig1_focus = 0.5  # mm for high-res focus
    N_fig1_focus = 512

    fig1, axes1 = plt.subplots(2, 6, figsize=(20, 7))

    for idx, z in enumerate(z_focus_distances):
        # Check if this is near focus - use high-resolution
        is_near_focus = abs(z - focal_length) < 0.1 * focal_length

        if is_near_focus:
            # Use high-resolution zoom propagation for focus region
            field_hr, x_hr, y_hr = fresnel_propagate_zoom(
                field_after_lens, z, k, L, L_fig1_focus, N_fig1_focus
            )
            intensity_z = np.abs(field_hr)**2
            phase_z = np.angle(field_hr)
            extent_z = [x_hr.min(), x_hr.max(), y_hr.min(), y_hr.max()]
            plot_range = L_fig1_focus / 2
            title_suffix = ' (high-res)'
        else:
            # Use standard propagation
            field_z = focus_propagated_fields[z]
            intensity_z = np.abs(field_z)**2
            phase_z = np.angle(field_z)
            extent_z = [x.min(), x.max(), y.min(), y.max()]
            plot_range = get_plot_range(z)
            title_suffix = ''

        # Intensity (top row)
        ax_int = axes1[0, idx]
        im_int = ax_int.imshow(intensity_z, extent=extent_z, cmap='hot', origin='lower')
        ax_int.set_title(f'z = {z/10:.1f} cm{title_suffix}')
        ax_int.set_xlabel('x (mm)')
        ax_int.set_xlim([-plot_range, plot_range])
        ax_int.set_ylim([-plot_range, plot_range])
        if idx == 0:
            ax_int.set_ylabel('Intensity\ny (mm)')
        else:
            ax_int.set_ylabel('y (mm)')
        plt.colorbar(im_int, ax=ax_int)

        # Phase (bottom row)
        ax_ph = axes1[1, idx]
        im_ph = ax_ph.imshow(phase_z, extent=extent_z, cmap='twilight', origin='lower',
                              vmin=-np.pi, vmax=np.pi)
        ax_ph.set_title(f'z = {z/10:.1f} cm{title_suffix}')
        ax_ph.set_xlabel('x (mm)')
        ax_ph.set_xlim([-plot_range, plot_range])
        ax_ph.set_ylim([-plot_range, plot_range])
        if idx == 0:
            ax_ph.set_ylabel('Phase\ny (mm)')
        else:
            ax_ph.set_ylabel('y (mm)')
        plt.colorbar(im_ph, ax=ax_ph, label='rad')

    fig1.suptitle(f'Gaussian Beam with Aperture (r={aperture_radius:.0f}mm, {aperture_distance_before_lens/10:.0f}cm before lens, f={focal_length/10:.0f}cm)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'unblocked_beam_2D_{_m2_tag}.png', dpi=300)

    # --- Figure 1B: Log Intensity at Focus with Unblocked Beam Comparison ---
    print("Computing high-resolution fields for focus comparison...")

    # Compute high-res fields at focus for both blocked and unblocked
    field_blocked_hr, x_focus_hr, y_focus_hr = fresnel_propagate_zoom(
        field_after_lens, true_focus_z, k, L, L_fig1_focus, N_fig1_focus
    )
    field_unblocked_hr, _, _ = fresnel_propagate_zoom(
        field_after_lens_unblocked, true_focus_z_unblocked, k, L, L_fig1_focus, N_fig1_focus
    )

    I_blocked_hr = np.abs(field_blocked_hr)**2
    I_unblocked_hr = np.abs(field_unblocked_hr)**2

    # Store peak intensities for reference
    I_peak_blocked_hr = I_blocked_hr.max()
    I_peak_unblocked_hr = I_unblocked_hr.max()
    intensity_ratio_hr = I_peak_blocked_hr / I_peak_unblocked_hr

    # Normalize BOTH to unblocked beam's peak intensity for direct comparison
    I_blocked_norm_common = I_blocked_hr / I_peak_unblocked_hr  # max will be < 1 due to aperture
    I_unblocked_norm_common = I_unblocked_hr / I_peak_unblocked_hr  # max will be 1

    # Also keep self-normalized versions for shape comparison
    I_blocked_norm = I_blocked_hr / I_blocked_hr.max()
    I_unblocked_norm = I_unblocked_hr / I_unblocked_hr.max()

    # Cross-sections (common normalization for intensity comparison)
    center_fig1b = N_fig1_focus // 2
    I_blocked_x_common = I_blocked_norm_common[center_fig1b, :]
    I_unblocked_x_common = I_unblocked_norm_common[center_fig1b, :]

    # Cross-sections (self-normalized for shape comparison)
    I_blocked_x = I_blocked_norm[center_fig1b, :]
    I_unblocked_x = I_unblocked_norm[center_fig1b, :]
    I_blocked_y = I_blocked_norm[:, center_fig1b]
    I_unblocked_y = I_unblocked_norm[:, center_fig1b]

    # Center coordinates
    x_centered_hr = x_focus_hr - x_focus_hr[np.argmax(I_blocked_x)]
    y_centered_hr = y_focus_hr - y_focus_hr[np.argmax(I_blocked_y)]

    # Create Figure 1B (3x3 layout for more comparisons)
    fig1b, axes1b = plt.subplots(3, 3, figsize=(18, 16))

    extent_hr = [x_focus_hr.min(), x_focus_hr.max(), y_focus_hr.min(), y_focus_hr.max()]

    # Row 1, Col 1: Blocked beam 2D (normalized to unblocked peak)
    ax1b_1 = axes1b[0, 0]
    im1b_1 = ax1b_1.imshow(I_blocked_norm_common, extent=extent_hr, cmap='hot', origin='lower', vmin=0, vmax=1, interpolation='bicubic')
    ax1b_1.set_title(f'Blocked Beam\n(I/I_unblocked_max = {intensity_ratio_hr:.3f})', fontsize=12)
    ax1b_1.set_xlabel('x (mm)')
    ax1b_1.set_ylabel('y (mm)')
    plt.colorbar(im1b_1, ax=ax1b_1, label='I/I_unblocked_max')

    # Row 1, Col 2: Unblocked beam 2D (normalized to its peak)
    ax1b_2 = axes1b[0, 1]
    im1b_2 = ax1b_2.imshow(I_unblocked_norm_common, extent=extent_hr, cmap='hot', origin='lower', vmin=0, vmax=1, interpolation='bicubic')
    ax1b_2.set_title(f'Unblocked Beam\n(reference)', fontsize=12)
    ax1b_2.set_xlabel('x (mm)')
    ax1b_2.set_ylabel('y (mm)')
    plt.colorbar(im1b_2, ax=ax1b_2, label='I/I_max')

    # Row 1, Col 3: Blocked beam 2D (log, normalized to unblocked)
    ax1b_3 = axes1b[0, 2]
    I_blocked_log_common = np.log10(I_blocked_norm_common + 1e-10)
    im1b_3 = ax1b_3.imshow(I_blocked_log_common, extent=extent_hr, cmap='hot', origin='lower', vmin=-4, vmax=0, interpolation='bicubic')
    ax1b_3.set_title(f'Blocked Beam (log)\n(normalized to unblocked)', fontsize=12)
    ax1b_3.set_xlabel('x (mm)')
    ax1b_3.set_ylabel('y (mm)')
    plt.colorbar(im1b_3, ax=ax1b_3, label='log₁₀(I/I_unblocked_max)')

    # Row 2, Col 1: X cross-section comparison (common normalization, linear)
    ax1b_4 = axes1b[1, 0]
    ax1b_4.plot(x_centered_hr * 1e3, I_blocked_x_common, 'b-', linewidth=2, label='Blocked')
    ax1b_4.plot(x_centered_hr * 1e3, I_unblocked_x_common, 'r--', linewidth=2, label='Unblocked')
    ax1b_4.axhline(y=intensity_ratio_hr, color='blue', linestyle=':', alpha=0.5, label=f'Blocked peak={intensity_ratio_hr:.3f}')
    ax1b_4.set_xlabel('x (μm)', fontsize=12)
    ax1b_4.set_ylabel('I / I_unblocked_max', fontsize=12)
    ax1b_4.set_title('X Cross-section (common norm, linear)', fontsize=12)
    ax1b_4.legend(fontsize=8)
    ax1b_4.grid(True, alpha=0.3)
    ax1b_4.set_xlim([-L_fig1_focus/2 * 1e3, L_fig1_focus/2 * 1e3])
    ax1b_4.set_ylim([0, 1.1])

    # Row 2, Col 2: X cross-section comparison (common normalization, log)
    ax1b_5 = axes1b[1, 1]
    ax1b_5.semilogy(x_centered_hr * 1e3, I_blocked_x_common + 1e-10, 'b-', linewidth=2, label='Blocked')
    ax1b_5.semilogy(x_centered_hr * 1e3, I_unblocked_x_common + 1e-10, 'r--', linewidth=2, label='Unblocked')
    ax1b_5.axhline(y=intensity_ratio_hr, color='blue', linestyle=':', alpha=0.5)
    ax1b_5.set_xlabel('x (μm)', fontsize=12)
    ax1b_5.set_ylabel('I / I_unblocked_max (log)', fontsize=12)
    ax1b_5.set_title('X Cross-section (common norm, log)', fontsize=12)
    ax1b_5.legend(fontsize=8)
    ax1b_5.grid(True, alpha=0.3, which='both')
    ax1b_5.set_xlim([-L_fig1_focus/2 * 1e3, L_fig1_focus/2 * 1e3])
    ax1b_5.set_ylim([1e-4, 2])

    # Row 2, Col 3: Shape comparison (self-normalized)
    ax1b_6 = axes1b[1, 2]
    ax1b_6.plot(x_centered_hr * 1e3, I_blocked_x, 'b-', linewidth=2, label='Blocked (self-norm)')
    ax1b_6.plot(x_centered_hr * 1e3, I_unblocked_x, 'r--', linewidth=2, label='Unblocked (self-norm)')
    ax1b_6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax1b_6.axhline(y=1/np.e**2, color='gray', linestyle=':', alpha=0.5, label='1/e²')
    ax1b_6.set_xlabel('x (μm)', fontsize=12)
    ax1b_6.set_ylabel('Normalized Intensity', fontsize=12)
    ax1b_6.set_title('Shape Comparison (self-normalized)', fontsize=12)
    ax1b_6.legend(fontsize=8)
    ax1b_6.grid(True, alpha=0.3)
    ax1b_6.set_xlim([-L_fig1_focus/2 * 1e3, L_fig1_focus/2 * 1e3])
    ax1b_6.set_ylim([0, 1.1])

    # Row 3, Col 1: Phase comparison
    ax1b_7 = axes1b[2, 0]
    phase_blocked_hr = np.angle(field_blocked_hr)
    phase_unblocked_hr = np.angle(field_unblocked_hr)
    phase_blocked_x = phase_blocked_hr[center_fig1b, :]
    phase_unblocked_x = phase_unblocked_hr[center_fig1b, :]
    ax1b_7.plot(x_centered_hr * 1e3, phase_blocked_x, 'b-', linewidth=2, label='Blocked')
    ax1b_7.plot(x_centered_hr * 1e3, phase_unblocked_x, 'r--', linewidth=2, label='Unblocked')
    ax1b_7.set_xlabel('x (μm)', fontsize=12)
    ax1b_7.set_ylabel('Phase (rad)', fontsize=12)
    ax1b_7.set_title('Phase Cross-section', fontsize=12)
    ax1b_7.legend(fontsize=8)
    ax1b_7.grid(True, alpha=0.3)
    ax1b_7.set_xlim([-L_fig1_focus/2 * 1e3, L_fig1_focus/2 * 1e3])
    ax1b_7.set_ylim([-np.pi, np.pi])

    # Row 3, Col 2: 2D Phase comparison - Blocked
    ax1b_8 = axes1b[2, 1]
    im1b_8 = ax1b_8.imshow(phase_blocked_hr, extent=extent_hr, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax1b_8.set_title('Blocked Beam Phase', fontsize=12)
    ax1b_8.set_xlabel('x (mm)')
    ax1b_8.set_ylabel('y (mm)')
    plt.colorbar(im1b_8, ax=ax1b_8, label='Phase (rad)')

    # Row 3, Col 3: Rayleigh range comparison summary
    ax1b_9 = axes1b[2, 2]
    ax1b_9.axis('off')

    rayleigh_text = f"""
    FOCUS AND RAYLEIGH RANGE COMPARISON
    {'='*50}

    BLOCKED BEAM (aperture r = {aperture_radius:.0f} mm):
    ─────────────────────────────────────────────
      Focus position: z = {true_focus_z:.2f} mm
      Spot size w0: {w0_blocked*1e3:.2f} μm
      Rayleigh range zR: {zR_blocked:.2f} mm
      Peak intensity ratio: {intensity_ratio_hr:.4f}

    UNBLOCKED BEAM (no aperture):
    ─────────────────────────────────────────────
      Focus position: z = {true_focus_z_unblocked:.2f} mm
      Spot size w0: {w0_unblocked*1e3:.2f} μm
      Rayleigh range zR: {zR_unblocked:.2f} mm
      Peak intensity: 1.0 (reference)

    COMPARISON:
    ─────────────────────────────────────────────
      Spot size ratio: {w0_blocked/w0_unblocked:.3f}
      Rayleigh range ratio: {zR_blocked/zR_unblocked:.3f}
      Aperture transmission: {aperture_transmission:.2f}%

    {'='*50}
    """
    ax1b_9.text(0.02, 0.98, rayleigh_text, transform=ax1b_9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig1b.suptitle('Focus Intensity Comparison: Blocked vs Unblocked Beam', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'unblocked_beam_focus_comparison_{_m2_tag}.png', dpi=300)
    print("Focus comparison figure saved to 'unblocked_beam_focus_comparison.png'")

    fig2, axes2 = plt.subplots(4, 4, figsize=(22, 20))

    # Extent for high-res plot (x/y in μm for better readability, z in mm)
    extent_focus_hr_x = [x_xz_hires.min() * 1e3, x_xz_hires.max() * 1e3,
                         z_focus_prop.min(), z_focus_prop.max()]
    extent_focus_hr_y = [x_xz_hires.min() * 1e3, x_xz_hires.max() * 1e3,  # y uses same grid
                         z_focus_prop.min(), z_focus_prop.max()]

    focus_idx = np.argmin(np.abs(z_focus_prop - true_focus_z))
    focus_idx_ub = np.argmin(np.abs(z_focus_prop_ub - true_focus_z_unblocked))

    # ===== ROW 1-2: x-z PLANES =====
    # Row 1, Col 1: Blocked Intensity x-z
    ax = axes2[0, 0]
    im = ax.imshow(xz_intensity_hires, extent=extent_focus_hr_x, aspect='auto', cmap='hot', origin='lower', interpolation='bicubic')
    ax.axhline(y=focal_length, color='cyan', linestyle='--', linewidth=1.5, label=f'f={focal_length:.0f}mm')
    ax.axhline(y=true_focus_z, color='lime', linestyle=':', linewidth=1.5, label=f'focus={true_focus_z:.2f}mm')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('BLOCKED: Intensity (x-z)', fontsize=10)
    ax.legend(loc='upper right', fontsize=6)
    plt.colorbar(im, ax=ax, label='I')

    # Row 1, Col 2: Blocked Intensity x cross-section
    ax = axes2[0, 1]
    I_cross_x_b = xz_intensity_hires[focus_idx, :]
    I_cross_x_norm_b = I_cross_x_b / I_cross_x_b.max()
    ax.plot(x_xz_hires * 1e3, I_cross_x_norm_b, 'b-', linewidth=2, label='Blocked X')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
    ax.axhline(y=1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('Norm. Intensity', fontsize=10)
    ax.set_title(f'BLOCKED: X Focus (z={true_focus_z:.2f}mm)', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 3: Unblocked Intensity x-z
    ax = axes2[0, 2]
    im = ax.imshow(xz_intensity_unblocked, extent=extent_focus_hr_x, aspect='auto', cmap='hot', origin='lower', interpolation='bicubic')
    ax.axhline(y=focal_length, color='cyan', linestyle='--', linewidth=1.5, label=f'f={focal_length:.0f}mm')
    ax.axhline(y=true_focus_z_unblocked, color='lime', linestyle=':', linewidth=1.5, label=f'focus={true_focus_z_unblocked:.2f}mm')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('UNBLOCKED: Intensity (x-z)', fontsize=10)
    ax.legend(loc='upper right', fontsize=6)
    plt.colorbar(im, ax=ax, label='I')

    # Row 1, Col 4: Unblocked Intensity x cross-section
    ax = axes2[0, 3]
    I_cross_x_ub = xz_intensity_unblocked[focus_idx_ub, :]
    I_cross_x_norm_ub = I_cross_x_ub / I_cross_x_ub.max()
    ax.plot(x_xz_hires * 1e3, I_cross_x_norm_ub, 'r-', linewidth=2, label='Unblocked X')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
    ax.axhline(y=1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('Norm. Intensity', fontsize=10)
    ax.set_title(f'UNBLOCKED: X Focus (z={true_focus_z_unblocked:.2f}mm)', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1: Blocked Phase x-z
    ax = axes2[1, 0]
    im = ax.imshow(xz_phase_hires, extent=extent_focus_hr_x, aspect='auto', cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax.axhline(y=focal_length, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=true_focus_z, color='lime', linestyle=':', linewidth=1.5)
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('BLOCKED: Phase (x-z)', fontsize=10)
    plt.colorbar(im, ax=ax, label='φ (rad)')

    # Row 2, Col 2: Blocked Phase x cross-section
    ax = axes2[1, 1]
    phase_x_b = xz_phase_hires[focus_idx, :]
    ax.plot(x_xz_hires * 1e3, phase_x_b, 'b-', linewidth=2, label='Blocked X')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('Phase (rad)', fontsize=10)
    ax.set_title('BLOCKED: X Phase at Focus', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=6)

    # Row 2, Col 3: Unblocked Phase x-z
    ax = axes2[1, 2]
    im = ax.imshow(xz_phase_unblocked, extent=extent_focus_hr_x, aspect='auto', cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax.axhline(y=focal_length, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=true_focus_z_unblocked, color='lime', linestyle=':', linewidth=1.5)
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('UNBLOCKED: Phase (x-z)', fontsize=10)
    plt.colorbar(im, ax=ax, label='φ (rad)')

    # Row 2, Col 4: Unblocked Phase x cross-section
    ax = axes2[1, 3]
    phase_x_ub = xz_phase_unblocked[focus_idx_ub, :]
    ax.plot(x_xz_hires * 1e3, phase_x_ub, 'r-', linewidth=2, label='Unblocked X')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('Phase (rad)', fontsize=10)
    ax.set_title('UNBLOCKED: X Phase at Focus', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=6)

    # ===== ROW 3-4: y-z PLANES =====
    # Row 3, Col 1: Blocked Intensity y-z
    ax = axes2[2, 0]
    im = ax.imshow(yz_intensity_hires, extent=extent_focus_hr_y, aspect='auto', cmap='hot', origin='lower', interpolation='bicubic')
    ax.axhline(y=focal_length, color='cyan', linestyle='--', linewidth=1.5, label=f'f={focal_length:.0f}mm')
    ax.axhline(y=true_focus_z, color='lime', linestyle=':', linewidth=1.5, label=f'focus={true_focus_z:.2f}mm')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('BLOCKED: Intensity (y-z)', fontsize=10)
    ax.legend(loc='upper right', fontsize=6)
    plt.colorbar(im, ax=ax, label='I')

    # Row 3, Col 2: Blocked Intensity y cross-section
    ax = axes2[2, 1]
    I_cross_y_b = yz_intensity_hires[focus_idx, :]
    I_cross_y_norm_b = I_cross_y_b / I_cross_y_b.max()
    ax.plot(x_xz_hires * 1e3, I_cross_y_norm_b, 'b-', linewidth=2, label='Blocked Y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
    ax.axhline(y=1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('Norm. Intensity', fontsize=10)
    ax.set_title(f'BLOCKED: Y Focus (z={true_focus_z:.2f}mm)', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 3, Col 3: Unblocked Intensity y-z
    ax = axes2[2, 2]
    im = ax.imshow(yz_intensity_unblocked, extent=extent_focus_hr_y, aspect='auto', cmap='hot', origin='lower', interpolation='bicubic')
    ax.axhline(y=focal_length, color='cyan', linestyle='--', linewidth=1.5, label=f'f={focal_length:.0f}mm')
    ax.axhline(y=true_focus_z_unblocked, color='lime', linestyle=':', linewidth=1.5, label=f'focus={true_focus_z_unblocked:.2f}mm')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('UNBLOCKED: Intensity (y-z)', fontsize=10)
    ax.legend(loc='upper right', fontsize=6)
    plt.colorbar(im, ax=ax, label='I')

    # Row 3, Col 4: Unblocked Intensity y cross-section
    ax = axes2[2, 3]
    I_cross_y_ub = yz_intensity_unblocked[focus_idx_ub, :]
    I_cross_y_norm_ub = I_cross_y_ub / I_cross_y_ub.max()
    ax.plot(x_xz_hires * 1e3, I_cross_y_norm_ub, 'r-', linewidth=2, label='Unblocked Y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
    ax.axhline(y=1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('Norm. Intensity', fontsize=10)
    ax.set_title(f'UNBLOCKED: Y Focus (z={true_focus_z_unblocked:.2f}mm)', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 4, Col 1: Blocked Phase y-z
    ax = axes2[3, 0]
    im = ax.imshow(yz_phase_hires, extent=extent_focus_hr_y, aspect='auto', cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax.axhline(y=focal_length, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=true_focus_z, color='lime', linestyle=':', linewidth=1.5)
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('BLOCKED: Phase (y-z)', fontsize=10)
    plt.colorbar(im, ax=ax, label='φ (rad)')

    # Row 4, Col 2: Blocked Phase y cross-section
    ax = axes2[3, 1]
    phase_y_b = yz_phase_hires[focus_idx, :]
    ax.plot(x_xz_hires * 1e3, phase_y_b, 'b-', linewidth=2, label='Blocked Y')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('Phase (rad)', fontsize=10)
    ax.set_title('BLOCKED: Y Phase at Focus', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=6)

    # Row 4, Col 3: Unblocked Phase y-z
    ax = axes2[3, 2]
    im = ax.imshow(yz_phase_unblocked, extent=extent_focus_hr_y, aspect='auto', cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax.axhline(y=focal_length, color='white', linestyle='--', linewidth=1.5)
    ax.axhline(y=true_focus_z_unblocked, color='lime', linestyle=':', linewidth=1.5)
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('z (mm)', fontsize=10)
    ax.set_title('UNBLOCKED: Phase (y-z)', fontsize=10)
    plt.colorbar(im, ax=ax, label='φ (rad)')

    # Row 4, Col 4: Unblocked Phase y cross-section
    ax = axes2[3, 3]
    phase_y_ub = yz_phase_unblocked[focus_idx_ub, :]
    ax.plot(x_xz_hires * 1e3, phase_y_ub, 'r-', linewidth=2, label='Unblocked Y')
    ax.set_xlabel('y (μm)', fontsize=10)
    ax.set_ylabel('Phase (rad)', fontsize=10)
    ax.set_title('UNBLOCKED: Y Phase at Focus', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=6)

    # Add statistics text
    peak_ratio_x = I_cross_x_b.max() / I_cross_x_ub.max()
    peak_ratio_y = I_cross_y_b.max() / I_cross_y_ub.max()
    fig2.text(0.5, 0.01, f'Peak Intensity Ratio (Blocked/Unblocked): X={peak_ratio_x:.3f}, Y={peak_ratio_y:.3f}',
              ha='center', fontsize=11, style='italic')

    fig2.suptitle(f'High-Resolution x-z and y-z Propagation: Blocked vs Unblocked (dx={dx_xz_focus*1e3:.2f}μm)', fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(f'blocked_vs_unblocked_xz_yz_highres_{_m2_tag}.png', dpi=300)
    print("High-resolution x-z and y-z comparison figure saved to 'blocked_vs_unblocked_xz_yz_highres.png'.")

    TIMER.start_section("Figure 6 - Focus beam shape")
    # =============================================================================
    # FIGURE 6: NORMALIZED SHAPE DISTRIBUTION AT FOCUS (HIGH RESOLUTION)
    # =============================================================================
    print("Computing high-resolution normalized shape distribution at focus...")

    # High-resolution parameters for Figure 6
    L_fig6 = 0.5  # mm - small grid for high resolution
    N_fig6 = 1024  # Grid points
    dx_fig6 = L_fig6 / N_fig6
    print(f"  Fig6 high-res grid: L={L_fig6}mm, N={N_fig6}, dx={dx_fig6*1e3:.2f}μm")

    # Compute high-resolution blocked beam at focus
    field_focus_hr, x_fig6, y_fig6 = fresnel_propagate_zoom(
        field_after_lens, true_focus_z, k, L, L_fig6, N_fig6
    )
    I_focus_hr = np.abs(field_focus_hr)**2

    # Compute high-resolution unblocked beam at focus
    field_focus_unblocked_hr, _, _ = fresnel_propagate_zoom(
        field_after_lens_unblocked, true_focus_z_unblocked, k, L, L_fig6, N_fig6
    )
    I_focus_unblocked_hr = np.abs(field_focus_unblocked_hr)**2

    # Normalize to maximum
    I_normalized = I_focus_hr / I_focus_hr.max()
    I_normalized_unblocked = I_focus_unblocked_hr / I_focus_unblocked_hr.max()

    # Cross-sections
    center_fig6 = N_fig6 // 2
    I_x_cross = I_normalized[center_fig6, :]  # along x (y=0)
    I_y_cross = I_normalized[:, center_fig6]  # along y (x=0)

    # Calculate beam parameters from normalized distribution
    # FWHM
    half_max_x = I_x_cross > 0.5
    half_max_y = I_y_cross > 0.5
    fwhm_x = np.sum(half_max_x) * dx_fig6
    fwhm_y = np.sum(half_max_y) * dx_fig6

    # 1/e² width (86.5% of power for Gaussian)
    e2_threshold = 1/np.e**2
    above_e2_x = I_x_cross > e2_threshold
    above_e2_y = I_y_cross > e2_threshold
    w_x = np.sum(above_e2_x) * dx_fig6 / 2  # radius
    w_y = np.sum(above_e2_y) * dx_fig6 / 2

    # Second moment beam radius (ISO 11146)
    # w² = 4 * ∫r²I(r)dr / ∫I(r)dr
    x_vals = x_fig6 - x_fig6[np.argmax(I_x_cross)]  # center at peak
    y_vals = y_fig6 - y_fig6[np.argmax(I_y_cross)]
    second_moment_x = 4 * np.sum(x_vals**2 * I_x_cross) / np.sum(I_x_cross)
    second_moment_y = 4 * np.sum(y_vals**2 * I_y_cross) / np.sum(I_y_cross)
    w_iso_x = np.sqrt(second_moment_x)
    w_iso_y = np.sqrt(second_moment_y)

    # Generate ideal Gaussian for comparison
    x_centered = x_fig6 - x_fig6[np.argmax(I_x_cross)]
    y_centered = y_fig6 - y_fig6[np.argmax(I_y_cross)]
    # Use measured 1/e² width
    gaussian_x = np.exp(-2 * x_centered**2 / w_x**2) if w_x > 0 else np.zeros_like(x_fig6)
    gaussian_y = np.exp(-2 * y_centered**2 / w_y**2) if w_y > 0 else np.zeros_like(y_fig6)

    # Generate Airy pattern for comparison (for aperture-limited case)
    # First zero at r = 1.22 λf/D
    D_effective = 2 * aperture_radius  # effective aperture diameter at lens
    airy_radius = 1.22 * wavelength * focal_length / D_effective
    r_airy_x = x_centered / airy_radius
    r_airy_y = y_centered / airy_radius
    # Airy function: I = [2*J1(pi*r)/(pi*r)]² (avoid division by zero)
    from scipy.special import j1 as _j1
    def _airy_pattern(r):
        """Airy disk intensity: [2*J1(pi*r)/(pi*r)]², normalized to 1 at r=0."""
        pr = np.pi * r
        result = np.ones_like(r, dtype=float)
        nz = pr != 0
        result[nz] = (2.0 * _j1(pr[nz]) / pr[nz])**2
        return result
    airy_x = _airy_pattern(r_airy_x) if airy_radius > 0 else np.zeros_like(x_fig6)
    airy_y = _airy_pattern(r_airy_y) if airy_radius > 0 else np.zeros_like(y_fig6)

    # Cross-sections for unblocked beam
    I_x_cross_unblocked = I_normalized_unblocked[center_fig6, :]
    I_y_cross_unblocked = I_normalized_unblocked[:, center_fig6]

    # Phase for blocked and unblocked beams (high-resolution)
    phase_focus_blocked = np.angle(field_focus_hr)
    phase_focus_unblocked = np.angle(field_focus_unblocked_hr)
    phase_x_blocked = phase_focus_blocked[center_fig6, :]
    phase_x_unblocked = phase_focus_unblocked[center_fig6, :]
    phase_y_blocked = phase_focus_blocked[:, center_fig6]
    phase_y_unblocked = phase_focus_unblocked[:, center_fig6]

    # Create figure with 4x3 layout to include log scale, unblocked comparison, and phase
    fig9, axes9 = plt.subplots(4, 3, figsize=(18, 20))

    # Plot range based on beam size
    shape_plot_range = max(fwhm_x, fwhm_y) * 5  # 5x FWHM for good visibility

    # Extent for high-resolution grid
    extent_fig6 = [x_fig6.min(), x_fig6.max(), y_fig6.min(), y_fig6.max()]

    # Row 1, Col 1: 2D normalized intensity (blocked)
    ax9_1 = axes9[0, 0]
    im9_1 = ax9_1.imshow(I_normalized,
                          extent=extent_fig6,
                          cmap='hot', origin='lower', vmin=0, vmax=1, interpolation='bicubic')
    ax9_1.set_title('Blocked Beam Intensity (HR)\n(I / I_max)', fontsize=12)
    ax9_1.set_xlabel('x (mm)')
    ax9_1.set_ylabel('y (mm)')
    plt.colorbar(im9_1, ax=ax9_1, label='I/I_max')

    # Row 1, Col 2: Log scale 2D intensity (blocked)
    ax9_2 = axes9[0, 1]
    I_log = np.log10(I_normalized + 1e-10)  # avoid log(0)
    im9_2 = ax9_2.imshow(I_log,
                          extent=extent_fig6,
                          cmap='hot', origin='lower', vmin=-4, vmax=0, interpolation='bicubic')
    ax9_2.set_title('Blocked Beam Log₁₀(I/I_max) (HR)\n(shows sidelobes)', fontsize=12)
    ax9_2.set_xlabel('x (mm)')
    ax9_2.set_ylabel('y (mm)')
    cbar9_2 = plt.colorbar(im9_2, ax=ax9_2, label='log₁₀(I/I_max)')

    # Row 1, Col 3: Unblocked beam 2D (log scale)
    ax9_3 = axes9[0, 2]
    I_log_unblocked = np.log10(I_normalized_unblocked + 1e-10)
    im9_3 = ax9_3.imshow(I_log_unblocked,
                          extent=extent_fig6,
                          cmap='hot', origin='lower', vmin=-4, vmax=0, interpolation='bicubic')
    ax9_3.set_title('Unblocked Beam Log₁₀(I/I_max) (HR)', fontsize=12)
    ax9_3.set_xlabel('x (mm)')
    ax9_3.set_ylabel('y (mm)')
    plt.colorbar(im9_3, ax=ax9_3, label='log₁₀(I/I_max)')

    # Row 2, Col 1: X cross-section with Gaussian and Airy comparison (linear)
    ax9_4 = axes9[1, 0]
    ax9_4.plot(x_centered * 1e3, I_x_cross, 'b-', linewidth=2, label='Blocked')
    ax9_4.plot(x_centered * 1e3, I_x_cross_unblocked, 'r--', linewidth=1.5, label='Unblocked')
    ax9_4.plot(x_centered * 1e3, gaussian_x, 'g:', linewidth=1.5, label='Gaussian fit')
    ax9_4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax9_4.axhline(y=1/np.e**2, color='gray', linestyle=':', alpha=0.5, label='1/e²')
    ax9_4.set_xlabel('x (μm)', fontsize=12)
    ax9_4.set_ylabel('Normalized Intensity', fontsize=12)
    ax9_4.set_title('X Cross-section (linear scale)', fontsize=12)
    ax9_4.legend(fontsize=8, loc='upper right')
    ax9_4.grid(True, alpha=0.3)
    ax9_4.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_4.set_ylim([0, 1.1])

    # Row 2, Col 2: X cross-section (log scale)
    ax9_5 = axes9[1, 1]
    ax9_5.semilogy(x_centered * 1e3, I_x_cross + 1e-10, 'b-', linewidth=2, label='Blocked')
    ax9_5.semilogy(x_centered * 1e3, I_x_cross_unblocked + 1e-10, 'r--', linewidth=1.5, label='Unblocked')
    ax9_5.semilogy(x_centered * 1e3, gaussian_x + 1e-10, 'g:', linewidth=1.5, label='Gaussian')
    ax9_5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax9_5.axhline(y=1/np.e**2, color='gray', linestyle=':', alpha=0.5, label='1/e²')
    ax9_5.set_xlabel('x (μm)', fontsize=12)
    ax9_5.set_ylabel('Normalized Intensity (log)', fontsize=12)
    ax9_5.set_title('X Cross-section (log scale)', fontsize=12)
    ax9_5.legend(fontsize=8, loc='upper right')
    ax9_5.grid(True, alpha=0.3, which='both')
    ax9_5.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_5.set_ylim([1e-4, 2])

    # Row 2, Col 3: Y cross-section (linear)
    ax9_6 = axes9[1, 2]
    ax9_6.plot(y_centered * 1e3, I_y_cross, 'b-', linewidth=2, label='Blocked')
    ax9_6.plot(y_centered * 1e3, I_y_cross_unblocked, 'r--', linewidth=1.5, label='Unblocked')
    ax9_6.plot(y_centered * 1e3, gaussian_y, 'g:', linewidth=1.5, label='Gaussian fit')
    ax9_6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax9_6.axhline(y=1/np.e**2, color='gray', linestyle=':', alpha=0.5, label='1/e²')
    ax9_6.set_xlabel('y (μm)', fontsize=12)
    ax9_6.set_ylabel('Normalized Intensity', fontsize=12)
    ax9_6.set_title('Y Cross-section (linear scale)', fontsize=12)
    ax9_6.legend(fontsize=8, loc='upper right')
    ax9_6.grid(True, alpha=0.3)
    ax9_6.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_6.set_ylim([0, 1.1])

    # Row 3, Col 1: Y cross-section (log scale)
    ax9_7 = axes9[2, 0]
    ax9_7.semilogy(y_centered * 1e3, I_y_cross + 1e-10, 'b-', linewidth=2, label='Blocked')
    ax9_7.semilogy(y_centered * 1e3, I_y_cross_unblocked + 1e-10, 'r--', linewidth=1.5, label='Unblocked')
    ax9_7.semilogy(y_centered * 1e3, gaussian_y + 1e-10, 'g:', linewidth=1.5, label='Gaussian')
    ax9_7.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax9_7.axhline(y=1/np.e**2, color='gray', linestyle=':', alpha=0.5, label='1/e²')
    ax9_7.set_xlabel('y (μm)', fontsize=12)
    ax9_7.set_ylabel('Normalized Intensity (log)', fontsize=12)
    ax9_7.set_title('Y Cross-section (log scale)', fontsize=12)
    ax9_7.legend(fontsize=8, loc='upper right')
    ax9_7.grid(True, alpha=0.3, which='both')
    ax9_7.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_7.set_ylim([1e-4, 2])

    # Row 3, Col 2: Contour plot comparison (high-resolution)
    ax9_8 = axes9[2, 1]
    # Use high-resolution coordinates (already centered)
    X_fig6, Y_fig6 = np.meshgrid(x_centered, y_centered)

    # Contour levels
    levels = [1/np.e**2, 0.5, 0.9]
    cs_blocked = ax9_8.contour(X_fig6 * 1e3, Y_fig6 * 1e3, I_normalized,
                        levels=levels, colors=['blue', 'blue', 'blue'], linestyles=['-', '--', ':'])
    cs_unblocked = ax9_8.contour(X_fig6 * 1e3, Y_fig6 * 1e3, I_normalized_unblocked,
                        levels=levels, colors=['red', 'red', 'red'], linestyles=['-', '--', ':'])
    ax9_8.set_title('Contour Comparison (HR)\n(Blocked=blue, Unblocked=red)', fontsize=12)
    ax9_8.set_xlabel('x (μm)')
    ax9_8.set_ylabel('y (μm)')
    ax9_8.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_8.set_ylim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_8.set_aspect('equal')
    ax9_8.grid(True, alpha=0.3)

    # Row 4, Col 1: 2D Phase comparison - Blocked (high-resolution)
    ax9_10 = axes9[3, 0]
    im9_10 = ax9_10.imshow(phase_focus_blocked, extent=extent_fig6,
                            cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax9_10.set_title('Blocked Beam Phase (HR)', fontsize=12)
    ax9_10.set_xlabel('x (mm)')
    ax9_10.set_ylabel('y (mm)')
    plt.colorbar(im9_10, ax=ax9_10, label='Phase (rad)')

    # Row 4, Col 2: 2D Phase comparison - Unblocked (high-resolution)
    ax9_11 = axes9[3, 1]
    im9_11 = ax9_11.imshow(phase_focus_unblocked, extent=extent_fig6,
                            cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax9_11.set_title('Unblocked Beam Phase (HR)', fontsize=12)
    ax9_11.set_xlabel('x (mm)')
    ax9_11.set_ylabel('y (mm)')
    plt.colorbar(im9_11, ax=ax9_11, label='Phase (rad)')

    # Row 4, Col 3: Phase cross-section comparison
    ax9_12 = axes9[3, 2]
    ax9_12.plot(x_centered * 1e3, phase_x_blocked, 'b-', linewidth=2, label='Blocked')
    ax9_12.plot(x_centered * 1e3, phase_x_unblocked, 'r--', linewidth=1.5, label='Unblocked')
    ax9_12.set_xlabel('x (μm)', fontsize=12)
    ax9_12.set_ylabel('Phase (rad)', fontsize=12)
    ax9_12.set_title('X Phase Cross-section', fontsize=12)
    ax9_12.legend(fontsize=8)
    ax9_12.grid(True, alpha=0.3)
    ax9_12.set_xlim([-shape_plot_range * 1e3, shape_plot_range * 1e3])
    ax9_12.set_ylim([-np.pi, np.pi])

    # Row 3, Col 3: Beam shape parameters summary (moved to row 3)
    ax9_9 = axes9[2, 2]
    ax9_9.axis('off')

    # Calculate beam quality (compare to ideal Gaussian)
    # Overlap integral with Gaussian
    overlap_x = np.sum(I_x_cross * gaussian_x) / np.sqrt(np.sum(I_x_cross**2) * np.sum(gaussian_x**2))
    overlap_y = np.sum(I_y_cross * gaussian_y) / np.sqrt(np.sum(I_y_cross**2) * np.sum(gaussian_y**2))

    # Calculate unblocked beam parameters
    half_max_x_ub = I_x_cross_unblocked > 0.5
    fwhm_x_ub = np.sum(half_max_x_ub) * dx_fig6
    above_e2_x_ub = I_x_cross_unblocked > 1/np.e**2
    w_x_ub = np.sum(above_e2_x_ub) * dx_fig6 / 2

    # Strehl ratio estimate (peak intensity compared to ideal Airy pattern)
    # For reference: Airy pattern first zero at 1.22 λf/D
    airy_first_zero = 1.22 * wavelength * focal_length / D_effective

    shape_text = f"""
    FOCUS BEAM COMPARISON
    {'='*45}

    BLOCKED BEAM (aperture r={aperture_radius:.0f}mm):
    ─────────────────────────────────────────────
      FWHM: {fwhm_x*1e3:.2f} x {fwhm_y*1e3:.2f} μm
      1/e² radius: {w_x*1e3:.2f} x {w_y*1e3:.2f} μm
      Gaussian overlap: {overlap_x:.4f} (x)
      Rayleigh range: {zR_blocked:.2f} mm

    UNBLOCKED BEAM:
    ─────────────────────────────────────────────
      FWHM: {fwhm_x_ub*1e3:.2f} μm
      1/e² radius: {w_x_ub*1e3:.2f} μm
      Rayleigh range: {zR_unblocked:.2f} mm

    COMPARISON:
    ─────────────────────────────────────────────
      FWHM ratio: {fwhm_x/fwhm_x_ub:.3f}
      Spot size ratio: {w_x/w_x_ub:.3f}
      Rayleigh ratio: {zR_blocked/zR_unblocked:.3f}
      Airy first zero: {airy_first_zero*1e3:.2f} μm
    {'='*45}
    """
    ax9_9.text(0.05, 0.95, shape_text, transform=ax9_9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig9.suptitle(f'Focus Beam Shape Distribution (f = {focal_length/10:.0f} cm, aperture = {aperture_radius:.0f} mm)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'unblocked_beam_focus_shape_{_m2_tag}.png', dpi=300)

    TIMER.start_section("Figure 9B - High-res focus analysis")
    # =============================================================================
    # FIGURE 9B: HIGH-RESOLUTION FOCUS ANALYSIS (Zoomed Grid)
    # =============================================================================
    print("\n" + "="*60)
    print("HIGH-RESOLUTION FOCUS ANALYSIS")
    print("="*60)

    # High-resolution output grid parameters
    L_focus = 1.0  # mm - small grid for high resolution
    N_focus = 4096  # Grid points
    dx_focus = L_focus / N_focus
    print(f"High-res grid: L = {L_focus} mm, N = {N_focus}, dx = {dx_focus*1e3:.3f} μm")
    print(f"Resolution improvement: {dx/dx_focus:.1f}x")

    # Propagate from lens to focus with zoom - BLOCKED BEAM (single field)
    print("Computing high-resolution field at focus (blocked beam)...")
    t_start = time.perf_counter()
    field_focus_hires, x_hires, y_hires = fresnel_propagate_zoom(field_after_lens, true_focus_z, k, L, L_focus, N_focus)
    I_focus_hires = np.abs(field_focus_hires)**2
    t_elapsed = time.perf_counter() - t_start
    print(f"Blocked zoom completed in {t_elapsed:.2f} s")

    # Propagate from lens to focus with zoom - UNBLOCKED BEAM (single field)
    print("Computing high-resolution field at focus (unblocked beam)...")
    t_start_ub = time.perf_counter()
    field_focus_hires_unblocked, _, _ = fresnel_propagate_zoom(field_after_lens_unblocked, true_focus_z_unblocked, k, L, L_focus, N_focus)
    I_focus_hires_ub = np.abs(field_focus_hires_unblocked)**2
    t_elapsed_ub = time.perf_counter() - t_start_ub
    print(f"Unblocked zoom completed in {t_elapsed_ub:.2f} s")

    # Peak intensities
    I_peak_blocked_hires = I_focus_hires.max()
    I_peak_unblocked_hires = I_focus_hires_ub.max()
    intensity_ratio_hires = I_peak_blocked_hires / I_peak_unblocked_hires
    print(f"  Peak intensity ratio (blocked/unblocked): {intensity_ratio_hires:.4f}")

    # Self-normalized (for beam shape measurements like FWHM)
    I_hires_self_norm = I_focus_hires / I_peak_blocked_hires
    I_hires_self_norm_ub = I_focus_hires_ub / I_peak_unblocked_hires

    # Common normalization (both to unblocked peak for fair comparison)
    I_hires_norm = I_focus_hires / I_peak_unblocked_hires  # max will be < 1
    I_hires_norm_ub = I_focus_hires_ub / I_peak_unblocked_hires  # max will be = 1

    # Cross-sections (self-normalized for beam parameter measurements)
    center_hires = N_focus // 2
    I_x_hires = I_hires_self_norm[center_hires, :]
    I_y_hires = I_hires_self_norm[:, center_hires]

    # Cross-sections - Unblocked (self-normalized)
    I_x_hires_ub = I_hires_self_norm_ub[center_hires, :]
    I_y_hires_ub = I_hires_self_norm_ub[:, center_hires]

    # Cross-sections with common normalization (for intensity comparison plots)
    I_x_hires_common = I_hires_norm[center_hires, :]
    I_y_hires_common = I_hires_norm[:, center_hires]
    I_x_hires_ub_common = I_hires_norm_ub[center_hires, :]
    I_y_hires_ub_common = I_hires_norm_ub[:, center_hires]

    # Calculate beam parameters with high resolution
    # FWHM
    half_max_x_hr = I_x_hires > 0.5
    half_max_y_hr = I_y_hires > 0.5
    fwhm_x_hr = np.sum(half_max_x_hr) * dx_focus
    fwhm_y_hr = np.sum(half_max_y_hr) * dx_focus

    # 1/e² width
    e2_threshold_hr = 1/np.e**2
    above_e2_x_hr = I_x_hires > e2_threshold_hr
    above_e2_y_hr = I_y_hires > e2_threshold_hr
    w_x_hr = np.sum(above_e2_x_hr) * dx_focus / 2
    w_y_hr = np.sum(above_e2_y_hr) * dx_focus / 2

    # ISO 11146 Second moment
    x_peak_idx = np.argmax(I_x_hires)
    y_peak_idx = np.argmax(I_y_hires)
    x_hr_centered = x_hires - x_hires[x_peak_idx]
    y_hr_centered = y_hires - y_hires[y_peak_idx]
    second_moment_x_hr = 4 * np.sum(x_hr_centered**2 * I_x_hires) / np.sum(I_x_hires)
    second_moment_y_hr = 4 * np.sum(y_hr_centered**2 * I_y_hires) / np.sum(I_y_hires)
    w_iso_x_hr = np.sqrt(second_moment_x_hr)
    w_iso_y_hr = np.sqrt(second_moment_y_hr)

    # Generate ideal Gaussian for comparison
    gaussian_x_hr = np.exp(-2 * x_hr_centered**2 / w_x_hr**2) if w_x_hr > 0 else np.zeros_like(x_hires)
    gaussian_y_hr = np.exp(-2 * y_hr_centered**2 / w_y_hr**2) if w_y_hr > 0 else np.zeros_like(y_hires)

    # Gaussian overlap
    overlap_x_hr = np.sum(I_x_hires * gaussian_x_hr) / np.sqrt(np.sum(I_x_hires**2) * np.sum(gaussian_x_hr**2))
    overlap_y_hr = np.sum(I_y_hires * gaussian_y_hr) / np.sqrt(np.sum(I_y_hires**2) * np.sum(gaussian_y_hr**2))

    print(f"\nHigh-resolution beam measurements:")
    print(f"  FWHM: {fwhm_x_hr*1e3:.2f} x {fwhm_y_hr*1e3:.2f} μm")
    print(f"  1/e^2 radius: {w_x_hr*1e3:.2f} x {w_y_hr*1e3:.2f} um")
    print(f"  ISO 11146 width: {w_iso_x_hr*1e3:.2f} x {w_iso_y_hr*1e3:.2f} μm")
    print(f"  Gaussian overlap: {overlap_x_hr:.4f} (x), {overlap_y_hr:.4f} (y)")
    print(f"  Points across FWHM: {fwhm_x_hr/dx_focus:.0f} x {fwhm_y_hr/dx_focus:.0f}")

    # Create Figure 9B with 3x3 layout to include Y cross-sections
    fig9b, axes9b = plt.subplots(3, 3, figsize=(18, 16))

    # Plot range
    plot_range_hr = max(fwhm_x_hr, fwhm_y_hr) * 5

    # Row 1, Col 1: 2D blocked beam intensity (normalized to unblocked peak)
    ax9b_1 = axes9b[0, 0]
    im9b_1 = ax9b_1.imshow(I_hires_norm,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='hot', origin='lower', vmin=0, vmax=1, interpolation='bicubic')
    ax9b_1.set_title(f'Blocked Beam (norm. to unblocked)\npeak = {intensity_ratio_hires:.3f}', fontsize=11)
    ax9b_1.set_xlabel('x (mm)')
    ax9b_1.set_ylabel('y (mm)')
    plt.colorbar(im9b_1, ax=ax9b_1, label='I/I_unblocked_max')

    # Row 1, Col 2: Log scale
    ax9b_2 = axes9b[0, 1]
    I_log_hr = np.log10(I_hires_norm + 1e-10)
    im9b_2 = ax9b_2.imshow(I_log_hr,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='hot', origin='lower', vmin=-4, vmax=0, interpolation='bicubic')
    ax9b_2.set_title('Log₁₀(I/I_max)\n(High Resolution)', fontsize=11)
    ax9b_2.set_xlabel('x (mm)')
    ax9b_2.set_ylabel('y (mm)')
    plt.colorbar(im9b_2, ax=ax9b_2, label='log₁₀(I/I_max)')

    # Row 1, Col 3: Unblocked beam 2D (log scale)
    ax9b_3 = axes9b[0, 2]
    I_log_hr_ub = np.log10(I_hires_norm_ub + 1e-10)
    im9b_3 = ax9b_3.imshow(I_log_hr_ub,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='hot', origin='lower', vmin=-4, vmax=0, interpolation='bicubic')
    ax9b_3.set_title('Unblocked Beam (High-Res)\nLog₁₀(I/I_max)', fontsize=11)
    ax9b_3.set_xlabel('x (mm)')
    ax9b_3.set_ylabel('y (mm)')
    plt.colorbar(im9b_3, ax=ax9b_3, label='log₁₀(I/I_max)')

    # Row 2, Col 1: X cross-section comparison (linear, common normalization)
    ax9b_4 = axes9b[1, 0]
    ax9b_4.plot(x_hr_centered * 1e3, I_x_hires_common, 'b-', linewidth=2, label=f'Blocked (peak={intensity_ratio_hires:.3f})')
    ax9b_4.plot(x_hr_centered * 1e3, I_x_hires_ub_common, 'r--', linewidth=1.5, label='Unblocked (peak=1)')
    ax9b_4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax9b_4.axhline(y=intensity_ratio_hires*0.5, color='blue', linestyle='--', alpha=0.3, label='Blocked FWHM')
    ax9b_4.set_xlabel('x (μm)', fontsize=11)
    ax9b_4.set_ylabel('Intensity (I/I_unblocked_max)', fontsize=11)
    ax9b_4.set_title('X Cross-section (common normalization)', fontsize=11)
    ax9b_4.legend(fontsize=7)
    ax9b_4.grid(True, alpha=0.3)
    ax9b_4.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])
    ax9b_4.set_ylim([0, 1.1])

    # Row 2, Col 2: X cross-section comparison (log, common normalization)
    ax9b_5 = axes9b[1, 1]
    ax9b_5.semilogy(x_hr_centered * 1e3, I_x_hires_common + 1e-10, 'b-', linewidth=2, label='Blocked')
    ax9b_5.semilogy(x_hr_centered * 1e3, I_x_hires_ub_common + 1e-10, 'r--', linewidth=1.5, label='Unblocked')
    ax9b_5.axhline(y=intensity_ratio_hires, color='blue', linestyle=':', alpha=0.5, label=f'Blocked peak ({intensity_ratio_hires:.3f})')
    ax9b_5.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Unblocked peak (1.0)')
    ax9b_5.set_xlabel('x (μm)', fontsize=11)
    ax9b_5.set_ylabel('Intensity (log)', fontsize=11)
    ax9b_5.set_title('X Cross-section (log, common norm.)', fontsize=11)
    ax9b_5.legend(fontsize=7)
    ax9b_5.grid(True, alpha=0.3, which='both')
    ax9b_5.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])
    ax9b_5.set_ylim([1e-4, 2])

    # Row 2, Col 3: Y cross-section comparison (linear, common normalization)
    ax9b_6 = axes9b[1, 2]
    ax9b_6.plot(y_hr_centered * 1e3, I_y_hires_common, 'b-', linewidth=2, label=f'Blocked (peak={intensity_ratio_hires:.3f})')
    ax9b_6.plot(y_hr_centered * 1e3, I_y_hires_ub_common, 'r--', linewidth=1.5, label='Unblocked (peak=1)')
    ax9b_6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax9b_6.axhline(y=intensity_ratio_hires*0.5, color='blue', linestyle='--', alpha=0.3, label='Blocked FWHM')
    ax9b_6.set_xlabel('y (μm)', fontsize=11)
    ax9b_6.set_ylabel('Intensity (I/I_unblocked_max)', fontsize=11)
    ax9b_6.set_title('Y Cross-section (common normalization)', fontsize=11)
    ax9b_6.legend(fontsize=7)
    ax9b_6.grid(True, alpha=0.3)
    ax9b_6.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])
    ax9b_6.set_ylim([0, 1.1])

    # Row 3, Col 1: Y cross-section comparison (log, common normalization)
    ax9b_7 = axes9b[2, 0]
    ax9b_7.semilogy(y_hr_centered * 1e3, I_y_hires_common + 1e-10, 'b-', linewidth=2, label='Blocked')
    ax9b_7.semilogy(y_hr_centered * 1e3, I_y_hires_ub_common + 1e-10, 'r--', linewidth=1.5, label='Unblocked')
    ax9b_7.axhline(y=intensity_ratio_hires, color='blue', linestyle=':', alpha=0.5, label=f'Blocked peak ({intensity_ratio_hires:.3f})')
    ax9b_7.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Unblocked peak (1.0)')
    ax9b_7.set_xlabel('y (μm)', fontsize=11)
    ax9b_7.set_ylabel('Intensity (log)', fontsize=11)
    ax9b_7.set_title('Y Cross-section (log, common norm.)', fontsize=11)
    ax9b_7.legend(fontsize=7)
    ax9b_7.grid(True, alpha=0.3, which='both')
    ax9b_7.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])
    ax9b_7.set_ylim([1e-4, 2])

    # Row 3, Col 2: X vs Y comparison (self-normalized)
    ax9b_8 = axes9b[2, 1]
    ax9b_8.plot(x_hr_centered * 1e3, I_x_hires, 'b-', linewidth=2, label='Blocked X')
    ax9b_8.plot(y_hr_centered * 1e3, I_y_hires, 'b--', linewidth=1.5, label='Blocked Y')
    ax9b_8.plot(x_hr_centered * 1e3, I_x_hires_ub, 'r-', linewidth=2, label='Unblocked X')
    ax9b_8.plot(y_hr_centered * 1e3, I_y_hires_ub, 'r--', linewidth=1.5, label='Unblocked Y')
    ax9b_8.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM')
    ax9b_8.set_xlabel('x/y (μm)', fontsize=11)
    ax9b_8.set_ylabel('Normalized Intensity', fontsize=11)
    ax9b_8.set_title('X vs Y Comparison (self-normalized)', fontsize=11)
    ax9b_8.legend(fontsize=7, ncol=2)
    ax9b_8.grid(True, alpha=0.3)
    ax9b_8.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])
    ax9b_8.set_ylim([0, 1.1])

    # Row 3, Col 3: Comparison summary
    ax9b_9 = axes9b[2, 2]
    ax9b_9.axis('off')

    # Calculate unblocked beam parameters (high-res) for X and Y
    half_max_x_hr_ub = I_x_hires_ub > 0.5
    fwhm_x_hr_ub = np.sum(half_max_x_hr_ub) * dx_focus
    above_e2_x_hr_ub = I_x_hires_ub > 1/np.e**2
    w_x_hr_ub = np.sum(above_e2_x_hr_ub) * dx_focus / 2

    half_max_y_hr_ub = I_y_hires_ub > 0.5
    fwhm_y_hr_ub = np.sum(half_max_y_hr_ub) * dx_focus
    above_e2_y_hr_ub = I_y_hires_ub > 1/np.e**2
    w_y_hr_ub = np.sum(above_e2_y_hr_ub) * dx_focus / 2

    comparison_text = f"""
    HIGH-RESOLUTION BLOCKED vs UNBLOCKED
    {'='*45}

    INTENSITY (norm. to unblocked peak):
      Blocked peak: {intensity_ratio_hires:.4f}
      Unblocked peak: 1.0000

    BLOCKED BEAM (r={aperture_radius:.0f}mm):
      FWHM X: {fwhm_x_hr*1e3:.2f} μm
      FWHM Y: {fwhm_y_hr*1e3:.2f} μm
      1/e² X: {w_x_hr*1e3:.2f} μm
      1/e² Y: {w_y_hr*1e3:.2f} μm
      Overlap X: {overlap_x_hr:.4f}
      Overlap Y: {overlap_y_hr:.4f}

    UNBLOCKED BEAM:
      FWHM X: {fwhm_x_hr_ub*1e3:.2f} μm
      FWHM Y: {fwhm_y_hr_ub*1e3:.2f} μm
      1/e² X: {w_x_hr_ub*1e3:.2f} μm
      1/e² Y: {w_y_hr_ub*1e3:.2f} μm

    RATIO (Blocked/Unblocked):
      Peak: {intensity_ratio_hires:.4f}
      FWHM X: {fwhm_x_hr/fwhm_x_hr_ub:.3f}
      FWHM Y: {fwhm_y_hr/fwhm_y_hr_ub:.3f}

    GRID: dx={dx_focus*1e3:.3f}μm ({dx/dx_focus:.0f}x)
    {'='*45}
    """
    ax9b_9.text(0.02, 0.98, comparison_text, transform=ax9b_9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    fig9b.suptitle(f'High-Resolution Focus Analysis (dx = {dx_focus*1e3:.3f} μm, {dx/dx_focus:.0f}x zoom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'unblocked_beam_focus_highres_{_m2_tag}.png', dpi=300)
    print("High-resolution focus figure saved to 'unblocked_beam_focus_highres.png'")

    TIMER.start_section("Figure 9C - Phase analysis")
    # =============================================================================
    # FIGURE 9C: HIGH-RESOLUTION PHASE ANALYSIS AT FOCUS (BLOCKED vs UNBLOCKED)
    # =============================================================================
    print("Computing high-resolution phase at focus (blocked and unblocked)...")

    # Get phase from high-resolution field - BLOCKED
    phase_focus_hires = np.angle(field_focus_hires)
    phase_x_hires = phase_focus_hires[center_hires, :]
    phase_y_hires = phase_focus_hires[:, center_hires]
    phase_x_unwrapped = np.unwrap(phase_x_hires)
    phase_y_unwrapped = np.unwrap(phase_y_hires)

    # Get phase from high-resolution field - UNBLOCKED
    phase_focus_hires_ub = np.angle(field_focus_hires_unblocked)
    phase_x_hires_ub = phase_focus_hires_ub[center_hires, :]
    phase_y_hires_ub = phase_focus_hires_ub[:, center_hires]
    phase_x_unwrapped_ub = np.unwrap(phase_x_hires_ub)
    phase_y_unwrapped_ub = np.unwrap(phase_y_hires_ub)

    # Remove linear trend (tilt) to see residual phase
    # Fit only in high-intensity region - BLOCKED
    intensity_mask_x = I_x_hires > 0.1 * I_x_hires.max()
    intensity_mask_y = I_y_hires > 0.1 * I_y_hires.max()

    # Linear fit for blocked beam
    if np.sum(intensity_mask_x) > 10:
        x_fit = x_hires[intensity_mask_x]
        phase_fit_x = phase_x_unwrapped[intensity_mask_x]
        coeffs_x = np.polyfit(x_fit, phase_fit_x, 1)
        phase_linear_x = np.polyval(coeffs_x, x_hires)
        phase_residual_x = phase_x_unwrapped - phase_linear_x
    else:
        coeffs_x = [0, 0]
        phase_residual_x = phase_x_unwrapped

    if np.sum(intensity_mask_y) > 10:
        y_fit = y_hires[intensity_mask_y]
        phase_fit_y = phase_y_unwrapped[intensity_mask_y]
        coeffs_y = np.polyfit(y_fit, phase_fit_y, 1)
        phase_linear_y = np.polyval(coeffs_y, y_hires)
        phase_residual_y = phase_y_unwrapped - phase_linear_y
    else:
        coeffs_y = [0, 0]
        phase_residual_y = phase_y_unwrapped

    # Linear fit for unblocked beam
    intensity_mask_x_ub = I_x_hires_ub > 0.1 * I_x_hires_ub.max()
    intensity_mask_y_ub = I_y_hires_ub > 0.1 * I_y_hires_ub.max()

    if np.sum(intensity_mask_x_ub) > 10:
        x_fit_ub = x_hires[intensity_mask_x_ub]
        phase_fit_x_ub = phase_x_unwrapped_ub[intensity_mask_x_ub]
        coeffs_x_ub = np.polyfit(x_fit_ub, phase_fit_x_ub, 1)
        phase_linear_x_ub = np.polyval(coeffs_x_ub, x_hires)
        phase_residual_x_ub = phase_x_unwrapped_ub - phase_linear_x_ub
    else:
        coeffs_x_ub = [0, 0]
        phase_residual_x_ub = phase_x_unwrapped_ub

    if np.sum(intensity_mask_y_ub) > 10:
        y_fit_ub = y_hires[intensity_mask_y_ub]
        phase_fit_y_ub = phase_y_unwrapped_ub[intensity_mask_y_ub]
        coeffs_y_ub = np.polyfit(y_fit_ub, phase_fit_y_ub, 1)
        phase_linear_y_ub = np.polyval(coeffs_y_ub, y_hires)
        phase_residual_y_ub = phase_y_unwrapped_ub - phase_linear_y_ub
    else:
        coeffs_y_ub = [0, 0]
        phase_residual_y_ub = phase_y_unwrapped_ub

    # Phase flatness in high-intensity region
    phase_std_x = np.std(phase_residual_x[intensity_mask_x]) if np.sum(intensity_mask_x) > 0 else 0
    phase_std_y = np.std(phase_residual_y[intensity_mask_y]) if np.sum(intensity_mask_y) > 0 else 0
    phase_std_x_ub = np.std(phase_residual_x_ub[intensity_mask_x_ub]) if np.sum(intensity_mask_x_ub) > 0 else 0
    phase_std_y_ub = np.std(phase_residual_y_ub[intensity_mask_y_ub]) if np.sum(intensity_mask_y_ub) > 0 else 0

    # Create Figure 9C for phase comparison (blocked vs unblocked) with 3x3 layout for Y direction
    fig9c, axes9c = plt.subplots(3, 3, figsize=(18, 16))

    # Row 1, Col 1: 2D Phase - Blocked beam (masked by intensity)
    ax9c_1 = axes9c[0, 0]
    phase_masked_blocked = np.where(I_hires_self_norm > 0.01, phase_focus_hires, np.nan)
    im9c_1 = ax9c_1.imshow(phase_masked_blocked,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax9c_1.set_title('BLOCKED Beam Phase\n(masked by I > 1%)', fontsize=11)
    ax9c_1.set_xlabel('x (mm)')
    ax9c_1.set_ylabel('y (mm)')
    plt.colorbar(im9c_1, ax=ax9c_1, label='Phase (rad)')

    # Row 1, Col 2: 2D Phase - Unblocked beam (masked by intensity)
    ax9c_2 = axes9c[0, 1]
    phase_masked_unblocked = np.where(I_hires_self_norm_ub > 0.01, phase_focus_hires_ub, np.nan)
    im9c_2 = ax9c_2.imshow(phase_masked_unblocked,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax9c_2.set_title('UNBLOCKED Beam Phase\n(masked by I > 1%)', fontsize=11)
    ax9c_2.set_xlabel('x (mm)')
    ax9c_2.set_ylabel('y (mm)')
    plt.colorbar(im9c_2, ax=ax9c_2, label='Phase (rad)')

    # Row 1, Col 3: Phase difference (where both beams have significant intensity)
    ax9c_3 = axes9c[0, 2]
    # Only show phase difference where both beams have intensity
    both_mask = (I_hires_self_norm > 0.01) & (I_hires_self_norm_ub > 0.01)
    phase_diff = np.angle(np.exp(1j * (phase_focus_hires - phase_focus_hires_ub)))
    phase_diff_masked = np.where(both_mask, phase_diff, np.nan)
    im9c_3 = ax9c_3.imshow(phase_diff_masked,
                            extent=[x_hires.min(), x_hires.max(), y_hires.min(), y_hires.max()],
                            cmap='RdBu_r', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
    ax9c_3.set_title('Phase Difference\n(Blocked - Unblocked)', fontsize=11)
    ax9c_3.set_xlabel('x (mm)')
    ax9c_3.set_ylabel('y (mm)')
    plt.colorbar(im9c_3, ax=ax9c_3, label='Δφ (rad)')

    # Row 2, Col 1: X phase cross-section comparison (unwrapped)
    ax9c_4 = axes9c[1, 0]
    ax9c_4.plot(x_hr_centered * 1e3, phase_x_unwrapped, 'b-', linewidth=2, label='Blocked X')
    ax9c_4.plot(x_hr_centered * 1e3, phase_x_unwrapped_ub, 'r--', linewidth=1.5, label='Unblocked X')
    ax9c_4.set_xlabel('x (μm)', fontsize=11)
    ax9c_4.set_ylabel('Phase (rad)', fontsize=11)
    ax9c_4.set_title('X Phase Cross-section (unwrapped)', fontsize=11)
    ax9c_4.legend(fontsize=7)
    ax9c_4.grid(True, alpha=0.3)
    ax9c_4.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])

    # Row 2, Col 2: X Phase residual comparison (after removing tilt)
    ax9c_5 = axes9c[1, 1]
    ax9c_5.plot(x_hr_centered * 1e3, phase_residual_x, 'b-', linewidth=2, label=f'Blocked X (σ={phase_std_x:.4f})')
    ax9c_5.plot(x_hr_centered * 1e3, phase_residual_x_ub, 'r--', linewidth=1.5, label=f'Unblocked X (σ={phase_std_x_ub:.4f})')
    ax9c_5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax9c_5.set_xlabel('x (μm)', fontsize=11)
    ax9c_5.set_ylabel('Phase Residual (rad)', fontsize=11)
    ax9c_5.set_title('X Phase Residual (tilt removed)', fontsize=11)
    ax9c_5.legend(fontsize=7)
    ax9c_5.grid(True, alpha=0.3)
    ax9c_5.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])

    # Row 2, Col 3: Y phase cross-section comparison (unwrapped)
    ax9c_6 = axes9c[1, 2]
    ax9c_6.plot(y_hr_centered * 1e3, phase_y_unwrapped, 'b-', linewidth=2, label='Blocked Y')
    ax9c_6.plot(y_hr_centered * 1e3, phase_y_unwrapped_ub, 'r--', linewidth=1.5, label='Unblocked Y')
    ax9c_6.set_xlabel('y (μm)', fontsize=11)
    ax9c_6.set_ylabel('Phase (rad)', fontsize=11)
    ax9c_6.set_title('Y Phase Cross-section (unwrapped)', fontsize=11)
    ax9c_6.legend(fontsize=7)
    ax9c_6.grid(True, alpha=0.3)
    ax9c_6.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])

    # Row 3, Col 1: Y Phase residual comparison (after removing tilt)
    ax9c_7 = axes9c[2, 0]
    ax9c_7.plot(y_hr_centered * 1e3, phase_residual_y, 'b-', linewidth=2, label=f'Blocked Y (σ={phase_std_y:.4f})')
    ax9c_7.plot(y_hr_centered * 1e3, phase_residual_y_ub, 'r--', linewidth=1.5, label=f'Unblocked Y (σ={phase_std_y_ub:.4f})')
    ax9c_7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax9c_7.set_xlabel('y (μm)', fontsize=11)
    ax9c_7.set_ylabel('Phase Residual (rad)', fontsize=11)
    ax9c_7.set_title('Y Phase Residual (tilt removed)', fontsize=11)
    ax9c_7.legend(fontsize=7)
    ax9c_7.grid(True, alpha=0.3)
    ax9c_7.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])

    # Row 3, Col 2: X vs Y phase comparison (blocked beam)
    ax9c_8 = axes9c[2, 1]
    ax9c_8.plot(x_hr_centered * 1e3, phase_residual_x, 'b-', linewidth=2, label='Blocked X')
    ax9c_8.plot(y_hr_centered * 1e3, phase_residual_y, 'b--', linewidth=1.5, label='Blocked Y')
    ax9c_8.plot(x_hr_centered * 1e3, phase_residual_x_ub, 'r-', linewidth=2, label='Unblocked X')
    ax9c_8.plot(y_hr_centered * 1e3, phase_residual_y_ub, 'r--', linewidth=1.5, label='Unblocked Y')
    ax9c_8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax9c_8.set_xlabel('x/y (μm)', fontsize=11)
    ax9c_8.set_ylabel('Phase Residual (rad)', fontsize=11)
    ax9c_8.set_title('X vs Y Phase Residual Comparison', fontsize=11)
    ax9c_8.legend(fontsize=7, ncol=2)
    ax9c_8.grid(True, alpha=0.3)
    ax9c_8.set_xlim([-plot_range_hr * 1e3, plot_range_hr * 1e3])

    # Row 3, Col 3: Phase analysis summary
    ax9c_9 = axes9c[2, 2]
    ax9c_9.axis('off')

    phase_summary = f"""
    PHASE COMPARISON: BLOCKED vs UNBLOCKED
    {'='*45}

    Grid Resolution:
      dx = {dx_focus*1e3:.3f} μm ({dx/dx_focus:.0f}x zoom)

    BLOCKED BEAM (r={aperture_radius:.0f}mm):
      Focus: z = {true_focus_z:.2f} mm
      X tilt: {coeffs_x[0]*1e3:.4f} rad/μm
      Y tilt: {coeffs_y[0]*1e3:.4f} rad/μm
      X RMS: {phase_std_x:.4f} rad
      Y RMS: {phase_std_y:.4f} rad
      Wavefront X: {phase_std_x*wavelength/(2*np.pi)*1e6:.2f} nm
      Wavefront Y: {phase_std_y*wavelength/(2*np.pi)*1e6:.2f} nm

    UNBLOCKED BEAM:
      Focus: z = {true_focus_z_unblocked:.2f} mm
      X tilt: {coeffs_x_ub[0]*1e3:.4f} rad/μm
      Y tilt: {coeffs_y_ub[0]*1e3:.4f} rad/μm
      X RMS: {phase_std_x_ub:.4f} rad
      Y RMS: {phase_std_y_ub:.4f} rad

    Interpretation:
      • At focus, phase should be flat
      • Blocked beam shows diffraction
      • RMS < λ/14 (≈{wavelength*1e6/14:.0f}nm) is
        diffraction-limited
    {'='*45}
    """
    ax9c_9.text(0.02, 0.98, phase_summary, transform=ax9c_9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig9c.suptitle(f'High-Resolution Phase Comparison at Focus (dx = {dx_focus*1e3:.3f} μm)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'unblocked_beam_focus_phase_highres_{_m2_tag}.png', dpi=300)
    print("High-resolution phase figure saved to 'unblocked_beam_focus_phase_highres.png'")

    # =============================================================================
    # MULTI-MASK OPTICAL COMPARISON — Computation
    # =============================================================================
    TIMER.start_section("Multi-mask comparison computation")
    print("\n" + "="*60)
    print("MULTI-MASK OPTICAL COMPARISON")
    print("="*60)

    mask_disp = {
        'none':     ('Unblocked', '#6E6E6E'),
        'circular': ('Circular',  '#2F6F9F'),
        'twosided': ('Two-side',  '#B3262E'),
        'diagonal': ('Diagonal',  '#218C4A'),
    }
    mc_masks_all = ['none', 'circular', 'twosided', 'diagonal']
    mc_data = {}

    # Downsample factor for 2D focus maps (4096 -> 1024)
    N_mc_2d = 1024
    ds_mc = max(N_focus // N_mc_2d, 1)
    extent_focus_mc = [x_hires.min()*1e3, x_hires.max()*1e3,
                       y_hires.min()*1e3, y_hires.max()*1e3]

    # --- Unblocked (from existing variables) ---
    ub_fwhm_x_mc = np.sum(I_x_hires_ub > 0.5) * dx_focus
    ub_fwhm_y_mc = np.sum(I_y_hires_ub > 0.5) * dx_focus
    ub_w0_x_mc = np.sum(I_x_hires_ub > 1/np.e**2) * dx_focus / 2
    ub_w0_y_mc = np.sum(I_y_hires_ub > 1/np.e**2) * dx_focus / 2

    mc_data['none'] = {
        'xz_I': xz_intensity_unblocked,
        'yz_I': yz_intensity_unblocked,
        'xz_gouy': xz_gouy_phase_unblocked,
        'yz_gouy': yz_gouy_phase_unblocked,
        'true_focus_z': true_focus_z_unblocked,
        'trans': 100.0,
        'focus_I_x': I_x_hires_ub,
        'focus_I_y': I_y_hires_ub,
        'focus_pres_x': phase_residual_x_ub,
        'focus_pres_y': phase_residual_y_ub,
        'fwhm_x': ub_fwhm_x_mc, 'fwhm_y': ub_fwhm_y_mc,
        'w0_x': ub_w0_x_mc, 'w0_y': ub_w0_y_mc,
        'I_peak': I_peak_unblocked_hires,
        'I_2d': (I_focus_hires_ub / I_peak_unblocked_hires)[::ds_mc, ::ds_mc],
        'phase_2d': phase_focus_hires_ub[::ds_mc, ::ds_mc],
    }
    print(f"  none (unblocked): focus={true_focus_z_unblocked:.3f}mm, FWHM={ub_fwhm_x_mc*1e3:.1f}x{ub_fwhm_y_mc*1e3:.1f}um")

    # --- Circular (from existing variables) ---
    mc_data['circular'] = {
        'xz_I': xz_intensity_hires,
        'yz_I': yz_intensity_hires,
        'xz_gouy': xz_gouy_phase_hires,
        'yz_gouy': yz_gouy_phase_hires,
        'true_focus_z': true_focus_z,
        'trans': compute_mask_transmission('circular', mask_params, field_at_aperture, X, Y) * 100,
        'focus_I_x': I_x_hires,
        'focus_I_y': I_y_hires,
        'focus_pres_x': phase_residual_x,
        'focus_pres_y': phase_residual_y,
        'fwhm_x': fwhm_x_hr, 'fwhm_y': fwhm_y_hr,
        'w0_x': w_x_hr, 'w0_y': w_y_hr,
        'I_peak': I_peak_blocked_hires,
        'I_2d': (I_focus_hires / I_peak_blocked_hires)[::ds_mc, ::ds_mc],
        'phase_2d': phase_focus_hires[::ds_mc, ::ds_mc],
    }
    print(f"  circular: focus={true_focus_z:.3f}mm, trans={aperture_transmission:.1f}%, FWHM={fwhm_x_hr*1e3:.1f}x{fwhm_y_hr*1e3:.1f}um")

    # --- Twosided and Diagonal (new propagation needed) ---
    for mc_mtype in ['twosided', 'diagonal']:
        print(f"\n  Computing {mc_mtype} mask...")
        mc_mask = build_mask(X, Y, mc_mtype, mask_params)
        mc_trans = compute_mask_transmission(mc_mtype, mask_params, field_at_aperture, X, Y) * 100
        print(f"    Transmission: {mc_trans:.1f}%")

        # Propagate single field through mask -> lens
        f_masked = field_at_aperture * mc_mask
        mc_field_after_lens = propagate_field(f_masked, aperture_distance_before_lens, k)
        mc_field_after_lens = thin_lens(mc_field_after_lens, R2, focal_length, k)
        del mc_mask

        # Find focus
        mc_focus_z, _ = find_true_focus(mc_field_after_lens, z_search, k, center_idx, dx, verbose=True)
        print(f"    Focus: z = {mc_focus_z:.3f} mm (shift: {mc_focus_z - focal_length:.3f} mm)")

        # x-z / y-z propagation (200 z-steps)
        mc_xz_I = np.zeros((n_z_steps_hr, N_xz_focus))
        mc_yz_I = np.zeros((n_z_steps_hr, N_xz_focus))
        mc_xz_gouy = np.zeros((n_z_steps_hr, N_xz_focus))
        mc_yz_gouy = np.zeros((n_z_steps_hr, N_xz_focus))

        for mc_i, mc_z in enumerate(z_focus_prop):
            if mc_i % 50 == 0:
                print(f"    z = {mc_z:.2f} mm ({mc_i+1}/{n_z_steps_hr})")
            mc_chr = N_xz_focus // 2
            mc_fz, _, _ = fresnel_propagate_zoom(mc_field_after_lens, mc_z, k, L, L_xz_focus, N_xz_focus)
            mc_xz_I[mc_i, :] = np.abs(mc_fz[mc_chr, :])**2
            mc_yz_I[mc_i, :] = np.abs(mc_fz[:, mc_chr])**2
            mc_fz_np = mc_fz * np.exp(-1j * k * mc_z)
            mc_xz_gouy[mc_i, :] = np.angle(mc_fz_np[mc_chr, :])
            mc_yz_gouy[mc_i, :] = np.angle(mc_fz_np[:, mc_chr])

        # Focus plane high-res field
        mc_ff, _, _ = fresnel_propagate_zoom(mc_field_after_lens, mc_focus_z, k, L, L_focus, N_focus)
        mc_If = np.abs(mc_ff)**2
        mc_Ip = max(mc_If.max(), 1e-30)
        mc_In = mc_If / mc_Ip
        mc_cfc = N_focus // 2

        mc_Ix = mc_In[mc_cfc, :]
        mc_Iy = mc_In[:, mc_cfc]
        mc_fwhm_x = np.sum(mc_Ix > 0.5) * dx_focus
        mc_fwhm_y = np.sum(mc_Iy > 0.5) * dx_focus
        mc_w0x = np.sum(mc_Ix > 1/np.e**2) * dx_focus / 2
        mc_w0y = np.sum(mc_Iy > 1/np.e**2) * dx_focus / 2

        # Phase at focus: unwrap and remove tilt
        mc_ph = np.angle(mc_ff)
        mc_phx = np.unwrap(mc_ph[mc_cfc, :])
        mc_phy = np.unwrap(mc_ph[:, mc_cfc])
        mc_imx = mc_Ix > 0.1
        mc_imy = mc_Iy > 0.1
        if np.sum(mc_imx) > 10:
            mc_cx = np.polyfit(x_hires[mc_imx], mc_phx[mc_imx], 1)
            mc_pres_x = mc_phx - np.polyval(mc_cx, x_hires)
        else:
            mc_pres_x = mc_phx
        if np.sum(mc_imy) > 10:
            mc_cy = np.polyfit(y_hires[mc_imy], mc_phy[mc_imy], 1)
            mc_pres_y = mc_phy - np.polyval(mc_cy, y_hires)
        else:
            mc_pres_y = mc_phy

        mc_data[mc_mtype] = {
            'xz_I': mc_xz_I,
            'yz_I': mc_yz_I,
            'xz_gouy': mc_xz_gouy,
            'yz_gouy': mc_yz_gouy,
            'true_focus_z': mc_focus_z,
            'trans': mc_trans,
            'focus_I_x': mc_Ix.copy(),
            'focus_I_y': mc_Iy.copy(),
            'focus_pres_x': mc_pres_x.copy(),
            'focus_pres_y': mc_pres_y.copy(),
            'fwhm_x': mc_fwhm_x, 'fwhm_y': mc_fwhm_y,
            'w0_x': mc_w0x, 'w0_y': mc_w0y,
            'I_peak': mc_Ip,
            'I_2d': mc_In[::ds_mc, ::ds_mc].copy(),
            'phase_2d': mc_ph[::ds_mc, ::ds_mc].copy(),
        }

        del mc_field_after_lens, mc_ff, mc_If, mc_In, mc_ph
        gc.collect()
        print(f"    FWHM: {mc_fwhm_x*1e3:.1f} x {mc_fwhm_y*1e3:.1f} um, w0: {mc_w0x*1e3:.1f} x {mc_w0y*1e3:.1f} um")

    print("\n  Multi-mask computation complete.")
    print(f"  {'Mask':12s} {'Trans%':>7s} {'Focus(mm)':>10s} {'FWHM_x(um)':>11s} {'FWHM_y(um)':>11s} {'w0_x(um)':>9s} {'w0_y(um)':>9s}")
    for mc_mn in mc_masks_all:
        d = mc_data[mc_mn]
        print(f"  {mc_mn:12s} {d['trans']:7.1f} {d['true_focus_z']:10.3f} {d['fwhm_x']*1e3:11.2f} {d['fwhm_y']*1e3:11.2f} {d['w0_x']*1e3:9.2f} {d['w0_y']*1e3:9.2f}")

    # =============================================================================
    # Figure MC-1: x-z / y-z Propagation Comparison (zoomed, per-column norm)
    # =============================================================================
    TIMER.start_section("Figure MC-1 - Propagation comparison")
    print("\nGenerating MC-1: x-z / y-z propagation comparison (zoomed)...")

    from matplotlib.gridspec import GridSpec

    mc_col_order = ['circular', 'twosided', 'diagonal', 'none']

    # Zoom to ±100 μm
    mc_zoom_um = 100
    mc_zoom_mm = mc_zoom_um * 1e-3
    mc_x_lo = np.searchsorted(x_xz_hires, -mc_zoom_mm)
    mc_x_hi = np.searchsorted(x_xz_hires, mc_zoom_mm)
    mc_x_sl = slice(mc_x_lo, mc_x_hi)
    mc_extent_zoom = [-mc_zoom_um, mc_zoom_um, z_focus_prop.min(), z_focus_prop.max()]
    mc_x_um_zoom = x_xz_hires[mc_x_sl] * 1e3  # μm

    # Per-column intensity max (for independent normalization)
    mc_vmax_col = {}
    for mname in mc_col_order:
        xz_crop = mc_data[mname]['xz_I'][:, mc_x_sl]
        yz_crop = mc_data[mname]['yz_I'][:, mc_x_sl]
        mc_vmax_col[mname] = max(xz_crop.max(), yz_crop.max())

    # Layout: 6 rows x 4 cols  (imshow / cross-section / Gouy) x (xz, yz)
    fig_mc1 = plt.figure(figsize=(24, 30))
    gs_mc1 = GridSpec(6, 4, figure=fig_mc1, height_ratios=[3, 1, 2, 3, 1, 2],
                      hspace=0.25, wspace=0.15)

    for col_idx, mname in enumerate(mc_col_order):
        label_m, color_m = mask_disp[mname]
        d = mc_data[mname]
        vmax_I = mc_vmax_col[mname]
        fz = d['true_focus_z']
        fz_idx = np.argmin(np.abs(z_focus_prop - fz))

        for block, (I_key, gouy_key, dir_label) in enumerate([
            ('xz_I', 'xz_gouy', 'x'),
            ('yz_I', 'yz_gouy', 'y'),
        ]):
            row_base = block * 3   # 0 for xz, 3 for yz

            # --- Intensity imshow ---
            ax = fig_mc1.add_subplot(gs_mc1[row_base, col_idx])
            data_I = d[I_key][:, mc_x_sl]
            im = ax.imshow(data_I, extent=mc_extent_zoom, origin='lower', aspect='auto',
                           cmap='hot', vmin=0, vmax=vmax_I, interpolation='bicubic')
            ax.axhline(focal_length, color='cyan', ls='--', lw=0.8, alpha=0.6)
            ax.axhline(fz, color='lime', ls=':', lw=0.8, alpha=0.8)
            if row_base == 0:
                ax.set_title(f'{label_m} ({d["trans"]:.0f}%)', fontsize=12, color=color_m)
            if col_idx == 0:
                ax.set_ylabel(f'I({dir_label},z)\nz (mm)', fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            if col_idx == 3:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # --- Cross-section at focus ---
            ax = fig_mc1.add_subplot(gs_mc1[row_base + 1, col_idx])
            cross = d[I_key][fz_idx, mc_x_sl]
            cross_max = cross.max() if cross.max() > 0 else 1.0
            cross_norm = cross / cross_max
            ax.plot(mc_x_um_zoom, cross_norm, '-', color=color_m, lw=1.5)
            ax.axhline(0.5, color='gray', ls='--', lw=0.5, alpha=0.5)
            ax.axhline(1/np.e**2, color='gray', ls=':', lw=0.5, alpha=0.5)
            ax.set_xlim(-mc_zoom_um, mc_zoom_um)
            ax.set_ylim(-0.05, 1.1)
            if col_idx == 0:
                ax.set_ylabel(f'I({dir_label}) norm', fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(True, alpha=0.2)

            # --- Gouy phase ---
            ax = fig_mc1.add_subplot(gs_mc1[row_base + 2, col_idx])
            data_gouy = d[gouy_key][:, mc_x_sl]
            im = ax.imshow(data_gouy, extent=mc_extent_zoom, origin='lower', aspect='auto',
                           cmap='twilight', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
            ax.axhline(focal_length, color='white', ls='--', lw=0.8, alpha=0.6)
            ax.axhline(fz, color='lime', ls=':', lw=0.8, alpha=0.8)
            if col_idx == 0:
                ax.set_ylabel(f'Phase({dir_label},z)\nz (mm)', fontsize=10)
            else:
                ax.set_yticklabels([])
            if row_base + 2 == 5:
                ax.set_xlabel(f'{dir_label} (μm)', fontsize=10)
            else:
                ax.set_xticklabels([])
            if col_idx == 3:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_mc1.suptitle(f'Multi-Mask Propagation Comparison Near Focus (±{mc_zoom_um}μm, per-column norm)',
                     fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'mask_comparison_xz_yz_propagation_{_m2_tag}.png', dpi=400)
    print(f"  Saved: mask_comparison_xz_yz_propagation_{_m2_tag}.png")

    # =============================================================================
    # Figure MC-2: Focus Intensity Comparison
    # =============================================================================
    TIMER.start_section("Figure MC-2 - Focus intensity comparison")
    print("Generating MC-2: focus intensity comparison...")

    fig_mc2, axes_mc2 = plt.subplots(5, 4, figsize=(22, 26))
    mc_x_um = x_hires * 1e3
    mc_y_um = y_hires * 1e3
    mc_I_peak_ub = mc_data['none']['I_peak']  # reference for absolute normalization

    # Row 1: 2D intensity (log scale, self-normalized)
    for col_idx, mname in enumerate(mc_col_order):
        ax = axes_mc2[0, col_idx]
        I2d = mc_data[mname]['I_2d']
        I2d_log = np.log10(np.clip(I2d, 1e-6, None))
        ax.imshow(I2d_log, extent=extent_focus_mc, origin='lower', cmap='hot',
                  vmin=-4, vmax=0, interpolation='bicubic')
        label, color = mask_disp[mname]
        ratio_str = f'  ({mc_data[mname]["I_peak"]/mc_I_peak_ub:.2f}x)' if mname != 'none' else ''
        ax.set_title(f'{label}\nFWHM: {mc_data[mname]["fwhm_x"]*1e3:.1f}x{mc_data[mname]["fwhm_y"]*1e3:.1f} um{ratio_str}',
                     fontsize=10, color=color)
        plot_hw = max(mc_data[mname]['fwhm_x'], mc_data[mname]['fwhm_y']) * 5 * 1e3
        plot_hw = max(plot_hw, 50)
        ax.set_xlim(-plot_hw, plot_hw)
        ax.set_ylim(-plot_hw, plot_hw)
        if col_idx == 0:
            ax.set_ylabel('y (um)')
        ax.set_xlabel('x (um)')

    # Row 2: x cross-sections (self-normalized shape)
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc2[1, col_idx]
        d = mc_data[mname]
        ax.plot(mc_x_um, d['focus_I_x'], color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_x_um, mc_data['none']['focus_I_x'], color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('I / I_max (self-norm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'x cross-section (shape)', fontsize=10)

    ax = axes_mc2[1, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(mc_x_um, mc_data[mname]['focus_I_x'], color=color, linewidth=1.5, label=label)
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('I / I_max (self-norm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('x shape overlay', fontsize=10)

    # Row 3: y cross-sections (self-normalized shape)
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc2[2, col_idx]
        d = mc_data[mname]
        ax.plot(mc_y_um, d['focus_I_y'], color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_y_um, mc_data['none']['focus_I_y'], color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('y (um)')
        ax.set_ylabel('I / I_max (self-norm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'y cross-section (shape)', fontsize=10)

    ax = axes_mc2[2, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(mc_y_um, mc_data[mname]['focus_I_y'], color=color, linewidth=1.5, label=label)
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('y (um)')
    ax.set_ylabel('I / I_max (self-norm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('y shape overlay', fontsize=10)

    # Row 4: x cross-sections (ABSOLUTE — all normalized to unblocked peak)
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc2[3, col_idx]
        d = mc_data[mname]
        I_abs_x = d['focus_I_x'] * d['I_peak'] / mc_I_peak_ub
        I_abs_x_ub = mc_data['none']['focus_I_x']  # already =1 at peak
        ax.plot(mc_x_um, I_abs_x, color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_x_um, I_abs_x_ub, color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('I / I_peak,unblocked')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'x absolute intensity', fontsize=10)

    ax = axes_mc2[3, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        d = mc_data[mname]
        I_abs_x = d['focus_I_x'] * d['I_peak'] / mc_I_peak_ub
        ax.plot(mc_x_um, I_abs_x, color=color, linewidth=1.5, label=f'{label} ({d["I_peak"]/mc_I_peak_ub:.2f}x)')
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('I / I_peak,unblocked')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('x absolute overlay', fontsize=10)

    # Row 5: y cross-sections (ABSOLUTE — all normalized to unblocked peak)
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc2[4, col_idx]
        d = mc_data[mname]
        I_abs_y = d['focus_I_y'] * d['I_peak'] / mc_I_peak_ub
        I_abs_y_ub = mc_data['none']['focus_I_y']
        ax.plot(mc_y_um, I_abs_y, color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_y_um, I_abs_y_ub, color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel('y (um)')
        ax.set_ylabel('I / I_peak,unblocked')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'y absolute intensity', fontsize=10)

    ax = axes_mc2[4, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        d = mc_data[mname]
        I_abs_y = d['focus_I_y'] * d['I_peak'] / mc_I_peak_ub
        ax.plot(mc_y_um, I_abs_y, color=color, linewidth=1.5, label=f'{label} ({d["I_peak"]/mc_I_peak_ub:.2f}x)')
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('y (um)')
    ax.set_ylabel('I / I_peak,unblocked')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('y absolute overlay', fontsize=10)

    fig_mc2.suptitle('Multi-Mask Focus Intensity Comparison', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'mask_comparison_focus_intensity_{_m2_tag}.png', dpi=400)
    print(f"  Saved: mask_comparison_focus_intensity_{_m2_tag}.png")

    # =============================================================================
    # Figure MC-3: Focus Phase Comparison
    # =============================================================================
    TIMER.start_section("Figure MC-3 - Focus phase comparison")
    print("Generating MC-3: focus phase comparison...")

    fig_mc3, axes_mc3 = plt.subplots(3, 4, figsize=(22, 16))

    # Row 1: 2D phase maps (masked by intensity)
    for col_idx, mname in enumerate(mc_col_order):
        ax = axes_mc3[0, col_idx]
        I2d = mc_data[mname]['I_2d']
        ph2d = mc_data[mname]['phase_2d'].copy()
        ph2d[I2d < 0.01] = np.nan
        ax.imshow(ph2d, extent=extent_focus_mc, origin='lower', cmap='twilight',
                  vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
        label, color = mask_disp[mname]
        ax.set_title(f'{label} phase at focus', fontsize=10, color=color)
        plot_hw = max(mc_data[mname]['fwhm_x'], mc_data[mname]['fwhm_y']) * 5 * 1e3
        plot_hw = max(plot_hw, 50)
        ax.set_xlim(-plot_hw, plot_hw)
        ax.set_ylim(-plot_hw, plot_hw)
        if col_idx == 0:
            ax.set_ylabel('y (um)')
        ax.set_xlabel('x (um)')

    # Row 2: x residual phase cross-sections
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc3[1, col_idx]
        d = mc_data[mname]
        ax.plot(mc_x_um, d['focus_pres_x'], color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_x_um, mc_data['none']['focus_pres_x'], color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('Residual phase (rad)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('x phase (tilt removed)', fontsize=10)

    # Row 2, Col 3: overlay
    ax = axes_mc3[1, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(mc_x_um, mc_data[mname]['focus_pres_x'], color=color, linewidth=1.5, label=label)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('Residual phase (rad)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('x phase overlay', fontsize=10)

    # Row 3: y residual phase cross-sections
    for col_idx, mname in enumerate(mc_col_order[:3]):
        ax = axes_mc3[2, col_idx]
        d = mc_data[mname]
        ax.plot(mc_y_um, d['focus_pres_y'], color=mask_disp[mname][1], linewidth=1.5, label=mask_disp[mname][0])
        ax.plot(mc_y_um, mc_data['none']['focus_pres_y'], color='gray', linewidth=1, alpha=0.5, label='Unblocked')
        ax.set_xlim(-100, 100)
        ax.set_xlabel('y (um)')
        ax.set_ylabel('Residual phase (rad)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('y phase (tilt removed)', fontsize=10)

    # Row 3, Col 3: overlay
    ax = axes_mc3[2, 3]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(mc_y_um, mc_data[mname]['focus_pres_y'], color=color, linewidth=1.5, label=label)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('y (um)')
    ax.set_ylabel('Residual phase (rad)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('y phase overlay', fontsize=10)

    fig_mc3.suptitle('Multi-Mask Focus Phase Comparison (residual after tilt removal)', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'mask_comparison_focus_phase_{_m2_tag}.png', dpi=400)
    print(f"  Saved: mask_comparison_focus_phase_{_m2_tag}.png")

    # =============================================================================
    # Figure MC-4: On-Axis Diagnostics
    # =============================================================================
    TIMER.start_section("Figure MC-4 - On-axis diagnostics")
    print("Generating MC-4: on-axis diagnostics...")

    fig_mc4, axes_mc4 = plt.subplots(2, 3, figsize=(18, 12))

    # Extract on-axis quantities and beam widths from stored cross-section data
    mc_center_xz = np.argmin(np.abs(x_xz_hires))
    mc_onaxis = {}
    for mname in mc_masks_all:
        d = mc_data[mname]
        xz_center = d['xz_I'][:, mc_center_xz]
        yz_center = d['yz_I'][:, mc_center_xz]
        # On-axis intensity (x-z and y-z should be same on-axis, use xz)
        I_onaxis = xz_center / max(xz_center.max(), 1e-30)
        # On-axis Gouy phase (unwrap along z)
        gouy_onaxis = np.unwrap(d['xz_gouy'][:, mc_center_xz])
        # Gouy gradient
        dgouy_dz = np.gradient(gouy_onaxis, z_focus_prop)  # rad/mm
        # Beam FWHM_x vs z (interpolated half-max crossings for smooth curves)
        x_xz_um = x_xz_hires * 1e3  # um
        fwhm_x_z = np.zeros(n_z_steps_hr)
        fwhm_y_z = np.zeros(n_z_steps_hr)
        for i in range(n_z_steps_hr):
            for axis_label, I_row, fwhm_arr in [('x', d['xz_I'][i, :], fwhm_x_z),
                                                  ('y', d['yz_I'][i, :], fwhm_y_z)]:
                Imax = I_row.max()
                if Imax <= 0:
                    continue
                I_norm = I_row / Imax
                half = 0.5
                above = I_norm >= half
                # Find first and last crossing by interpolation
                diffs = np.diff(above.astype(int))
                rises = np.where(diffs == 1)[0]   # crossing upward
                falls = np.where(diffs == -1)[0]  # crossing downward
                if len(rises) > 0 and len(falls) > 0:
                    # Interpolate left edge (first rise)
                    j = rises[0]
                    frac_l = (half - I_norm[j]) / (I_norm[j+1] - I_norm[j])
                    x_left = x_xz_um[j] + frac_l * (x_xz_um[j+1] - x_xz_um[j])
                    # Interpolate right edge (last fall)
                    j = falls[-1]
                    frac_r = (half - I_norm[j]) / (I_norm[j+1] - I_norm[j])
                    x_right = x_xz_um[j] + frac_r * (x_xz_um[j+1] - x_xz_um[j])
                    fwhm_arr[i] = abs(x_right - x_left)
                else:
                    fwhm_arr[i] = np.sum(above) * (x_xz_um[1] - x_xz_um[0])
        mc_onaxis[mname] = {
            'I': I_onaxis, 'gouy': gouy_onaxis, 'dgouy': dgouy_dz,
            'fwhm_x': fwhm_x_z, 'fwhm_y': fwhm_y_z,
        }

    # (0,0) On-axis intensity vs z
    ax = axes_mc4[0, 0]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(z_focus_prop, mc_onaxis[mname]['I'], color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('I_on-axis (normalized)')
    ax.set_title('On-Axis Intensity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) On-axis Gouy phase vs z
    ax = axes_mc4[0, 1]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(z_focus_prop, mc_onaxis[mname]['gouy'], color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Gouy phase (rad)')
    ax.set_title('On-Axis Gouy Phase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) dGouy/dz vs z
    ax = axes_mc4[0, 2]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(z_focus_prop, mc_onaxis[mname]['dgouy'], color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('d(Gouy)/dz (rad/mm)')
    ax.set_title('Gouy Phase Gradient (key for HHG)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) FWHM_x vs z
    ax = axes_mc4[1, 0]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(z_focus_prop, mc_onaxis[mname]['fwhm_x'], color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('FWHM_x (um)')
    ax.set_title('Beam Width in x vs z')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) FWHM_y vs z
    ax = axes_mc4[1, 1]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        ax.plot(z_focus_prop, mc_onaxis[mname]['fwhm_y'], color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('FWHM_y (um)')
    ax.set_title('Beam Width in y vs z')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) FWHM_y / FWHM_x ratio vs z
    ax = axes_mc4[1, 2]
    for mname in mc_masks_all:
        label, color = mask_disp[mname]
        fwhm_x_z = mc_onaxis[mname]['fwhm_x']
        fwhm_y_z = mc_onaxis[mname]['fwhm_y']
        ratio = np.where(fwhm_x_z > 0, fwhm_y_z / fwhm_x_z, 1.0)
        ax.plot(z_focus_prop, ratio, color=color, linewidth=1.5, label=label)
    ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('FWHM_y / FWHM_x')
    ax.set_title('Beam Asymmetry Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig_mc4.suptitle('Multi-Mask On-Axis Diagnostics', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'mask_comparison_onaxis_diagnostics_{_m2_tag}.png', dpi=400)
    print(f"  Saved: mask_comparison_onaxis_diagnostics_{_m2_tag}.png")

    # =============================================================================
    # Figure MC-5: Beam Parameters Summary
    # =============================================================================
    TIMER.start_section("Figure MC-5 - Beam parameters summary")
    print("Generating MC-5: beam parameters summary...")

    fig_mc5, axes_mc5 = plt.subplots(2, 3, figsize=(16, 10))
    mc_bar_x = np.arange(len(mc_masks_all))
    mc_bar_w = 0.35
    mc_bar_colors = [mask_disp[m][1] for m in mc_masks_all]
    mc_bar_labels = [mask_disp[m][0] for m in mc_masks_all]

    # (0,0) Peak intensity at focus
    ax = axes_mc5[0, 0]
    peaks = [mc_data[m]['I_peak'] for m in mc_masks_all]
    peaks_norm = [p / mc_data['none']['I_peak'] for p in peaks]
    bars = ax.bar(mc_bar_x, peaks_norm, color=mc_bar_colors, alpha=0.8)
    for b, v in zip(bars, peaks_norm):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('Peak I / I_unblocked')
    ax.set_title('Peak Intensity at Focus')
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1) FWHM x and y (grouped bars)
    ax = axes_mc5[0, 1]
    fwhm_x_vals = [mc_data[m]['fwhm_x']*1e3 for m in mc_masks_all]
    fwhm_y_vals = [mc_data[m]['fwhm_y']*1e3 for m in mc_masks_all]
    b1 = ax.bar(mc_bar_x - mc_bar_w/2, fwhm_x_vals, mc_bar_w, color=mc_bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    b2 = ax.bar(mc_bar_x + mc_bar_w/2, fwhm_y_vals, mc_bar_w, color=mc_bar_colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('FWHM (um)')
    ax.set_title('FWHM at Focus (solid=x, hatched=y)')
    ax.grid(True, alpha=0.3, axis='y')

    # (0,2) 1/e^2 radius x and y
    ax = axes_mc5[0, 2]
    w0x_vals = [mc_data[m]['w0_x']*1e3 for m in mc_masks_all]
    w0y_vals = [mc_data[m]['w0_y']*1e3 for m in mc_masks_all]
    ax.bar(mc_bar_x - mc_bar_w/2, w0x_vals, mc_bar_w, color=mc_bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(mc_bar_x + mc_bar_w/2, w0y_vals, mc_bar_w, color=mc_bar_colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('w0 (um)')
    ax.set_title('1/e^2 Radius at Focus (solid=x, hatched=y)')
    ax.grid(True, alpha=0.3, axis='y')

    # (1,0) Rayleigh range estimate from on-axis intensity (half-max width in z)
    ax = axes_mc5[1, 0]
    zR_vals = []
    for mname in mc_masks_all:
        I_oa = mc_onaxis[mname]['I']
        half = 0.5
        above = I_oa > half
        zR_est = np.sum(above) * (z_focus_prop[1] - z_focus_prop[0]) / 2  # half-width
        zR_vals.append(zR_est)
    bars = ax.bar(mc_bar_x, zR_vals, color=mc_bar_colors, alpha=0.8)
    for b, v in zip(bars, zR_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('Rayleigh range (mm)')
    ax.set_title('Confocal Parameter (on-axis half-max)')
    ax.grid(True, alpha=0.3, axis='y')

    # (1,1) Focus shift from geometric focus
    ax = axes_mc5[1, 1]
    shifts = [(mc_data[m]['true_focus_z'] - focal_length)*1e3 for m in mc_masks_all]  # in um
    bars = ax.bar(mc_bar_x, shifts, color=mc_bar_colors, alpha=0.8)
    for b, v in zip(bars, shifts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5 if v >= 0 else v - 2,
                f'{v:.1f}', ha='center', fontsize=9)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('Focus shift (um)')
    ax.set_title('Focus Shift from f')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,2) Transmission
    ax = axes_mc5[1, 2]
    trans_vals = [mc_data[m]['trans'] for m in mc_masks_all]
    bars = ax.bar(mc_bar_x, trans_vals, color=mc_bar_colors, alpha=0.8)
    for b, v in zip(bars, trans_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.1f}%',
                ha='center', fontsize=9)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=9)
    ax.set_ylabel('Transmission (%)')
    ax.set_title('Power Transmission')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    fig_mc5.suptitle('Multi-Mask Beam Parameters Summary', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'mask_comparison_beam_parameters_{_m2_tag}.png', dpi=400)
    print(f"  Saved: mask_comparison_beam_parameters_{_m2_tag}.png")

    # =============================================================================
    # Figures MC-6a/b/c/d: Per-Mask High-Resolution x-z/y-z Propagation (like Fig 2)
    # =============================================================================
    # Reference unblocked data from mc_data
    mc_ref = mc_data['none']
    mc_ref_xz_ext = [x_xz_hires.min()*1e3, x_xz_hires.max()*1e3,
                     z_focus_prop.min(), z_focus_prop.max()]
    mc_ref_x_um = x_hires * 1e3  # focus-plane coordinates (N_focus)
    mc_ref_y_um = y_hires * 1e3

    for mc6_mname in ['circular', 'twosided', 'diagonal', 'none']:
        TIMER.start_section(f"Figure MC-6 - {mc6_mname} propagation detail")
        mc6_label, mc6_color = mask_disp[mc6_mname]
        print(f"\nGenerating MC-6: High-res propagation for {mc6_label}...")

        d_m = mc_data[mc6_mname]
        d_ub = mc_data['none']
        fz_m = d_m['true_focus_z']
        fz_ub = d_ub['true_focus_z']
        focus_idx_m = np.argmin(np.abs(z_focus_prop - fz_m))
        focus_idx_ub = np.argmin(np.abs(z_focus_prop - fz_ub))

        # Intensity threshold for Gouy phase masking
        gouy_thresh_frac = 0.005

        fig_mc6, axes_mc6 = plt.subplots(4, 4, figsize=(22, 20))

        # ===== ROW 1: x-z intensity + x focus cross-section =====
        # (0,0) Masked intensity x-z
        ax = axes_mc6[0, 0]
        im = ax.imshow(d_m['xz_I'], extent=mc_ref_xz_ext, aspect='auto', cmap='hot',
                       origin='lower', interpolation='bicubic')
        ax.axhline(focal_length, color='cyan', linestyle='--', linewidth=1.5,
                   label=f'f={focal_length:.0f}mm')
        ax.axhline(fz_m, color='lime', linestyle=':', linewidth=1.5,
                   label=f'focus={fz_m:.2f}mm')
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title(f'{mc6_label}: Intensity (x-z)', fontsize=10)
        ax.legend(loc='upper right', fontsize=6)
        plt.colorbar(im, ax=ax, label='I')

        # (0,1) Masked x focus cross-section
        ax = axes_mc6[0, 1]
        I_cross_x_m = d_m['xz_I'][focus_idx_m, :]
        I_cross_x_m_n = I_cross_x_m / max(I_cross_x_m.max(), 1e-30)
        ax.plot(x_xz_hires*1e3, I_cross_x_m_n, color=mc6_color, linewidth=2,
                label=f'{mc6_label} X')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
        ax.axhline(1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
        ax.set_ylabel('Norm. Intensity', fontsize=10)
        ax.set_title(f'{mc6_label}: X Focus (z={fz_m:.2f}mm)', fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # (0,2) Unblocked intensity x-z
        ax = axes_mc6[0, 2]
        im = ax.imshow(d_ub['xz_I'], extent=mc_ref_xz_ext, aspect='auto', cmap='hot',
                       origin='lower', interpolation='bicubic')
        ax.axhline(focal_length, color='cyan', linestyle='--', linewidth=1.5,
                   label=f'f={focal_length:.0f}mm')
        ax.axhline(fz_ub, color='lime', linestyle=':', linewidth=1.5,
                   label=f'focus={fz_ub:.2f}mm')
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title('UNBLOCKED: Intensity (x-z)', fontsize=10)
        ax.legend(loc='upper right', fontsize=6)
        plt.colorbar(im, ax=ax, label='I')

        # (0,3) Unblocked x focus cross-section
        ax = axes_mc6[0, 3]
        I_cross_x_ub = d_ub['xz_I'][focus_idx_ub, :]
        I_cross_x_ub_n = I_cross_x_ub / max(I_cross_x_ub.max(), 1e-30)
        ax.plot(x_xz_hires*1e3, I_cross_x_ub_n, 'gray', linewidth=2, label='Unblocked X')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
        ax.axhline(1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
        ax.set_ylabel('Norm. Intensity', fontsize=10)
        ax.set_title(f'UNBLOCKED: X Focus (z={fz_ub:.2f}mm)', fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # ===== ROW 2: x-z phase + x phase cross-section =====
        # (1,0) Masked Gouy phase x-z
        ax = axes_mc6[1, 0]
        im = ax.imshow(d_m['xz_gouy'], extent=mc_ref_xz_ext, aspect='auto', cmap='twilight',
                       origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
        ax.axhline(focal_length, color='white', linestyle='--', linewidth=1.5)
        ax.axhline(fz_m, color='lime', linestyle=':', linewidth=1.5)
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title(f'{mc6_label}: Gouy Phase (x-z)', fontsize=10)
        plt.colorbar(im, ax=ax, label='φ (rad)')

        # (1,1) Masked x phase at focus (residual)
        ax = axes_mc6[1, 1]
        ax.plot(mc_ref_x_um, d_m['focus_pres_x'], color=mc6_color, linewidth=2,
                label=f'{mc6_label} X')
        ax.set_ylabel('Residual Phase (rad)', fontsize=10)
        ax.set_title(f'{mc6_label}: X Phase at Focus', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xlim([-200, 200])

        # (1,2) Unblocked Gouy phase x-z
        ax = axes_mc6[1, 2]
        im = ax.imshow(d_ub['xz_gouy'], extent=mc_ref_xz_ext, aspect='auto', cmap='twilight',
                       origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
        ax.axhline(focal_length, color='white', linestyle='--', linewidth=1.5)
        ax.axhline(fz_ub, color='lime', linestyle=':', linewidth=1.5)
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title('UNBLOCKED: Gouy Phase (x-z)', fontsize=10)
        plt.colorbar(im, ax=ax, label='φ (rad)')

        # (1,3) Unblocked x phase at focus
        ax = axes_mc6[1, 3]
        ax.plot(mc_ref_x_um, d_ub['focus_pres_x'], 'gray', linewidth=2, label='Unblocked X')
        ax.set_ylabel('Residual Phase (rad)', fontsize=10)
        ax.set_title('UNBLOCKED: X Phase at Focus', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xlim([-200, 200])

        # ===== ROW 3: y-z intensity + y focus cross-section =====
        # (2,0) Masked intensity y-z
        ax = axes_mc6[2, 0]
        im = ax.imshow(d_m['yz_I'], extent=mc_ref_xz_ext, aspect='auto', cmap='hot',
                       origin='lower', interpolation='bicubic')
        ax.axhline(focal_length, color='cyan', linestyle='--', linewidth=1.5,
                   label=f'f={focal_length:.0f}mm')
        ax.axhline(fz_m, color='lime', linestyle=':', linewidth=1.5,
                   label=f'focus={fz_m:.2f}mm')
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title(f'{mc6_label}: Intensity (y-z)', fontsize=10)
        ax.legend(loc='upper right', fontsize=6)
        plt.colorbar(im, ax=ax, label='I')

        # (2,1) Masked y focus cross-section
        ax = axes_mc6[2, 1]
        I_cross_y_m = d_m['yz_I'][focus_idx_m, :]
        I_cross_y_m_n = I_cross_y_m / max(I_cross_y_m.max(), 1e-30)
        ax.plot(x_xz_hires*1e3, I_cross_y_m_n, color=mc6_color, linewidth=2,
                label=f'{mc6_label} Y')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
        ax.axhline(1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
        ax.set_ylabel('Norm. Intensity', fontsize=10)
        ax.set_title(f'{mc6_label}: Y Focus (z={fz_m:.2f}mm)', fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # (2,2) Unblocked intensity y-z
        ax = axes_mc6[2, 2]
        im = ax.imshow(d_ub['yz_I'], extent=mc_ref_xz_ext, aspect='auto', cmap='hot',
                       origin='lower', interpolation='bicubic')
        ax.axhline(focal_length, color='cyan', linestyle='--', linewidth=1.5,
                   label=f'f={focal_length:.0f}mm')
        ax.axhline(fz_ub, color='lime', linestyle=':', linewidth=1.5,
                   label=f'focus={fz_ub:.2f}mm')
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title('UNBLOCKED: Intensity (y-z)', fontsize=10)
        ax.legend(loc='upper right', fontsize=6)
        plt.colorbar(im, ax=ax, label='I')

        # (2,3) Unblocked y focus cross-section
        ax = axes_mc6[2, 3]
        I_cross_y_ub = d_ub['yz_I'][focus_idx_ub, :]
        I_cross_y_ub_n = I_cross_y_ub / max(I_cross_y_ub.max(), 1e-30)
        ax.plot(x_xz_hires*1e3, I_cross_y_ub_n, 'gray', linewidth=2, label='Unblocked Y')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='FWHM')
        ax.axhline(1/np.e**2, color='orange', linestyle=':', alpha=0.5, label='1/e²')
        ax.set_ylabel('Norm. Intensity', fontsize=10)
        ax.set_title(f'UNBLOCKED: Y Focus (z={fz_ub:.2f}mm)', fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # ===== ROW 4: y-z phase + y phase cross-section =====
        # (3,0) Masked Gouy phase y-z
        ax = axes_mc6[3, 0]
        im = ax.imshow(d_m['yz_gouy'], extent=mc_ref_xz_ext, aspect='auto', cmap='twilight',
                       origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
        ax.axhline(focal_length, color='white', linestyle='--', linewidth=1.5)
        ax.axhline(fz_m, color='lime', linestyle=':', linewidth=1.5)
        ax.set_xlabel('y (μm)', fontsize=10)
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title(f'{mc6_label}: Gouy Phase (y-z)', fontsize=10)
        plt.colorbar(im, ax=ax, label='φ (rad)')

        # (3,1) Masked y phase at focus
        ax = axes_mc6[3, 1]
        ax.plot(mc_ref_y_um, d_m['focus_pres_y'], color=mc6_color, linewidth=2,
                label=f'{mc6_label} Y')
        ax.set_xlabel('y (μm)', fontsize=10)
        ax.set_ylabel('Residual Phase (rad)', fontsize=10)
        ax.set_title(f'{mc6_label}: Y Phase at Focus', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xlim([-200, 200])

        # (3,2) Unblocked Gouy phase y-z
        ax = axes_mc6[3, 2]
        im = ax.imshow(d_ub['yz_gouy'], extent=mc_ref_xz_ext, aspect='auto', cmap='twilight',
                       origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='bicubic')
        ax.axhline(focal_length, color='white', linestyle='--', linewidth=1.5)
        ax.axhline(fz_ub, color='lime', linestyle=':', linewidth=1.5)
        ax.set_xlabel('y (μm)', fontsize=10)
        ax.set_ylabel('z (mm)', fontsize=10)
        ax.set_title('UNBLOCKED: Gouy Phase (y-z)', fontsize=10)
        plt.colorbar(im, ax=ax, label='φ (rad)')

        # (3,3) Unblocked y phase at focus
        ax = axes_mc6[3, 3]
        ax.plot(mc_ref_y_um, d_ub['focus_pres_y'], 'gray', linewidth=2, label='Unblocked Y')
        ax.set_xlabel('y (μm)', fontsize=10)
        ax.set_ylabel('Residual Phase (rad)', fontsize=10)
        ax.set_title('UNBLOCKED: Y Phase at Focus', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xlim([-200, 200])

        # Peak intensity ratio
        I_ratio_x = d_m['I_peak'] / max(d_ub['I_peak'], 1e-30)
        fig_mc6.suptitle(f'High-Resolution x-z and y-z Propagation: {mc6_label} vs Unblocked '
                         f'(dx={dx_xz_focus*1e3:.2f}μm, Peak ratio={I_ratio_x:.3f})',
                         fontsize=13, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname_mc6 = f'propagation_detail_{mc6_mname}_{_m2_tag}.png'
        plt.savefig(fname_mc6, dpi=400)
        print(f"  Saved: {fname_mc6}")

# =============================================================================
# HHG PHASE MISMATCH CALCULATION
# =============================================================================
TIMER.start_section("HHG Phase Mismatch")
print(f"\n{'='*60}")
print("HHG PHASE MISMATCH CALCULATION")
print(f"{'='*60}")

# --- User-configurable HHG parameters ---
hhg_harmonic_order = 21          # q - harmonic order
hhg_gas_pressure = 125.0          # Gas pressure in mbar
hhg_gas_type = 'argon'           # Gas species
hhg_peak_intensity_Wcm2 = 2.0e14 # Peak laser intensity (W/cm^2)
pulse_fwhm_fs = 55.0              # FWHM of Gaussian pulse envelope (fs)
# Experimental H21 yield vs intensity (from Mathematica analysis of lab data 24-12-3)
# Multi-harmonic unblocked yield vs intensity (same 7 intensity points, low→high)
# XUV absorption cross-sections (Mb) per harmonic in argon
# Argon has Cooper minimum near H17 (47 nm); H11-H15 have large σ
sigma_xuv_multi_Mb = {
    11: 33.0,   # 73 nm, 17.1 eV
    13: 30.0,   # 62 nm, 20.2 eV
    15: 27.0,   # 53 nm, 23.3 eV
    17: 23.0,   # 47 nm, 26.4 eV (Cooper minimum is at ~48 eV, not here)
    19: 20.0,   # 42 nm, 29.5 eV
    21: 17.0,   # 38 nm, 32.6 eV
}

# deconv_alpha/Is: set by TDSE LUT bridge below (dummy values for compatibility)


# Experimental enhancement data (blocked/unblocked yield ratio)
# Intensities: [950, 890, 850, 790, 770, 680, 580] * 3/1000 in units of 10^14 W/cm²
# --- HHG Helper Functions ---

def get_gas_properties(gas_type):
    """Return refractive index data and ionization potential for HHG gases."""
    # delta_n = (n_IR - n_q) per bar at STP (approximate values)
    # N_atm = number density at 1 bar, 293K (atoms/m^3)
    # Ip_eV = ionization potential in eV
    gas_data = {
        'argon':   {'delta_n': 2.8e-4, 'Ip_eV': 15.76, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
        'neon':    {'delta_n': 6.6e-5, 'Ip_eV': 21.56, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
        'helium':  {'delta_n': 3.5e-5, 'Ip_eV': 24.59, 'N_atm': 2.65e25, 'Z': 1, 'l': 0, 'm': 0, 'alpha_tl': 7.0},
        'krypton': {'delta_n': 4.2e-4, 'Ip_eV': 14.00, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
    }
    if gas_type.lower() not in gas_data:
        raise ValueError(f"Unknown gas type: {gas_type}. Available: {list(gas_data.keys())}")
    return gas_data[gas_type.lower()]

def ionization_fraction_bsi(I_Wcm2, Ip_eV):
    """
    Ionization fraction using smooth BSI model.
    Calibrated to ADK cycle-averaged rates (~10 optical cycles at 800nm).
    nf = 1 - exp(-(I/I_BSI / a)^n), a=0.89, n=5.
    """
    Ip_au = Ip_eV / 27.2114
    I_BSI_au = Ip_au**4 / 16.0
    I_BSI_Wcm2 = I_BSI_au * 3.5094e16
    ratio = I_Wcm2 / I_BSI_Wcm2
    n_f = 1.0 - np.exp(-(ratio / 0.89)**5)
    return np.clip(n_f, 0.0, 1.0)

def w_tbi_rate_au(F_au, Ip_au, Zc=1, l=1, m=0, alpha_tl=9.0):
    """Tong-Lin TBI ionization rate in atomic units (Eq. 2-3, Tong & Lin JPB 2005)."""
    kappa = np.sqrt(2.0 * Ip_au)
    n_star = Zc / kappa
    # C_{n*,l*}^2 from standard ADK formula
    Cl2 = (2.0 * np.e / n_star)**(2.0 * n_star) / (2.0 * np.pi * n_star)
    # Combinatorial factor f(l,m)
    Glm = (2*l + 1) * factorial(l + abs(m)) / (2**abs(m) * factorial(abs(m)) * factorial(l - abs(m)))
    F_safe = np.maximum(np.abs(F_au), 1e-30)
    # W_TI: standard ADK rate (Eq. 3)
    w_ti = (Cl2 * Glm / (2.0 * kappa**(2*Zc/kappa - 1))) * \
           (2.0 * kappa**3 / F_safe)**(2*Zc/kappa - abs(m) - 1) * \
           np.exp(-2.0 * kappa**3 / (3.0 * F_safe))
    # TBI correction factor (Eq. 2)
    correction = np.exp(-alpha_tl * (Zc**2 / Ip_au) * (F_safe / kappa**3))
    w_tbi = w_ti * correction
    return np.where(np.abs(F_au) < 1e-20, 0.0, w_tbi)

def ionization_fraction_tbi_pulse(I_peak_Wcm2, Ip_au, Zc, l, m, alpha_tl, tau_fwhm_fs):
    """Time-integrated ionization fraction using TBI rate over Gaussian pulse (Eq. 6)."""
    if I_peak_Wcm2 <= 0:
        return 0.0
    # Convert peak intensity to peak field (a.u.)
    I_au = I_peak_Wcm2 / 3.5094e16
    E0_au = np.sqrt(I_au)
    # Pulse parameters in a.u.
    tau_au = tau_fwhm_fs * 1e-15 / 2.4189e-17  # FWHM in a.u.
    omega_au = laser_omega_au
    T_cycle = 2.0 * np.pi / omega_au
    # Time grid: +/-5*FWHM/2, step = T_cycle/20
    t_max = 5.0 * tau_au / 2.0
    dt = T_cycle / 20.0
    t = np.arange(-t_max, t_max, dt)
    # Gaussian envelope field: E(t) = E0 * exp(-2*ln2*t^2/tau^2)
    E_env = E0_au * np.exp(-2.0 * np.log(2.0) * t**2 / tau_au**2)
    # Cycle-average: average W_TBI over |sin(phase)| within one cycle
    n_phase = 20
    phases = np.linspace(0, np.pi, n_phase, endpoint=False) + np.pi / (2 * n_phase)
    sin_vals = np.abs(np.sin(phases))
    # F_inst[i,j] = E_env[i] * sin_vals[j]
    F_inst = E_env[:, None] * sin_vals[None, :]
    w_inst = w_tbi_rate_au(F_inst, Ip_au, Zc, l, m, alpha_tl)
    w_avg = np.mean(w_inst, axis=1)  # cycle-averaged rate
    # P = 1 - exp(-integral w(t) dt)  (Eq. 6)
    integral = np.sum(w_avg) * dt
    n_f = 1.0 - np.exp(-integral)
    return np.clip(n_f, 0.0, 1.0)

def build_tbi_ionization_lut(gas_params, tau_fwhm_fs, I_min=1e12, I_max=5e14, n_points=500):
    """Build a 1D lookup table for TBI time-integrated ionization fraction."""
    Ip_au = gas_params['Ip_eV'] / 27.2114
    Zc = gas_params['Z']
    l = gas_params['l']
    m = gas_params['m']
    alpha_tl = gas_params['alpha_tl']
    I_grid = np.logspace(np.log10(I_min), np.log10(I_max), n_points)
    nf_grid = np.array([ionization_fraction_tbi_pulse(I, Ip_au, Zc, l, m, alpha_tl, tau_fwhm_fs)
                        for I in I_grid])
    # Interpolate in log(I) vs log(1-n_f) space for numerical stability
    log_I = np.log(I_grid)
    log_survival = np.log(np.maximum(1.0 - nf_grid, 1e-30))

    def ionization_func(I_Wcm2):
        I_arr = np.asarray(I_Wcm2, dtype=np.float64)
        scalar = I_arr.ndim == 0
        I_arr = np.atleast_1d(I_arr)
        result = np.zeros_like(I_arr)
        valid = I_arr > I_min
        if np.any(valid):
            log_I_query = np.log(np.clip(I_arr[valid], I_min, I_max))
            log_surv_interp = np.interp(log_I_query, log_I, log_survival)
            result[valid] = 1.0 - np.exp(log_surv_interp)
        result = np.clip(result, 0.0, 1.0)
        return float(result[0]) if scalar else result

    return ionization_func

def calc_dk_neutral(q, P_mbar, lambda_0_m, n_f, delta_n_per_bar):
    """Neutral gas dispersion phase mismatch (1/m)."""
    P_bar = P_mbar / 1000.0
    return (2 * np.pi * q / lambda_0_m) * P_bar * (1 - n_f) * delta_n_per_bar

def calc_dk_plasma(q, P_mbar, lambda_0_m, n_f, N_atm):
    """Plasma dispersion phase mismatch (1/m). Negative sign."""
    r_e = 2.8179403227e-15  # classical electron radius (m)
    P_bar = P_mbar / 1000.0
    N_e = n_f * N_atm * P_bar  # free electron density (m^-3)
    return -(q - 1.0 / q) * N_e * r_e * lambda_0_m

# --- Unit conversions ---
lambda_0_m = wavelength * 1e-3             # 800e-9 m
# Get gas properties
gas = get_gas_properties(hhg_gas_type)
print(f"Gas: {hhg_gas_type}, Ip = {gas['Ip_eV']:.2f} eV, delta_n = {gas['delta_n']:.2e} /bar")
print(f"Harmonic order q = {hhg_harmonic_order}, lambda_q = {wavelength * 1e6 / hhg_harmonic_order:.1f} nm")
print(f"Pressure = {hhg_gas_pressure:.1f} mbar, Peak I = {hhg_peak_intensity_Wcm2:.2e} W/cm^2")

# Build Tong-Lin TBI ionization LUT (time-integrated over pulse)
print(f"Building TBI ionization LUT (Tong-Lin, tau={pulse_fwhm_fs:.0f} fs)...")
_tbi_lut = build_tbi_ionization_lut(gas, pulse_fwhm_fs)

def ionization_fraction(I_Wcm2, Ip_eV):
    """Drop-in replacement: Tong-Lin TBI + pulse integration via LUT."""
    return _tbi_lut(I_Wcm2)

# Comparison: BSI vs TBI at reference intensity
I_test = hhg_peak_intensity_Wcm2
nf_bsi_test = ionization_fraction_bsi(I_test, gas['Ip_eV'])
nf_tbi_test = ionization_fraction(I_test, gas['Ip_eV'])
print(f"  At I={I_test:.1e}: BSI n_f={nf_bsi_test:.4f}, TBI n_f={nf_tbi_test:.4f}")


# =============================================================================
# TDSE DIPOLE LUT (replaces empirical alpha/Is fitting and SFA Lewenstein)
# =============================================================================
print('\n' + '='*60)
print('USING TDSE DIPOLE LUT FOR HHG YIELD COMPUTATION')
print('='*60)

multi_q_list = [11, 13, 15, 17, 19, 21]

# --- Pre-compute variables needed by yield section ---
lambda_0_m = wavelength * 1e-3
dx_hhg_m = (x_hhg_2d[1] - x_hhg_2d[0]) * 1e-3
I_scale_factor_2d = hhg_peak_intensity_Wcm2 / xz_intensity_unblocked.max()
z_gas_2d_b_m = z_gas_2d_b * 1e-3
z_gas_2d_ub_m = z_gas_2d_ub * 1e-3

gas = get_gas_properties(hhg_gas_type)
P_bar_gas = hhg_gas_pressure / 1000.0
n_gas_density = gas['N_atm'] * P_bar_gas
print(f"Building TBI ionization LUT (tau={pulse_fwhm_fs:.0f} fs)...")
_tbi_lut = build_tbi_ionization_lut(gas, pulse_fwhm_fs)
def ionization_fraction(I_Wcm2, Ip_eV):
    return _tbi_lut(I_Wcm2)

# Gouy phase gradient (complex-domain method)
field_env_b = np.sqrt(I_2d_gas_b) * np.exp(1j * phase_geom_2d_gas_b)
field_env_ub = np.sqrt(I_2d_gas_ub) * np.exp(1j * phase_geom_2d_gas_ub)
dphase_dz_3d_b = np.zeros_like(phase_geom_2d_gas_b)
dphase_dz_3d_ub = np.zeros_like(phase_geom_2d_gas_ub)
dz_arr_b = np.diff(z_gas_2d_b_m)
dz_arr_ub = np.diff(z_gas_2d_ub_m)
for j in range(1, len(z_gas_2d_b_m)):
    dphi_b = np.angle(field_env_b[j] * np.conj(field_env_b[j-1]))
    dphase_dz_3d_b[j] = dphi_b / dz_arr_b[j-1]
dphase_dz_3d_b[0] = dphase_dz_3d_b[1]
for j in range(1, len(z_gas_2d_ub_m)):
    dphi_ub = np.angle(field_env_ub[j] * np.conj(field_env_ub[j-1]))
    dphase_dz_3d_ub[j] = dphi_ub / dz_arr_ub[j-1]
dphase_dz_3d_ub[0] = dphase_dz_3d_ub[1]
I_thresh_b = 0.01 * I_2d_gas_b.max()
I_thresh_ub = 0.01 * I_2d_gas_ub.max()
dphase_dz_3d_b[I_2d_gas_b < I_thresh_b] = 0.0
dphase_dz_3d_ub[I_2d_gas_ub < I_thresh_ub] = 0.0
del field_env_b, field_env_ub
gc.collect()

# --- TDSE LUT loading ---
_lut_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'_lut_cache_tdse_{hhg_gas_type}.npz')
if not os.path.exists(_lut_cache_file):
    raise RuntimeError(f"TDSE LUT cache not found: {_lut_cache_file}\n"
                       f"Run build_tdse_lut.py first to generate the matching {laser_wavelength_nm:.0f} nm LUT.")
print(f"  [TDSE LUT] Loading from {_lut_cache_file}")
_lut_data = np.load(_lut_cache_file, allow_pickle=True)
if 'sfa_omega' in _lut_data.files:
    cached_sfa_omega = float(_lut_data['sfa_omega'])
    if not np.isclose(cached_sfa_omega, laser_omega_au, rtol=1e-3, atol=0.0):
        raise RuntimeError(
            f"TDSE LUT cache was generated with sfa_omega={cached_sfa_omega:.8f}, "
            f"but this run is configured for {laser_wavelength_nm:.0f} nm ({laser_omega_au:.8f}). "
            "Regenerate the TDSE LUT cache."
        )
else:
    raise RuntimeError("TDSE LUT cache is missing sfa_omega metadata. Regenerate it with build_tdse_lut.py.")
if 'laser_wavelength_nm' in _lut_data.files:
    cached_wavelength_nm = float(_lut_data['laser_wavelength_nm'])
    if not np.isclose(cached_wavelength_nm, laser_wavelength_nm, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            f"TDSE LUT cache was generated for {cached_wavelength_nm:.3f} nm, "
            f"but this run is configured for {laser_wavelength_nm:.3f} nm. "
            "Regenerate the TDSE LUT cache."
        )
I_lut = _lut_data['I_lut']
n_lut = len(I_lut)
I_lut_min = float(_lut_data['I_lut_min'])
I_lut_max = float(_lut_data['I_lut_max'])
multi_lut = {int(k): {'mag': v['mag'], 'phase': v['phase']}
             for k, v in _lut_data['multi_lut'].item().items()}
multi_lut_interp = {}
phase_interp_per_q = {}
mag_interp_per_q = {}
for q in multi_q_list:
    multi_lut_interp[q] = {
        'mag': interp1d(I_lut, multi_lut[q]['mag'], kind='cubic',
                        bounds_error=False, fill_value=0.0),
        'phase': interp1d(I_lut, multi_lut[q]['phase'], kind='cubic',
                          bounds_error=False, fill_value=0.0),
    }
    phase_interp_per_q[q] = multi_lut_interp[q]['phase']
    mag_interp_per_q[q] = multi_lut_interp[q]['mag']
del _lut_data
print(f"  Loaded TDSE LUT ({n_lut} points, {len(multi_q_list)} harmonics)")
print(f"  I range: {I_lut_min:.2e} - {I_lut_max:.2e} W/cm^2")
for q in multi_q_list:
    peak_idx = np.argmax(multi_lut[q]['mag'])
    print(f"  H{q}: peak |d_q| at I={I_lut[peak_idx]:.2e} W/cm^2")

# Dummy variables for backward compatibility with downstream code
# numpy trapz compatibility
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# --- Detection geometry (needed by mask comparison) ---
hhg_slit_width_mm = 2.0
hhg_slit_height_mm = 10.0
hhg_slit_distance = 1.2
hhg_aperture_radius_mm = 1.0
hhg_aperture_distance = 0.3
slit_half_angle_x = (hhg_slit_width_mm / 2) * 1e-3 / hhg_slit_distance
slit_half_angle_y = (hhg_slit_height_mm / 2) * 1e-3 / hhg_slit_distance
aperture_half_angle = hhg_aperture_radius_mm * 1e-3 / hhg_aperture_distance

# Far-field angular grid for H21 (used by plotting code)
lambda_q_ref = lambda_0_m / hhg_harmonic_order
dtheta_ref = lambda_q_ref / (N_hhg_2d * dx_hhg_m)
theta_axis = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dtheta_ref
theta_mrad = theta_axis * 1e3
dtheta = dtheta_ref
r_theta_grid = np.sqrt(theta_axis[:, None]**2 + theta_axis[None, :]**2)

# Near-field / far-field plot extents
x_hhg_um = x_hhg_2d * 1e3  # mm -> um
hhg_extent = [x_hhg_um[0], x_hhg_um[-1], x_hhg_um[0], x_hhg_um[-1]]
theta_extent_mc = [theta_mrad[0], theta_mrad[-1], theta_mrad[0], theta_mrad[-1]]
aperture_half_mrad_mc = aperture_half_angle * 1e3

# =============================================================================
# Figure HHG-10: Mask Shape Comparison
# =============================================================================
print("\n" + "="*60)
print("MASK SHAPE COMPARISON SCAN")
print("="*60)

mask_configs = [
    ('none', {}),
    ('circular', {}),
    ('twosided', {}),
    ('diagonal', {}),
]

mask_results = {}
# Reuse unblocked beam's HHG data as the common denominator
# We already have: I_2d_gas_ub, phase_geom_2d_gas_ub, z_gas_2d_ub
# and the unblocked far-field yield (yield_ap_ub) from the main computation.

# We need to recompute the "masked" beam for each mask shape.
# Loop over all HG modes per mask, sum intensities incoherently (weighted).
# Use HG00 phase for geometric phase / Gouy phase extraction.

for mask_name, mask_extra in mask_configs:
    print(f"\n--- Mask: {mask_name} ---")

    # Build mask
    msk = build_mask(X, Y, mask_name, mask_params)
    # Transmission
    field_masked = field_at_aperture * msk
    trans = np.sum(np.abs(field_masked)**2) / np.sum(np.abs(field_at_aperture)**2) * 100
    print(f"  Transmission: {trans:.1f}%")

    # Propagate single field through mask -> lens -> gas
    field_lens_masked = propagate_field(field_masked, aperture_distance_before_lens, k)
    field_lens_masked = thin_lens(field_lens_masked, R2, focal_length, k)

    I_gas_list = []
    phase_gas_list = []
    z_gas_list_m = []
    for i, z in enumerate(z_focus_prop):
        if gas_z_start_prop <= z <= gas_z_end_prop:
            fz, _, _ = fresnel_propagate_zoom(field_lens_masked, z, k, L, L_xz_focus, N_xz_focus)
            I_gas_list.append(np.abs(fz[hhg_crop, hhg_crop])**2)
            fz_nopw = fz * np.exp(-1j * k * z)
            phase_gas_list.append(np.angle(fz_nopw[hhg_crop, hhg_crop]))
            z_gas_list_m.append(z)

    I_gas = np.array(I_gas_list)
    phase_gas = np.array(phase_gas_list)
    print(f"  Propagated through gas ({len(z_gas_list_m)} z-steps)")

    z_gas_mm = np.array(z_gas_list_m)
    z_gas_m = z_gas_mm * 1e-3
    print(f"  Gas slices: {len(z_gas_mm)}, shape: {I_gas.shape}")

    # Scale to W/cm²
    I_gas_Wcm2 = I_gas * I_scale_factor_2d

    # Ionization
    nf_m = ionization_fraction(I_gas_Wcm2, gas['Ip_eV'])

    # Geometric phase gradient
    field_env = np.sqrt(I_gas) * np.exp(1j * phase_gas)
    dphase_dz = np.zeros_like(phase_gas)
    dz_arr = np.diff(z_gas_m)
    for j in range(1, len(z_gas_m)):
        dphase_dz[j] = np.angle(field_env[j] * np.conj(field_env[j-1])) / dz_arr[j-1]
    dphase_dz[0] = dphase_dz[1]
    I_thresh = 0.01 * I_gas.max()
    dphase_dz[I_gas < I_thresh] = 0.0
    # Peak-position Gouy phase gradient (use beam peak, not grid center)
    cg = I_gas.shape[1] // 2
    focus_iz = np.argmin(np.abs(z_gas_mm - focal_length))
    peak_pos = np.unravel_index(np.argmax(I_gas[focus_iz]), I_gas[focus_iz].shape)
    gouy_grad_onaxis = dphase_dz[:, peak_pos[0], peak_pos[1]]
    peak_I_m = I_gas_Wcm2.max()
    py, px = peak_pos
    focus_idx_m = np.argmin(np.abs(z_gas_mm - focal_length))
    print(f"    Beam peak at focus: ({py}, {px}) vs grid center ({cg}, {cg})")

    # --- Per-harmonic HHG yield for ALL harmonics ---
    I_clip = np.clip(I_gas_Wcm2, I_lut_min, I_lut_max)
    below_min = I_gas_Wcm2 < I_lut_min
    mask_hq = {}  # per-harmonic results for this mask

    for q in multi_q_list:
        # Phase mismatch (q-dependent)
        dk_neut_q = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_m, gas['delta_n'])
        dk_plas_q = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_m, gas['N_atm'])
        dk_geom_q = -(q - 1.0 / q) * dphase_dz
        dk_tot_q = dk_neut_q + dk_plas_q + dk_geom_q
        Phi_q = np.zeros_like(dk_tot_q)
        Phi_q[1:] = cumulative_trapezoid(dk_tot_q, z_gas_m, axis=0)

        # XUV absorption (q-dependent)
        sigma_q = sigma_xuv_multi_Mb.get(q, 10.0) * 1e-22
        mu_q = sigma_q * n_gas_density * (1.0 - nf_m)
        mu_cum_q = np.zeros_like(mu_q)
        mu_cum_q[1:] = cumulative_trapezoid(mu_q, z_gas_m, axis=0)
        tau_q = mu_cum_q[-1:] - mu_cum_q
        abs_q = np.exp(-tau_q / 2.0)

        # d_q from per-harmonic fitted parameters
        # TDSE: both magnitude and phase from LUT
        dq_3d = mag_interp_per_q[q](I_clip) * np.exp(1j * phase_interp_per_q[q](I_clip))
        dq_3d[below_min] = 0.0

        # Macroscopic integration
        integrand_q = dq_3d * (1.0 - nf_m) * np.exp(1j * Phi_q) * abs_q
        E_q_m = np.trapz(integrand_q, z_gas_m, axis=0)
        yield_nf_q = np.sum(np.abs(E_q_m)**2) * dx_hhg_m**2

        # Far-field
        E_ff_m = np.fft.fftshift(np.fft.fft2(E_q_m)) * dx_hhg_m**2
        I_ff_m = np.abs(E_ff_m)**2
        lambda_q_m = lambda_0_m / q
        dt_q = lambda_q_m / (N_hhg_2d * dx_hhg_m)
        theta_ax_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q

        # Slit aperture yield
        slit_mask_q = ((np.abs(theta_ax_q[None, :]) <= slit_half_angle_x) &
                       (np.abs(theta_ax_q[:, None]) <= slit_half_angle_y)).astype(float)
        yield_slit_q = np.sum(I_ff_m * slit_mask_q) * dt_q**2

        # Circular aperture yield
        r_th_q = np.sqrt(theta_ax_q[:, None]**2 + theta_ax_q[None, :]**2)
        circ_mask_q = (r_th_q <= aperture_half_angle).astype(float)
        yield_circ_q = np.sum(I_ff_m * circ_mask_q) * dt_q**2

        # On-axis buildup
        buildup_q = np.zeros(len(z_gas_m), dtype=complex)
        if len(z_gas_m) > 1:
            buildup_q[1:] = cumulative_trapezoid(integrand_q[:, py, px], z_gas_m)

        # Total yield buildup vs z (NF + slit + circ)
        yield_vs_z_q = None
        yield_vs_z_slit_q = None
        yield_vs_z_circ_q = None
        if len(z_gas_m) > 1:
            n_z_q = len(z_gas_m)
            dz_q = np.diff(z_gas_m)
            buildup_2d_q = np.zeros((n_z_q, N_hhg_2d, N_hhg_2d), dtype=complex)
            for j in range(1, n_z_q):
                buildup_2d_q[j] = buildup_2d_q[j-1] + 0.5 * (integrand_q[j-1] + integrand_q[j]) * dz_q[j-1]
            yield_vs_z_q = np.zeros(n_z_q)
            yield_vs_z_slit_q = np.zeros(n_z_q)
            yield_vs_z_circ_q = np.zeros(n_z_q)
            for j in range(n_z_q):
                yield_vs_z_q[j] = np.sum(np.abs(buildup_2d_q[j])**2) * dx_hhg_m**2
                if USE_SCIPY_FFT:
                    E_ff_j = scipy_fft.fftshift(scipy_fft.fft2(buildup_2d_q[j], workers=-1)) * dx_hhg_m**2
                else:
                    E_ff_j = np.fft.fftshift(np.fft.fft2(buildup_2d_q[j])) * dx_hhg_m**2
                I_ff_j = np.abs(E_ff_j)**2
                yield_vs_z_slit_q[j] = np.sum(I_ff_j * slit_mask_q) * dt_q**2
                yield_vs_z_circ_q[j] = np.sum(I_ff_j * circ_mask_q) * dt_q**2
            del buildup_2d_q

        mask_hq[q] = {
            'yield_nf': yield_nf_q,
            'yield_slit': yield_slit_q,
            'yield_circ': yield_circ_q,
            'E_q': E_q_m.copy(),
            'E_ff': E_ff_m.copy(),
            'I_ff': I_ff_m.copy(),
            'dtheta': dt_q,
            'theta_axis': theta_ax_q,
            'dk_total_onaxis': dk_tot_q[:, py, px].copy(),
            'buildup_onaxis': buildup_q,
            'yield_vs_z': yield_vs_z_q,
            'yield_vs_z_slit': yield_vs_z_slit_q,
            'yield_vs_z_circ': yield_vs_z_circ_q,
        }

        # Save full cross-section data for reference harmonic (used by HHG-MC figures)
        if q == hhg_harmonic_order:
            _mc_dk_data = {
                'dk_total_onaxis': dk_tot_q[:, py, px].copy(),
                'dk_neut_onaxis': dk_neut_q[:, py, px].copy(),
                'dk_plas_onaxis': dk_plas_q[:, py, px].copy(),
                'dk_geom_onaxis': dk_geom_q[:, py, px].copy(),
                'Phi_onaxis': Phi_q[:, py, px].copy(),
                'dk_total_xz': dk_tot_q[:, py, :].copy(),
                'L_coh_xz': (np.pi / np.clip(np.abs(dk_tot_q[:, py, :]), 1e-10, None)).copy(),
                'I_xz_Wcm2': I_gas_Wcm2[:, py, :].copy(),
                'dk_focus_x': dk_tot_q[focus_idx_m, py, :].copy(),
                'dk_neut_focus_x': dk_neut_q[focus_idx_m, py, :].copy(),
                'dk_plas_focus_x': dk_plas_q[focus_idx_m, py, :].copy(),
                'dk_geom_focus_x': dk_geom_q[focus_idx_m, py, :].copy(),
                'I_focus_x_Wcm2': I_gas_Wcm2[focus_idx_m, py, :].copy(),
                'nf_focus_x': nf_m[focus_idx_m, py, :].copy(),
            }

        del dk_neut_q, dk_plas_q, dk_geom_q, dk_tot_q, Phi_q
        del mu_q, mu_cum_q, tau_q, abs_q, dq_3d, integrand_q

    # Store results (backward-compatible: keep H21 as default for old figures)
    _q0 = hhg_harmonic_order
    mask_results[mask_name] = {
        'transmission': trans,
        'yield_nf': mask_hq[_q0]['yield_nf'],
        'yield_ap': mask_hq[_q0]['yield_slit'],
        'yield_ff': mask_hq[_q0]['yield_slit'],
        'onaxis': mask_hq[_q0]['I_ff'][N_hhg_2d // 2, N_hhg_2d // 2],
        'peak_I': peak_I_m,
        'E_q': mask_hq[_q0]['E_q'],
        'E_ff': mask_hq[_q0]['E_ff'],
        'I_ff': mask_hq[_q0]['I_ff'],
        'gouy_grad': gouy_grad_onaxis,
        'z_gas_mm': z_gas_mm,
        'I_onaxis_Wcm2': I_gas_Wcm2[:, py, px].copy(),
        'nf_onaxis': nf_m[:, py, px].copy(),
        'gouy_onaxis': phase_gas[:, py, px].copy(),
        'buildup_onaxis': mask_hq[_q0]['buildup_onaxis'],
        'focus_idx': focus_idx_m,
        # --- Per-harmonic data ---
        'per_q': mask_hq,
    }
    # Merge H21 dk cross-section data for HHG-MC figures
    mask_results[mask_name].update(_mc_dk_data)

    print(f"  Peak I: {peak_I_m:.2e} W/cm^2")
    for q in multi_q_list:
        _r = mask_hq[q]
        print(f"    H{q}: NF={_r['yield_nf']:.3e}, slit={_r['yield_slit']:.3e}, circ={_r['yield_circ']:.3e}")

    del I_gas, phase_gas, I_gas_Wcm2, nf_m, field_env, dphase_dz
    gc.collect()

# --- Print comparison table ---
print("\n  === MASK COMPARISON ===")
ref_name = 'none'
ref_ap = mask_results[ref_name]['yield_ap']
ref_nf = mask_results[ref_name]['yield_nf']
print(f"  {'Mask':12s} {'Trans%':>7s} {'Peak I':>12s} {'NF ratio':>10s} {'AP ratio':>10s}")
for mname in ['none', 'circular', 'twosided', 'diagonal']:
    r = mask_results[mname]
    nf_r = r['yield_nf'] / ref_nf if ref_nf > 0 else 0
    ap_r = r['yield_ap'] / ref_ap if ref_ap > 0 else 0
    print(f"  {mname:12s} {r['transmission']:7.1f} {r['peak_I']:12.2e} {nf_r:10.4f} {ap_r:10.4f}")

# --- Figure HHG-10: Mask comparison (2×2) ---
save_tdse_mask_comparison_npz(
    mask_results,
    f'hhg_mask_comparison_data_tdse_rps_P{hhg_gas_pressure:.0f}mbar_{_m2_tag}.npz',
)

fig10, axes10 = plt.subplots(2, 2, figsize=(15, 10.5))
mask_colors = {'none': '#6E6E6E', 'circular': '#2F6F9F', 'twosided': '#B3262E', 'diagonal': '#218C4A'}
mask_labels = {'none': 'No mask', 'circular': 'Circular', 'twosided': 'Two-side', 'diagonal': 'Diagonal'}

# (0,0) Bar chart: aperture ratio for each mask
ax = axes10[0, 0]
mnames = ['none', 'circular', 'twosided', 'diagonal']
ap_ratios_all = [mask_results[m]['yield_ap'] / ref_ap for m in mnames]
bars = ax.bar(range(len(mnames)), ap_ratios_all,
              color=[mask_colors[m] for m in mnames], alpha=0.92,
              edgecolor='black', linewidth=0.5)
for i, (b, v) in enumerate(zip(bars, ap_ratios_all)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{v:.2f}',
            ha='center', fontsize=10.5)
ax.set_xticks(range(len(mnames)))
ax.set_xticklabels([mask_labels[m] for m in mnames])
ax.set_ylabel('Aperture Yield Ratio (mask / no-mask)')
ax.set_title('Far-Field Aperture Yield')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
style_paper_axis(ax, grid=True)

# (0,1) 2D near-field HHG for each mask (2×2 subgrid) — common normalization + log scale
ax = axes10[0, 1]
ax.set_visible(False)
gs_inner = fig10.add_gridspec(2, 2, left=0.55, right=0.95, top=0.88, bottom=0.52,
                               wspace=0.15, hspace=0.25)
# Common normalization: max across all masks
nf_global_max = max((np.abs(mask_results[m]['E_q'])**2).max()
                    for m in mnames if 'E_q' in mask_results[m])
ext_nf = [x_hhg_2d.min()*1e3, x_hhg_2d.max()*1e3,
          x_hhg_2d.min()*1e3, x_hhg_2d.max()*1e3]
for idx, mname in enumerate(mnames):
    ax_sub = fig10.add_subplot(gs_inner[idx // 2, idx % 2])
    I_nf = np.abs(mask_results[mname]['E_q'])**2 if 'E_q' in mask_results[mname] else np.zeros((N_hhg_2d, N_hhg_2d))
    I_nf_log = np.log10(np.clip(I_nf / max(nf_global_max, 1e-30), 1e-6, None))
    ax_sub.imshow(I_nf_log, extent=ext_nf, origin='lower', cmap='magma', vmin=-4, vmax=0)
    nf_ratio = I_nf.max() / max(nf_global_max, 1e-30)
    ax_sub.set_title(f'{mask_labels[mname]} ({nf_ratio:.2f}x)', fontsize=11)
    if idx >= 2:
        ax_sub.set_xlabel(r'x ($\mu$m)')
    if idx % 2 == 0:
        ax_sub.set_ylabel(r'y ($\mu$m)')
    style_paper_axis(ax_sub)

# (1,0) Far-field angular lineouts
ax = axes10[1, 0]
for mname in mnames:
    I_ff_line = mask_results[mname]['I_ff'][N_hhg_2d // 2, :]
    I_ff_line = I_ff_line / max(I_ff_line.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_line, color=mask_colors[mname],
                label=mask_labels[mname], linewidth=2.0)
ax.axvline(-slit_half_angle_x*1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75)
ax.axvline(slit_half_angle_x*1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75, label='Slit')
ax.axvline(-aperture_half_angle*1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75)
ax.axvline(aperture_half_angle*1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75, label='Circ')
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'Far-field $|E|^2$ (self norm.)')
ax.set_title('Far-Field Angular Lineout')
ax.set_xlim([-10, 10])
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (1,1) On-axis Gouy phase gradient for each mask
ax = axes10[1, 1]
for mname in mnames:
    r = mask_results[mname]
    ax.plot(r['z_gas_mm'], r['gouy_grad'], color=mask_colors[mname],
            label=mask_labels[mname], linewidth=2.0)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'd$\phi$/dz (rad/m)')
ax.set_title('On-Axis Gouy Phase Gradient')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

fig10.suptitle(f'Mask Shape Comparison — H{hhg_harmonic_order}, '
               f'P={hhg_gas_pressure:.0f} mbar', fontsize=17, fontweight='bold')
finalize_paper_figure(fig10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'hhg_mask_comparison_{_m2_tag}.png', dpi=400)
print(f"\nMask comparison figure saved to 'hhg_mask_comparison.png'")

# =============================================================================
# Per-Harmonic Mask Comparison Figures (H11-H21)
# =============================================================================
print("\n" + "="*60)
print("PER-HARMONIC MASK COMPARISON FIGURES")
print("="*60)

_mnames = ['none', 'circular', 'twosided', 'diagonal']
_mlabels = {'none': 'No mask', 'circular': 'Circular', 'twosided': 'Two-side', 'diagonal': 'Diagonal'}
_mcolors = {'none': '#7F7F7F', 'circular': '#2F5597', 'twosided': '#B04A4A', 'diagonal': '#4F8B5B'}
_ref_name = 'none'

# --- Print full comparison table (all harmonics) ---
print("\n  === PER-HARMONIC MASK COMPARISON (slit / circ / NF) ===")
for q in multi_q_list:
    ref_slit = mask_results[_ref_name]['per_q'][q]['yield_slit']
    ref_circ = mask_results[_ref_name]['per_q'][q]['yield_circ']
    ref_nf_q = mask_results[_ref_name]['per_q'][q]['yield_nf']
    print(f"\n  H{q}:")
    print(f"    {'Mask':12s} {'Slit enh':>10s} {'Circ enh':>10s} {'NF enh':>10s}")
    for mn in _mnames:
        rq = mask_results[mn]['per_q'][q]
        s_r = rq['yield_slit'] / ref_slit if ref_slit > 0 else 0
        c_r = rq['yield_circ'] / ref_circ if ref_circ > 0 else 0
        n_r = rq['yield_nf'] / ref_nf_q if ref_nf_q > 0 else 0
        print(f"    {mn:12s} {s_r:10.4f} {c_r:10.4f} {n_r:10.4f}")

# --- Figure HHG-PH: Per-harmonic mask comparison (one figure per harmonic) ---
# Layout like HHG-10: (0,0) bar chart, (0,1) NF 2×2, (1,0) FF lineout, (1,1) Gouy phase
# Bar chart: 3 groups (Slit / Circ / NF), each with 4 bars (masks)
x_hhg_um_ph = x_hhg_2d * 1e3
ext_nf_ph = [x_hhg_um_ph[0], x_hhg_um_ph[-1], x_hhg_um_ph[0], x_hhg_um_ph[-1]]

for q in multi_q_list:
    print(f"\n  Generating per-harmonic figure for H{q}...")
    ref_slit = mask_results[_ref_name]['per_q'][q]['yield_slit']
    ref_circ = mask_results[_ref_name]['per_q'][q]['yield_circ']
    ref_nf_q = mask_results[_ref_name]['per_q'][q]['yield_nf']

    fig_ph, axes_ph = plt.subplots(2, 3, figsize=(18.5, 9.6))

    # --- (0,0) Bar chart: 3 detection groups × 4 masks ---
    ax = axes_ph[0, 0]
    _det_names = ['Slit', 'Circular', 'Near-field']
    n_det = 3
    n_mask = len(_mnames)
    x_group = np.arange(n_det)
    w = 0.18
    offsets = np.arange(n_mask) * w - (n_mask - 1) * w / 2
    _ratio_max_ph = 1.0

    for im, mn in enumerate(_mnames):
        rq = mask_results[mn]['per_q'][q]
        enh_slit = rq['yield_slit'] / ref_slit if ref_slit > 0 else 0
        enh_circ = rq['yield_circ'] / ref_circ if ref_circ > 0 else 0
        enh_nf = rq['yield_nf'] / ref_nf_q if ref_nf_q > 0 else 0
        vals = [enh_slit, enh_circ, enh_nf]
        _ratio_max_ph = max(_ratio_max_ph, float(np.nanmax(vals)))
        bars = ax.bar(x_group + offsets[im], vals, w, label=_mlabels[mn],
                      color=_mcolors[mn], alpha=0.92,
                      edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.2f}', ha='center', fontsize=8.5, rotation=90)
    ax.set_xticks(x_group)
    ax.set_xticklabels(_det_names)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Enhancement (mask / no-mask)')
    ax.set_title(f'H{q} Enhancement')
    ax.set_ylim(0, max(1.15, _ratio_max_ph * 1.16))
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (0,1) Near-field 2×2: 4 masks ---
    ax_main = axes_ph[0, 1]
    ax_main.set_visible(False)
    nf_global = max((np.abs(mask_results[m]['per_q'][q]['E_q'])**2).max()
                    for m in _mnames)
    gs_nf = fig_ph.add_gridspec(2, 2, left=0.38, right=0.63, top=0.88, bottom=0.52,
                                 wspace=0.15, hspace=0.25)
    for idx, mn in enumerate(_mnames):
        ax_sub = fig_ph.add_subplot(gs_nf[idx // 2, idx % 2])
        I_nf = np.abs(mask_results[mn]['per_q'][q]['E_q'])**2
        rel = I_nf.max() / max(nf_global, 1e-30)
        if nf_global > 0:
            ax_sub.imshow(I_nf.T / nf_global, extent=ext_nf_ph, origin='lower', cmap='magma', vmin=0, vmax=1)
        ax_sub.set_title(f'{_mlabels[mn]} ({rel:.2f}x)', fontsize=11)
        ax_sub.set_xlim([-80, 80]); ax_sub.set_ylim([-80, 80])
        style_paper_axis(ax_sub)
        if idx >= 2: ax_sub.set_xlabel('x (μm)')
        if idx % 2 == 0: ax_sub.set_ylabel('y (μm)')

        if idx >= 2:
            ax_sub.set_xlabel(r'x ($\mu$m)')
        if idx % 2 == 0:
            ax_sub.set_ylabel(r'y ($\mu$m)')

    # --- (1,0) Far-field angular lineout ---
    ax = axes_ph[1, 0]
    for mn in _mnames:
        rq = mask_results[mn]['per_q'][q]
        dt_q = rq['dtheta']
        th_ax = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q * 1e3  # mrad
        lineout = rq['I_ff'][N_hhg_2d // 2, :]
        if lineout.max() > 0:
                ax.semilogy(th_ax, lineout / lineout.max(), color=_mcolors[mn],
                            label=_mlabels[mn], linewidth=2.0)
    # Mark slit and circular aperture boundaries
    ax.axvline(-slit_half_angle_x * 1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75)
    ax.axvline(slit_half_angle_x * 1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75, label='Slit')
    ax.axvline(-aperture_half_angle * 1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75)
    ax.axvline(aperture_half_angle * 1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75, label='Circ')
    ax.set_xlabel('θx (mrad)')
    ax.set_ylabel('Far-field |E|² (self-norm, log)')
    ax.set_title('Far-Field Angular Lineout')
    ax.set_xlim([-10, 10])
    ax.set_xlabel(r'$\theta_x$ (mrad)')
    ax.set_ylabel(r'Far-field $|E|^2$ (self norm.)')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (1,1) On-axis Gouy phase gradient ---
    ax = axes_ph[1, 1]
    for mn in _mnames:
        r = mask_results[mn]
        ax.plot(r['z_gas_mm'], r['gouy_grad'], color=_mcolors[mn],
                label=_mlabels[mn], linewidth=2.0)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('dφ/dz (rad/m)')
    ax.set_ylabel(r'd$\phi$/dz (rad/m)')
    ax.set_title('On-Axis Gouy Phase Gradient')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (0,2) HHG near-field lineouts (x and y) ---
    ax = axes_ph[0, 2]
    center_ph = N_hhg_2d // 2
    for mn in _mnames:
        I_nf_mn = np.abs(mask_results[mn]['per_q'][q]['E_q'])**2
        ax.plot(x_hhg_um_ph, I_nf_mn[center_ph, :] / max(nf_global, 1e-30),
                color=_mcolors[mn], linewidth=2.0, label=f'{_mlabels[mn]} (x)')
        ax.plot(x_hhg_um_ph, I_nf_mn[:, center_ph] / max(nf_global, 1e-30),
                color=_mcolors[mn], linewidth=1.4, linestyle='--')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel(r'$|E_q|^2$ (normalized)')
    ax.set_title('HHG Near-Field Lineouts')
    ax.set_xlim([-50, 50])
    ax.set_xlabel(r'Position ($\mu$m)')
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (1,2) Total yield buildup vs z (slit + circ + NF) ---
    ax = axes_ph[1, 2]
    ref_yvz_slit = mask_results[_ref_name]['per_q'][q].get('yield_vs_z_slit')
    ref_yvz_circ = mask_results[_ref_name]['per_q'][q].get('yield_vs_z_circ')
    ref_yvz_nf = mask_results[_ref_name]['per_q'][q].get('yield_vs_z')
    yvz_norm_slit_ph = max(ref_yvz_slit[-1], 1e-30) if ref_yvz_slit is not None else 1e-30
    yvz_norm_circ_ph = max(ref_yvz_circ[-1], 1e-30) if ref_yvz_circ is not None else 1e-30
    yvz_norm_nf_ph = max(ref_yvz_nf[-1], 1e-30) if ref_yvz_nf is not None else 1e-30
    for mn in _mnames:
        z_mm = mask_results[mn]['z_gas_mm']
        yvz_s = mask_results[mn]['per_q'][q].get('yield_vs_z_slit')
        yvz_c = mask_results[mn]['per_q'][q].get('yield_vs_z_circ')
        yvz_n = mask_results[mn]['per_q'][q].get('yield_vs_z')
        if yvz_s is not None:
            ax.plot(z_mm, yvz_s / yvz_norm_slit_ph,
                    color=_mcolors[mn], linewidth=2.4, label=f'{_mlabels[mn]} (slit)')
        if yvz_c is not None:
            ax.plot(z_mm, yvz_c / yvz_norm_circ_ph,
                    color=_mcolors[mn], linewidth=1.8, linestyle='--')
        if yvz_n is not None:
            ax.plot(z_mm, yvz_n / yvz_norm_nf_ph,
                    color=_mcolors[mn], linewidth=1.8, linestyle=':')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Integrated yield (normalized)')
    ax.set_title('Yield buildup')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    fig_ph.suptitle(f'Mask Comparison -- H{q}, {hhg_gas_type.capitalize()}, '
                     f'P={hhg_gas_pressure:.0f} mbar, M$^2$=({M2x},{M2y})',
                     fontsize=17, fontweight='bold')
    finalize_paper_figure(fig_ph)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'hhg_mask_H{q}_{_m2_tag}.png', dpi=400)
    print(f"    Saved: hhg_mask_H{q}_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-1: On-axis Phase Mismatch & Physics Comparison
# =============================================================================
TIMER.start_section("Figure HHG-MC-1 - On-axis physics")
print("\nGenerating HHG-MC-1: on-axis phase mismatch & physics comparison...")

fig_hmc1, axes_hmc1 = plt.subplots(2, 3, figsize=(20, 12))
hmc_masks = ['none', 'circular', 'twosided', 'diagonal']

# (0,0) dk_total on-axis for all masks
ax = axes_hmc1[0, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(r['z_gas_mm'], r['dk_total_onaxis'], color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Delta k_{total}$ (1/mm)')
ax.set_title(r'On-Axis Total $\Delta k$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,1) |dk_total| on-axis (log scale) + L_coh
ax = axes_hmc1[0, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    dk_abs = np.abs(r['dk_total_onaxis'])
    L_coh = np.pi / np.clip(dk_abs, 1e-10, None)
    ax.semilogy(r['z_gas_mm'], L_coh, color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$L_{coh} = \pi/|\Delta k|$ (mm)')
ax.set_title('On-Axis Coherence Length')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,2) Cumulative phase Phi(z)
ax = axes_hmc1[0, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(r['z_gas_mm'], r['Phi_onaxis'], color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.axhline(np.pi, color='orange', linestyle='--', linewidth=0.8, alpha=0.6, label=r'$\pm\pi$')
ax.axhline(-np.pi, color='orange', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Phi(z)$ (rad)')
ax.set_title('On-Axis Cumulative Phase')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,0) On-axis intensity
ax = axes_hmc1[1, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(r['z_gas_mm'], r['I_onaxis_Wcm2'], color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Intensity (W/cm^2)')
ax.set_title('On-Axis Intensity')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,1) Ionization fraction
ax = axes_hmc1[1, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(r['z_gas_mm'], r['nf_onaxis'], color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Ionization fraction')
ax.set_title(f'Ionization ({hhg_gas_type.capitalize()}, Ip={gas["Ip_eV"]:.2f} eV)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,2) On-axis Gouy phase (unwrapped)
ax = axes_hmc1[1, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    gouy_uw = np.unwrap(r['gouy_onaxis'])
    gouy_uw -= gouy_uw[len(gouy_uw)//2]  # center at focus
    ax.plot(r['z_gas_mm'], gouy_uw, color=color, linewidth=2.0, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Gouy phase (rad)')
ax.set_title('On-Axis Gouy Phase')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig_hmc1.suptitle(f'On-Axis HHG Physics — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc1)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc1_onaxis_physics_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc1_onaxis_physics_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-2: 2D Phase Mismatch x-z Maps Comparison
# =============================================================================
TIMER.start_section("Figure HHG-MC-2 - 2D phase mismatch maps")
print("Generating HHG-MC-2: 2D phase mismatch maps comparison...")

fig_hmc2, axes_hmc2 = plt.subplots(2, 4, figsize=(24, 10))
hmc2_col = ['none', 'circular', 'twosided', 'diagonal']

# Determine common color scale
dk_vmax = max(np.percentile(np.abs(mask_results[mn]['dk_total_xz']), 95) for mn in hmc2_col)

for col_idx, mn in enumerate(hmc2_col):
    r = mask_results[mn]
    z_mm = r['z_gas_mm']
    dk_xz = r['dk_total_xz']
    L_coh_xz = r['L_coh_xz']
    # Mask low-intensity regions
    I_xz = r['I_xz_Wcm2']
    I_mask = I_xz > 0.01 * I_xz.max()
    dk_xz_masked = np.where(I_mask, dk_xz, np.nan)
    L_coh_mm = L_coh_xz * 1e3  # convert m → mm
    L_coh_masked = np.where(I_mask, np.log10(np.clip(L_coh_mm, 1e-3, 100)), np.nan)

    ext_hmc2 = [x_hhg_2d.min()*1e3, x_hhg_2d.max()*1e3, z_mm.min(), z_mm.max()]

    # Row 1: dk_total
    ax = axes_hmc2[0, col_idx]
    ax.imshow(dk_xz_masked, extent=ext_hmc2, origin='lower', aspect='auto',
              cmap='RdBu_r', vmin=-dk_vmax, vmax=dk_vmax, interpolation='bicubic')
    label, color = mask_disp[mn]
    ax.set_title(f'{label}', fontsize=11, color=color)
    if col_idx == 0:
        ax.set_ylabel(r'$\Delta k_{total}$' + '\nz (mm)')
    ax.set_xlabel('x (um)')

    # Row 2: L_coh (log10)
    ax = axes_hmc2[1, col_idx]
    im = ax.imshow(L_coh_masked, extent=ext_hmc2, origin='lower', aspect='auto',
                   cmap='hot', vmin=-1, vmax=2, interpolation='bicubic')
    if col_idx == 0:
        ax.set_ylabel(r'$\log_{10}(L_{coh}$ / mm)' + '\nz (mm)')
    ax.set_xlabel('x (um)')
    if col_idx == 3:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig_hmc2.suptitle(f'2D Phase Mismatch Maps (x-z plane) — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}',
                   fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc2)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc2_2d_phase_mismatch_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc2_2d_phase_mismatch_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-3: Phase Mismatch at Focus Plane Comparison
# =============================================================================
TIMER.start_section("Figure HHG-MC-3 - Focus plane mismatch")
print("Generating HHG-MC-3: phase mismatch at focus plane comparison...")

fig_hmc3, axes_hmc3 = plt.subplots(2, 3, figsize=(20, 12))
hmc3_x_um = x_hhg_2d * 1e3

# (0,0) dk_total at focus
ax = axes_hmc3[0, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(hmc3_x_um, r['dk_focus_x'], color=color, linewidth=2.0, label=label)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('x (um)')
ax.set_ylabel(r'$\Delta k_{total}$ (1/mm)')
ax.set_title(r'$\Delta k_{total}$ at Focus')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,1) |dk_total| at focus
ax = axes_hmc3[0, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(hmc3_x_um, np.abs(r['dk_focus_x']), color=color, linewidth=2.0, label=label)
ax.set_xlabel('x (um)')
ax.set_ylabel(r'$|\Delta k|$ (1/mm)')
ax.set_title(r'$|\Delta k|$ at Focus')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,2) L_coh at focus
ax = axes_hmc3[0, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    L_coh_f = np.pi / np.clip(np.abs(r['dk_focus_x']), 1e-10, None)
    ax.semilogy(hmc3_x_um, L_coh_f, color=color, linewidth=2.0, label=label)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Gas length (1 mm)')
ax.set_xlabel('x (um)')
ax.set_ylabel(r'$L_{coh}$ (mm)')
ax.set_title('Coherence Length at Focus')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,0) Intensity at focus
ax = axes_hmc3[1, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(hmc3_x_um, r['I_focus_x_Wcm2'], color=color, linewidth=2.0, label=label)
ax.set_xlabel('x (um)')
ax.set_ylabel('Intensity (W/cm^2)')
ax.set_title('Intensity at Focus')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,1) Ionization at focus
ax = axes_hmc3[1, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    ax.plot(hmc3_x_um, r['nf_focus_x'], color=color, linewidth=2.0, label=label)
ax.set_xlabel('x (um)')
ax.set_ylabel('Ionization fraction')
ax.set_title('Ionization at Focus')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,2) On-axis HHG buildup
ax = axes_hmc3[1, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    buildup = np.abs(r['buildup_onaxis'])**2
    if buildup.max() > 0:
        buildup = buildup / buildup.max()
    ax.plot(r['z_gas_mm'], buildup, color=color, linewidth=2.4, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=1.1)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'On-axis $|E_q(z)|^2$ (normalized)')
ax.set_title('HHG Buildup Along Gas')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

fig_hmc3.suptitle(f'Phase Mismatch at Focus — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc3_focus_mismatch_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc3_focus_mismatch_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-4: HHG Yield & Far-field Details Comparison
# =============================================================================
TIMER.start_section("Figure HHG-MC-4 - Yield details")
print("Generating HHG-MC-4: HHG yield & far-field comparison...")

fig_hmc4, axes_hmc4 = plt.subplots(2, 3, figsize=(20, 12))
hmc4_nf_center = N_hhg_2d // 2

# (0,0) Near-field x lineout
ax = axes_hmc4[0, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_nf = np.abs(r['E_q'])**2
    I_nf_x = I_nf[hmc4_nf_center, :]
    I_nf_x_norm = I_nf_x / max(I_nf_x.max(), 1e-30)
    ax.plot(hmc3_x_um, I_nf_x_norm, color=color, linewidth=2.0, label=label)
ax.set_xlabel('x (um)')
ax.set_ylabel(r'$|E_q|^2$ (self-norm)')
ax.set_title('Near-Field x Lineout')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (0,1) Near-field y lineout
ax = axes_hmc4[0, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_nf = np.abs(r['E_q'])**2
    I_nf_y = I_nf[:, hmc4_nf_center]
    I_nf_y_norm = I_nf_y / max(I_nf_y.max(), 1e-30)
    ax.plot(hmc3_x_um, I_nf_y_norm, color=color, linewidth=2.0, label=label)
ax.set_xlabel('y (um)')
ax.set_ylabel(r'$|E_q|^2$ (self-norm)')
ax.set_title('Near-Field y Lineout')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (0,2) On-axis HHG buildup (absolute, not normalized)
ax = axes_hmc4[0, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    buildup = np.abs(r['buildup_onaxis'])**2
    ax.plot(r['z_gas_mm'], buildup, color=color, linewidth=2.4, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=1.1)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$|E_q(z)|^2$')
ax.set_title('On-Axis HHG Buildup (absolute)')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (1,0) Far-field x lineout
ax = axes_hmc4[1, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_ff_x = r['I_ff'][hmc4_nf_center, :]
    I_ff_x_norm = I_ff_x / max(I_ff_x.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_x_norm, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=1.1)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=1.1)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
ax.set_title('Far-Field x Lineout')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (1,1) Far-field y lineout
ax = axes_hmc4[1, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_ff_y = r['I_ff'][:, hmc4_nf_center]
    I_ff_y_norm = I_ff_y / max(I_ff_y.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_y_norm, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=1.1)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=1.1)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
ax.set_title('Far-Field y Lineout')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

# (1,2) Cumulative yield vs acceptance angle
ax = axes_hmc4[1, 2]
r_theta_grid = np.sqrt(theta_axis[:, None]**2 + theta_axis[None, :]**2)
angle_scan = np.linspace(0.5, theta_mrad.max(), 50) * 1e-3  # in rad
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    cum_yield = np.array([np.sum(r['I_ff'] * (r_theta_grid <= a).astype(float)) * dtheta**2
                          for a in angle_scan])
    cum_yield_norm = cum_yield / max(cum_yield[-1], 1e-30)
    ax.plot(angle_scan * 1e3, cum_yield_norm, color=color, linewidth=2.0, label=label)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=1.1, label=f'Aperture ({aperture_half_angle*1e3:.1f} mrad)')
ax.set_xlabel('Acceptance half-angle (mrad)')
ax.set_ylabel('Cumulative yield (normalized)')
ax.set_title('Cumulative Yield vs Acceptance')
ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='0.75')
style_paper_axis(ax, grid=True)

fig_hmc4.suptitle(f'HHG Yield & Far-Field — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc4)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc4_yield_farfield_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc4_yield_farfield_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-5: Near-Field 2D HHG Yield for All Masks
# =============================================================================
TIMER.start_section("Figure HHG-MC-5 - Near-field 2D maps")
print("Generating HHG-MC-5: near-field 2D HHG for all masks...")

fig_hmc5, axes_hmc5 = plt.subplots(2, 4, figsize=(24, 11))
hmc5_col = ['none', 'circular', 'twosided', 'diagonal']
hmc5_nf_center = N_hhg_2d // 2

# Compute near-field data per mask
hmc5_I_nf = {}
hmc5_I_max_global = 0
for mn in hmc5_col:
    I_nf = np.abs(mask_results[mn]['E_q'])**2
    hmc5_I_nf[mn] = I_nf
    hmc5_I_max_global = max(hmc5_I_max_global, I_nf.max())

# Row 1: 2D near-field |E_q|² (common normalization, log scale)
for col_idx, mn in enumerate(hmc5_col):
    ax = axes_hmc5[0, col_idx]
    I_nf = hmc5_I_nf[mn]
    I_nf_log = np.log10(np.clip(I_nf / max(hmc5_I_max_global, 1e-30), 1e-6, None))
    im = ax.imshow(I_nf_log, extent=hhg_extent, aspect='equal', origin='lower',
                   cmap='magma', vmin=-4, vmax=0, interpolation='bicubic')
    label, color = mask_disp[mn]
    nf_yield = mask_results[mn]['yield_nf']
    peak_ratio = I_nf.max() / max(hmc5_I_max_global, 1e-30)
    ax.set_title(f'{label}\nyield={nf_yield:.2e}, peak={peak_ratio:.2f}x', fontsize=10, color=color)
    ax.set_xlabel('x (um)')
    if col_idx == 0:
        ax.set_ylabel('y (um)')
    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    if col_idx == 3:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10(I/I_max)')

# Row 2: near-field lineouts (absolute scale, all normalized to global max)
for col_idx, mn in enumerate(hmc5_col):
    ax = axes_hmc5[1, col_idx]
    I_nf = hmc5_I_nf[mn]
    I_x = I_nf[hmc5_nf_center, :] / max(hmc5_I_max_global, 1e-30)
    I_y = I_nf[:, hmc5_nf_center] / max(hmc5_I_max_global, 1e-30)
    label, color = mask_disp[mn]
    ax.plot(x_hhg_um, I_x, color=color, linewidth=2.0, label='x')
    ax.plot(x_hhg_um, I_y, color=color, linewidth=1.6, linestyle='--', alpha=0.8, label='y')
    # Overlay unblocked for reference
    I_ref_x = hmc5_I_nf['none'][hmc5_nf_center, :] / max(hmc5_I_max_global, 1e-30)
    ax.plot(x_hhg_um, I_ref_x, color='gray', linewidth=0.8, alpha=0.5, label='No mask')
    ax.set_xlabel('Position (um)')
    if col_idx == 0:
        ax.set_ylabel(r'$|E_q|^2$ / $|E_q|^2_{max,all}$')
    ax.set_title(f'{label} lineouts (abs.)', fontsize=10)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)
    ax.set_xlim([-60, 60])

fig_hmc5.suptitle(f'Near-Field HHG Yield — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar, Grid={N_hhg_2d}x{N_hhg_2d}', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc5)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc5_nearfield_2d_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc5_nearfield_2d_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-6: Far-Field 2D HHG for All Masks
# =============================================================================
TIMER.start_section("Figure HHG-MC-6 - Far-field 2D maps")
print("Generating HHG-MC-6: far-field 2D HHG for all masks...")

fig_hmc6, axes_hmc6 = plt.subplots(2, 4, figsize=(24, 11))
theta_extent_mc = [theta_mrad[0], theta_mrad[-1], theta_mrad[0], theta_mrad[-1]]
aperture_half_mrad_mc = aperture_half_angle * 1e3

# Row 1: 2D far-field (common normalization, log scale, with aperture circle)
hmc6_ff_max = max(mask_results[mn]['I_ff'].max() for mn in hmc5_col)
for col_idx, mn in enumerate(hmc5_col):
    ax = axes_hmc6[0, col_idx]
    I_ff = mask_results[mn]['I_ff']
    I_ff_log = np.log10(np.clip(I_ff / max(hmc6_ff_max, 1e-30), 1e-6, None))
    im = ax.imshow(I_ff_log, extent=theta_extent_mc, aspect='equal', origin='lower',
                   cmap='magma', vmin=-4, vmax=0, interpolation='bicubic')
    circ = plt.Circle((0, 0), aperture_half_mrad_mc, fill=False, color='white',
                       linestyle='--', linewidth=1.5)
    ax.add_patch(circ)
    label, color = mask_disp[mn]
    ap_yield = mask_results[mn]['yield_ap']
    peak_ratio = I_ff.max() / max(hmc6_ff_max, 1e-30)
    ax.set_title(f'{label}\nap_yield={ap_yield:.2e}, peak={peak_ratio:.2f}x', fontsize=10, color=color)
    ax.set_xlabel(r'$\theta_x$ (mrad)')
    if col_idx == 0:
        ax.set_ylabel(r'$\theta_y$ (mrad)')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    if col_idx == 3:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10(I/I_max)')

# Row 2: far-field lineouts (x and y separately)
for col_idx, mn in enumerate(hmc5_col):
    ax = axes_hmc6[1, col_idx]
    I_ff = mask_results[mn]['I_ff']
    c_ff = N_hhg_2d // 2
    I_ff_x = I_ff[c_ff, :]
    I_ff_y = I_ff[:, c_ff]
    I_ff_x_n = I_ff_x / max(I_ff_x.max(), 1e-30)
    I_ff_y_n = I_ff_y / max(I_ff_y.max(), 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_x_n, color=color, linewidth=2.0, label=r'$\theta_x$')
    ax.semilogy(theta_mrad, I_ff_y_n, color=color, linewidth=1.6, linestyle='--', alpha=0.8, label=r'$\theta_y$')
    ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Angle (mrad)')
    if col_idx == 0:
        ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
    ax.set_title(f'{label} angular', fontsize=10)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)
    ax.set_xlim([-10, 10])
    ax.set_ylim([1e-6, 1.5])

fig_hmc6.suptitle(f'Far-Field HHG — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'Aperture: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc6)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc6_farfield_2d_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc6_farfield_2d_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MC-7: Far-Field Angular Comparison (all masks overlaid, ±10 mrad)
# =============================================================================
TIMER.start_section("Figure HHG-MC-7 - Far-field angular comparison")
print("Generating HHG-MC-7: far-field angular comparison (all masks)...")

fig_hmc7, axes_hmc7 = plt.subplots(2, 2, figsize=(16, 12))
hmc7_masks = ['none', 'circular', 'twosided', 'diagonal']
hmc7_c = N_hhg_2d // 2
# Global max for absolute normalization
hmc7_ff_peak_max = max(mask_results[mn]['I_ff'].max() for mn in hmc7_masks)

# (0,0) x-lineout overlay (self-normalized, log)
ax = axes_hmc7[0, 0]
for mn in hmc7_masks:
    I_ff_x = mask_results[mn]['I_ff'][hmc7_c, :]
    I_ff_x_n = I_ff_x / max(I_ff_x.max(), 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_x_n, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm)')
ax.set_title(r'$\theta_x$ lineout — shape comparison')
ax.legend(fontsize=9)
style_paper_axis(ax, grid=True)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (0,1) y-lineout overlay (self-normalized, log)
ax = axes_hmc7[0, 1]
for mn in hmc7_masks:
    I_ff_y = mask_results[mn]['I_ff'][:, hmc7_c]
    I_ff_y_n = I_ff_y / max(I_ff_y.max(), 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_y_n, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm)')
ax.set_title(r'$\theta_y$ lineout — shape comparison')
ax.legend(fontsize=9)
style_paper_axis(ax, grid=True)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (1,0) x-lineout overlay (absolute, normalized to global max)
ax = axes_hmc7[1, 0]
for mn in hmc7_masks:
    I_ff_x = mask_results[mn]['I_ff'][hmc7_c, :]
    I_ff_x_abs = I_ff_x / max(hmc7_ff_peak_max, 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_x_abs, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ / $|E_{ff}|^2_{max,all}$')
ax.set_title(r'$\theta_x$ lineout — absolute intensity')
ax.legend(fontsize=9)
style_paper_axis(ax, grid=True)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (1,1) y-lineout overlay (absolute, normalized to global max)
ax = axes_hmc7[1, 1]
for mn in hmc7_masks:
    I_ff_y = mask_results[mn]['I_ff'][:, hmc7_c]
    I_ff_y_abs = I_ff_y / max(hmc7_ff_peak_max, 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_y_abs, color=color, linewidth=2.0, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ / $|E_{ff}|^2_{max,all}$')
ax.set_title(r'$\theta_y$ lineout — absolute intensity')
ax.legend(fontsize=9)
style_paper_axis(ax, grid=True)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

fig_hmc7.suptitle(f'Far-Field Angular Comparison (±10 mrad) — {hhg_gas_type.capitalize()}, '
                   f'H{hhg_harmonic_order}, P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
finalize_paper_figure(fig_hmc7)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc7_farfield_angular_comparison_{_m2_tag}.png', dpi=400)
print(f"  Saved: hhg_mc7_farfield_angular_comparison_{_m2_tag}.png")

# =============================================================================
# Figures HHG-MC-8a/b: Per-Mask Macroscopic HHG Yield (like HHG-4)
# =============================================================================
# Reference: unblocked data
r_ub = mask_results['none']
I_q_ub_mc = np.abs(r_ub['E_q'])**2
yield_ub_mc = r_ub['yield_nf']
c_ub_mc = N_hhg_2d // 2
I_q_x_ub_mc = I_q_ub_mc[c_ub_mc, :]
I_q_y_ub_mc = I_q_ub_mc[:, c_ub_mc]

for mc8_mname in ['twosided', 'diagonal']:
    TIMER.start_section(f"Figure HHG-MC-8 - {mc8_mname} yield")
    mc8_label, mc8_color = mask_disp[mc8_mname]
    print(f"\nGenerating HHG-MC-8: Macroscopic HHG Yield for {mc8_label}...")

    r_m = mask_results[mc8_mname]
    I_q_m_mc = np.abs(r_m['E_q'])**2
    yield_m_mc = r_m['yield_nf']
    yield_ratio_mc = yield_m_mc / max(yield_ub_mc, 1e-30)
    I_q_x_m = I_q_m_mc[c_ub_mc, :]
    I_q_y_m = I_q_m_mc[:, c_ub_mc]

    vmax_shared_mc = max(I_q_m_mc.max(), I_q_ub_mc.max())

    fig_mc8, axes_mc8 = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) 2D near-field — masked beam (self-normalized)
    ax = axes_mc8[0, 0]
    vmax_m = max(I_q_m_mc.max(), 1e-30)
    im = ax.imshow(I_q_m_mc.T / vmax_m, extent=hhg_extent, aspect='equal',
                   origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Self-norm.')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title(f'{mc8_label} HHG (H{hhg_harmonic_order}) '
                 f'[peak={vmax_m/max(vmax_shared_mc,1e-30):.2e} rel]')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])

    # (0,1) 2D near-field — unblocked (common normalization)
    ax = axes_mc8[0, 1]
    im = ax.imshow(I_q_ub_mc.T / max(vmax_shared_mc, 1e-30), extent=hhg_extent,
                   aspect='equal', origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalized')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title(f'Unblocked HHG (H{hhg_harmonic_order})')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])

    # (0,2) Bar chart: yield comparison
    ax = axes_mc8[0, 2]
    bars = ax.bar([mc8_label, 'Unblocked'], [yield_m_mc, yield_ub_mc],
                  color=[mc8_color, 'salmon'], edgecolor='black')
    ax.set_ylabel('Total HHG Yield (arb. units)')
    ax.set_title(f'Integrated Yield (ratio = {yield_ratio_mc:.3f})')
    style_paper_axis(ax, grid=True)
    for bar, val in zip(bars, [yield_m_mc, yield_ub_mc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)

    # (1,0) x and y lineouts
    ax = axes_mc8[1, 0]
    lineout_max_mc = max(I_q_x_m.max(), I_q_y_m.max(),
                         I_q_x_ub_mc.max(), I_q_y_ub_mc.max(), 1e-30)
    ax.plot(x_hhg_um, I_q_x_m / lineout_max_mc, color=mc8_color, linewidth=2,
            label=f'{mc8_label} (x)')
    ax.plot(x_hhg_um, I_q_y_m / lineout_max_mc, color=mc8_color, linewidth=1.6,
            linestyle='--', label=f'{mc8_label} (y)')
    ax.plot(x_hhg_um, I_q_x_ub_mc / lineout_max_mc, 'gray', linewidth=1.6,
            alpha=0.6, label='Unblocked (x)')
    ax.plot(x_hhg_um, I_q_y_ub_mc / lineout_max_mc, 'gray', linewidth=1,
            linestyle='--', alpha=0.4, label='Unblocked (y)')
    ax.set_xlabel('Position (um)')
    ax.set_ylabel(r'$|E_q|^2$ (normalized)')
    ax.set_title('HHG Lineouts (x and y)')
    ax.legend(fontsize=8.5, loc='best', frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)
    ax.set_xlim([-50, 50])

    # (1,1) On-axis buildup vs z
    ax = axes_mc8[1, 1]
    bu_m = r_m['buildup_onaxis']
    bu_ub = r_ub['buildup_onaxis']
    bu_m_norm = np.abs(bu_m)**2 / max(np.abs(bu_m[-1])**2, 1e-30)
    bu_ub_norm = np.abs(bu_ub)**2 / max(np.abs(bu_ub[-1])**2, 1e-30)
    ax.plot(r_m['z_gas_mm'], bu_m_norm, color=mc8_color, linewidth=2.4,
            label=mc8_label)
    ax.plot(r_ub['z_gas_mm'], bu_ub_norm, 'gray', linewidth=1.8, linestyle='--',
            alpha=0.7, label='Unblocked')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel(r'On-axis $|E_q|^2$ (normalized)')
    ax.set_title('On-axis HHG Buildup')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # (1,2) Far-field 2D (log scale)
    ax = axes_mc8[1, 2]
    I_ff_m = mask_results[mc8_mname]['I_ff']
    I_ff_ub = mask_results['none']['I_ff']
    ff_max_mc = max(I_ff_m.max(), I_ff_ub.max())
    I_ff_log = np.log10(np.clip(I_ff_m / max(ff_max_mc, 1e-30), 1e-6, None))
    im = ax.imshow(I_ff_log, extent=theta_extent_mc, aspect='equal',
                   origin='lower', cmap='magma', vmin=-4, vmax=0,
                   interpolation='bicubic')
    circ = plt.Circle((0, 0), aperture_half_mrad_mc, fill=False, color='white',
                       linestyle='--', linewidth=1.5)
    ax.add_patch(circ)
    plt.colorbar(im, ax=ax, label='log10(I/I_max)')
    ax.set_xlabel(r'$\theta_x$ (mrad)')
    ax.set_ylabel(r'$\theta_y$ (mrad)')
    ap_yield_m = r_m['yield_ap']
    ap_yield_ub = r_ub['yield_ap']
    ax.set_title(f'Far-field (ap ratio={ap_yield_m/max(ap_yield_ub,1e-30):.2f}x)')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])

    fig_mc8.suptitle(f'Macroscopic HHG Yield — {mc8_label} Mask, {hhg_gas_type.capitalize()}, '
                     f'P={hhg_gas_pressure:.0f} mbar, H{hhg_harmonic_order}, '
                     f'Grid={N_hhg_2d}x{N_hhg_2d}', fontsize=13)
    finalize_paper_figure(fig_mc8)
    plt.tight_layout()
    fname_mc8 = f'hhg_yield_{mc8_mname}_{_m2_tag}.png'
    plt.savefig(fname_mc8, dpi=400)
    print(f"  Saved: {fname_mc8}")

# =============================================================================
# TIMING SUMMARY
# =============================================================================
TIMER.summary()

plt.show()
