"""
Fitting + HHG Yield: Random Phase Screen (RPS) Beam Model
==========================================================
Merged pipeline: parametric fitting of dipole response d_q(I) followed by
macroscopic HHG yield computation using fitted parameters.

Workflow:
  1. Beam propagation (RPS model, M²-degraded Gaussian)
  2. Optical diagnostics (controlled by PLOT_OPTICAL_DIAGNOSTICS flag)
  3. Parametric fitting: |d_q(I)| = sqrt(I^alpha * exp(-I/Is))
     → outputs best_alphas[q], best_Is_vals[q] per harmonic
  4. HHG yield computation using fitted parameters
  5. Multi-mask comparison (circular, two-side, diagonal)

Phase from Lewenstein SFA; magnitude from empirical peaked response.
All units normalized to mm for beam propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ThreadPoolExecutor
import os
import time
import gc  # For memory management
from math import factorial

import scipy.signal as signal
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, brentq


def configure_paper_matplotlib():
    """Use consistent paper-style typography and axes for generated figures."""
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
        'axes.linewidth': 1.25,
        'lines.linewidth': 1.9,
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
wavelength = 790e-6     # wavelength (mm), 790 nm
k = 2 * np.pi / wavelength  # wave number (1/mm)

# Beam quality factors (M² = 1 for ideal Gaussian)
M2x = 1.5  # M² in x direction
M2y = 1.5  # M² in y direction
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
PROPAGATION_METHOD = 'asm'

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
beam_radius_eff = max(w0x_measured, w0y_measured)

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


def get_cache_stats():
    """Return cache statistics for performance monitoring."""
    czt_total = _cache_stats['czt_hits'] + _cache_stats['czt_misses']
    quad_total = _cache_stats['quad_hits'] + _cache_stats['quad_misses']

    czt_rate = _cache_stats['czt_hits'] / czt_total * 100 if czt_total > 0 else 0
    quad_rate = _cache_stats['quad_hits'] / quad_total * 100 if quad_total > 0 else 0

    return {
        'czt_hits': _cache_stats['czt_hits'],
        'czt_misses': _cache_stats['czt_misses'],
        'czt_hit_rate': czt_rate,
        'quad_hits': _cache_stats['quad_hits'],
        'quad_misses': _cache_stats['quad_misses'],
        'quad_hit_rate': quad_rate,
        'czt_cache_size': len(_czt_cache),
        'quad_cache_size': len(_quad_phase_cache)
    }


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
    'none':     ('Unblocked', 'gray'),
    'circular': ('Circular',  'blue'),
    'twosided': ('Two-side',  'red'),
    'diagonal': ('Diagonal',  'green'),
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
        'none':     ('Unblocked', 'gray'),
        'circular': ('Circular',  'blue'),
        'twosided': ('Two-side',  'red'),
        'diagonal': ('Diagonal',  'green'),
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
    mc_bp_title_fs = 16
    mc_bp_label_fs = 15
    mc_bp_tick_fs = 12
    mc_bp_value_fs = 11
    mc_bp_suptitle_fs = 19
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
                ha='center', fontsize=mc_bp_value_fs)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
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
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
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
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
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
                ha='center', fontsize=mc_bp_value_fs)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
    ax.set_ylabel('Rayleigh range (mm)')
    ax.set_title('Confocal Parameter (on-axis half-max)')
    ax.grid(True, alpha=0.3, axis='y')

    # (1,1) Focus shift from geometric focus
    ax = axes_mc5[1, 1]
    shifts = [(mc_data[m]['true_focus_z'] - focal_length)*1e3 for m in mc_masks_all]  # in um
    bars = ax.bar(mc_bar_x, shifts, color=mc_bar_colors, alpha=0.8)
    for b, v in zip(bars, shifts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5 if v >= 0 else v - 2,
                f'{v:.1f}', ha='center', fontsize=mc_bp_value_fs)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
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
                ha='center', fontsize=mc_bp_value_fs)
    ax.set_xticks(mc_bar_x)
    ax.set_xticklabels(mc_bar_labels, fontsize=mc_bp_tick_fs)
    ax.set_ylabel('Transmission (%)')
    ax.set_title('Power Transmission')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    for ax in axes_mc5.flat:
        ax.set_title(ax.get_title(), fontsize=mc_bp_title_fs, fontweight='bold')
        ax.xaxis.label.set_fontsize(mc_bp_label_fs)
        ax.yaxis.label.set_fontsize(mc_bp_label_fs)
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        ax.tick_params(axis='both', which='major', labelsize=mc_bp_tick_fs)

    fig_mc5.suptitle('Multi-Mask Beam Parameters Summary', fontsize=mc_bp_suptitle_fs, y=0.98)
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
# SFA LEWENSTEIN MODEL FUNCTIONS (refactored from SFALeweinsteinVe6pPT.py)
# =============================================================================
# These functions have all globals converted to explicit parameters.

@njit
def sfa_E_flist(t, E0, omega, Thalf, Tcut, omega2, chirp_b):
    """Compute electric field array for SFA calculation."""
    Eflist = np.zeros_like(t)
    for count in range(len(t)):
        if t[count] < Tcut:
            Eflist[count] = E0 * np.cos(omega * (t[count] - Thalf) + chirp_b * (t[count] - Thalf)**2) * np.sin(omega2 * t[count])**2
        else:
            Eflist[count] = 0.0
    return Eflist

@njit
def sfa_A_flist(tlist, E0, omega, Thalf, Tcut, omega2, chirp_b):
    """Compute vector potential array for SFA calculation."""
    Aflist = np.zeros_like(tlist, dtype=np.complex128)
    for count in range(len(tlist)):
        if tlist[count] < Tcut:
            Eflist = E0 * np.cos(omega * (tlist[0:count] - Thalf) + chirp_b * (tlist[0:count] - Thalf)**2) * np.sin(omega2 * tlist[0:count])**2
            Aflist[count] = -np.trapz(Eflist, tlist[0:count])
        else:
            break
    if count + 1 < len(tlist):
        Aflist[count+1:] = Aflist[count]
    return Aflist

@njit
def sfa_dipole_momion(Aflist, Eflist, tlist, Ip, omega, use_ppt):
    """Lewenstein SFA dipole moment. use_ppt=True includes PPT ionization depletion."""
    eta = 0.001
    dmomlist = np.zeros_like(tlist, dtype=np.complex128)
    ionizationprob = np.zeros_like(tlist, dtype=np.complex128)
    gammapara = 1
    remain = np.zeros_like(tlist)
    wadk = 0.0
    ns = 1 / np.sqrt(2 * Ip)
    ls = ns - 1
    m = 0
    Glm = 3.0
    Cnl2 = 4.11546
    F0 = (2 * Ip)**(3.0 / 2.0)
    remain[0] = 1
    dt = tlist[1] - tlist[0]
    for tcount in range(len(tlist)):
        if tcount != 0:
            pslist = np.zeros(tcount, dtype=np.complex128)
            dval = np.zeros(tcount, dtype=np.complex128)
            dvalconj = np.zeros(tcount, dtype=np.complex128)
            tau = np.zeros(tcount, dtype=np.complex128)
            Sactlist = np.zeros(tcount, dtype=np.complex128)
            for i in range(tcount):
                pslist[i] = -np.trapz(Aflist[i:tcount], tlist[i:tcount]) / (tlist[tcount] - tlist[i])
                tau[i] = (tlist[tcount] - tlist[i])
                p = pslist[i] + Aflist[i]
                dval[i] = 1j * p * ((p**2 + gammapara**2) + (0.5 * p**2 + Ip)) * (gammapara + 2 * np.sqrt(2 * Ip)) / ((p**2 + gammapara**2)**(3.0 / 2.0) * (p**2 / 2 + Ip)**2 * (2 * np.pi * (2 * Ip)**(-1.0 / 4.0)))
                p = pslist[i] + Aflist[tcount]
                dvalconj[i] = 1j * p * ((p**2 + gammapara**2) + (0.5 * p**2 + Ip)) * (gammapara + 2 * np.sqrt(2 * Ip)) / ((p**2 + gammapara**2)**(3.0 / 2.0) * (p**2 / 2 + Ip)**2 * (2 * np.pi * (2 * Ip)**(-1.0 / 4.0)))
                dvalconj[i] = np.conjugate(dvalconj[i])
                Sactlist[i] = np.trapz((pslist[i] + Aflist[i:tcount])**2 / 2 + Ip, tlist[i:tcount])

            if use_ppt:
                # PPT ionization rate
                E_inst = np.abs(Eflist[tcount])
                if E_inst > 1e-9:
                    kappa = np.sqrt(2 * Ip)
                    gamma_k = omega * kappa / E_inst
                    term1 = (1 + 1 / (2 * gamma_k**2)) * np.arcsinh(gamma_k)
                    term2 = np.sqrt(1 + gamma_k**2) / (2 * gamma_k)
                    g_gamma = (3.0 / (2 * gamma_k)) * (term1 - term2)
                    prefactor_corr = 1
                    adk_exponent = -2 * F0 / (3 * E_inst)
                    wadk = Cnl2 * Glm * Ip * (2 * F0 / E_inst)**(2 * ns - 1) * prefactor_corr * np.exp(adk_exponent * g_gamma)
                else:
                    wadk = 0.0
                remain[tcount] = remain[tcount-1] - wadk * dt * remain[tcount-1]
                if remain[tcount] < 0:
                    remain[tcount] = 0
            else:
                remain[tcount] = 1.0
            a = 1j * remain[tcount] * (np.pi / (eta + 1j * (tau) / 2))**(3.0 / 2.0) * Eflist[0:tcount] * dval * np.exp(-1j * Sactlist) * dvalconj
            dmomlist[tcount] = np.trapz(a, tlist[0:tcount])
            ionizationprob[tcount] = np.trapz(1j * remain[tcount] * Eflist[0:tcount] * dval * np.exp(-1j * Sactlist), tlist[0:tcount])
        else:
            dmomlist[0] = 0
            ionizationprob[0] = 0
    return dmomlist, ionizationprob, remain

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
hhg_trajectory = 'short'         # 'short' or 'long' electron trajectory
pulse_fwhm_fs = 55.0              # FWHM of Gaussian pulse envelope (fs)
hhg_lut_include_ppt = False      # False = pure SFA without PPT depletion (avoids double-counting with macroscopic (1-nf))
hhg_lut_mode = 'deconvolved'     # 'lewenstein', 'powerlaw', 'experimental', or 'deconvolved'
hhg_lut_powerlaw_exp = 4.0       # Power law exponent n (only used if mode='powerlaw')

# Experimental H21 yield vs intensity (from Mathematica analysis of lab data 24-12-3)
exp_yield_I_Wcm2 = np.array([1.74, 2.04, 2.31, 2.37, 2.55, 2.67, 2.85]) * 1e14  # W/cm²
exp_yield_H21 = np.array([255.50, 424.63, 464.11, 486.05, 448.98, 472.43, 506.14])  # arb. units

# Multi-harmonic unblocked yield vs intensity (same 7 intensity points, low→high)
exp_yield_multi = {
    11: np.array([1254.17, 1917.40, 2071.34, 2138.20, 2251.15, 2282.78, 2308.66]),
    13: np.array([1307.14, 2683.21, 3163.65, 3527.96, 4192.40, 4468.43, 4304.46]),
    15: np.array([918.94, 1413.24, 1586.04, 1668.70, 1741.57, 1808.46, 1887.21]),
    17: np.array([715.44, 1113.52, 1206.96, 1277.79, 1387.12, 1442.31, 1522.66]),
    19: np.array([433.74, 728.48, 796.15, 789.09, 932.30, 1007.91, 973.39]),
    21: np.array([255.50, 424.63, 464.11, 486.05, 448.98, 472.43, 506.14]),
}

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

exp_extrap_mode = 'global_fit'  # '2point', 'global_fit', 'constant', or float (manual slope)
hhg_sigma_xuv_Mb = sigma_xuv_multi_Mb.get(hhg_harmonic_order, 10.0)  # XUV absorption cross-section (Mb) per harmonic
deconv_alpha_per_h = {11: 5.14, 13: 5.23, 15: 5.00, 17: 0.50, 19: 4.05, 21: 4.13}
deconv_Is_per_h = {11: 3.00e13, 13: 3.00e13, 15: 3.00e13, 17: 3.76e13, 19: 3.00e13, 21: 3.00e13}
deconv_alpha = deconv_alpha_per_h.get(hhg_harmonic_order, 4.13)
deconv_Is = deconv_Is_per_h.get(hhg_harmonic_order, 3.00e13)

# Experimental enhancement data (blocked/unblocked yield ratio)
# Intensities: [950, 890, 850, 790, 770, 680, 580] * 3/1000 in units of 10^14 W/cm²
exp_intensities_1e14 = np.array([950, 890, 850, 790, 770, 680, 580]) * 3 / 1000  # ×10^14 W/cm²
exp_enhancement = {
    'H21': np.array([3.6198571818344334, 3.947580624839355, 3.952835241524503,
                     3.111838270232141, 2.974128776413735, 2.2874821273374644,
                     1.8477975230871022]),
    'H19': np.array([3.1585861921359997, 3.2060265895626054, 3.1856280390634577,
                     3.3378030161306866, 2.755735920378966, 2.089089494773861,
                     1.6911895482241863]),
    'H17': np.array([2.7409846165393317, 3.0292535805550607, 2.8994146827099674,
                     2.8594297064682084, 2.48762531513724, 2.0371835443037973,
                     1.5704815247451605]),
    'H15': np.array([2.3435167569149717, 2.6185116528910384, 2.5571625438507746,
                     2.3536019678877027, 2.321838666336403, 1.9102383748258394,
                     1.6348677051270093]),
    'H13': np.array([1.1246807712709908, 1.1739569679337576, 1.1219293891547395,
                     1.259469422160253, 1.2123975058984673, 1.076141933789314,
                     1.3366278201377781]),
    'H11': np.array([1.9307029331595225, 2.070968085117479, 1.972864605423059,
                     1.906308808234767, 1.7556442185339343, 1.4577961669228838,
                     1.3884154432912577])
}

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
    omega_au = 0.05767513  # 790 nm angular frequency in a.u.
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

def calc_dk_gouy_from_sim(gouy_phase_onaxis, z_m, q):
    """
    Gouy phase mismatch from simulation data.
    Input phase must have plane wave already removed (via field * exp(-ikz)).
    Returns Dk_Gouy in 1/m.
    """
    phase_uw = np.unwrap(gouy_phase_onaxis)  # works: ~pi/200 change per step
    phase_uw -= phase_uw[len(phase_uw) // 2]  # center at 0
    dphase_dz = np.gradient(phase_uw, z_m)
    return -(q - 1.0 / q) * dphase_dz

def calc_dk_dipole(I_onaxis_Wcm2, z_m, trajectory='short'):
    """Atomic dipole phase mismatch: -alpha * dI/dz (1/m)."""
    alpha_dict = {
        'short': 1.0e-14,   # rad * cm^2 / W
        'long':  5.0e-14,
    }
    alpha_cgs = alpha_dict[trajectory]
    alpha_SI = alpha_cgs * 1e-4  # rad * m^2 / W
    # Convert intensity to W/m^2 for gradient, then dI/dz in W/m^2/m
    I_Wm2 = I_onaxis_Wcm2 * 1e4
    dI_dz = np.gradient(I_Wm2, z_m)
    return -alpha_SI * dI_dz

# --- Unit conversions ---
z_m = z_focus_prop * 1e-3                  # z positions in meters
lambda_0_m = wavelength * 1e-3             # 790e-9 m
z_focus_ub_m = true_focus_z_unblocked * 1e-3

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
# FITTING PIPELINE (from fitting with yield random phase screen.py)
# =============================================================================

# --- Fitting config flags ---
SENSITIVITY_CHECK_MODE = False  # False = run full simulation
LAVG_MM_LIST = [1.0]                 # single pass (smoothing disabled)
USE_TEMPORAL_AVG = False              # disabled
N_TEMPORAL_PTS = 11
PLOT_TEMPORAL_PM_DIAGNOSTIC = False   # disabled for merged file

# --- Color palette for fitting figures ---
COLORS_HQ = {
    11: '#0072B2',  # blue
    13: '#E69F00',  # orange
    15: '#009E73',  # teal
    17: '#CC79A7',  # pink
    19: '#D55E00',  # vermillion
    21: '#56B4E9',  # sky blue
}
COLORS_LIST = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
COLOR_SIM = '#0072B2'
COLOR_EXP = '#D55E00'
COLOR_REF = '#999999'

# --- Aperture transmission helper ---
# --- Aperture-variation experimental data ---
# Aperture-variation experimental data (fixed power, varying iris opening)
# Experiment at I = 2.37e14 W/cm^2 (unblocked peak), different day from power-variation
exp_aperture_x = np.arange(6.0, 7.3, 0.1)  # 13 iris settings
exp_aperture_power_mW = np.array([780, 765, 760, 747, 738, 710, 690, 670, 645, 613, 575, 550, 510])
exp_aperture_T = exp_aperture_power_mW / 780.0  # transmission fractions
exp_aperture_I_Wcm2 = 2.37e14  # peak intensity without blocking

exp_aperture_yield = {
    11: np.array([1578.32, 1601.33, 1770.69, 1738.11, 1955.45, 2070.43, 2078.13, 2116.87, 1991.53, 1790.85, 1157.99, 739.02, 324.63]),
    13: np.array([2000.36, 1917.87, 2023.91, 1820.43, 2069.47, 2184.82, 2044.77, 2021.46, 1963.54, 1894.81, 1198.70, 922.58, 292.86]),
    15: np.array([1342.27, 1334.36, 1489.60, 1555.04, 1800.85, 1941.45, 1875.08, 2059.73, 1962.93, 1637.29, 1020.43, 665.88, 323.31]),
    17: np.array([958.45, 1080.71, 1221.05, 1269.30, 1467.97, 1603.97, 1809.79, 1732.13, 1413.00, 1442.86, 860.18, 426.69, 525.75]),
    19: np.array([559.91, 664.61, 773.52, 811.49, 967.45, 1057.02, 1149.65, 1226.18, 1195.77, 905.52, 837.23, 693.40, 367.86]),
    21: np.array([323.22, 363.73, 408.20, 464.03, 555.06, 682.30, 663.83, 645.54, 644.83, 712.53, 617.20, 296.91, 144.05]),
}


# --- Fitting hyperparameters ---
APERTURE_SUBSAMPLE_STEP = 2  # use every 2nd aperture for fitting
ap_indices = list(range(0, len(exp_aperture_T), APERTURE_SUBSAMPLE_STEP))  # [0,2,4,6,8,10,12]
n_ap_sub = len(ap_indices)  # 7
ap_ref_idx = 0  # normalize to least-blocked aperture
lambda_ap = 1.0  # weight for aperture cost relative to power cost

# Joint optimization hyperparameters
lambda_smooth_alpha = 0.3   # 2nd-order difference penalty on alpha
lambda_smooth_Is = 0.5      # 2nd-order difference penalty on log(Is)
lambda_alpha_box = 0.2      # weak soft-box penalty outside the alpha range below
alpha_box_min = 2.0
alpha_box_max = 6.0
sigma_alpha_box = 1.5
lambda_prior = 0.1          # soft Is prior weight
log_Is_prior_center = np.log(3e13)  # SFA deconv median
log_Is_prior_width = 1.5            # ~1 order of magnitude

def alpha_soft_box_penalty(alpha_val):
    below = np.maximum((alpha_box_min - alpha_val) / sigma_alpha_box, 0.0)
    above = np.maximum((alpha_val - alpha_box_max) / sigma_alpha_box, 0.0)
    return lambda_alpha_box * (below**2 + above**2)

# Parameter bounds (hard)
ALPHA_MIN, ALPHA_MAX = 1.0, 8.0     # α range 1-8
LOG_IS_MIN, LOG_IS_MAX = None, None  # set from Is_grid after parameter grid definition


# --- Temporal ionization LUT (for temporal averaging) ---
def build_tbi_ionization_lut_temporal(gas_params, tau_fwhm_fs, n_temporal=11,
                                      I_min=1e12, I_max=5e14, n_I=500):
    """Build temporal ionization LUTs: n_temporal 1D interpolators.

    For each time slice t_j, ionization_luts[j](I_peak) gives the
    cumulative ionization fraction nf(t_j) for a pulse with peak intensity I_peak.
    Also returns intensity fractions g²(t_j) at each time point.
    """
    Ip_au = gas_params['Ip_eV'] / 27.2114
    Zc, l, m, alpha_tl = gas_params['Z'], gas_params['l'], gas_params['m'], gas_params['alpha_tl']

    tau_au = tau_fwhm_fs * 1e-15 / 2.4189e-17
    sigma_au = tau_au / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    omega_au = 0.05767513
    T_cycle = 2.0 * np.pi / omega_au

    # Output time points: span FULL pulse (-1.5σ to +1.5σ)
    t_out_au = np.linspace(-1.5 * sigma_au, 1.5 * sigma_au, n_temporal)
    I_fracs = np.exp(-2.0 * np.log(2.0) * (2.0 * t_out_au / tau_au)**2)

    I_grid = np.logspace(np.log10(I_min), np.log10(I_max), n_I)
    nf_2d = np.zeros((n_I, n_temporal))

    for i, I_peak in enumerate(I_grid):
        I_au = I_peak / 3.5094e16
        E0_au = np.sqrt(I_au)

        # Fine time grid for integration
        t_max = 5.0 * tau_au / 2.0
        dt = T_cycle / 20.0
        t_fine = np.arange(-t_max, t_max, dt)
        E_env = E0_au * np.exp(-2.0 * np.log(2.0) * t_fine**2 / tau_au**2)

        # Cycle-averaged rate
        n_phase = 20
        phases = np.linspace(0, np.pi, n_phase, endpoint=False) + np.pi / (2 * n_phase)
        sin_vals = np.abs(np.sin(phases))
        F_inst = E_env[:, None] * sin_vals[None, :]
        w_inst = w_tbi_rate_au(F_inst, Ip_au, Zc, l, m, alpha_tl)
        w_avg = np.mean(w_inst, axis=1)

        # Cumulative integral at fine time points
        cum_integral = np.cumsum(w_avg) * dt

        # Interpolate to output time points
        for j, t_j in enumerate(t_out_au):
            idx = np.searchsorted(t_fine, t_j)
            idx = min(max(idx, 0), len(cum_integral) - 1)
            nf_2d[i, j] = 1.0 - np.exp(-cum_integral[idx])

    # Build n_temporal 1D interpolators (same log-space scheme as existing LUT)
    log_I = np.log(I_grid)
    luts = []
    for j in range(n_temporal):
        log_surv = np.log(np.maximum(1.0 - nf_2d[:, j], 1e-30))

        def make_interp(log_I_ref, log_surv_ref, I_min_ref):
            def func(I_Wcm2):
                I_arr = np.asarray(I_Wcm2, dtype=np.float64)
                scalar = I_arr.ndim == 0
                I_arr = np.atleast_1d(I_arr)
                result = np.zeros_like(I_arr)
                valid = I_arr > I_min_ref
                if np.any(valid):
                    lq = np.log(np.clip(I_arr[valid], I_min_ref, I_max))
                    ls = np.interp(lq, log_I_ref, log_surv_ref)
                    result[valid] = 1.0 - np.exp(ls)
                return float(result[0]) if scalar else np.clip(result, 0, 1)
            return func

        luts.append(make_interp(log_I.copy(), log_surv.copy(), I_min))

    return luts, I_fracs


# --- Intensity calibration & Gouy phase gradient ---
# Build temporal ionization LUTs (cumulative nf at each time slice)
if USE_TEMPORAL_AVG or PLOT_TEMPORAL_PM_DIAGNOSTIC:
    print(f"Building temporal TBI LUTs ({N_TEMPORAL_PTS} time slices, cumulative ionization)...")
    _tbi_temporal_luts, _temporal_I_fracs = build_tbi_ionization_lut_temporal(
        gas, pulse_fwhm_fs, n_temporal=N_TEMPORAL_PTS)
    # Diagnostic: print rising vs falling edge nf asymmetry
    print(f"  Temporal I_fracs (g²): {np.array2string(_temporal_I_fracs, precision=3)}")
    nf_rise = _tbi_temporal_luts[0](I_test)
    nf_mid = _tbi_temporal_luts[N_TEMPORAL_PTS // 2](I_test)
    nf_fall = _tbi_temporal_luts[-1](I_test)
    print(f"  At I_peak={I_test:.1e}: nf(t_0/rise)={nf_rise:.4f}, "
          f"nf(t_mid)={nf_mid:.4f}, nf(t_end/fall)={nf_fall:.4f}")
else:
    _tbi_temporal_luts = None
    _temporal_I_fracs = None

# Intensity calibration: scale sim -> physical units using unblocked peak
I_scale_factor_2d = hhg_peak_intensity_Wcm2 / xz_intensity_unblocked.max()

# Gas region data in physical units
z_gas_2d_b_m = z_gas_2d_b * 1e-3
z_gas_2d_ub_m = z_gas_2d_ub * 1e-3
I_2d_gas_b_Wcm2 = I_2d_gas_b * I_scale_factor_2d
I_2d_gas_ub_Wcm2 = I_2d_gas_ub * I_scale_factor_2d

dx_hhg_m = (x_hhg_2d[1] - x_hhg_2d[0]) * 1e-3  # mm -> m

# Gouy phase gradient (complex-domain method, robust against wrapping)
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

# Mask low-intensity phase gradient
I_thresh_b = 0.01 * I_2d_gas_b.max()
I_thresh_ub = 0.01 * I_2d_gas_ub.max()
dphase_dz_3d_b[I_2d_gas_b < I_thresh_b] = 0.0
dphase_dz_3d_ub[I_2d_gas_ub < I_thresh_ub] = 0.0

del field_env_b, field_env_ub
gc.collect()

print(f"  Gas region (blocked):   {z_gas_2d_b[0]:.2f} to {z_gas_2d_b[-1]:.2f} mm ({len(z_gas_2d_b)} z-points)")
print(f"  Gas region (unblocked): {z_gas_2d_ub[0]:.2f} to {z_gas_2d_ub[-1]:.2f} mm ({len(z_gas_2d_ub)} z-points)")
print(f"  Peak I (unblocked, on-axis): {I_2d_gas_ub_Wcm2.max():.3e} W/cm^2")


# =============================================================================
# STEP 1: BUILD SFA PHASE LUT (Lewenstein deconvolved method)
# =============================================================================
TIMER.start_section("Step 1: Build SFA phase LUT")
print("Building Lewenstein SFA phase look-up tables for each harmonic...")

# Deconvolved magnitude parameters (per-harmonic)
deconv_alpha_per_h = {11: 3.32, 13: 5.29, 15: 4.77, 17: 3.30, 19: 1.73, 21: 3.24}
deconv_Is_per_h = {11: 8.02e13, 13: 2.77e13, 15: 2.60e13, 17: 1.51e13, 19: 3.06e13, 21: 1.64e13}

# LUT parameters
n_lut = 80
I_lut_min = 1e13
I_lut_max = hhg_peak_intensity_Wcm2
sfa_omega = 0.05767513
sfa_Ip_au = gas['Ip_eV'] / 27.2114
sfa_I_to_E0 = 3.5094e16
hhg_lut_include_ppt = False
multi_q_list_lut = [11, 13, 15, 17, 19, 21]

# --- Check LUT cache ---
_lut_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'_lut_cache_{hhg_gas_type}_n{n_lut}_Imax{hhg_peak_intensity_Wcm2:.0e}.npz')

if os.path.exists(_lut_cache_file):
    print(f"  [LUT CACHE] Loading from {_lut_cache_file}")
    _lut_data = np.load(_lut_cache_file, allow_pickle=True)
    I_lut = _lut_data['I_lut']
    n_lut = len(I_lut)
    I_lut_min = float(_lut_data['I_lut_min'])
    I_lut_max = float(_lut_data['I_lut_max'])
    # Build phase_interp_per_q from cached multi_lut
    multi_lut_cached = {int(k): v for k, v in _lut_data['multi_lut'].item().items()}
    phase_interp_per_q = {}
    for q_lut in multi_q_list_lut:
        phase_interp_per_q[q_lut] = interp1d(I_lut, multi_lut_cached[q_lut]['phase'],
                                              kind='cubic', bounds_error=False, fill_value=0.0)
    del _lut_data, multi_lut_cached
    print(f"  Loaded {n_lut} intensity points, {len(multi_q_list_lut)} harmonics")
    for q_lut in multi_q_list_lut:
        print(f"    H{q_lut}: deconv_alpha={deconv_alpha_per_h[q_lut]:.2f}, deconv_Is={deconv_Is_per_h[q_lut]:.2e}")
else:
    # --- Full LUT computation ---
    sfa_n_cycles = 8
    sfa_dt = 0.5
    sfa_Tfull = 1000.0
    sfa_chirp = 0.0
    sfa_Thalf = np.pi / sfa_omega * sfa_n_cycles
    sfa_Tcut = 2 * sfa_Thalf
    sfa_omega2 = sfa_omega / (2 * sfa_n_cycles)
    I_lut = np.logspace(np.log10(I_lut_min), np.log10(I_lut_max), n_lut)

    sfa_ti = np.linspace(0, sfa_Tfull, num=int(sfa_Tfull / sfa_dt) + 1)
    sfa_window = signal.windows.flattop(len(sfa_ti))
    sfa_omegalist = np.fft.rfftfreq(len(sfa_ti), d=sfa_dt) * 2 * np.pi / sfa_omega

    print(f"  Ip = {sfa_Ip_au:.4f} a.u. ({gas['Ip_eV']:.2f} eV), omega = {sfa_omega} a.u.")
    print(f"  Pulse: {sfa_n_cycles} cycles, dt = {sfa_dt} a.u., {len(sfa_ti)} time points")
    print(f"  Intensity LUT: {n_lut} points, {I_lut_min:.1e} to {I_lut_max:.1e} W/cm^2")

    lut_start = time.time()
    E0_warmup = np.sqrt(I_lut[0] / sfa_I_to_E0)
    Ef_warmup = sfa_E_flist(sfa_ti, E0_warmup, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
    Af_warmup = sfa_A_flist(sfa_ti, E0_warmup, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
    dm_warmup, _, _ = sfa_dipole_momion(Af_warmup, Ef_warmup, sfa_ti, sfa_Ip_au, sfa_omega, hhg_lut_include_ppt)
    print(f"  JIT warmup done ({time.time()-lut_start:.1f}s)")

    phase_interp_per_q = {}
    multi_lut_save = {}

    for q_lut in multi_q_list_lut:
        sfa_idx_q = np.argmin(np.abs(sfa_omegalist - q_lut))
        d_q_lut = np.zeros(n_lut, dtype=np.complex128)
        print(f"  Building LUT for H{q_lut} (idx={sfa_idx_q})...")
        for i_lut in range(n_lut):
            E0_i = np.sqrt(I_lut[i_lut] / sfa_I_to_E0)
            Ef_i = sfa_E_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
            Af_i = sfa_A_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
            dm_i, _, _ = sfa_dipole_momion(Af_i, Ef_i, sfa_ti, sfa_Ip_au, sfa_omega, hhg_lut_include_ppt)
            HHG_i = np.fft.rfft(np.real(dm_i) * sfa_window)
            d_q_lut[i_lut] = HHG_i[sfa_idx_q]
        dq_phase_q = np.unwrap(np.angle(d_q_lut))
        phase_interp_per_q[q_lut] = interp1d(I_lut, dq_phase_q, kind='cubic',
                                               bounds_error=False, fill_value=0.0)
        multi_lut_save[q_lut] = {'mag': np.abs(d_q_lut), 'phase': dq_phase_q}

    lut_time = time.time() - lut_start
    print(f"  All SFA phase LUTs complete in {lut_time:.1f}s ({lut_time/60:.1f} min)")

    # Save cache (compatible with scan files)
    np.savez_compressed(_lut_cache_file,
        I_lut=I_lut, I_lut_min=I_lut_min, I_lut_max=I_lut_max,
        dq_mag=multi_lut_save[21]['mag'], dq_phase=multi_lut_save[21]['phase'],
        multi_lut=multi_lut_save,
        sfa_omega=sfa_omega, sfa_Ip_au=sfa_Ip_au, sfa_I_to_E0=sfa_I_to_E0)
    print(f"  [LUT CACHE] Saved to {_lut_cache_file}")

    for q_lut in multi_q_list_lut:
        print(f"    H{q_lut}: deconv_alpha={deconv_alpha_per_h[q_lut]:.2f}, deconv_Is={deconv_Is_per_h[q_lut]:.2e}")


# =============================================================================
# STEP 1b: MULTI-APERTURE BEAM PROPAGATION
# =============================================================================
TIMER.start_section("Multi-aperture propagation")
print("\nComputing multi-aperture beams for aperture-variation fitting...")
print(f"  Subsampling: {n_ap_sub} of {len(exp_aperture_T)} apertures (step={APERTURE_SUBSAMPLE_STEP})")

# Compute intensity at aperture (for transmission calculation)
I_at_aperture = np.abs(field_at_aperture)**2
power_total_at_aperture = np.sum(I_at_aperture)
R_grid_ap = np.sqrt(X**2 + Y**2)
gas_data_ap = {}  # gas_data_ap[i_ap] = {I_2d_Wcm2, dphase_dz, z_gas_m}

for i_sub, i_ap in enumerate(ap_indices):
    T = exp_aperture_T[i_ap]
    print(f"\n  Aperture {i_sub+1}/{n_ap_sub}: iris={exp_aperture_x[i_ap]:.1f}, T={T:.3f}")

    if T > 0.995:
        print("    T > 0.995 — reusing unblocked beam data")
        gas_data_ap[i_ap] = {
            'I_2d_Wcm2': I_2d_gas_ub_Wcm2,
            'dphase_dz': dphase_dz_3d_ub,
            'z_gas_m': z_gas_2d_ub_m,
        }
        continue

    # Find aperture radius for this transmission
    def _T_residual_ap(r):
        mask = np.where(R_grid_ap <= r, 1.0, 0.0)
        return np.sum(I_at_aperture * mask) / power_total_at_aperture - T
    r_ap = brentq(_T_residual_ap, 0.5, 19.0)
    print(f"    Aperture radius = {r_ap:.3f} mm")

    # Apply circular iris mask
    mask_ap = np.where(R_grid_ap <= r_ap, 1.0, 0.0)

    T_check = np.sum(I_at_aperture * mask_ap) / power_total_at_aperture
    print(f"    Transmission check: {T_check:.4f} (target {T:.4f})")

    # Single-field propagation through iris -> lens
    f_masked_ap = field_at_aperture * mask_ap
    f_at_lens_ap = propagate_field(f_masked_ap, aperture_distance_before_lens, k)
    f_after_lens_ap = thin_lens(f_at_lens_ap, R2, focal_length, k)
    del f_masked_ap, f_at_lens_ap

    # High-res propagation through gas region (single field)
    I_2d_gas_ap_list = []
    phase_geom_2d_gas_ap_list = []
    z_gas_ap_list = []

    for i, z in enumerate(z_focus_prop):
        if i % 50 == 0:
            print(f"    z-step {i+1}/{n_z_steps_hr}")
        if gas_z_start_prop <= z <= gas_z_end_prop:
            field_z, x_out, y_out = fresnel_propagate_zoom(
                f_after_lens_ap, z, k, L, L_xz_focus, N_xz_focus)
            field_no_pw = field_z * np.exp(-1j * k * z)
            I_2d_gas_ap_list.append(np.abs(field_z[hhg_crop, hhg_crop])**2)
            phase_geom_2d_gas_ap_list.append(np.angle(field_no_pw[hhg_crop, hhg_crop]))
            z_gas_ap_list.append(z)

    del f_after_lens_ap

    I_2d_gas_ap = np.array(I_2d_gas_ap_list)
    phase_geom_ap = np.array(phase_geom_2d_gas_ap_list)
    z_gas_ap = np.array(z_gas_ap_list)
    del I_2d_gas_ap_list, phase_geom_2d_gas_ap_list, z_gas_ap_list

    z_gas_ap_m = z_gas_ap * 1e-3
    I_2d_gas_ap_Wcm2 = I_2d_gas_ap * I_scale_factor_2d

    # Gouy phase gradient (complex-domain method)
    field_env_ap = np.sqrt(I_2d_gas_ap) * np.exp(1j * phase_geom_ap)
    dphase_dz_ap = np.zeros_like(phase_geom_ap)
    dz_arr_ap = np.diff(z_gas_ap_m)
    for j in range(1, len(z_gas_ap_m)):
        dphi = np.angle(field_env_ap[j] * np.conj(field_env_ap[j-1]))
        dphase_dz_ap[j] = dphi / dz_arr_ap[j-1]
    dphase_dz_ap[0] = dphase_dz_ap[1]
    I_thresh_ap = 0.01 * I_2d_gas_ap.max()
    dphase_dz_ap[I_2d_gas_ap < I_thresh_ap] = 0.0
    del field_env_ap, I_2d_gas_ap, phase_geom_ap

    gas_data_ap[i_ap] = {
        'I_2d_Wcm2': I_2d_gas_ap_Wcm2,
        'dphase_dz': dphase_dz_ap,
        'z_gas_m': z_gas_ap_m,
    }

    print(f"    Gas region: {len(z_gas_ap_m)} z-points, I_peak = {I_2d_gas_ap_Wcm2.max():.3e} W/cm^2")
    gc.collect()

print(f"\n  Multi-aperture propagation complete: {len(gas_data_ap)} apertures stored")
mem_total = sum(gd['I_2d_Wcm2'].nbytes + gd['dphase_dz'].nbytes
                for gd in gas_data_ap.values()) / 1e9
print(f"  Total aperture data memory: {mem_total:.1f} GB")


# =============================================================================
# STEP 2: MULTI-POWER MULTI-HARMONIC GRID SCAN
# =============================================================================
TIMER.start_section("Step 2: Grid scan (power x harmonic x params)")


# Handle np.trapz deprecation in NumPy >= 2.0
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

multi_q_list = [11, 13, 15, 17, 19, 21]
n_q = len(multi_q_list)
n_P = len(exp_yield_I_Wcm2)

# Parameter grid
alpha_grid = np.arange(1.0, 8.5, 0.5)
Is_grid = np.logspace(np.log10(0.1e14), np.log10(15e14), 24)
LOG_IS_MIN, LOG_IS_MAX = np.log(Is_grid[0]), np.log(Is_grid[-1])
n_alpha = len(alpha_grid)
n_Is = len(Is_grid)

# Detection geometry: 'slit' (spectrometer) or 'circular'
hhg_acceptance_type = 'slit'

# Slit parameters (spectrometer entrance slit: y open, x narrow)
hhg_slit_width_mm = 2.0        # mm, slit width (x direction)
hhg_slit_height_mm = 10.0      # mm, slit height (y direction)
hhg_slit_distance = 1.2        # m, distance from gas jet to slit

# Circular aperture parameters (fallback)
hhg_aperture_radius_mm = 1.0   # mm
hhg_aperture_distance = 0.3    # m

if hhg_acceptance_type == 'slit':
    slit_half_angle_x = (hhg_slit_width_mm / 2) * 1e-3 / hhg_slit_distance  # rad
    slit_half_angle_y = (hhg_slit_height_mm / 2) * 1e-3 / hhg_slit_distance  # rad
    print(f"  Slit acceptance: {hhg_slit_width_mm}x{hhg_slit_height_mm} mm at {hhg_slit_distance} m")
    print(f"    x half-angle: {slit_half_angle_x*1e3:.2f} mrad, y half-angle: {slit_half_angle_y*1e3:.2f} mrad")
else:
    aperture_half_angle = hhg_aperture_radius_mm * 1e-3 / hhg_aperture_distance
    print(f"  Circular aperture: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m, half-angle: {aperture_half_angle*1e3:.2f} mrad")

# Precompute far-field aperture masks (harmonic-dependent, reused across all scans)
ff_ap_masks = {}
for q in multi_q_list:
    lambda_q_m = lambda_0_m / q
    dtheta_q = lambda_q_m / (N_hhg_2d * dx_hhg_m)
    theta_axis_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dtheta_q
    if hhg_acceptance_type == 'slit':
        mask_q = ((np.abs(theta_axis_q[None, :]) <= slit_half_angle_x) &
                  (np.abs(theta_axis_q[:, None]) <= slit_half_angle_y)).astype(float)
    else:
        r_theta_q = np.sqrt(theta_axis_q[:, None]**2 + theta_axis_q[None, :]**2)
        mask_q = (r_theta_q <= aperture_half_angle).astype(float)
    # Build both slit and circular masks for dual far-field comparison
    circ_half_angle = hhg_aperture_radius_mm * 1e-3 / hhg_aperture_distance
    r_theta_q = np.sqrt(theta_axis_q[:, None]**2 + theta_axis_q[None, :]**2)
    mask_circ_q = (r_theta_q <= circ_half_angle).astype(float)
    mask_slit_q = ((np.abs(theta_axis_q[None, :]) <= slit_half_angle_x) &
                   (np.abs(theta_axis_q[:, None]) <= slit_half_angle_y)).astype(float)
    if hhg_acceptance_type == 'slit':
        mask_q = mask_slit_q
    else:
        mask_q = mask_circ_q
    ff_ap_masks[q] = {
        'mask': mask_q,
        'mask_slit': mask_slit_q,
        'mask_circ': mask_circ_q,
        'dtheta': dtheta_q,
        'theta_axis': theta_axis_q,
    }

for LAVG_MM in LAVG_MM_LIST:
    lavg_tag = f'Lavg{LAVG_MM:.1f}'
    print("\n" + "#"*60)
    print(f"  LAVG SWEEP: LAVG_MM = {LAVG_MM} mm ({lavg_tag})")
    print("#"*60)

    # Result arrays
    yield_ub = np.zeros((n_q, n_P, n_alpha, n_Is))

    P_bar_gas = hhg_gas_pressure / 1000.0
    n_gas_density = gas['N_atm'] * P_bar_gas

    print(f"  Grid: {n_alpha} alpha x {n_Is} I_s = {n_alpha*n_Is} parameter pairs")
    print(f"  Total integrations: {n_P} x {n_q} x {n_alpha*n_Is} = {n_P*n_q*n_alpha*n_Is}")

    nz_ub = len(z_gas_2d_ub_m)

    for iP in range(n_P):
        I_exp = exp_yield_I_Wcm2[iP]
        power_scale = I_exp / hhg_peak_intensity_Wcm2
        print(f"\n  Power level {iP+1}/{n_P}: I = {I_exp:.2e} W/cm^2 (scale = {power_scale:.3f})")

        # Scaled 3D intensity
        I_3d = I_2d_gas_ub_Wcm2 * power_scale
        log_I_3d = np.log(np.maximum(I_3d, 1e-30))

        # Ionization at this power
        nf_3d = ionization_fraction(I_3d, gas['Ip_eV'])

        for iq, q in enumerate(multi_q_list):
            lambda_q_m = lambda_0_m / q

            # Phase mismatch terms (no dk_dip — dipole phase is in complex dipole)
            dk_neut = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d, gas['delta_n'])
            dk_plas = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d, gas['N_atm'])
            dk_geom = -(q - 1.0 / q) * dphase_dz_3d_ub
            dk_total = dk_neut + dk_plas + dk_geom

            # Cumulative phase
            Phi_3d = np.zeros_like(dk_total)
            Phi_3d[1:] = cumulative_trapezoid(dk_total, z_gas_2d_ub_m, axis=0)

            # XUV absorption
            sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22
            mu_3d = sigma_q_m2 * n_gas_density * (1.0 - nf_3d)
            mu_cumfwd = np.zeros_like(mu_3d)
            mu_cumfwd[1:] = cumulative_trapezoid(mu_3d, z_gas_2d_ub_m, axis=0)
            tau = mu_cumfwd[-1:] - mu_cumfwd
            abs_factor = np.exp(-tau / 2.0)

            # Base integrand with SFA phase pre-multiplied (parameter-independent)
            base = (1.0 - nf_3d) * np.exp(1j * Phi_3d) * abs_factor
            I_clipped = np.clip(I_3d, I_lut_min, I_lut_max)
            sfa_phase_3d = phase_interp_per_q[q](I_clipped)
            base_complex = base * np.exp(1j * sfa_phase_3d)
            base_complex[I_3d < I_lut_min] = 0.0

            ff = ff_ap_masks[q]

            # Inner loop: (alpha, I_s) parameter grid
            for ia in range(n_alpha):
                alpha = alpha_grid[ia]
                for js in range(n_Is):
                    I_s = Is_grid[js]

                    dq_mag = np.exp(alpha / 2.0 * log_I_3d - I_3d / (2.0 * I_s))
                    dq_mag[I_3d < I_lut_min] = 0.0

                    E_q_2d = _trapz(dq_mag * base_complex, z_gas_2d_ub_m, axis=0)

                    if USE_SCIPY_FFT:
                        E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d, workers=-1)) * dx_hhg_m**2
                    else:
                        E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d)) * dx_hhg_m**2

                    yield_ub[iq, iP, ia, js] = np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2

            del dk_neut, dk_plas, dk_geom, dk_total, Phi_3d, mu_3d, mu_cumfwd, tau, abs_factor, base, base_complex

        del I_3d, log_I_3d, nf_3d

    gc.collect()
    print(f"\n  Power grid scan complete. yield_ub shape: {yield_ub.shape}")

    # --- Aperture-variation grid scan ---
    print("\n  Computing aperture-variation yield grid...")
    yield_ap = np.zeros((n_q, n_ap_sub, n_alpha, n_Is))
    power_scale_ap = exp_aperture_I_Wcm2 / hhg_peak_intensity_Wcm2
    print(f"  Aperture power scale: {power_scale_ap:.3f} (I = {exp_aperture_I_Wcm2:.2e} W/cm^2)")
    print(f"  Total aperture integrations: {n_ap_sub} x {n_q} x {n_alpha*n_Is} = {n_ap_sub*n_q*n_alpha*n_Is}")

    for i_sub, i_ap in enumerate(ap_indices):
        gd = gas_data_ap[i_ap]
        I_3d_ap = gd['I_2d_Wcm2'] * power_scale_ap
        log_I_3d_ap = np.log(np.maximum(I_3d_ap, 1e-30))
        nf_3d_ap = ionization_fraction(I_3d_ap, gas['Ip_eV'])
        z_m_ap = gd['z_gas_m']

        print(f"\n  Aperture {i_sub+1}/{n_ap_sub}: iris={exp_aperture_x[i_ap]:.1f}, T={exp_aperture_T[i_ap]:.3f}")

        for iq, q in enumerate(multi_q_list):
            lambda_q_m = lambda_0_m / q

            dk_neut = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d_ap, gas['delta_n'])
            dk_plas = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d_ap, gas['N_atm'])
            dk_geom = -(q - 1.0 / q) * gd['dphase_dz']
            dk_total = dk_neut + dk_plas + dk_geom

            Phi_3d = np.zeros_like(dk_total)
            Phi_3d[1:] = cumulative_trapezoid(dk_total, z_m_ap, axis=0)

            sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22
            mu_3d = sigma_q_m2 * n_gas_density * (1.0 - nf_3d_ap)
            mu_cumfwd = np.zeros_like(mu_3d)
            mu_cumfwd[1:] = cumulative_trapezoid(mu_3d, z_m_ap, axis=0)
            tau = mu_cumfwd[-1:] - mu_cumfwd
            abs_factor = np.exp(-tau / 2.0)

            base = (1.0 - nf_3d_ap) * np.exp(1j * Phi_3d) * abs_factor
            I_clipped_ap = np.clip(I_3d_ap, I_lut_min, I_lut_max)
            sfa_phase_3d_ap = phase_interp_per_q[q](I_clipped_ap)
            base_complex = base * np.exp(1j * sfa_phase_3d_ap)
            base_complex[I_3d_ap < I_lut_min] = 0.0
            ff = ff_ap_masks[q]

            for ia in range(n_alpha):
                alpha = alpha_grid[ia]
                for js in range(n_Is):
                    I_s = Is_grid[js]

                    dq_mag = np.exp(alpha / 2.0 * log_I_3d_ap - I_3d_ap / (2.0 * I_s))
                    dq_mag[I_3d_ap < I_lut_min] = 0.0

                    E_q_2d = _trapz(dq_mag * base_complex, z_m_ap, axis=0)
                    if USE_SCIPY_FFT:
                        E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d, workers=-1)) * dx_hhg_m**2
                    else:
                        E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d)) * dx_hhg_m**2

                    yield_ap[iq, i_sub, ia, js] = np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2

            del dk_neut, dk_plas, dk_geom, dk_total, Phi_3d, mu_3d, mu_cumfwd, tau, abs_factor, base, base_complex

        del I_3d_ap, log_I_3d_ap, nf_3d_ap

    gc.collect()
    print(f"\n  Aperture grid scan complete. yield_ap shape: {yield_ap.shape}")

    print(f"  Total grid scan complete (power + aperture)")



    if True:  # single fitting pass (no pressure fitting)
        pressure_tag = "noP"
        print("\n" + "="*60)
        print("  FITTING PASS: power + aperture only")
        print("="*60)

        # =============================================================================
        # STEP 3: COMBINED COST FUNCTION — LOG-SPACE NORMALIZED RESIDUALS
        # =============================================================================
        TIMER.start_section("Step 3: Combined cost function evaluation")

        P_ref_idx = 3   # middle power level (I = 2.37e14)

        error_grid = np.zeros((n_alpha, n_Is))
        error_grid_power = np.zeros((n_alpha, n_Is))
        error_grid_aperture = np.zeros((n_alpha, n_Is))
        error_grid_power_q = np.full((n_q, n_alpha, n_Is), np.inf, dtype=np.float64)
        error_grid_aperture_q = np.full((n_q, n_alpha, n_Is), np.inf, dtype=np.float64)
        error_grid_power_chi2_q = np.full_like(error_grid_power_q, np.inf)
        error_grid_aperture_chi2_q = np.full_like(error_grid_aperture_q, np.inf)

        for ia in range(n_alpha):
            for js in range(n_Is):
                valid = True

                # Power-variation cost
                chi2_power_q = np.zeros(n_q, dtype=np.float64)
                for iq, q in enumerate(multi_q_list):
                    y_sim = yield_ub[iq, :, ia, js]
                    y_exp = exp_yield_multi[q]
                    if y_sim[P_ref_idx] <= 0 or np.any(y_sim <= 0):
                        valid = False
                        break
                    if y_exp[P_ref_idx] <= 0 or np.any(y_exp <= 0):
                        valid = False
                        break
                    y_sim_norm = y_sim / y_sim[P_ref_idx]
                    y_exp_norm = y_exp / y_exp[P_ref_idx]
                    log_y_sim_norm = np.log(y_sim_norm)
                    chi2_power_q[iq] = np.sum((log_y_sim_norm - np.log(y_exp_norm))**2)

                # Aperture-variation cost
                chi2_ap_q = np.zeros(n_q, dtype=np.float64)
                if valid:
                    for iq, q in enumerate(multi_q_list):
                        y_sim_ap = yield_ap[iq, :, ia, js]
                        y_exp_ap = exp_aperture_yield[q][ap_indices]
                        if y_sim_ap[ap_ref_idx] <= 0 or np.any(y_sim_ap <= 0):
                            valid = False
                            break
                        if y_exp_ap[ap_ref_idx] <= 0 or np.any(y_exp_ap <= 0):
                            valid = False
                            break
                        y_sim_ap_norm = y_sim_ap / y_sim_ap[ap_ref_idx]
                        y_exp_ap_norm = y_exp_ap / y_exp_ap[ap_ref_idx]
                        log_y_sim_ap_norm = np.log(y_sim_ap_norm)
                        log_y_exp_ap_norm = np.log(y_exp_ap_norm)
                        chi2_ap_q[iq] = np.sum((log_y_sim_ap_norm - log_y_exp_ap_norm)**2)

                if valid:
                    power_chi2_q = chi2_power_q / n_P
                    aperture_chi2_q = chi2_ap_q / n_ap_sub
                    cost_p_q = power_chi2_q
                    cost_a_q = aperture_chi2_q
                    cost_p = float(np.mean(cost_p_q))
                    cost_a = float(np.mean(cost_a_q))
                    error_grid_power[ia, js] = cost_p
                    error_grid_aperture[ia, js] = cost_a
                    error_grid_power_q[:, ia, js] = cost_p_q
                    error_grid_aperture_q[:, ia, js] = cost_a_q
                    error_grid_power_chi2_q[:, ia, js] = power_chi2_q
                    error_grid_aperture_chi2_q[:, ia, js] = aperture_chi2_q
                    error_grid[ia, js] = np.nan  # filled after cost-scale normalization
                else:
                    error_grid[ia, js] = np.inf
                    error_grid_power[ia, js] = np.inf
                    error_grid_aperture[ia, js] = np.inf

        # Normalize each harmonic cost separately before averaging.
        C_P_chi2_q_scale_arr = np.ones(n_q, dtype=np.float64)
        C_Aq_scale_arr = np.ones(n_q, dtype=np.float64)
        for iq, q in enumerate(multi_q_list):
            _finite_power_chi2 = error_grid_power_chi2_q[iq][
                np.isfinite(error_grid_power_chi2_q[iq]) & (error_grid_power_chi2_q[iq] > 0)
            ]
            _finite_aperture = error_grid_aperture_q[iq][
                np.isfinite(error_grid_aperture_q[iq]) & (error_grid_aperture_q[iq] > 0)
            ]
            C_P_chi2_q_scale_arr[iq] = max(float(np.nanmedian(_finite_power_chi2)) if _finite_power_chi2.size else 1.0, 1e-12)
            C_Aq_scale_arr[iq] = max(float(np.nanmedian(_finite_aperture)) if _finite_aperture.size else 1.0, 1e-12)

        C_Pq_scale_arr = C_P_chi2_q_scale_arr.copy()  # legacy output name
        C_P_scale = float(np.nanmedian(C_P_chi2_q_scale_arr))
        C_A_scale = float(np.nanmedian(C_Aq_scale_arr))
        valid_grid = np.all(
            np.isfinite(error_grid_power_q) & np.isfinite(error_grid_aperture_q),
            axis=0
        )
        error_grid_power_q_scaled = np.full_like(error_grid_power_q, np.inf)
        error_grid_power_chi2_q_scaled = np.full_like(error_grid_power_q, np.inf)
        error_grid_aperture_q_scaled = np.full_like(error_grid_aperture_q, np.inf)
        error_grid_power_scaled = np.full_like(error_grid_power, np.inf)
        error_grid_aperture_scaled = np.full_like(error_grid_aperture, np.inf)
        for iq in range(n_q):
            error_grid_power_chi2_q_scaled[iq, valid_grid] = (
                error_grid_power_chi2_q[iq, valid_grid] / C_P_chi2_q_scale_arr[iq]
            )
            error_grid_power_q_scaled[iq, valid_grid] = error_grid_power_chi2_q_scaled[iq, valid_grid]
            error_grid_aperture_q_scaled[iq, valid_grid] = (
                error_grid_aperture_q[iq, valid_grid] / C_Aq_scale_arr[iq]
            )
        error_grid_power_scaled[valid_grid] = np.mean(error_grid_power_q_scaled[:, valid_grid], axis=0)
        error_grid_aperture_scaled[valid_grid] = np.mean(error_grid_aperture_q_scaled[:, valid_grid], axis=0)
        alpha_box_grid = np.asarray(alpha_soft_box_penalty(alpha_grid), dtype=np.float64)
        error_grid_alpha_box = np.repeat(alpha_box_grid[:, None], n_Is, axis=1)
        error_grid[:] = np.inf
        error_grid[valid_grid] = (
            error_grid_power_scaled[valid_grid]
            + lambda_ap * error_grid_aperture_scaled[valid_grid]
            + error_grid_alpha_box[valid_grid]
        )
        print("  Per-harmonic cost normalization scales:")
        for iq, q in enumerate(multi_q_list):
            print(f"    H{q}: C_P_chi2_scale={C_P_chi2_q_scale_arr[iq]:.6g}, C_Aq_scale={C_Aq_scale_arr[iq]:.6g}")
        print(f"  Median scales for reference: C_P_scale={C_P_scale:.6g}, C_A_scale={C_A_scale:.6g}")

        # Find best grid point
        best_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
        best_alpha_grid = alpha_grid[best_idx[0]]
        best_Is_grid = Is_grid[best_idx[1]]
        best_error_grid = error_grid[best_idx]

        print(f"  Best grid point: alpha = {best_alpha_grid:.1f}, I_s = {best_Is_grid:.2e} W/cm^2")
        print(f"  Best combined error: {best_error_grid:.6f}")
        print(f"    Power cost:    {error_grid_power[best_idx]:.6f} (norm {error_grid_power_scaled[best_idx]:.6f})")
        print(f"    Aperture cost: {error_grid_aperture[best_idx]:.6f} (norm {error_grid_aperture_scaled[best_idx]:.6f})")

        if SENSITIVITY_CHECK_MODE:
            print("\n=== SENSITIVITY CHECK MODE: stopping after grid scan ===")
            print(f"  Beam model: phase screen (seed={PHASE_SCREEN_SEED})")
            print(f"  M2x = {M2x}, M2y = {M2y}")
            print(f"  Best alpha = {best_alpha_grid}, Best Is = {best_Is_grid:.2e}")
            print(f"  Error: power={error_grid_power[best_idx]:.6f}, "
                  f"aperture={error_grid_aperture[best_idx]:.6f}")
            TIMER.summary()
            import sys; sys.exit(0)

        # =============================================================================
        # STEP 4: SCIPY.OPTIMIZE FINE-TUNING (Nelder-Mead) — INTERPOLATION-BASED
        # =============================================================================
        TIMER.start_section("Step 4: scipy.optimize fine-tuning (interpolation)")

        # _compute_yield_at_intensity: core yield computation at a single intensity level
        def _compute_yield_at_intensity(I_3d, dphase_dz, z_m, q,
                                        alpha_val, I_s_val, P_mbar=None,
                                        nf_func=None, return_nearfield=False):
            """Compute macroscopic HHG yield (or near-field) for a single intensity slice.
            Uses Lewenstein SFA phase + empirical magnitude (deconvolved method)."""
            P_use = P_mbar if P_mbar is not None else hhg_gas_pressure
            if nf_func is not None:
                nf_3d = nf_func(I_3d)
            else:
                nf_3d = ionization_fraction(I_3d, gas['Ip_eV'])

            dk_neut = calc_dk_neutral(q, P_use, lambda_0_m, nf_3d, gas['delta_n'])
            dk_plas = calc_dk_plasma(q, P_use, lambda_0_m, nf_3d, gas['N_atm'])
            dk_geom = -(q - 1.0 / q) * dphase_dz
            dk_total = dk_neut + dk_plas + dk_geom

            Phi_3d = np.zeros_like(dk_total)
            Phi_3d[1:] = cumulative_trapezoid(dk_total, z_m, axis=0)

            sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22
            P_bar_use = P_use / 1000.0
            n_density_use = gas['N_atm'] * P_bar_use
            mu_3d = sigma_q_m2 * n_density_use * (1.0 - nf_3d)
            mu_cumfwd = np.zeros_like(mu_3d)
            mu_cumfwd[1:] = cumulative_trapezoid(mu_3d, z_m, axis=0)
            tau = mu_cumfwd[-1:] - mu_cumfwd
            abs_factor = np.exp(-tau / 2.0)

            # Base integrand with SFA phase
            base = (1.0 - nf_3d) * np.exp(1j * Phi_3d) * abs_factor
            I_clipped_local = np.clip(I_3d, I_lut_min, I_lut_max)
            sfa_phase_local = phase_interp_per_q[q](I_clipped_local)
            base_complex = base * np.exp(1j * sfa_phase_local)
            base_complex[I_3d < I_lut_min] = 0.0

            log_I_3d_local = np.log(np.maximum(I_3d, 1e-30))
            dq_mag = np.exp(alpha_val / 2.0 * log_I_3d_local - I_3d / (2.0 * I_s_val))
            dq_mag[I_3d < I_lut_min] = 0.0

            E_q_2d = _trapz(dq_mag * base_complex, z_m, axis=0)

            if return_nearfield:
                return E_q_2d  # complex near-field for coherent temporal sum

            if USE_SCIPY_FFT:
                E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d, workers=-1)) * dx_hhg_m**2
            else:
                E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d)) * dx_hhg_m**2

            ff = ff_ap_masks[q]
            return np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2

        # compute_yield_single: wrapper with optional temporal averaging
        def compute_yield_single(I_3d, nf_3d, dphase_dz, z_m, q, alpha_val, I_s_val,
                                 P_mbar=None):
            """Compute macroscopic HHG yield, optionally averaged over pulse envelope.
            Uses Lewenstein deconvolved method (SFA phase + empirical magnitude).
            """
            if USE_TEMPORAL_AVG and _tbi_temporal_luts is not None:
                n_t = len(_temporal_I_fracs)
                I_3d_peak = I_3d  # unscaled = peak intensity in each voxel
                E_q_2d_sum = None
                for j in range(n_t):
                    f = _temporal_I_fracs[j]
                    nf_lut_j = _tbi_temporal_luts[j]
                    E_q_2d_j = _compute_yield_at_intensity(
                        I_3d_peak * f, dphase_dz, z_m, q,
                        alpha_val, I_s_val, P_mbar,
                        nf_func=lambda I, _lut=nf_lut_j, _Ipk=I_3d_peak: _lut(_Ipk),
                        return_nearfield=True)
                    if E_q_2d_sum is None:
                        E_q_2d_sum = E_q_2d_j.copy()
                    else:
                        E_q_2d_sum += E_q_2d_j
                # Single FFT on coherent temporal sum
                E_q_2d_avg = E_q_2d_sum / n_t
                if USE_SCIPY_FFT:
                    E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d_avg, workers=-1)) * dx_hhg_m**2
                else:
                    E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d_avg)) * dx_hhg_m**2
                ff = ff_ap_masks[q]
                return np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2
            else:
                return _compute_yield_at_intensity(
                    I_3d, dphase_dz, z_m, q,
                    alpha_val, I_s_val, P_mbar)

        # Build interpolators from precomputed grid scan results (log-space for smooth interpolation)
        print("  Building interpolators from grid scan results...")
        log_Is_grid = np.log(Is_grid)

        # Interpolators for power-variation yields: yield_ub[iq, iP, ia, js]
        interp_ub = {}
        for iq in range(n_q):
            for iP in range(n_P):
                data = yield_ub[iq, iP, :, :]  # shape (n_alpha, n_Is)
                # Use log(yield) for smoother interpolation; handle zeros
                log_data = np.log(np.maximum(data, 1e-300))
                interp_ub[(iq, iP)] = RegularGridInterpolator(
                    (alpha_grid, log_Is_grid), log_data, method='linear',
                    bounds_error=False, fill_value=-700.0)

        # Interpolators for aperture-variation yields: yield_ap[iq, i_sub, ia, js]
        interp_ap = {}
        for iq in range(n_q):
            for i_sub in range(n_ap_sub):
                data = yield_ap[iq, i_sub, :, :]
                log_data = np.log(np.maximum(data, 1e-300))
                interp_ap[(iq, i_sub)] = RegularGridInterpolator(
                    (alpha_grid, log_Is_grid), log_data, method='linear',
                    bounds_error=False, fill_value=-700.0)

        print(f"  Built {len(interp_ub) + len(interp_ap)} interpolators")

        def cost_components_per_harmonic(params, iq, q):
            """Return (C_P_chi2, C_Aq) for one harmonic."""
            alpha_q, log_Is = params
            if alpha_q < alpha_grid[0] or alpha_q > alpha_grid[-1]:
                return 1e6, 1e6
            if log_Is < log_Is_grid[0] or log_Is > log_Is_grid[-1]:
                return 1e6, 1e6
            pt = np.array([alpha_q, log_Is])

            # Power
            y_sim = np.array([np.exp(interp_ub[(iq, iP)](pt)) for iP in range(n_P)])
            y_exp = exp_yield_multi[q]
            if y_sim[P_ref_idx] <= 0 or np.any(y_sim <= 0):
                return 1e6, 1e6
            log_y_sim_norm = np.log(y_sim / y_sim[P_ref_idx])
            chi2_p = np.sum((log_y_sim_norm - np.log(y_exp / y_exp[P_ref_idx]))**2)

            # Aperture
            y_sim_ap = np.array([np.exp(interp_ap[(iq, i_sub)](pt)) for i_sub in range(n_ap_sub)])
            y_exp_ap = exp_aperture_yield[q][ap_indices]
            if y_sim_ap[ap_ref_idx] <= 0 or np.any(y_sim_ap <= 0):
                return 1e6, 1e6
            log_y_sim_ap_norm = np.log(y_sim_ap / y_sim_ap[ap_ref_idx])
            log_y_exp_ap_norm = np.log(y_exp_ap / y_exp_ap[ap_ref_idx])
            chi2_a = np.sum((log_y_sim_ap_norm - log_y_exp_ap_norm)**2)

            C_P_chi2 = float(np.ravel(chi2_p / n_P)[0])
            C_Aq = float(np.ravel(chi2_a / n_ap_sub)[0])
            return C_P_chi2, C_Aq

        def cost_per_harmonic(params, iq, q):
            """Combined cost after per-harmonic scale normalization."""
            C_P_chi2, C_Aq = cost_components_per_harmonic(params, iq, q)
            if C_P_chi2 >= 1e6 or C_Aq >= 1e6:
                return 1e6
            C_P_norm = C_P_chi2 / C_P_chi2_q_scale_arr[iq]
            return C_P_norm + lambda_ap * C_Aq / C_Aq_scale_arr[iq] + alpha_soft_box_penalty(params[0])

        # Joint cost function: 12-parameter vector [alpha_0, logIs_0, ..., alpha_5, logIs_5]
        def joint_cost(params):
            """Joint cost over all harmonics with 2nd-order smoothness regularization."""
            n_h = len(multi_q_list)
            # Hard bounds check
            for ih in range(n_h):
                a_h = params[2 * ih]
                li_h = params[2 * ih + 1]
                if a_h < ALPHA_MIN or a_h > ALPHA_MAX:
                    return 1e6
                if li_h < LOG_IS_MIN or li_h > LOG_IS_MAX:
                    return 1e6

            # C_shape: sum of per-harmonic shape costs
            C_shape = 0.0
            for ih in range(n_h):
                c_h = cost_per_harmonic([params[2*ih], params[2*ih+1]], ih, multi_q_list[ih])
                if c_h >= 1e6:
                    return 1e6
                C_shape += c_h
            C_shape /= n_h

            # C_smooth: 2nd-order difference penalty
            C_smooth_alpha = 0.0
            C_smooth_Is = 0.0
            for ih in range(n_h - 2):
                d2_a = params[2*(ih+2)] - 2*params[2*(ih+1)] + params[2*ih]
                d2_li = params[2*(ih+2)+1] - 2*params[2*(ih+1)+1] + params[2*ih+1]
                C_smooth_alpha += d2_a**2
                C_smooth_Is += d2_li**2
            C_smooth_alpha /= max(n_h - 2, 1)
            C_smooth_Is /= max(n_h - 2, 1)

            # C_prior: soft Is prior
            C_prior = 0.0
            for ih in range(n_h):
                C_prior += ((params[2*ih+1] - log_Is_prior_center) / log_Is_prior_width)**2
            C_prior /= n_h

            return (C_shape
                    + lambda_smooth_alpha * C_smooth_alpha
                    + lambda_smooth_Is * C_smooth_Is
                    + lambda_prior * C_prior)

        # Joint optimization across all harmonics
        print(f"  Starting joint Nelder-Mead from grid best: alpha={best_alpha_grid:.2f}, I_s={best_Is_grid:.2e}")
        x0_joint = []
        for iq, q in enumerate(multi_q_list):
            x0_joint.extend([best_alpha_grid, np.log(best_Is_grid)])
        x0_joint = np.array(x0_joint)

        _jcnt = [0]
        def _joint_cost_log(params, _cnt=_jcnt):
            _cnt[0] += 1
            val = joint_cost(params)
            if _cnt[0] % 50 == 0:
                alphas_str = ', '.join(f'{params[2*i]:.2f}' for i in range(n_q))
                print(f"    Joint eval {_cnt[0]}: cost={val:.6f}, alphas=[{alphas_str}]")
            return val

        result_joint = minimize(_joint_cost_log, x0_joint, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 0.005, 'fatol': 1e-6, 'adaptive': True})

        best_alphas = {}
        best_Is_vals = {}
        best_costs = {}
        best_costs_power = {}
        best_costs_aperture = {}
        for iq, q in enumerate(multi_q_list):
            best_alphas[q] = result_joint.x[2 * iq]
            best_Is_vals[q] = np.exp(result_joint.x[2 * iq + 1])
            C_P_chi2, C_Aq = cost_components_per_harmonic(
                [result_joint.x[2*iq], result_joint.x[2*iq+1]], iq, q)
            C_P_norm = C_P_chi2 / C_P_chi2_q_scale_arr[iq]
            best_costs_power[q] = C_P_norm
            best_costs_aperture[q] = C_Aq
            best_costs[q] = C_P_norm + lambda_ap * C_Aq / C_Aq_scale_arr[iq] + alpha_soft_box_penalty(best_alphas[q])

        best_error = sum(best_costs.values()) / n_q

        _eval_total = _jcnt[0]
        print(f"\n  Joint optimization complete ({_eval_total} evaluations, final cost={result_joint.fun:.6f})")
        for q in multi_q_list:
            print(f"    H{q}: alpha={best_alphas[q]:.3f}, I_s={best_Is_vals[q]:.2e}, cost={best_costs[q]:.4f}")
        print(f"  Mean per-harmonic cost: {best_error:.6f}")


        # --- Per-harmonic cost landscape diagnostic ---
        print("\n  Generating per-harmonic cost landscape figures...")
        cost_map_n_alpha = 50
        cost_map_n_Is = 50
        cost_map_alpha_range = np.linspace(ALPHA_MIN, ALPHA_MAX, cost_map_n_alpha)
        cost_map_logIs_range = np.linspace(LOG_IS_MIN, LOG_IS_MAX, cost_map_n_Is)
        cost_map_Is_range = np.exp(cost_map_logIs_range)
        _per_harmonic_cost_maps = np.full(
            (n_q, cost_map_n_alpha, cost_map_n_Is), np.nan, dtype=np.float64
        )
        _per_harmonic_cost_maps_power = np.full_like(_per_harmonic_cost_maps, np.nan)
        _per_harmonic_cost_maps_aperture = np.full_like(_per_harmonic_cost_maps, np.nan)
        _per_harmonic_cost_maps_power_raw = np.full_like(_per_harmonic_cost_maps, np.nan)
        _per_harmonic_cost_maps_aperture_raw = np.full_like(_per_harmonic_cost_maps, np.nan)

        for iq, q in enumerate(multi_q_list):
            cost_map = np.zeros((cost_map_n_alpha, cost_map_n_Is))
            cost_map_power = np.zeros((cost_map_n_alpha, cost_map_n_Is))
            cost_map_aperture = np.zeros((cost_map_n_alpha, cost_map_n_Is))
            cost_map_power_raw = np.zeros((cost_map_n_alpha, cost_map_n_Is))
            cost_map_aperture_raw = np.zeros((cost_map_n_alpha, cost_map_n_Is))
            for ia_cm in range(cost_map_n_alpha):
                for js_cm in range(cost_map_n_Is):
                    C_P_chi2, C_Aq = cost_components_per_harmonic(
                        [cost_map_alpha_range[ia_cm], cost_map_logIs_range[js_cm]], iq, q)
                    if C_P_chi2 >= 1e6 or C_Aq >= 1e6:
                        cost_map_power_raw[ia_cm, js_cm] = 1e6
                        cost_map_aperture_raw[ia_cm, js_cm] = 1e6
                        cost_map_power[ia_cm, js_cm] = 1e6
                        cost_map_aperture[ia_cm, js_cm] = 1e6
                        cost_map[ia_cm, js_cm] = 1e6
                    else:
                        C_P_norm = C_P_chi2 / C_P_chi2_q_scale_arr[iq]
                        cost_map_power_raw[ia_cm, js_cm] = C_P_chi2
                        cost_map_aperture_raw[ia_cm, js_cm] = C_Aq
                        cost_map_power[ia_cm, js_cm] = C_P_norm
                        cost_map_aperture[ia_cm, js_cm] = C_Aq / C_Aq_scale_arr[iq]
                        cost_map[ia_cm, js_cm] = (
                            cost_map_power[ia_cm, js_cm] + lambda_ap * cost_map_aperture[ia_cm, js_cm]
                            + alpha_soft_box_penalty(cost_map_alpha_range[ia_cm])
                        )
            cost_map[cost_map >= 1e6] = np.nan
            cost_map_power[cost_map_power >= 1e6] = np.nan
            cost_map_aperture[cost_map_aperture >= 1e6] = np.nan
            cost_map_power_raw[cost_map_power_raw >= 1e6] = np.nan
            cost_map_aperture_raw[cost_map_aperture_raw >= 1e6] = np.nan
            _per_harmonic_cost_maps[iq] = cost_map
            _per_harmonic_cost_maps_power[iq] = cost_map_power
            _per_harmonic_cost_maps_aperture[iq] = cost_map_aperture
            _per_harmonic_cost_maps_power_raw[iq] = cost_map_power_raw
            _per_harmonic_cost_maps_aperture_raw[iq] = cost_map_aperture_raw

        cost_plot_items = [(iq, q) for iq, q in enumerate(multi_q_list) if 13 <= q <= 17]
        if not cost_plot_items:
            cost_plot_items = list(enumerate(multi_q_list))
        fig_cm, axes_cm = plt.subplots(
            1, len(cost_plot_items), figsize=(4.9 * len(cost_plot_items), 4.35),
            sharex=True, sharey=True
        )
        axes_cm = np.atleast_1d(axes_cm)
        for ip_cm, (iq, q) in enumerate(cost_plot_items):
            ax = axes_cm[ip_cm]
            cost_map = _per_harmonic_cost_maps[iq]
            finite_cm = cost_map[np.isfinite(cost_map)]
            if finite_cm.size:
                vmin_cm = np.nanpercentile(finite_cm, 2)
                vmax_cm = np.nanpercentile(finite_cm, 98)
                if not np.isfinite(vmin_cm) or not np.isfinite(vmax_cm) or vmin_cm >= vmax_cm:
                    vmin_cm, vmax_cm = np.nanmin(finite_cm), np.nanmax(finite_cm)
            else:
                vmin_cm, vmax_cm = None, None
            im = ax.pcolormesh(
                cost_map_Is_range, cost_map_alpha_range, cost_map,
                shading='auto', cmap='cividis', vmin=vmin_cm, vmax=vmax_cm
            )
            ax.set_xscale('log')
            ax.set_xlim(cost_map_Is_range.min(), cost_map_Is_range.max())
            ax.set_ylim(cost_map_alpha_range.min(), cost_map_alpha_range.max())
            ax.plot(
                best_Is_vals[q], best_alphas[q], marker='*', linestyle='None',
                markersize=16, markeredgecolor='white', markeredgewidth=0.9,
                color='#D7191C', zorder=5
            )
            ax.plot(
                best_Is_grid, best_alpha_grid, marker='x', linestyle='None',
                markersize=10, markeredgewidth=3.2, color='black', alpha=0.55, zorder=5
            )
            ax.plot(
                best_Is_grid, best_alpha_grid, marker='x', linestyle='None',
                markersize=9.2, markeredgewidth=2.0, color='white', zorder=6
            )
            ax.set_xlabel(r'$I_s$ (W/cm$^2$)')
            if ip_cm == 0:
                ax.set_ylabel(r'$\alpha$')
            ax.set_title(f'H{q}', fontsize=16, fontweight='bold', pad=8)
            style_paper_axis(ax)

            cbar = fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
            cbar.set_label('Cost', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=9.5, width=0.9, length=3.5)
            cbar.outline.set_linewidth(0.9)

        legend_handles_cm = [
            Line2D([0], [0], marker='*', linestyle='None', markersize=13,
                   markerfacecolor='#D7191C', markeredgecolor='white', label='Joint opt'),
            Line2D([0], [0], marker='x', linestyle='None', markersize=9,
                   markeredgewidth=2.0, color='black', label='Grid best'),
        ]
        fig_cm.legend(
            handles=legend_handles_cm, loc='upper center', ncol=2, frameon=False,
            fontsize=12, bbox_to_anchor=(0.5, 1.01)
        )

        fig_cm.suptitle(f'Per-Harmonic Cost Landscape ({lavg_tag})',
                        fontsize=17, fontweight='bold', y=0.995)
        fig_cm.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.4)
        cost_map_path = f'hhg_cost_landscape_per_harmonic_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
        fig_cm.savefig(cost_map_path, dpi=400, bbox_inches='tight')
        plt.close(fig_cm)
        print(f"  Saved cost landscape: {cost_map_path}")

        def save_component_cost_landscape(component_maps, component_label, component_title, filename_token):
            fig_comp, axes_comp = plt.subplots(
                1, len(cost_plot_items), figsize=(4.9 * len(cost_plot_items), 4.35),
                sharex=True, sharey=True
            )
            axes_comp = np.atleast_1d(axes_comp)
            for ip_comp, (iq, q) in enumerate(cost_plot_items):
                ax = axes_comp[ip_comp]
                component_map = component_maps[iq]
                finite_comp = component_map[np.isfinite(component_map)]
                if finite_comp.size:
                    vmin_comp = np.nanpercentile(finite_comp, 2)
                    vmax_comp = np.nanpercentile(finite_comp, 98)
                    if (not np.isfinite(vmin_comp) or not np.isfinite(vmax_comp)
                            or vmin_comp >= vmax_comp):
                        vmin_comp, vmax_comp = np.nanmin(finite_comp), np.nanmax(finite_comp)
                else:
                    vmin_comp, vmax_comp = None, None

                im = ax.pcolormesh(
                    cost_map_Is_range, cost_map_alpha_range, component_map,
                    shading='auto', cmap='cividis', vmin=vmin_comp, vmax=vmax_comp
                )
                ax.set_xscale('log')
                ax.set_xlim(cost_map_Is_range.min(), cost_map_Is_range.max())
                ax.set_ylim(cost_map_alpha_range.min(), cost_map_alpha_range.max())
                ax.plot(
                    best_Is_vals[q], best_alphas[q], marker='*', linestyle='None',
                    markersize=16, markeredgecolor='white', markeredgewidth=0.9,
                    color='#D7191C', zorder=5
                )
                ax.plot(
                    best_Is_grid, best_alpha_grid, marker='x', linestyle='None',
                    markersize=10, markeredgewidth=3.2, color='black', alpha=0.55, zorder=5
                )
                ax.plot(
                    best_Is_grid, best_alpha_grid, marker='x', linestyle='None',
                    markersize=9.2, markeredgewidth=2.0, color='white', zorder=6
                )
                ax.set_xlabel(r'$I_s$ (W/cm$^2$)')
                if ip_comp == 0:
                    ax.set_ylabel(r'$\alpha$')
                ax.set_title(f'H{q}', fontsize=16, fontweight='bold', pad=8)
                style_paper_axis(ax)

                cbar = fig_comp.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
                cbar.set_label(component_label, fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=9.5, width=0.9, length=3.5)
                cbar.outline.set_linewidth(0.9)

            fig_comp.legend(
                handles=legend_handles_cm, loc='upper center', ncol=2, frameon=False,
                fontsize=12, bbox_to_anchor=(0.5, 1.01)
            )
            fig_comp.suptitle(f'Per-Harmonic {component_title} ({lavg_tag})',
                              fontsize=17, fontweight='bold', y=0.995)
            fig_comp.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.4)
            component_path = (
                f'hhg_cost_landscape_per_harmonic_{filename_token}_'
                f'{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
            )
            fig_comp.savefig(component_path, dpi=400, bbox_inches='tight')
            plt.close(fig_comp)
            print(f"  Saved {component_title.lower()}: {component_path}")
            return component_path

        cost_map_power_path = save_component_cost_landscape(
            _per_harmonic_cost_maps_power, r'$C_{P,q}^{norm}$', r'Per-Harmonic Normalized Power Cost $C_{P,q}^{norm}$', 'power'
        )
        cost_map_aperture_path = save_component_cost_landscape(
            _per_harmonic_cost_maps_aperture, r'$C_{A,q}/s_{A,q}$', r'Per-Harmonic Normalized Aperture Cost $C_{A,q}/s_{A,q}$', 'aperture'
        )

        cost_map_npz_path = f'hhg_cost_landscape_per_harmonic_data_{lavg_tag}_{pressure_tag}_{_m2_tag}.npz'
        np.savez_compressed(
            cost_map_npz_path,
            harmonic_orders=np.array(multi_q_list, dtype=np.int32),
            per_harmonic_cost_maps=_per_harmonic_cost_maps,
            per_harmonic_cost_maps_power=_per_harmonic_cost_maps_power,
            per_harmonic_cost_maps_aperture=_per_harmonic_cost_maps_aperture,
            per_harmonic_cost_maps_power_raw=_per_harmonic_cost_maps_power_raw,
            per_harmonic_cost_maps_aperture_raw=_per_harmonic_cost_maps_aperture_raw,
            per_harmonic_cost_alpha_grid=cost_map_alpha_range,
            per_harmonic_cost_logIs_grid=cost_map_logIs_range,
            per_harmonic_cost_Is_grid=cost_map_Is_range,
            grid_alpha_grid=alpha_grid,
            grid_logIs_grid=log_Is_grid,
            grid_Is_grid=Is_grid,
            error_grid_power_chi2_q=error_grid_power_chi2_q,
            error_grid_power_chi2_q_scaled=error_grid_power_chi2_q_scaled,
            error_grid_power_q_scaled=error_grid_power_q_scaled,
            error_grid_aperture_chi2_q=error_grid_aperture_chi2_q,
            best_alphas_arr=np.array([best_alphas[q] for q in multi_q_list], dtype=np.float64),
            best_Is_arr=np.array([best_Is_vals[q] for q in multi_q_list], dtype=np.float64),
            best_costs_arr=np.array([best_costs[q] for q in multi_q_list], dtype=np.float64),
            best_costs_power_arr=np.array([best_costs_power[q] for q in multi_q_list], dtype=np.float64),
            best_costs_aperture_arr=np.array([best_costs_aperture[q] for q in multi_q_list], dtype=np.float64),
            best_costs_power_scaled_arr=np.array([best_costs_power[q] for q in multi_q_list], dtype=np.float64),
            best_costs_aperture_scaled_arr=np.array([best_costs_aperture[q] / C_Aq_scale_arr[iq] for iq, q in enumerate(multi_q_list)], dtype=np.float64),
            best_alpha_grid=np.float64(best_alpha_grid),
            best_Is_grid=np.float64(best_Is_grid),
            best_error_grid=np.float64(best_error_grid),
            best_error=np.float64(best_error),
            lambda_ap=np.float64(lambda_ap),
            lambda_alpha_box=np.float64(lambda_alpha_box),
            alpha_box_min=np.float64(alpha_box_min),
            alpha_box_max=np.float64(alpha_box_max),
            sigma_alpha_box=np.float64(sigma_alpha_box),
            alpha_box_grid=alpha_box_grid,
            error_grid_alpha_box=error_grid_alpha_box,
            C_P_chi2_q_scale_arr=C_P_chi2_q_scale_arr,
            C_Pq_scale_arr=C_Pq_scale_arr,
            C_Aq_scale_arr=C_Aq_scale_arr,
            C_P_scale=np.float64(C_P_scale),
            C_A_scale=np.float64(C_A_scale),
            cost_map_power_path=np.array(cost_map_power_path),
            cost_map_aperture_path=np.array(cost_map_aperture_path),
            lavg_tag=np.array(lavg_tag),
            pressure_tag=np.array(pressure_tag),
            hhg_gas_type=np.array(hhg_gas_type),
            hhg_gas_pressure=np.float64(hhg_gas_pressure),
            M2x=np.float64(M2x),
            M2y=np.float64(M2y),
        )
        print(f"  Saved cost landscape data: {cost_map_npz_path} ({os.path.getsize(cost_map_npz_path) / 1024:.1f} KB)")


        # =============================================================================
        # STEP 5: ENHANCEMENT PREDICTION (BLOCKED BEAM VALIDATION)
        # =============================================================================
        TIMER.start_section("Step 5: Enhancement prediction")
        print("Computing yields for both beams at best (alpha, I_s)...")

        yield_ub_best = np.zeros((n_q, n_P))
        yield_b_best = np.zeros((n_q, n_P))

        for iP in range(n_P):
            I_exp = exp_yield_I_Wcm2[iP]
            power_scale = I_exp / hhg_peak_intensity_Wcm2
            print(f"  Power {iP+1}/{n_P}: I = {I_exp:.2e} W/cm^2")

            # ---- UNBLOCKED ----
            I_3d_ub = I_2d_gas_ub_Wcm2 * power_scale
            nf_3d_ub = ionization_fraction(I_3d_ub, gas['Ip_eV'])

            # ---- BLOCKED ----
            I_3d_b = I_2d_gas_b_Wcm2 * power_scale
            nf_3d_b = ionization_fraction(I_3d_b, gas['Ip_eV'])

            for iq, q in enumerate(multi_q_list):
                ff = ff_ap_masks[q]
                ap_mask_q = ff['mask']
                dtheta_q = ff['dtheta']

                sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22

                # -- Unblocked --
                dk_n_ub = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d_ub, gas['delta_n'])
                dk_p_ub = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d_ub, gas['N_atm'])
                dk_g_ub = -(q - 1.0 / q) * dphase_dz_3d_ub
                dk_t_ub = dk_n_ub + dk_p_ub + dk_g_ub

                Phi_ub = np.zeros_like(dk_t_ub)
                Phi_ub[1:] = cumulative_trapezoid(dk_t_ub, z_gas_2d_ub_m, axis=0)

                mu_ub = sigma_q_m2 * n_gas_density * (1.0 - nf_3d_ub)
                mu_cum_ub = np.zeros_like(mu_ub)
                mu_cum_ub[1:] = cumulative_trapezoid(mu_ub, z_gas_2d_ub_m, axis=0)
                tau_ub = mu_cum_ub[-1:] - mu_cum_ub
                abs_ub = np.exp(-tau_ub / 2.0)

                alpha_q = best_alphas[q]
                Is_q = best_Is_vals[q]

                # SFA phase + empirical magnitude (deconvolved)
                base_ub = (1.0 - nf_3d_ub) * np.exp(1j * Phi_ub) * abs_ub
                I_clipped_ub = np.clip(I_3d_ub, I_lut_min, I_lut_max)
                sfa_phase_ub = phase_interp_per_q[q](I_clipped_ub)
                base_complex_ub = base_ub * np.exp(1j * sfa_phase_ub)
                base_complex_ub[I_3d_ub < I_lut_min] = 0.0

                dq_mag_ub = I_3d_ub**(alpha_q / 2.0) * np.exp(-I_3d_ub / (2.0 * Is_q))
                dq_mag_ub[I_3d_ub < I_lut_min] = 0.0

                E_ub = _trapz(dq_mag_ub * base_complex_ub, z_gas_2d_ub_m, axis=0)
                if USE_SCIPY_FFT:
                    E_ff_ub = scipy_fft.fftshift(scipy_fft.fft2(E_ub, workers=-1)) * dx_hhg_m**2
                else:
                    E_ff_ub = np.fft.fftshift(np.fft.fft2(E_ub)) * dx_hhg_m**2
                yield_ub_best[iq, iP] = np.sum(np.abs(E_ff_ub)**2 * ap_mask_q) * dtheta_q**2

                # -- Blocked --
                dk_n_b = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d_b, gas['delta_n'])
                dk_p_b = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d_b, gas['N_atm'])
                dk_g_b = -(q - 1.0 / q) * dphase_dz_3d_b
                dk_t_b = dk_n_b + dk_p_b + dk_g_b

                Phi_b = np.zeros_like(dk_t_b)
                Phi_b[1:] = cumulative_trapezoid(dk_t_b, z_gas_2d_b_m, axis=0)

                mu_b = sigma_q_m2 * n_gas_density * (1.0 - nf_3d_b)
                mu_cum_b = np.zeros_like(mu_b)
                mu_cum_b[1:] = cumulative_trapezoid(mu_b, z_gas_2d_b_m, axis=0)
                tau_b = mu_cum_b[-1:] - mu_cum_b
                abs_b = np.exp(-tau_b / 2.0)

                # SFA phase + empirical magnitude (deconvolved)
                base_b = (1.0 - nf_3d_b) * np.exp(1j * Phi_b) * abs_b
                I_clipped_b = np.clip(I_3d_b, I_lut_min, I_lut_max)
                sfa_phase_b = phase_interp_per_q[q](I_clipped_b)
                base_complex_b = base_b * np.exp(1j * sfa_phase_b)
                base_complex_b[I_3d_b < I_lut_min] = 0.0

                dq_mag_b = I_3d_b**(alpha_q / 2.0) * np.exp(-I_3d_b / (2.0 * Is_q))
                dq_mag_b[I_3d_b < I_lut_min] = 0.0

                E_b = _trapz(dq_mag_b * base_complex_b, z_gas_2d_b_m, axis=0)
                if USE_SCIPY_FFT:
                    E_ff_b = scipy_fft.fftshift(scipy_fft.fft2(E_b, workers=-1)) * dx_hhg_m**2
                else:
                    E_ff_b = np.fft.fftshift(np.fft.fft2(E_b)) * dx_hhg_m**2
                yield_b_best[iq, iP] = np.sum(np.abs(E_ff_b)**2 * ap_mask_q) * dtheta_q**2

            del I_3d_ub, I_3d_b, nf_3d_ub, nf_3d_b
            gc.collect()

        # Enhancement = blocked / unblocked
        enhancement_sim = np.full((n_q, n_P), np.nan)
        for iq in range(n_q):
            for iP in range(n_P):
                if yield_ub_best[iq, iP] > 0:
                    enhancement_sim[iq, iP] = yield_b_best[iq, iP] / yield_ub_best[iq, iP]

        print("\n  Enhancement (sim) at highest power:")
        for iq, q in enumerate(multi_q_list):
            exp_key = f'H{q}'
            exp_enh = exp_enhancement[exp_key][0] if exp_key in exp_enhancement else float('nan')
            print(f"    H{q}: sim = {enhancement_sim[iq, 0]:.3f}, exp = {exp_enh:.3f}")

        # --- Aperture-variation yields at best parameters ---
        print("\nComputing aperture-variation yields at best (alpha, I_s)...")
        yield_ap_best = np.zeros((n_q, n_ap_sub))

        for i_sub, i_ap in enumerate(ap_indices):
            gd = gas_data_ap[i_ap]
            I_3d = gd['I_2d_Wcm2'] * power_scale_ap
            nf_3d = ionization_fraction(I_3d, gas['Ip_eV'])
            for iq, q in enumerate(multi_q_list):
                alpha_q = best_alphas[q]
                Is_q = best_Is_vals[q]
                yield_ap_best[iq, i_sub] = compute_yield_single(
                    I_3d, nf_3d, gd['dphase_dz'], gd['z_gas_m'], q, alpha_q, Is_q)

        print("  Aperture-variation yield (normalized) — sim vs exp:")
        for iq, q in enumerate(multi_q_list):
            y_sim_ap = yield_ap_best[iq, :]
            y_exp_ap = exp_aperture_yield[q][ap_indices]
            if y_sim_ap[ap_ref_idx] > 0:
                y_sim_norm = y_sim_ap / y_sim_ap[ap_ref_idx]
            else:
                y_sim_norm = y_sim_ap
            y_exp_norm = y_exp_ap / y_exp_ap[ap_ref_idx]
            print(f"    H{q}: sim=[{', '.join(f'{v:.2f}' for v in y_sim_norm)}]")
            print(f"         exp=[{', '.join(f'{v:.2f}' for v in y_exp_norm)}]")



        # =============================================================================
        # PRE-COMPUTE dk DIAGNOSTIC (on-axis slices for caching)
        # =============================================================================
        center_2d = N_hhg_2d // 2
        dk_geom_onaxis_ub = dphase_dz_3d_ub[:, center_2d, center_2d]
        nf_ref_onaxis = ionization_fraction(I_2d_gas_ub_Wcm2[:, center_2d, center_2d], gas['Ip_eV'])
        dk_n_ref_onaxis = calc_dk_neutral(21, hhg_gas_pressure, lambda_0_m, nf_ref_onaxis, gas['delta_n'])
        dk_p_ref_onaxis = calc_dk_plasma(21, hhg_gas_pressure, lambda_0_m, nf_ref_onaxis, gas['N_atm'])
        dk_g_ref_onaxis = -(21 - 1.0/21) * dk_geom_onaxis_ub

        # =============================================================================
        # STEP 6: FIGURES
        # =============================================================================
        TIMER.start_section("Step 6: Figures")

        # --- Figure HHG-EF-1: Fit Quality (3x3) ---
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))

        colors_q = COLORS_HQ
        I_plot = exp_yield_I_Wcm2 / 1e14   # in units of 10^14
        T_plot = np.array([exp_aperture_T[i] for i in ap_indices])  # transmission for aperture x-axis

        # (0,0) Yield vs power shape: sim vs exp (normalized)
        ax = axes1[0, 0]
        for iq, q in enumerate(multi_q_list):
            y_sim = yield_ub_best[iq, :]
            y_exp = exp_yield_multi[q]
            y_sim_norm = y_sim / y_sim[P_ref_idx]
            y_exp_norm = y_exp / y_exp[P_ref_idx]
            ax.plot(I_plot, y_sim_norm, '-o', color=colors_q[q], markersize=4, label=f'H{q} sim')
            ax.plot(I_plot, y_exp_norm, 's--', color=colors_q[q], markersize=5, alpha=0.6, label=f'H{q} exp')
        ax.set_xlabel(r'Peak Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel('Normalized Yield')
        ax.set_title('Power Variation: Sim vs Exp')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.15)

        # (0,1) Yield vs aperture shape: sim vs exp (normalized)
        ax = axes1[0, 1]
        for iq, q in enumerate(multi_q_list):
            y_sim_ap = yield_ap_best[iq, :]
            y_exp_ap = exp_aperture_yield[q][ap_indices]
            if y_sim_ap[ap_ref_idx] > 0:
                y_sim_ap_norm = y_sim_ap / y_sim_ap[ap_ref_idx]
            else:
                y_sim_ap_norm = y_sim_ap
            y_exp_ap_norm = y_exp_ap / y_exp_ap[ap_ref_idx]
            ax.plot(T_plot, y_sim_ap_norm, '-o', color=colors_q[q], markersize=4, label=f'H{q} sim')
            ax.plot(T_plot, y_exp_ap_norm, 's--', color=colors_q[q], markersize=5, alpha=0.6, label=f'H{q} exp')
        ax.set_xlabel('Aperture Transmission')
        ax.set_ylabel('Normalized Yield')
        ax.set_title('Aperture Variation: Sim vs Exp')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.15)
        ax.invert_xaxis()

        # (0,2) Best |d_q(I)|^2 curve — per-harmonic
        ax = axes1[0, 2]
        I_curve = np.linspace(0, 3.5e14, 500)
        for q in multi_q_list:
            aq = best_alphas[q]
            isq = best_Is_vals[q]
            dq2 = I_curve**aq * np.exp(-I_curve / isq)
            dq2_norm = dq2 / max(dq2.max(), 1e-300)
            ax.plot(I_curve / 1e14, dq2_norm, '-', color=colors_q[q],
                    label=f'H{q} (α={aq:.1f}, Is={isq:.1e})')
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel(r'$|d_q(I)|^2$ (normalized)')
        ax.set_title('Per-harmonic dipole fit')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

        # (1,0) Power residual heatmap (harmonic x power)
        ax = axes1[1, 0]
        residual_map = np.zeros((n_q, n_P))
        for iq, q in enumerate(multi_q_list):
            y_sim = yield_ub_best[iq, :]
            y_exp = exp_yield_multi[q]
            if y_sim[P_ref_idx] > 0 and y_exp[P_ref_idx] > 0:
                y_sim_norm = y_sim / y_sim[P_ref_idx]
                y_exp_norm = y_exp / y_exp[P_ref_idx]
                residual_map[iq, :] = np.log(y_sim_norm) - np.log(y_exp_norm)

        im = ax.imshow(residual_map, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-0.4, vmax=0.4, interpolation='nearest')
        ax.set_xticks(range(n_P))
        ax.set_xticklabels([f'{I:.2f}' for I in I_plot], fontsize=8, rotation=45)
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f'H{q}' for q in multi_q_list])
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel('Harmonic')
        ax.set_title('Power Residuals (log sim/exp)')
        plt.colorbar(im, ax=ax)

        # (1,1) Aperture residual heatmap (harmonic x aperture)
        ax = axes1[1, 1]
        residual_map_ap = np.zeros((n_q, n_ap_sub))
        for iq, q in enumerate(multi_q_list):
            y_sim_ap = yield_ap_best[iq, :]
            y_exp_ap = exp_aperture_yield[q][ap_indices]
            if y_sim_ap[ap_ref_idx] > 0 and y_exp_ap[ap_ref_idx] > 0:
                y_sim_ap_norm = y_sim_ap / y_sim_ap[ap_ref_idx]
                y_exp_ap_norm = y_exp_ap / y_exp_ap[ap_ref_idx]
                residual_map_ap[iq, :] = np.log(y_sim_ap_norm) - np.log(y_exp_ap_norm)

        im = ax.imshow(residual_map_ap, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-0.4, vmax=0.4, interpolation='nearest')
        ax.set_xticks(range(n_ap_sub))
        ax.set_xticklabels([f'{T:.2f}' for T in T_plot], fontsize=8, rotation=45)
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f'H{q}' for q in multi_q_list])
        ax.set_xlabel('Aperture Transmission')
        ax.set_ylabel('Harmonic')
        ax.set_title('Aperture Residuals (log sim/exp)')
        plt.colorbar(im, ax=ax)

        # (1,2) Summary text
        ax = axes1[1, 2]
        ax.axis('off')
        _ph_lines = "\n".join([f"  H{q}: α={best_alphas[q]:.2f}, Is={best_Is_vals[q]:.2e}" for q in multi_q_list])
        summary_text = f"""
        PER-HARMONIC FIT (Power + Aperture)
        {'='*45}

        {_ph_lines}

        Grid scan best:
          alpha = {best_alpha_grid:.1f}, I_s = {best_Is_grid:.2e}
          error = {best_error_grid:.6f}

        Optimized error = {best_error:.6f}
        Optimizer evals = {_eval_total}

        Data: {n_P} powers + {n_ap_sub} apertures
        Gas: {hhg_gas_type}, P = {hhg_gas_pressure} mbar
        Harmonics: H{multi_q_list[0]}-H{multi_q_list[-1]}
        lambda_ap = {lambda_ap:.1f}
        lambda_alpha_box = {lambda_alpha_box:.2f}
        C_P_scale = {C_P_scale:.3g}
        C_A_scale = {C_A_scale:.3g}
        {'='*45}
        """
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7', edgecolor='0.7', alpha=0.9))

        fig1.suptitle('HHG Pure-Experimental LUT: Fit Quality (Power + Aperture)', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'hhg_experimental_fit_quality_{lavg_tag}_{pressure_tag}_{_m2_tag}.png')
        print(f"  Saved: hhg_experimental_fit_quality_{lavg_tag}_{pressure_tag}_{_m2_tag}.png")


        # --- Figure HHG-EF-2: Enhancement Prediction (2x3) ---
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

        exp_I_plot = exp_intensities_1e14   # already in 10^14 units

        # (0,0) Enhancement bar chart (all q, highest power)
        ax = axes2[0, 0]
        x_bar = np.arange(n_q)
        width = 0.35
        enh_sim_hi = [enhancement_sim[iq, 0] for iq in range(n_q)]
        enh_exp_hi = [exp_enhancement[f'H{q}'][0] for q in multi_q_list]
        ax.bar(x_bar - width/2, enh_sim_hi, width, label='Simulation',
               color=COLOR_SIM, edgecolor='black', linewidth=0.5)
        ax.bar(x_bar + width/2, enh_exp_hi, width, label='Experiment',
               color=COLOR_EXP, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_bar)
        ax.set_xticklabels([f'H{q}' for q in multi_q_list])
        ax.set_ylabel('Enhancement Factor')
        ax.set_title('Enhancement at Highest Power')
        ax.legend()
        ax.grid(True, alpha=0.15, axis='y')

        # (0,1) Enhancement vs power (H21, H19, H17)
        ax = axes2[0, 1]
        for iq, q in enumerate([21, 19, 17]):
            idx_q = multi_q_list.index(q)
            exp_key = f'H{q}'
            ax.plot(exp_I_plot, enhancement_sim[idx_q, :], '-o', color=colors_q[q], markersize=7, label=f'H{q} sim')
            ax.plot(exp_I_plot, exp_enhancement[exp_key], 's--', color=colors_q[q], markersize=7, alpha=0.6, label=f'H{q} exp')
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel('Enhancement')
        ax.set_title('Enhancement vs Power (H17-H21)')
        ax.legend()
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1, color=COLOR_REF, linestyle='--', alpha=0.5)

        # (0,2) Enhancement vs power (H15, H13, H11)
        ax = axes2[0, 2]
        for iq, q in enumerate([15, 13, 11]):
            idx_q = multi_q_list.index(q)
            exp_key = f'H{q}'
            ax.plot(exp_I_plot, enhancement_sim[idx_q, :], '-o', color=colors_q[q], markersize=7, label=f'H{q} sim')
            ax.plot(exp_I_plot, exp_enhancement[exp_key], 's--', color=colors_q[q], markersize=7, alpha=0.6, label=f'H{q} exp')
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel('Enhancement')
        ax.set_title('Enhancement vs Power (H11-H15)')
        ax.legend()
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1, color=COLOR_REF, linestyle='--', alpha=0.5)

        # (1,0) Enhancement spectrum: sim vs exp at all power levels
        ax = axes2[1, 0]
        for iP in [0, 3, 6]:   # high, mid, low power
            enh_sim_spectrum = [enhancement_sim[iq, iP] for iq in range(n_q)]
            enh_exp_spectrum = [exp_enhancement[f'H{q}'][iP] for q in multi_q_list]
            label_P = f'I={exp_yield_I_Wcm2[iP]/1e14:.2f}'
            ax.plot(multi_q_list, enh_sim_spectrum, '-o', markersize=5, label=f'Sim {label_P}')
            ax.plot(multi_q_list, enh_exp_spectrum, 's--', markersize=5, alpha=0.5, label=f'Exp {label_P}')
        ax.set_xlabel('Harmonic Order')
        ax.set_ylabel('Enhancement')
        ax.set_title('Enhancement Spectrum (3 power levels)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1, color=COLOR_REF, linestyle='--', alpha=0.5)

        # (1,1) Aperture yield vs transmission (raw, not normalized) for selected harmonics
        ax = axes2[1, 1]
        for q in [11, 15, 19, 21]:
            iq = multi_q_list.index(q)
            y_sim_raw = yield_ap_best[iq, :]
            y_exp_raw = exp_aperture_yield[q][ap_indices]
            # Scale sim to match exp at reference aperture
            scale = y_exp_raw[ap_ref_idx] / y_sim_raw[ap_ref_idx] if y_sim_raw[ap_ref_idx] > 0 else 1.0
            ax.plot(T_plot, y_sim_raw * scale, '-o', color=colors_q[q], markersize=4, label=f'H{q} sim')
            ax.plot(T_plot, y_exp_raw, 's--', color=colors_q[q], markersize=5, alpha=0.6, label=f'H{q} exp')
        ax.set_xlabel('Aperture Transmission')
        ax.set_ylabel('Yield (scaled)')
        ax.set_title('Aperture Yield (scaled sim vs exp)')
        ax.legend()
        ax.grid(True, alpha=0.15)
        ax.invert_xaxis()

        # (1,2) dk diagnostic (uses pre-computed on-axis slices; dipole phase now in complex dipole)
        ax = axes2[1, 2]
        dk_scale = 1e-3   # 1/m -> 1/mm

        ax.plot(z_gas_2d_ub, dk_n_ref_onaxis * dk_scale, '-', color=COLORS_LIST[0], label=r'$\Delta k_{\mathrm{neutral}}$')
        ax.plot(z_gas_2d_ub, dk_p_ref_onaxis * dk_scale, '-', color=COLORS_LIST[3], label=r'$\Delta k_{\mathrm{plasma}}$')
        ax.plot(z_gas_2d_ub, dk_g_ref_onaxis * dk_scale, '-', color=COLORS_LIST[2], label=r'$\Delta k_{\mathrm{geom}}$')
        dk_tot_ref_onaxis = dk_n_ref_onaxis + dk_p_ref_onaxis + dk_g_ref_onaxis
        ax.plot(z_gas_2d_ub, dk_tot_ref_onaxis * dk_scale, '-', color='black', linewidth=2.8, label=r'$\Delta k_{\mathrm{total}}$')
        ax.axhline(y=0, color=COLOR_REF, linestyle='--', alpha=0.5)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel(r'$\Delta k$ (1/mm)')
        ax.set_title('Phase Mismatch (H21, on-axis, no dk_dip)')
        ax.legend()
        ax.grid(True, alpha=0.15)

        fig2.suptitle('HHG Pure-Experimental LUT: Enhancement Prediction', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'hhg_experimental_fit_enhancement_{lavg_tag}_{pressure_tag}_{_m2_tag}.png')
        print(f"  Saved: hhg_experimental_fit_enhancement_{lavg_tag}_{pressure_tag}_{_m2_tag}.png")


        # --- Figure HHG-EF-3: Error Landscape Comparison (1x3) ---
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (err_data, title_str) in enumerate([
            (error_grid_power_scaled, 'Power-Only Cost (norm.)'),
            (error_grid_aperture_scaled, 'Aperture-Only Cost (norm.)'),
            (error_grid, 'Combined Cost'),
        ]):
            ax = axes3[idx]
            err_plot = np.log10(np.where(np.isfinite(err_data), err_data, np.nan) + 1e-10)
            im = ax.pcolormesh(
                Is_grid / 1e14, alpha_grid, err_plot,
                shading='auto', cmap='viridis'
            )
            ax.set_xscale('log')
            ax.set_xlim(Is_grid[0] / 1e14, Is_grid[-1] / 1e14)
            ax.plot(best_Is_grid/1e14, best_alpha_grid, 'r*', markersize=15, label='Grid best')
            ax.plot(best_Is_vals[15]/1e14, best_alphas[15], 'wx', markersize=12, markeredgewidth=2,
                    label=f'Opt (H15: α={best_alphas[15]:.2f})')
            ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')
            ax.set_ylabel(r'$\alpha$')
            ax.set_title(title_str)
            ax.legend()
            plt.colorbar(im, ax=ax)

        fig3.suptitle('HHG Error Landscape: Power vs Aperture Constraints', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'hhg_experimental_fit_error_landscape_{lavg_tag}_{pressure_tag}_{_m2_tag}.png')
        print(f"  Saved: hhg_experimental_fit_error_landscape_{lavg_tag}_{pressure_tag}_{_m2_tag}.png")

        # --- Figure HHG-EF-3b: Error landscape comparison, one figure per harmonic ---
        for iq, q in enumerate(multi_q_list):
            fig3q, axes3q = plt.subplots(1, 3, figsize=(18, 5))
            for idx, (err_data, title_str) in enumerate([
                (_per_harmonic_cost_maps_power[iq], 'Power-Only Cost (norm.)'),
                (_per_harmonic_cost_maps_aperture[iq], 'Aperture-Only Cost (norm.)'),
                (_per_harmonic_cost_maps[iq], 'Combined Cost'),
            ]):
                ax = axes3q[idx]
                err_plot = np.log10(np.where(np.isfinite(err_data), err_data, np.nan) + 1e-10)
                im = ax.pcolormesh(
                    cost_map_Is_range / 1e14, cost_map_alpha_range, err_plot,
                    shading='auto', cmap='viridis'
                )
                ax.set_xscale('log')
                ax.set_xlim(cost_map_Is_range[0] / 1e14, cost_map_Is_range[-1] / 1e14)
                ax.plot(best_Is_grid / 1e14, best_alpha_grid, 'r*', markersize=15, label='Grid best')
                ax.plot(
                    best_Is_vals[q] / 1e14, best_alphas[q], 'wx',
                    markersize=12, markeredgewidth=2,
                    label=f'Opt (H{q}: alpha={best_alphas[q]:.2f})'
                )
                ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')
                ax.set_ylabel(r'$\alpha$')
                ax.set_title(title_str)
                ax.legend()
                plt.colorbar(im, ax=ax)

            fig3q.suptitle(f'HHG Error Landscape: Power vs Aperture Constraints -- H{q}', fontsize=15)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig3q_path = f'hhg_experimental_fit_error_landscape_H{q}_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
            plt.savefig(fig3q_path)
            plt.close(fig3q)
            print(f"  Saved: {fig3q_path}")

        # --- Figure HHG-EF-3c: All harmonics in one cost landscape figure ---
        cost_columns = [
            (_per_harmonic_cost_maps_power, 'Power-Only Cost (norm.)'),
            (_per_harmonic_cost_maps_aperture, 'Aperture-Only Cost (norm.)'),
            (_per_harmonic_cost_maps, 'Combined Cost'),
        ]
        fig3all, axes3all = plt.subplots(
            n_q, 3, figsize=(18, 3.0 * n_q), sharex=True, sharey=True
        )
        axes3all = np.atleast_2d(axes3all)
        ims3all = []

        for col_idx, (maps_all, title_str) in enumerate(cost_columns):
            finite_all = np.log10(maps_all[np.isfinite(maps_all)] + 1e-10)
            if finite_all.size:
                vmin_all = np.nanpercentile(finite_all, 2)
                vmax_all = np.nanpercentile(finite_all, 98)
                if not np.isfinite(vmin_all) or not np.isfinite(vmax_all) or vmin_all >= vmax_all:
                    vmin_all, vmax_all = np.nanmin(finite_all), np.nanmax(finite_all)
            else:
                vmin_all, vmax_all = None, None

            im_col = None
            for iq, q in enumerate(multi_q_list):
                ax = axes3all[iq, col_idx]
                err_data = maps_all[iq]
                err_plot = np.log10(np.where(np.isfinite(err_data), err_data, np.nan) + 1e-10)
                im_col = ax.pcolormesh(
                    cost_map_Is_range / 1e14, cost_map_alpha_range, err_plot,
                    shading='auto', cmap='viridis', vmin=vmin_all, vmax=vmax_all
                )
                ax.set_xscale('log')
                ax.set_xlim(cost_map_Is_range[0] / 1e14, cost_map_Is_range[-1] / 1e14)
                ax.plot(best_Is_grid / 1e14, best_alpha_grid, 'r*', markersize=10)
                ax.plot(
                    best_Is_vals[q] / 1e14, best_alphas[q], 'wx',
                    markersize=9, markeredgewidth=1.8
                )
                if iq == 0:
                    ax.set_title(title_str)
                if col_idx == 0:
                    ax.set_ylabel(f'H{q}\n' + r'$\alpha$')
                if iq == n_q - 1:
                    ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')

            ims3all.append(im_col)

        for col_idx, im_col in enumerate(ims3all):
            cbar = fig3all.colorbar(im_col, ax=axes3all[:, col_idx], fraction=0.025, pad=0.012)
            cbar.set_label(r'$\log_{10}(\mathrm{cost})$')

        fig3all.suptitle('HHG Error Landscape by Harmonic: Power vs Aperture Constraints', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig3all_path = f'hhg_experimental_fit_error_landscape_all_harmonics_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
        plt.savefig(fig3all_path)
        plt.close(fig3all)
        print(f"  Saved: {fig3all_path}")


        # =============================================================================
        # FINAL SUMMARY
        # =============================================================================
        TIMER.end_section()
        print("\n" + "="*60)
        print("PER-HARMONIC FIT — SUMMARY")
        print("="*60)
        for q in multi_q_list:
            print(f"  H{q}: alpha={best_alphas[q]:.3f}, I_s={best_Is_vals[q]:.2e}, cost={best_costs[q]:.4f}")
        print(f"  Total cost (mean): {best_error:.6f}")
        print(f"  Data: {n_P} power levels + {n_ap_sub} apertures")
        print(f"  Weights: lambda_ap = {lambda_ap:.1f}")
        print(f"           lambda_alpha_box = {lambda_alpha_box:.2f} (alpha {alpha_box_min:.1f}-{alpha_box_max:.1f}, sigma={sigma_alpha_box:.1f})")
        print(f"           C_P_scale = {C_P_scale:.6g}")
        print(f"           C_A_scale = {C_A_scale:.6g}")
        print(f"\n  Enhancement comparison (highest power):")
        for iq, q in enumerate(multi_q_list):
            exp_key = f'H{q}'
            e_sim = enhancement_sim[iq, 0]
            e_exp = exp_enhancement[exp_key][0]
            print(f"    H{q}: sim={e_sim:.2f}, exp={e_exp:.2f}, ratio={e_sim/e_exp:.2f}")
        print("="*60)

        # =============================================================================
        # STEP 6: POSITION SCAN VALIDATION  (REMOVED — saves ~50% runtime)
        # =============================================================================

        # =============================================================================
        # SAVE DATA FOR STANDALONE PLOTTING
        # =============================================================================
        _save_file = f'hhg_results_data_{lavg_tag}_{pressure_tag}_{_m2_tag}.npz'
        print(f"\n  Saving plot data to {_save_file} ...")

        # Convert dicts to arrays for npz storage (keyed by multi_q_list order)
        _best_alphas_arr = np.array([best_alphas[q] for q in multi_q_list])
        _best_Is_arr = np.array([best_Is_vals[q] for q in multi_q_list])
        _exp_yield_multi_arr = np.array([exp_yield_multi[q] for q in multi_q_list])
        _exp_aperture_yield_arr = np.array([exp_aperture_yield[q] for q in multi_q_list])
        _exp_enhancement_arr = np.array([exp_enhancement[f'H{q}'] for q in multi_q_list])

        _save_dict = dict(
            # Harmonics list
            multi_q_list=np.array(multi_q_list),
            # Fit yields
            yield_ub_best=yield_ub_best, yield_ap_best=yield_ap_best,
            enhancement_sim=enhancement_sim,
            # Error landscape
            error_grid_power=error_grid_power, error_grid_aperture=error_grid_aperture,
            error_grid_power_scaled=error_grid_power_scaled,
            error_grid_aperture_scaled=error_grid_aperture_scaled,
            alpha_box_grid=alpha_box_grid,
            error_grid_alpha_box=error_grid_alpha_box,
            error_grid_power_q=error_grid_power_q,
            error_grid_aperture_q=error_grid_aperture_q,
            error_grid_power_q_scaled=error_grid_power_q_scaled,
            error_grid_power_chi2_q_scaled=error_grid_power_chi2_q_scaled,
            error_grid_aperture_q_scaled=error_grid_aperture_q_scaled,
            error_grid_power_chi2_q=error_grid_power_chi2_q,
            error_grid_aperture_chi2_q=error_grid_aperture_chi2_q,
            error_grid=error_grid,
            alpha_grid=alpha_grid, Is_grid=Is_grid,
            # Per-harmonic cost landscape
            per_harmonic_cost_maps=_per_harmonic_cost_maps,
            per_harmonic_cost_maps_power=_per_harmonic_cost_maps_power,
            per_harmonic_cost_maps_aperture=_per_harmonic_cost_maps_aperture,
            per_harmonic_cost_maps_power_raw=_per_harmonic_cost_maps_power_raw,
            per_harmonic_cost_maps_aperture_raw=_per_harmonic_cost_maps_aperture_raw,
            per_harmonic_cost_alpha_grid=cost_map_alpha_range,
            per_harmonic_cost_logIs_grid=cost_map_logIs_range,
            per_harmonic_cost_Is_grid=cost_map_Is_range,
            # Per-harmonic fit params
            best_alphas_arr=_best_alphas_arr, best_Is_arr=_best_Is_arr,
            best_costs_arr=np.array([best_costs[q] for q in multi_q_list]),
            best_costs_power_arr=np.array([best_costs_power[q] for q in multi_q_list]),
            best_costs_aperture_arr=np.array([best_costs_aperture[q] for q in multi_q_list]),
            best_costs_power_scaled_arr=np.array([best_costs_power[q] for q in multi_q_list]),
            best_costs_aperture_scaled_arr=np.array([best_costs_aperture[q] / C_Aq_scale_arr[iq] for iq, q in enumerate(multi_q_list)]),
            best_alpha_grid=np.float64(best_alpha_grid),
            best_Is_grid=np.float64(best_Is_grid),
            best_error_grid=np.float64(best_error_grid),
            best_error=np.float64(best_error),
            # dk diagnostic (on-axis)
            z_gas_2d_ub=z_gas_2d_ub,
            dk_n_ref_onaxis=dk_n_ref_onaxis, dk_p_ref_onaxis=dk_p_ref_onaxis,
            dk_g_ref_onaxis=dk_g_ref_onaxis,
            # Experimental data
            exp_yield_I_Wcm2=exp_yield_I_Wcm2,
            exp_intensities_1e14=exp_intensities_1e14,
            exp_yield_multi_arr=_exp_yield_multi_arr,
            exp_aperture_yield_arr=_exp_aperture_yield_arr,
            exp_aperture_T=exp_aperture_T,
            exp_enhancement_arr=_exp_enhancement_arr,
            ap_indices=np.array(ap_indices),
            # Config
            n_q=np.int32(n_q), n_P=np.int32(n_P),
            n_ap_sub=np.int32(n_ap_sub),
            P_ref_idx=np.int32(P_ref_idx),
            ap_ref_idx=np.int32(ap_ref_idx),
            hhg_gas_pressure=np.float64(hhg_gas_pressure),
            lambda_ap=np.float64(lambda_ap),
            lambda_alpha_box=np.float64(lambda_alpha_box),
            alpha_box_min=np.float64(alpha_box_min),
            alpha_box_max=np.float64(alpha_box_max),
            sigma_alpha_box=np.float64(sigma_alpha_box),
            C_P_chi2_q_scale_arr=C_P_chi2_q_scale_arr,
            C_Pq_scale_arr=C_Pq_scale_arr,
            C_Aq_scale_arr=C_Aq_scale_arr,
            C_P_scale=np.float64(C_P_scale),
            C_A_scale=np.float64(C_A_scale),
            eval_total=np.int32(_eval_total),
            # String metadata
            lavg_tag=np.array(lavg_tag),
            pressure_tag=np.array(pressure_tag),
            hhg_gas_type=np.array(hhg_gas_type),
            M2x=np.float64(M2x),
            M2y=np.float64(M2y),
        )


        np.savez_compressed(_save_file, **_save_dict)
        print(f"  Saved: {_save_file} ({os.path.getsize(_save_file) / 1024:.1f} KB)")

        # =============================================================================

# =============================================================================
# USE FITTED PARAMETERS FOR HHG YIELD COMPUTATION
# =============================================================================
print('\n' + '='*60)
print('USING FITTED PARAMETERS FOR HHG YIELD COMPUTATION')
print('='*60)
deconv_alpha_per_h = best_alphas
deconv_Is_per_h = best_Is_vals
deconv_alpha = deconv_alpha_per_h.get(hhg_harmonic_order, 4.13)
deconv_Is = deconv_Is_per_h.get(hhg_harmonic_order, 3.00e13)
for q in multi_q_list:
    print(f'  H{q}: alpha={best_alphas[q]:.3f}, I_s={best_Is_vals[q]:.2e}')


# --- On-axis 1D calculation ---
center_xz = N_xz_focus // 2

# Blocked beam on-axis
I_onaxis_b = xz_intensity_hires[:, center_xz]
gouy_onaxis_b = xz_gouy_phase_hires[:, center_xz]

# Unblocked beam on-axis
I_onaxis_ub = xz_intensity_unblocked[:, center_xz]
gouy_onaxis_ub = xz_gouy_phase_unblocked[:, center_xz]

# Calibrate intensity: use UNBLOCKED beam as reference (preserves blocked/unblocked ratio)
I_scale_factor = hhg_peak_intensity_Wcm2 / I_onaxis_ub.max()
I_onaxis_ub_Wcm2 = I_onaxis_ub * I_scale_factor
I_onaxis_b_Wcm2 = I_onaxis_b * I_scale_factor  # same factor, preserves ratio
print(f"  Intensity ratio (blocked/unblocked peak): {I_onaxis_b_Wcm2.max()/I_onaxis_ub_Wcm2.max():.3f}")

# Ionization fractions
nf_b = ionization_fraction(I_onaxis_b_Wcm2, gas['Ip_eV'])
nf_ub = ionization_fraction(I_onaxis_ub_Wcm2, gas['Ip_eV'])

# Compute 4 Dk terms - BLOCKED
dk_neut_b = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_b, gas['delta_n'])
dk_plas_b = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_b, gas['N_atm'])
dk_gouy_b = calc_dk_gouy_from_sim(gouy_onaxis_b, z_m, hhg_harmonic_order)
dk_dip_b = calc_dk_dipole(I_onaxis_b_Wcm2, z_m, hhg_trajectory)
dk_total_b = dk_neut_b + dk_plas_b + dk_gouy_b + dk_dip_b

# Compute 4 Dk terms - UNBLOCKED
dk_neut_ub = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_ub, gas['delta_n'])
dk_plas_ub = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_ub, gas['N_atm'])
dk_gouy_ub = calc_dk_gouy_from_sim(gouy_onaxis_ub, z_m, hhg_harmonic_order)
dk_dip_ub = calc_dk_dipole(I_onaxis_ub_Wcm2, z_m, hhg_trajectory)
dk_total_ub = dk_neut_ub + dk_plas_ub + dk_gouy_ub + dk_dip_ub

# Coherence lengths
L_coh_b = np.pi / (np.abs(dk_total_b) + 1e-6)    # meters
L_coh_ub = np.pi / (np.abs(dk_total_ub) + 1e-6)

# Ideal Gaussian Gouy for comparison (use unblocked Rayleigh range)
# Estimate z_R from beam parameters: z_R = pi * w0^2 / (M^2 * lambda)
zR_focus_est = np.pi * (focus_spot_size)**2 / (M2x * wavelength)  # in mm
zR_focus_m = zR_focus_est * 1e-3

print(f"\nOn-axis results (blocked beam):")
print(f"  Max |Dk_neutral| = {np.max(np.abs(dk_neut_b)):.2e} /m")
print(f"  Max |Dk_plasma|  = {np.max(np.abs(dk_plas_b)):.2e} /m")
print(f"  Max |Dk_Gouy|    = {np.max(np.abs(dk_gouy_b)):.2e} /m")
print(f"  Max |Dk_dipole|  = {np.max(np.abs(dk_dip_b)):.2e} /m")
idx_min_dk_b = np.argmin(np.abs(dk_total_b))
print(f"  Min |Dk_total|   = {np.abs(dk_total_b[idx_min_dk_b]):.2e} /m at z = {z_focus_prop[idx_min_dk_b]:.3f} mm")
print(f"  Max coherence length = {np.max(L_coh_b)*1e3:.3f} mm")

# --- 2D x-z plane phase mismatch maps ---
print("\nComputing 2D phase mismatch maps...")

# 2D intensity calibration (shared scale factor from unblocked beam)
I_scale_factor_2d = hhg_peak_intensity_Wcm2 / xz_intensity_unblocked.max()
I_xz_ub_Wcm2 = xz_intensity_unblocked * I_scale_factor_2d
I_xz_b_Wcm2 = xz_intensity_hires * I_scale_factor_2d

# 2D ionization
nf_2d_b = ionization_fraction(I_xz_b_Wcm2, gas['Ip_eV'])
nf_2d_ub = ionization_fraction(I_xz_ub_Wcm2, gas['Ip_eV'])

# 2D neutral and plasma (element-wise, same formulas)
dk_neut_2d_b = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_2d_b, gas['delta_n'])
dk_plas_2d_b = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_2d_b, gas['N_atm'])

dk_neut_2d_ub = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_2d_ub, gas['delta_n'])
dk_plas_2d_ub = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_2d_ub, gas['N_atm'])

# 2D Gouy phase: use pre-computed gouy phase arrays (plane wave already removed)
phase_2d_b_geom = np.unwrap(xz_gouy_phase_hires, axis=0)
phase_2d_ub_geom = np.unwrap(xz_gouy_phase_unblocked, axis=0)

# Mask low-intensity regions for phase (avoid noise)
I_mask_b = I_xz_b_Wcm2 > 0.01 * hhg_peak_intensity_Wcm2
I_mask_ub = I_xz_ub_Wcm2 > 0.01 * hhg_peak_intensity_Wcm2

dphase_dz_2d_b = np.gradient(phase_2d_b_geom, z_m, axis=0)
dphase_dz_2d_ub = np.gradient(phase_2d_ub_geom, z_m, axis=0)
dk_geom_2d_b = -(hhg_harmonic_order - 1.0 / hhg_harmonic_order) * dphase_dz_2d_b
dk_geom_2d_ub = -(hhg_harmonic_order - 1.0 / hhg_harmonic_order) * dphase_dz_2d_ub

# 2D dipole phase
alpha_dict = {'short': 1.0e-14, 'long': 5.0e-14}
alpha_SI = alpha_dict[hhg_trajectory] * 1e-4  # rad * m^2 / W
dI_dz_2d_b = np.gradient(I_xz_b_Wcm2 * 1e4, z_m, axis=0)  # W/m^2 per m
dI_dz_2d_ub = np.gradient(I_xz_ub_Wcm2 * 1e4, z_m, axis=0)
dk_dip_2d_b = -alpha_SI * dI_dz_2d_b
dk_dip_2d_ub = -alpha_SI * dI_dz_2d_ub

# Total 2D
dk_total_2d_b = dk_neut_2d_b + dk_plas_2d_b + dk_geom_2d_b + dk_dip_2d_b
dk_total_2d_ub = dk_neut_2d_ub + dk_plas_2d_ub + dk_geom_2d_ub + dk_dip_2d_ub

# Apply mask: set masked regions to NaN for plotting
dk_total_2d_b_masked = np.where(I_mask_b, dk_total_2d_b, np.nan)
dk_total_2d_ub_masked = np.where(I_mask_ub, dk_total_2d_ub, np.nan)

L_coh_2d_b = np.pi / (np.abs(dk_total_2d_b) + 1e-6)
L_coh_2d_ub = np.pi / (np.abs(dk_total_2d_ub) + 1e-6)
L_coh_2d_b_masked = np.where(I_mask_b, L_coh_2d_b, np.nan)
L_coh_2d_ub_masked = np.where(I_mask_ub, L_coh_2d_ub, np.nan)

print("  2D phase mismatch computation complete.")

# =============================================================================
# Figure HHG-1: On-axis phase mismatch terms
# =============================================================================
fig_hhg1, axes_hhg1 = plt.subplots(2, 2, figsize=(14, 10))

z_plot = z_focus_prop  # mm

# Convert Dk from 1/m to 1/mm for plotting
dk_scale = 1e-3  # multiply by this to go from 1/m to 1/mm

# (0,0) Individual Dk terms - blocked
ax = axes_hhg1[0, 0]
ax.plot(z_plot, dk_neut_b * dk_scale, 'b-', label=r'$\Delta k_{neutral}$', linewidth=1.5)
ax.plot(z_plot, dk_plas_b * dk_scale, 'r-', label=r'$\Delta k_{plasma}$', linewidth=1.5)
ax.plot(z_plot, dk_gouy_b * dk_scale, 'g-', label=r'$\Delta k_{Gouy}$', linewidth=1.5)
ax.plot(z_plot, dk_dip_b * dk_scale, 'm-', label=r'$\Delta k_{dipole}$', linewidth=1.5)
ax.plot(z_plot, dk_total_b * dk_scale, 'k-', label=r'$\Delta k_{total}$', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=true_focus_z, color='gray', linestyle=':', alpha=0.5, label='Focus')
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Delta k$ (1/mm)')
ax.set_title(f'Blocked Beam - Phase Mismatch (H{hhg_harmonic_order})')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# (0,1) Individual Dk terms - unblocked
ax = axes_hhg1[0, 1]
ax.plot(z_plot, dk_neut_ub * dk_scale, 'b-', label=r'$\Delta k_{neutral}$', linewidth=1.5)
ax.plot(z_plot, dk_plas_ub * dk_scale, 'r-', label=r'$\Delta k_{plasma}$', linewidth=1.5)
ax.plot(z_plot, dk_gouy_ub * dk_scale, 'g-', label=r'$\Delta k_{Gouy}$', linewidth=1.5)
ax.plot(z_plot, dk_dip_ub * dk_scale, 'm-', label=r'$\Delta k_{dipole}$', linewidth=1.5)
ax.plot(z_plot, dk_total_ub * dk_scale, 'k-', label=r'$\Delta k_{total}$', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=true_focus_z_unblocked, color='gray', linestyle=':', alpha=0.5, label='Focus')
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Delta k$ (1/mm)')
ax.set_title(f'Unblocked Beam - Phase Mismatch (H{hhg_harmonic_order})')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# (1,0) Total |Dk| comparison (absolute value to show inverse relation with L_coh)
ax = axes_hhg1[1, 0]
ax.plot(z_plot, np.abs(dk_total_b) * dk_scale, 'b-', label='Blocked', linewidth=2)
ax.plot(z_plot, np.abs(dk_total_ub) * dk_scale, 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$|\Delta k_{total}|$ (1/mm)')
ax.set_title(r'Total $|\Delta k|$ Comparison')
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# (1,1) Coherence length comparison
ax = axes_hhg1[1, 1]
ax.plot(z_plot, L_coh_b * 1e3, 'b-', label='Blocked', linewidth=2)
ax.plot(z_plot, L_coh_ub * 1e3, 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$L_{coh}$ (mm)')
ax.set_title(r'Coherence Length $L_{coh} = \pi / |\Delta k|$')
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

fig_hhg1.suptitle(f'HHG Phase Mismatch - {hhg_gas_type.capitalize()}, P={hhg_gas_pressure} mbar, '
                   f'H{hhg_harmonic_order}, I₀={hhg_peak_intensity_Wcm2:.1e} W/cm², '
                   f'{hhg_trajectory} traj.', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_phase_mismatch_onaxis_{_m2_tag}.png', dpi=300)
print("HHG on-axis phase mismatch figure saved to 'hhg_phase_mismatch_onaxis.png'")

# =============================================================================
# Figure HHG-2: 2D phase mismatch maps
# =============================================================================
fig_hhg2, axes_hhg2 = plt.subplots(2, 3, figsize=(18, 10))

x_um = x_xz_hires * 1e3  # mm to um
extent_xz = [x_um[0], x_um[-1], z_plot[0], z_plot[-1]]

# Determine symmetric color limits for Dk
dk_max = np.nanpercentile(np.abs(dk_total_2d_b_masked * dk_scale), 95)

# (0,0) Total Dk - blocked
ax = axes_hhg2[0, 0]
im = ax.imshow(dk_total_2d_b_masked * dk_scale, aspect='auto', origin='lower',
               extent=extent_xz, cmap='RdBu_r', vmin=-dk_max, vmax=dk_max,
               interpolation='bicubic')
plt.colorbar(im, ax=ax, label=r'$\Delta k$ (1/mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title('Blocked - Total Δk')

# (0,1) Total Dk - unblocked
ax = axes_hhg2[0, 1]
im = ax.imshow(dk_total_2d_ub_masked * dk_scale, aspect='auto', origin='lower',
               extent=extent_xz, cmap='RdBu_r', vmin=-dk_max, vmax=dk_max,
               interpolation='bicubic')
plt.colorbar(im, ax=ax, label=r'$\Delta k$ (1/mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title('Unblocked - Total Δk')

# (0,2) Difference
dk_diff = dk_total_2d_b_masked - dk_total_2d_ub_masked
dk_diff_max = np.nanpercentile(np.abs(dk_diff * dk_scale), 95)
ax = axes_hhg2[0, 2]
im = ax.imshow(dk_diff * dk_scale, aspect='auto', origin='lower',
               extent=extent_xz, cmap='RdBu_r', vmin=-dk_diff_max, vmax=dk_diff_max,
               interpolation='bicubic')
plt.colorbar(im, ax=ax, label=r'$\Delta\Delta k$ (1/mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title('Difference (Blocked - Unblocked)')

# (1,0) and (1,1) Coherence length — shared color scale
L_coh_plot_b = np.clip(L_coh_2d_b_masked * 1e3, 1e-3, 100)  # in mm, clip for log
L_coh_plot_ub = np.clip(L_coh_2d_ub_masked * 1e3, 1e-3, 100)
lcoh_log_min = min(np.nanmin(np.log10(L_coh_plot_b)), np.nanmin(np.log10(L_coh_plot_ub)))
lcoh_log_max = max(np.nanmax(np.log10(L_coh_plot_b)), np.nanmax(np.log10(L_coh_plot_ub)))

ax = axes_hhg2[1, 0]
im = ax.imshow(np.log10(L_coh_plot_b), aspect='auto', origin='lower',
               extent=extent_xz, cmap='hot', vmin=lcoh_log_min, vmax=lcoh_log_max,
               interpolation='bicubic')
cbar = plt.colorbar(im, ax=ax, label=r'$\log_{10}(L_{coh}$ / mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title(r'Blocked - $L_{coh}$')

# (1,1) Coherence length - unblocked
ax = axes_hhg2[1, 1]
im = ax.imshow(np.log10(L_coh_plot_ub), aspect='auto', origin='lower',
               extent=extent_xz, cmap='hot', vmin=lcoh_log_min, vmax=lcoh_log_max,
               interpolation='bicubic')
plt.colorbar(im, ax=ax, label=r'$\log_{10}(L_{coh}$ / mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title(r'Unblocked - $L_{coh}$')

# (1,2) Ratio of coherence lengths
L_coh_ratio = L_coh_2d_b_masked / (L_coh_2d_ub_masked + 1e-20)
ax = axes_hhg2[1, 2]
ratio_max = np.nanpercentile(np.abs(np.log10(L_coh_ratio + 1e-20)), 95)
im = ax.imshow(np.log10(L_coh_ratio + 1e-20), aspect='auto', origin='lower',
               extent=extent_xz, cmap='RdBu_r', vmin=-ratio_max, vmax=ratio_max,
               interpolation='bicubic')
plt.colorbar(im, ax=ax, label=r'$\log_{10}(L_{coh,B}/L_{coh,UB})$')
ax.set_xlabel('x (μm)')
ax.set_ylabel('z (mm)')
ax.set_title(r'$L_{coh}$ Ratio (Blocked / Unblocked)')

fig_hhg2.suptitle(f'2D Phase Mismatch Maps (x-z plane) - {hhg_gas_type.capitalize()}, '
                   f'H{hhg_harmonic_order}', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_phase_mismatch_2d_{_m2_tag}.png', dpi=300)
print("HHG 2D phase mismatch figure saved to 'hhg_phase_mismatch_2d.png'")

# =============================================================================
# Figure HHG-2B: Phase mismatch at focus plane (radial profiles)
# =============================================================================
fig_hhg2b, axes_hhg2b = plt.subplots(2, 3, figsize=(18, 10))

x_um = x_xz_hires * 1e3  # mm to μm

# Extract all Dk components at focus z-index
# Blocked beam
dk_neut_focus_b = dk_neut_2d_b[focus_idx, :] * dk_scale
dk_plas_focus_b = dk_plas_2d_b[focus_idx, :] * dk_scale
dk_geom_focus_b = dk_geom_2d_b[focus_idx, :] * dk_scale
dk_dip_focus_b = dk_dip_2d_b[focus_idx, :] * dk_scale
dk_total_focus_b = dk_total_2d_b[focus_idx, :] * dk_scale
I_focus_b = I_xz_b_Wcm2[focus_idx, :]

# Unblocked beam
dk_neut_focus_ub = dk_neut_2d_ub[focus_idx_ub, :] * dk_scale
dk_plas_focus_ub = dk_plas_2d_ub[focus_idx_ub, :] * dk_scale
dk_geom_focus_ub = dk_geom_2d_ub[focus_idx_ub, :] * dk_scale
dk_dip_focus_ub = dk_dip_2d_ub[focus_idx_ub, :] * dk_scale
dk_total_focus_ub = dk_total_2d_ub[focus_idx_ub, :] * dk_scale
I_focus_ub = I_xz_ub_Wcm2[focus_idx_ub, :]

# Mask to beam region (>1% of peak)
mask_b = I_focus_b > 0.01 * I_focus_b.max()
mask_ub = I_focus_ub > 0.01 * I_focus_ub.max()

# (0,0) Individual Dk terms at focus - blocked
ax = axes_hhg2b[0, 0]
ax.plot(x_um, np.where(mask_b, dk_neut_focus_b, np.nan), 'b-', label=r'$\Delta k_{neutral}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_b, dk_plas_focus_b, np.nan), 'r-', label=r'$\Delta k_{plasma}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_b, dk_geom_focus_b, np.nan), 'g-', label=r'$\Delta k_{geom}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_b, dk_dip_focus_b, np.nan), 'm-', label=r'$\Delta k_{dipole}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_b, dk_total_focus_b, np.nan), 'k-', label=r'$\Delta k_{total}$', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$\Delta k$ (1/mm)')
ax.set_title(f'Blocked - Δk at Focus (z={true_focus_z:.2f} mm)')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 50])

# (0,1) Individual Dk terms at focus - unblocked
ax = axes_hhg2b[0, 1]
ax.plot(x_um, np.where(mask_ub, dk_neut_focus_ub, np.nan), 'b-', label=r'$\Delta k_{neutral}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_ub, dk_plas_focus_ub, np.nan), 'r-', label=r'$\Delta k_{plasma}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_ub, dk_geom_focus_ub, np.nan), 'g-', label=r'$\Delta k_{geom}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_ub, dk_dip_focus_ub, np.nan), 'm-', label=r'$\Delta k_{dipole}$', linewidth=1.5)
ax.plot(x_um, np.where(mask_ub, dk_total_focus_ub, np.nan), 'k-', label=r'$\Delta k_{total}$', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$\Delta k$ (1/mm)')
ax.set_title(f'Unblocked - Δk at Focus (z={true_focus_z_unblocked:.2f} mm)')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 50])

# (0,2) Total Dk comparison at focus
ax = axes_hhg2b[0, 2]
ax.plot(x_um, np.where(mask_b, dk_total_focus_b, np.nan), 'b-', label='Blocked', linewidth=2)
ax.plot(x_um, np.where(mask_ub, dk_total_focus_ub, np.nan), 'r--', label='Unblocked', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$\Delta k_{total}$ (1/mm)')
ax.set_title(r'Total $\Delta k$ Comparison at Focus')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 50])

# (1,0) |Dk| and intensity overlay - blocked
ax = axes_hhg2b[1, 0]
ax.plot(x_um, np.where(mask_b, np.abs(dk_total_focus_b), np.nan), 'b-', label=r'$|\Delta k_{total}|$', linewidth=2)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$|\Delta k|$ (1/mm)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(x_um, I_focus_b, 'r-', alpha=0.5, linewidth=1.5, label='Intensity')
ax2.set_ylabel(r'Intensity (W/cm$^2$)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax.set_title('Blocked - |Δk| & Intensity at Focus')
ax.set_xlim([-50, 50])

# (1,1) |Dk| and intensity overlay - unblocked
ax = axes_hhg2b[1, 1]
ax.plot(x_um, np.where(mask_ub, np.abs(dk_total_focus_ub), np.nan), 'b-', label=r'$|\Delta k_{total}|$', linewidth=2)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$|\Delta k|$ (1/mm)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(x_um, I_focus_ub, 'r-', alpha=0.5, linewidth=1.5, label='Intensity')
ax2.set_ylabel(r'Intensity (W/cm$^2$)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax.set_title('Unblocked - |Δk| & Intensity at Focus')
ax.set_xlim([-50, 50])

# (1,2) Coherence length at focus
L_coh_focus_b = np.pi / (np.abs(dk_total_2d_b[focus_idx, :]) + 1e-6) * 1e3  # mm
L_coh_focus_ub = np.pi / (np.abs(dk_total_2d_ub[focus_idx_ub, :]) + 1e-6) * 1e3
ax = axes_hhg2b[1, 2]
ax.plot(x_um, np.where(mask_b, L_coh_focus_b, np.nan), 'b-', label='Blocked', linewidth=2)
ax.plot(x_um, np.where(mask_ub, L_coh_focus_ub, np.nan), 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$L_{coh}$ (mm)')
ax.set_title(r'Coherence Length at Focus')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 50])

fig_hhg2b.suptitle(f'Phase Mismatch at Focus Plane - {hhg_gas_type.capitalize()}, P={hhg_gas_pressure} mbar, '
                    f'H{hhg_harmonic_order}, {hhg_trajectory} traj.', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_phase_mismatch_focus_{_m2_tag}.png', dpi=300)
print("HHG focus plane phase mismatch figure saved to 'hhg_phase_mismatch_focus.png'")

# =============================================================================
# Figure HHG-3: Supporting context (intensity, ionization, Gouy phase)
# =============================================================================
fig_hhg3, axes_hhg3 = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) On-axis intensity
ax = axes_hhg3[0, 0]
ax.plot(z_plot, I_onaxis_b_Wcm2, 'b-', label='Blocked', linewidth=2)
ax.plot(z_plot, I_onaxis_ub_Wcm2, 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'Intensity (W/cm$^2$)')
ax.set_title('On-axis Intensity')
ax.legend()
ax.grid(True, alpha=0.3)

# (0,1) Ionization fraction
ax = axes_hhg3[0, 1]
ax.plot(z_plot, nf_b, 'b-', label='Blocked', linewidth=2)
ax.plot(z_plot, nf_ub, 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Ionization fraction')
ax.set_title(f'Ionization Fraction ({hhg_gas_type.capitalize()}, Ip={gas["Ip_eV"]:.2f} eV)')
ax.legend()
ax.grid(True, alpha=0.3)

# (1,0) Extracted Gouy phase (plane wave removed in propagation loop)
gouy_b = np.unwrap(gouy_onaxis_b)
gouy_b -= gouy_b[len(gouy_b) // 2]

gouy_ub = np.unwrap(gouy_onaxis_ub)
gouy_ub -= gouy_ub[len(gouy_ub) // 2]

# Ideal Gaussian Gouy phase
z_rel_ub = z_m - z_focus_ub_m
gouy_ideal = np.arctan(z_rel_ub / zR_focus_m)
gouy_ideal -= gouy_ideal[len(gouy_ideal)//2]  # center at 0

ax = axes_hhg3[1, 0]
ax.plot(z_plot, gouy_b, 'b-', label='Blocked (sim)', linewidth=2)
ax.plot(z_plot, gouy_ub, 'r--', label='Unblocked (sim)', linewidth=2)
ax.plot(z_plot, gouy_ideal, 'k:', label='Ideal Gaussian', linewidth=1.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Gouy phase (rad)')
ax.set_title('Extracted Gouy Phase')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,1) Gouy phase gradient
ax = axes_hhg3[1, 1]
dgouy_b = np.gradient(gouy_b, z_m)
dgouy_ub = np.gradient(gouy_ub, z_m)
dgouy_ideal = np.gradient(gouy_ideal, z_m)
ax.plot(z_plot, dgouy_b, 'b-', label='Blocked (sim)', linewidth=2)
ax.plot(z_plot, dgouy_ub, 'r--', label='Unblocked (sim)', linewidth=2)
ax.plot(z_plot, dgouy_ideal, 'k:', label='Ideal Gaussian', linewidth=1.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'd$\phi_{Gouy}$/dz (rad/m)')
ax.set_title('Gouy Phase Gradient')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig_hhg3.suptitle(f'HHG Context - Intensity, Ionization & Gouy Phase', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_ionization_context_{_m2_tag}.png', dpi=300)
print("HHG ionization context figure saved to 'hhg_ionization_context.png'")

# --- HHG Summary ---
print(f"\n{'='*50}")
print("HHG Phase Mismatch Summary:")
print(f"{'='*50}")
print(f"  Harmonic: H{hhg_harmonic_order} ({wavelength * 1e6 / hhg_harmonic_order:.1f} nm)")
print(f"  Gas: {hhg_gas_type.capitalize()}, P = {hhg_gas_pressure} mbar")
print(f"  Peak intensity: {hhg_peak_intensity_Wcm2:.2e} W/cm^2")
print(f"  Trajectory: {hhg_trajectory}")
print(f"  --- Blocked beam ---")
print(f"    Min |Dk_total| = {np.min(np.abs(dk_total_b)):.2e} /m")
print(f"    at z = {z_focus_prop[np.argmin(np.abs(dk_total_b))]:.3f} mm")
print(f"    Max L_coh = {np.max(L_coh_b)*1e3:.3f} mm")
print(f"  --- Unblocked beam ---")
print(f"    Min |Dk_total| = {np.min(np.abs(dk_total_ub)):.2e} /m")
print(f"    at z = {z_focus_prop[np.argmin(np.abs(dk_total_ub))]:.3f} mm")
print(f"    Max L_coh = {np.max(L_coh_ub)*1e3:.3f} mm")
print(f"  Max ionization fraction (blocked):   {np.max(nf_b):.4f}")
print(f"  Max ionization fraction (unblocked): {np.max(nf_ub):.4f}")

# Print parameters
print(f"\n{'='*50}")
print("Simulation Parameters:")
print(f"{'='*50}")
print(f"Wavelength: {wavelength*1e6:.1f} nm")
print(f"M^2 x: {M2x:.2f}")
print(f"M^2 y: {M2y:.2f}")
print(f"Measured beam waist: {w0x_measured:.2f} x {w0y_measured:.2f} mm")
print(f"Fundamental mode waist: {w0x:.3f} x {w0y:.3f} mm")
print(f"Rayleigh range (real beam): {zRx/10:.2f} cm")
zR_fund_x = np.pi * w0x**2 / wavelength
print(f"Rayleigh range (fund. mode): {zR_fund_x/10:.2f} cm")
print(f"\nAperture Parameters:")
print(f"  Aperture radius: {aperture_radius:.1f} mm")
print(f"  Aperture position from waist: {aperture_position/10:.1f} cm")
print(f"  Distance before lens: {aperture_distance_before_lens/10:.0f} cm")
print(f"  Aperture transmission: {aperture_transmission:.2f}%")
print(f"\nLens position from waist: {lens_position/10:.1f} cm")
print(f"Focal length: {focal_length/10:.1f} cm")
print(f"True focus position: {true_focus_z/10:.2f} cm")
print(f"Focus shift from f: {(true_focus_z - focal_length):.3f} mm")

# Beam parameters at lens (using x direction, assuming symmetric for now)
wx_at_lens = w0x_measured * np.sqrt(1 + (lens_position / zRx)**2)
wy_at_lens = w0y_measured * np.sqrt(1 + (lens_position / zRy)**2)
print(f"Beam radius at lens (x, measured): {wx_at_lens:.4f} mm")
print(f"Beam radius at lens (y, measured): {wy_at_lens:.4f} mm")

# Focused spot size (theoretical for Gaussian: w_focus = M² * λf/(π*w_input))
# For real beam with M², the focused spot is M² times larger than ideal
w_focus_x_theoretical = M2x * wavelength * focal_length / (np.pi * wx_at_lens)
w_focus_y_theoretical = M2y * wavelength * focal_length / (np.pi * wy_at_lens)
print(f"Theoretical focus spot size (x): {w_focus_x_theoretical*1e3:.2f} um")
print(f"Theoretical focus spot size (y): {w_focus_y_theoretical*1e3:.2f} um")
w_focus_theoretical = (w_focus_x_theoretical + w_focus_y_theoretical) / 2  # average for resolution check

# Actual focus spot size from simulation (use high-res data if available)
if 'I_focus_hires' in dir():
    intensity_focus_sim = I_focus_hires.max()
    print(f"Simulated FWHM at focus (high-res): {fwhm_x_hr*1e3:.2f} x {fwhm_y_hr*1e3:.2f} um")
else:
    print("  (Skipped high-res focus diagnostics — PLOT_OPTICAL_DIAGNOSTICS is off)")

print(f"\n{'='*50}")
print("Numerical Resolution:")
print(f"{'='*50}")
print(f"Grid spacing (dx): {dx*1e3:.2f} um")
print(f"Focal spot size: {w_focus_theoretical*1e3:.2f} um")
print(f"Points across spot: {w_focus_theoretical/dx:.1f}")
if w_focus_theoretical/dx < 5:
    print("WARNING: Focal spot is undersampled! Consider increasing N or decreasing L.")

print(f"\n{'='*50}")
print("Propagation Method Info:")
print(f"{'='*50}")
print(f"Selected method: {PROPAGATION_METHOD}")
if PROPAGATION_METHOD == 'auto':
    print("  Auto-selection based on Fresnel number (N_F = a^2/lz):")
    for z_test in [lens_position, focal_length*0.5, focal_length, focal_length*1.5]:
        N_F_test = beam_radius_eff**2 / (wavelength * z_test)
        method_used = "Fresnel" if N_F_test < 0.5 else "ASM"
        print(f"    z = {z_test/10:.1f} cm: N_F = {N_F_test:.4f} → {method_used}")

print(f"\n{'='*50}")
print("Optimization Info:")
print(f"{'='*50}")
print(f"Using scipy.fft: {USE_SCIPY_FFT}")
print(f"Using Numba JIT: {USE_NUMBA}")
print(f"Number of parallel workers: {NUM_WORKERS}")

# Cache performance statistics
cache_stats = get_cache_stats()
print(f"\nCache Performance:")
print(f"  CZT Param cache:  {cache_stats['czt_hits']} hits / {cache_stats['czt_misses']} misses ({cache_stats['czt_hit_rate']:.1f}% hit rate)")
print(f"  Quad Phase cache: {cache_stats['quad_hits']} hits / {cache_stats['quad_misses']} misses ({cache_stats['quad_hit_rate']:.1f}% hit rate)")
print(f"  Cache entries: {cache_stats['czt_cache_size']} CZT params, {cache_stats['quad_cache_size']} quad phases")

# =============================================================================
# MACROSCOPIC HHG YIELD (Lewenstein LUT + Phase Matching)
# =============================================================================
TIMER.start_section("Macroscopic HHG Yield")
print(f"\n{'='*60}")
print("MACROSCOPIC HHG YIELD CALCULATION")
print(f"{'='*60}")


# --- SFA pulse parameters ---
sfa_omega = 0.05767513     # 790 nm in a.u.
sfa_n_cycles = 8
sfa_dt = 0.5               # a.u.
sfa_Tfull = 1000.0          # a.u., sufficient for 8 cycles
sfa_chirp = 0.0             # no chirp
sfa_Thalf = np.pi / sfa_omega * sfa_n_cycles
sfa_Tcut = 2 * sfa_Thalf
sfa_omega2 = sfa_omega / (2 * sfa_n_cycles)
sfa_Ip_au = gas['Ip_eV'] / 27.2114   # argon: 0.5793 a.u.
sfa_I_to_E0 = 3.5094e16              # I(W/cm^2) = E0^2 * 3.5094e16

# --- Gas region: use stored 2D data from propagation loops ---
hhg_gas_length = hhg_gas_length_prop   # mm (defined earlier, default 1.0)
z_gas_2d_b_m = z_gas_2d_b * 1e-3      # mm → m
z_gas_2d_ub_m = z_gas_2d_ub * 1e-3
n_z_gas_b = len(z_gas_2d_b)
n_z_gas_ub = len(z_gas_2d_ub)
print(f"  Gas region (blocked):   {z_gas_2d_b[0]:.2f} to {z_gas_2d_b[-1]:.2f} mm ({n_z_gas_b} z-points)")
print(f"  Gas region (unblocked): {z_gas_2d_ub[0]:.2f} to {z_gas_2d_ub[-1]:.2f} mm ({n_z_gas_ub} z-points)")
print(f"  2D grid: {N_hhg_2d}x{N_hhg_2d} = {N_hhg_2d**2} points per z-plane")

# --- Check LUT cache ---
n_lut = 80
_lut_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'_lut_cache_{hhg_gas_type}_n{n_lut}_Imax{hhg_peak_intensity_Wcm2:.0e}.npz')
_LUT_LOADED = False

if os.path.exists(_lut_cache_file):
    print(f"  [LUT CACHE] Loading from {_lut_cache_file}")
    _lut_data = np.load(_lut_cache_file, allow_pickle=True)
    I_lut = _lut_data['I_lut']
    n_lut = len(I_lut)
    I_lut_min = float(_lut_data['I_lut_min'])
    I_lut_max = float(_lut_data['I_lut_max'])
    dq_mag = _lut_data['dq_mag']
    dq_phase = _lut_data['dq_phase']
    dq_mag_interp = interp1d(I_lut, dq_mag, kind='cubic', bounds_error=False, fill_value=0.0)
    dq_phase_interp = interp1d(I_lut, dq_phase, kind='cubic', bounds_error=False, fill_value=0.0)
    def dq_complex_interp(I_arr):
        mag = dq_mag_interp(I_arr)
        phase = dq_phase_interp(I_arr)
        return mag * np.exp(1j * phase)
    multi_lut = {int(k): {'mag': v['mag'], 'phase': v['phase']} for k, v in _lut_data['multi_lut'].item().items()}
    sfa_omega = float(_lut_data['sfa_omega'])
    sfa_Ip_au = float(_lut_data['sfa_Ip_au'])
    sfa_I_to_E0 = float(_lut_data['sfa_I_to_E0'])
    multi_lut_interp = {}
    for q in [11, 13, 15, 17, 19, 21]:
        multi_lut_interp[q] = {
            'mag': interp1d(I_lut, multi_lut[q]['mag'], kind='cubic', bounds_error=False, fill_value=0.0),
            'phase': interp1d(I_lut, multi_lut[q]['phase'], kind='cubic', bounds_error=False, fill_value=0.0),
        }
    hhg_lut_include_ppt = False
    sfa_n_cycles = 8; sfa_dt = 0.5; sfa_Tfull = 1000.0; sfa_chirp = 0.0
    sfa_Thalf = np.pi / sfa_omega * sfa_n_cycles
    sfa_Tcut = 2 * sfa_Thalf
    sfa_omega2 = sfa_omega / (2 * sfa_n_cycles)
    sfa_ti = np.linspace(0, sfa_Tfull, num=int(sfa_Tfull / sfa_dt) + 1)
    sfa_window = signal.windows.flattop(len(sfa_ti))
    sfa_omegalist = np.fft.rfftfreq(len(sfa_ti), d=sfa_dt) * 2 * np.pi / sfa_omega
    _LUT_LOADED = True
    del _lut_data
    print(f"  Loaded all LUT data ({n_lut} points, 6 harmonics)")

if not _LUT_LOADED:
    # --- Build LUT: complex d_q(I) via Lewenstein model ---
    n_lut = 80
    I_lut_min = 1e13
    I_lut_max = hhg_peak_intensity_Wcm2
    I_lut = np.logspace(np.log10(I_lut_min), np.log10(I_lut_max), n_lut)
    d_q_lut = np.zeros(n_lut, dtype=np.complex128)

    sfa_ti = np.linspace(0, sfa_Tfull, num=int(sfa_Tfull / sfa_dt) + 1)
    sfa_window = signal.windows.flattop(len(sfa_ti))
    sfa_omegalist = np.fft.rfftfreq(len(sfa_ti), d=sfa_dt) * 2 * np.pi / sfa_omega
    sfa_idx_q = np.argmin(np.abs(sfa_omegalist - hhg_harmonic_order))
    print(f"  Building Lewenstein LUT: {n_lut} intensity points for H{hhg_harmonic_order}")
    print(f"  Intensity range: {I_lut_min:.1e} to {I_lut_max:.1e} W/cm^2")
    print(f"  Ip = {sfa_Ip_au:.4f} a.u. ({gas['Ip_eV']:.2f} eV), omega = {sfa_omega} a.u.")
    print(f"  Pulse: {sfa_n_cycles} cycles, dt = {sfa_dt} a.u., {len(sfa_ti)} time points")
    print(f"  PPT depletion in SFA: {'ON' if hhg_lut_include_ppt else 'OFF (pure SFA — ionization via macroscopic (1-nf) only)'}")

    lut_start = time.time()
    # Warm up Numba JIT with first call
    E0_warmup = np.sqrt(I_lut[0] / sfa_I_to_E0)
    Ef_warmup = sfa_E_flist(sfa_ti, E0_warmup, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
    Af_warmup = sfa_A_flist(sfa_ti, E0_warmup, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
    dm_warmup, _, _ = sfa_dipole_momion(Af_warmup, Ef_warmup, sfa_ti, sfa_Ip_au, sfa_omega, hhg_lut_include_ppt)
    HHG_warmup = np.fft.rfft(np.real(dm_warmup) * sfa_window)
    d_q_lut[0] = HHG_warmup[sfa_idx_q]
    print(f"  JIT warmup done ({time.time()-lut_start:.1f}s). Computing remaining {n_lut-1} points...")

    for i_lut in range(1, n_lut):
        E0_i = np.sqrt(I_lut[i_lut] / sfa_I_to_E0)
        Ef_i = sfa_E_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
        Af_i = sfa_A_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
        dm_i, _, _ = sfa_dipole_momion(Af_i, Ef_i, sfa_ti, sfa_Ip_au, sfa_omega, hhg_lut_include_ppt)
        HHG_i = np.fft.rfft(np.real(dm_i) * sfa_window)
        d_q_lut[i_lut] = HHG_i[sfa_idx_q]
        if (i_lut + 1) % 10 == 0:
            elapsed = time.time() - lut_start
            eta_s = elapsed / (i_lut + 1) * (n_lut - i_lut - 1)
            print(f"    [{i_lut+1}/{n_lut}] elapsed={elapsed:.0f}s, ETA={eta_s:.0f}s")

    lut_time = time.time() - lut_start
    print(f"  LUT complete in {lut_time:.1f}s ({lut_time/60:.1f} min)")

    # --- Build interpolators for magnitude and phase ---
    dq_mag = np.abs(d_q_lut)
    dq_phase = np.unwrap(np.angle(d_q_lut))
    dq_mag_interp = interp1d(I_lut, dq_mag, kind='cubic', bounds_error=False, fill_value=0.0)
    dq_phase_interp = interp1d(I_lut, dq_phase, kind='cubic', bounds_error=False, fill_value=0.0)

def dq_complex_interp(I_arr):
    """Interpolate complex d_q onto an intensity array of any shape."""
    mag = dq_mag_interp(I_arr)
    phase = dq_phase_interp(I_arr)
    return mag * np.exp(1j * phase)

# --- LUT mode override ---
if hhg_lut_mode == 'powerlaw':
    n_pl = hhg_lut_powerlaw_exp
    I_ref = I_lut[-1]
    dq_ref = dq_mag[-1]
    C_pl = dq_ref / I_ref**n_pl
    dq_mag_pl = C_pl * I_lut**n_pl
    dq_mag_interp_pl = interp1d(I_lut, dq_mag_pl, kind='cubic', bounds_error=False, fill_value=0.0)
    def dq_complex_interp(I_arr):
        mag = dq_mag_interp_pl(I_arr)
        phase = dq_phase_interp(I_arr)
        return mag * np.exp(1j * phase)
    print(f"  LUT magnitude replaced with power law: |d_q| = C * I^{n_pl}")
    print(f"  Anchored at I={I_ref:.2e}: |d_q|={dq_ref:.4e}")

elif hhg_lut_mode == 'experimental':
    # Experimental |d_q| from measured H21 yield: |d_q| ∝ sqrt(yield)
    exp_dq = np.sqrt(exp_yield_H21)
    # Normalize: anchor to Lewenstein at I_lut max (must be within LUT range)
    I_anchor = I_lut[-1]
    dq_lew_at_anchor = dq_mag_interp(I_anchor)
    # Interpolate experimental |d_q| at anchor point
    exp_dq_at_anchor = np.interp(I_anchor, exp_yield_I_Wcm2, exp_dq)
    scale_exp = dq_lew_at_anchor / exp_dq_at_anchor
    exp_dq_scaled = exp_dq * scale_exp
    # Extrapolation slope below experimental range
    if exp_extrap_mode == '2point':
        low_slope = np.log(exp_dq_scaled[1] / exp_dq_scaled[0]) / \
                    np.log(exp_yield_I_Wcm2[1] / exp_yield_I_Wcm2[0])
    elif exp_extrap_mode == 'global_fit':
        low_slope = np.polyfit(np.log(exp_yield_I_Wcm2), np.log(exp_dq_scaled), 1)[0]
    elif exp_extrap_mode == 'constant':
        low_slope = 0.0
    else:
        low_slope = float(exp_extrap_mode)  # manual value
    # Build full LUT: experimental in range, power-law extrapolation outside
    dq_mag_exp = np.zeros_like(I_lut)
    I_min_exp = exp_yield_I_Wcm2[0]
    I_max_exp = exp_yield_I_Wcm2[-1]
    for i, I_val in enumerate(I_lut):
        if I_val >= I_min_exp and I_val <= I_max_exp:
            dq_mag_exp[i] = np.interp(I_val, exp_yield_I_Wcm2, exp_dq_scaled)
        elif I_val < I_min_exp:
            dq_mag_exp[i] = exp_dq_scaled[0] * (I_val / I_min_exp)**low_slope
        else:
            dq_mag_exp[i] = exp_dq_scaled[-1]  # clamp above range
    dq_mag_interp_exp = interp1d(I_lut, dq_mag_exp, kind='cubic',
                                  bounds_error=False, fill_value=0.0)
    def dq_complex_interp(I_arr):
        mag = dq_mag_interp_exp(I_arr)
        phase = dq_phase_interp(I_arr)
        return mag * np.exp(1j * phase)
    print(f"  LUT magnitude replaced with experimental H21 data ({len(exp_yield_H21)} points)")
    print(f"  Experimental I range: {I_min_exp:.2e} to {I_max_exp:.2e} W/cm^2")
    print(f"  Low-I extrapolation slope: {low_slope:.2f} (mode: {exp_extrap_mode})")
    print(f"  Anchor: Lewenstein |d_q|={dq_lew_at_anchor:.4e} at I={I_anchor:.2e}")
elif hhg_lut_mode == 'deconvolved':
    # Hybrid mode: empirical magnitude + Lewenstein phase (no anchoring)
    # |d_q| = sqrt(I^alpha * exp(-I/Is)), phase from SFA
    dq_mag_deconv = np.sqrt(I_lut**deconv_alpha * np.exp(-I_lut / deconv_Is))
    dq_mag_deconv[I_lut < 1e13] = 0.0
    dq_mag_interp_deconv = interp1d(I_lut, dq_mag_deconv, kind='cubic',
                                      bounds_error=False, fill_value=0.0)
    def dq_complex_interp(I_arr):
        mag = dq_mag_interp_deconv(I_arr)
        phase = dq_phase_interp(I_arr)
        return mag * np.exp(1j * phase)
    print(f"  LUT: empirical magnitude + Lewenstein phase (no anchoring)")
    print(f"  alpha={deconv_alpha:.2f}, I_s={deconv_Is:.2e} W/cm^2")
    print(f"  Peak response at I = {deconv_alpha * deconv_Is:.2e} W/cm^2")
else:
    print(f"  Using original Lewenstein LUT")

# --- Scale stored 2D intensity to W/cm² ---
I_2d_gas_b_Wcm2 = I_2d_gas_b * I_scale_factor_2d
I_2d_gas_ub_Wcm2 = I_2d_gas_ub * I_scale_factor_2d

# --- Compute 2D ionization fraction ---
nf_3d_b = ionization_fraction(I_2d_gas_b_Wcm2, gas['Ip_eV'])
nf_3d_ub = ionization_fraction(I_2d_gas_ub_Wcm2, gas['Ip_eV'])

# --- Compute XUV absorption optical depth (z → end of gas jet) ---
sigma_xuv_m2 = hhg_sigma_xuv_Mb * 1e-22  # Mb → m²
P_bar_gas = hhg_gas_pressure / 1000.0
n_gas_density = gas['N_atm'] * P_bar_gas  # total gas density (m^-3)
# Absorption coefficient mu(x,y,z) = sigma × n_neutral = sigma × n_gas × (1-nf)
mu_3d_b = sigma_xuv_m2 * n_gas_density * (1.0 - nf_3d_b)
mu_3d_ub = sigma_xuv_m2 * n_gas_density * (1.0 - nf_3d_ub)
# Optical depth from z to L: tau(z) = integral_z^L mu(z') dz'
# = total_tau - forward_cumulative_tau
mu_cumfwd_b = np.zeros_like(mu_3d_b)
mu_cumfwd_b[1:] = cumulative_trapezoid(mu_3d_b, z_gas_2d_b_m, axis=0)
tau_b = mu_cumfwd_b[-1:] - mu_cumfwd_b
mu_cumfwd_ub = np.zeros_like(mu_3d_ub)
mu_cumfwd_ub[1:] = cumulative_trapezoid(mu_3d_ub, z_gas_2d_ub_m, axis=0)
tau_ub = mu_cumfwd_ub[-1:] - mu_cumfwd_ub
# Amplitude absorption factor
abs_factor_b = np.exp(-tau_b / 2.0)
abs_factor_ub = np.exp(-tau_ub / 2.0)
# Diagnostics
L_abs = 1.0 / (sigma_xuv_m2 * n_gas_density + 1e-30)
tau_total_onaxis_b = tau_b[0, tau_b.shape[1]//2, tau_b.shape[2]//2]
tau_total_onaxis_ub = tau_ub[0, tau_ub.shape[1]//2, tau_ub.shape[2]//2]
print(f"\n  --- XUV Absorption ---")
print(f"  sigma_XUV = {hhg_sigma_xuv_Mb:.1f} Mb, L_abs = {L_abs*1e3:.3f} mm (neutral gas)")
print(f"  On-axis optical depth (full gas): blocked={tau_total_onaxis_b:.2f}, unblocked={tau_total_onaxis_ub:.2f}")
print(f"  On-axis transmission: blocked={np.exp(-tau_total_onaxis_b):.4f}, unblocked={np.exp(-tau_total_onaxis_ub):.4f}")
del mu_3d_b, mu_3d_ub, mu_cumfwd_b, mu_cumfwd_ub

# --- Compute 2D phase mismatch terms ---
# Neutral dispersion
dk_neut_3d_b = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_3d_b, gas['delta_n'])
dk_neut_3d_ub = calc_dk_neutral(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_3d_ub, gas['delta_n'])

# Plasma dispersion
dk_plas_3d_b = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_3d_b, gas['N_atm'])
dk_plas_3d_ub = calc_dk_plasma(hhg_harmonic_order, hhg_gas_pressure, lambda_0_m, nf_3d_ub, gas['N_atm'])

# Geometric phase mismatch: -(q - 1/q) * d(phase_geom)/dz
# Use complex-domain phase difference method (robust against 2π wrapping for non-Gaussian beams)
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

# Mask phase gradient in very low intensity regions (noise-dominated)
I_thresh_b = 0.01 * I_2d_gas_b.max()
I_thresh_ub = 0.01 * I_2d_gas_ub.max()
dphase_dz_3d_b[I_2d_gas_b < I_thresh_b] = 0.0
dphase_dz_3d_ub[I_2d_gas_ub < I_thresh_ub] = 0.0

del field_env_b, field_env_ub

dk_geom_3d_b = -(hhg_harmonic_order - 1.0 / hhg_harmonic_order) * dphase_dz_3d_b
dk_geom_3d_ub = -(hhg_harmonic_order - 1.0 / hhg_harmonic_order) * dphase_dz_3d_ub

# Propagation phase mismatch (NO dk_dip — Lewenstein includes intrinsic dipole phase)
dk_prop_3d_b = dk_neut_3d_b + dk_plas_3d_b + dk_geom_3d_b
dk_prop_3d_ub = dk_neut_3d_ub + dk_plas_3d_ub + dk_geom_3d_ub

# --- Phase-matching diagnostic: on-axis at focus ---
_iz_foc_b = np.argmin(np.abs(z_gas_2d_b - focal_length))
_iz_foc_ub = np.argmin(np.abs(z_gas_2d_ub - focal_length))
_c2d = N_hhg_2d // 2

_dk_n_b  = dk_neut_3d_b[_iz_foc_b, _c2d, _c2d]
_dk_p_b  = dk_plas_3d_b[_iz_foc_b, _c2d, _c2d]
_dk_g_b  = dk_geom_3d_b[_iz_foc_b, _c2d, _c2d]
_dk_t_b  = dk_prop_3d_b[_iz_foc_b, _c2d, _c2d]
_nf_foc_b = nf_3d_b[_iz_foc_b, _c2d, _c2d]
_I_foc_b  = I_2d_gas_b_Wcm2[_iz_foc_b, _c2d, _c2d]

_dk_n_ub = dk_neut_3d_ub[_iz_foc_ub, _c2d, _c2d]
_dk_p_ub = dk_plas_3d_ub[_iz_foc_ub, _c2d, _c2d]
_dk_g_ub = dk_geom_3d_ub[_iz_foc_ub, _c2d, _c2d]
_dk_t_ub = dk_prop_3d_ub[_iz_foc_ub, _c2d, _c2d]
_nf_foc_ub = nf_3d_ub[_iz_foc_ub, _c2d, _c2d]
_I_foc_ub  = I_2d_gas_ub_Wcm2[_iz_foc_ub, _c2d, _c2d]

_Lcoh_b = np.pi / (np.abs(_dk_t_b) + 1e-6)
_Lcoh_ub = np.pi / (np.abs(_dk_t_ub) + 1e-6)

print(f"\n  === Phase-Matching Diagnostic (on-axis, at focus) ===")
print(f"  {'':30s} {'BLOCKED':>14s} {'UNBLOCKED':>14s}")
print(f"  {'z_focus (mm)':30s} {z_gas_2d_b[_iz_foc_b]:14.3f} {z_gas_2d_ub[_iz_foc_ub]:14.3f}")
print(f"  {'I_peak (W/cm2)':30s} {_I_foc_b:14.3e} {_I_foc_ub:14.3e}")
print(f"  {'Ionization fraction nf':30s} {_nf_foc_b:14.4f} {_nf_foc_ub:14.4f}")
print(f"  {'dk_neutral (1/m)':30s} {_dk_n_b:14.2f} {_dk_n_ub:14.2f}")
print(f"  {'dk_plasma  (1/m)':30s} {_dk_p_b:14.2f} {_dk_p_ub:14.2f}")
print(f"  {'dk_geom    (1/m)':30s} {_dk_g_b:14.2f} {_dk_g_ub:14.2f}")
print(f"  {'dk_total   (1/m)':30s} {_dk_t_b:14.2f} {_dk_t_ub:14.2f}")
print(f"  {'L_coh (mm)':30s} {_Lcoh_b*1e3:14.4f} {_Lcoh_ub*1e3:14.4f}")
print(f"  {'|dk_neut/dk_total|':30s} {np.abs(_dk_n_b)/max(np.abs(_dk_t_b),1e-6):14.4f} {np.abs(_dk_n_ub)/max(np.abs(_dk_t_ub),1e-6):14.4f}")
print(f"  {'|dk_plas/dk_total|':30s} {np.abs(_dk_p_b)/max(np.abs(_dk_t_b),1e-6):14.4f} {np.abs(_dk_p_ub)/max(np.abs(_dk_t_ub),1e-6):14.4f}")
print(f"  {'|dk_geom/dk_total|':30s} {np.abs(_dk_g_b)/max(np.abs(_dk_t_b),1e-6):14.4f} {np.abs(_dk_g_ub)/max(np.abs(_dk_t_ub),1e-6):14.4f}")

# --- Cumulative phase along z ---
Phi_3d_b = np.zeros_like(dk_prop_3d_b)
Phi_3d_ub = np.zeros_like(dk_prop_3d_ub)
Phi_3d_b[1:] = cumulative_trapezoid(dk_prop_3d_b, z_gas_2d_b_m, axis=0)
Phi_3d_ub[1:] = cumulative_trapezoid(dk_prop_3d_ub, z_gas_2d_ub_m, axis=0)

# --- Interpolate d_q from LUT onto 3D (z, x, y) grid ---
dq_3d_b = dq_complex_interp(np.clip(I_2d_gas_b_Wcm2, I_lut_min, I_lut_max))
dq_3d_b[I_2d_gas_b_Wcm2 < I_lut_min] = 0.0
dq_3d_ub = dq_complex_interp(np.clip(I_2d_gas_ub_Wcm2, I_lut_min, I_lut_max))
dq_3d_ub[I_2d_gas_ub_Wcm2 < I_lut_min] = 0.0

print(f"  {'Peak |d_q| at focus':30s} {np.abs(dq_3d_b[_iz_foc_b, _c2d, _c2d]):14.4e} {np.abs(dq_3d_ub[_iz_foc_ub, _c2d, _c2d]):14.4e}")
print(f"  {'Peak |d_q| global':30s} {np.abs(dq_3d_b).max():14.4e} {np.abs(dq_3d_ub).max():14.4e}")
if hhg_lut_mode == 'experimental':
    I_min_exp_check = exp_yield_I_Wcm2[0]
    frac_below_b = (I_2d_gas_b_Wcm2 < I_min_exp_check).sum() / I_2d_gas_b_Wcm2.size
    frac_below_ub = (I_2d_gas_ub_Wcm2 < I_min_exp_check).sum() / I_2d_gas_ub_Wcm2.size
    dq2_below_b = (np.abs(dq_3d_b[I_2d_gas_b_Wcm2 < I_min_exp_check])**2).sum()
    dq2_total_b = (np.abs(dq_3d_b)**2).sum()
    dq2_below_ub = (np.abs(dq_3d_ub[I_2d_gas_ub_Wcm2 < I_min_exp_check])**2).sum()
    dq2_total_ub = (np.abs(dq_3d_ub)**2).sum()
    print(f"  --- Experimental LUT diagnostics ---")
    print(f"  Voxels below exp. range ({I_min_exp_check:.1e}):  blocked={frac_below_b:.1%}, unblocked={frac_below_ub:.1%}")
    print(f"  |d_q|^2 from extrapolated region:  blocked={dq2_below_b/dq2_total_b:.1%}, unblocked={dq2_below_ub/dq2_total_ub:.1%}")

# --- Macroscopic HHG integration (total-field single integration) ---
# Uses total-field intensity for d_q, total-field geometric phase, single z-integration.
# Consistent with multi-mask HHG computation approach.
# All quantities (d_q, Phi, nf, abs_factor) already computed from total field above.

dx_hhg_m = (x_hhg_2d[1] - x_hhg_2d[0]) * 1e-3  # mm → m
lambda_q_m = lambda_0_m / hhg_harmonic_order      # H21 wavelength

print(f"\n  Computing macroscopic HHG (total-field integration)...")

# Total-field integrand
integrand_3d_b = dq_3d_b * (1.0 - nf_3d_b) * np.exp(1j * Phi_3d_b) * abs_factor_b
integrand_3d_ub = dq_3d_ub * (1.0 - nf_3d_ub) * np.exp(1j * Phi_3d_ub) * abs_factor_ub

# Near-field: macroscopic integral along z
E_q_2d_b = np.trapz(integrand_3d_b, z_gas_2d_b_m, axis=0)
E_q_2d_ub = np.trapz(integrand_3d_ub, z_gas_2d_ub_m, axis=0)

# Far-field via FFT
E_ff_b = np.fft.fftshift(np.fft.fft2(E_q_2d_b)) * dx_hhg_m**2
E_ff_ub = np.fft.fftshift(np.fft.fft2(E_q_2d_ub)) * dx_hhg_m**2

# Angular grid
dtheta = lambda_q_m / (N_hhg_2d * dx_hhg_m)  # rad/pixel
theta_axis = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dtheta
theta_mrad = theta_axis * 1e3

# Detection geometry parameters
# Circular aperture
hhg_aperture_radius_mm = 1.0   # mm
hhg_aperture_distance = 0.3    # m from gas jet

# Slit parameters (spectrometer entrance slit: y open, x narrow)
hhg_slit_width_mm = 2.0        # mm, slit width (x direction)
hhg_slit_height_mm = 10.0      # mm, slit height (y direction)
hhg_slit_distance = 1.2        # m, distance from gas jet to slit

# Build BOTH far-field acceptance masks
r_theta = np.sqrt(theta_axis[:, None]**2 + theta_axis[None, :]**2)

# Circular mask
circular_half_angle = hhg_aperture_radius_mm * 1e-3 / hhg_aperture_distance
mask_ff_circular = (r_theta <= circular_half_angle).astype(float)

# Slit mask
slit_half_angle_x = (hhg_slit_width_mm / 2) * 1e-3 / hhg_slit_distance
slit_half_angle_y = (hhg_slit_height_mm / 2) * 1e-3 / hhg_slit_distance
mask_ff_slit = ((np.abs(theta_axis[None, :]) <= slit_half_angle_x) &
                (np.abs(theta_axis[:, None]) <= slit_half_angle_y)).astype(float)

# For backward compatibility with plotting code
hhg_acceptance_type = 'slit'
aperture_mask_ff = mask_ff_slit
aperture_half_angle = slit_half_angle_x

# Far-field intensities
I_ff_b = np.abs(E_ff_b)**2
I_ff_ub = np.abs(E_ff_ub)**2

# On-axis brightness
c_ff = N_hhg_2d // 2
onaxis_b = I_ff_b[c_ff, c_ff]
onaxis_ub = I_ff_ub[c_ff, c_ff]
onaxis_ratio = onaxis_b / onaxis_ub if onaxis_ub > 0 else np.nan

# --- Circular aperture yield ---
yield_circ_b = np.sum(I_ff_b * mask_ff_circular) * dtheta**2
yield_circ_ub = np.sum(I_ff_ub * mask_ff_circular) * dtheta**2
ratio_circ = yield_circ_b / yield_circ_ub if yield_circ_ub > 0 else np.nan

# --- Slit yield ---
yield_slit_b = np.sum(I_ff_b * mask_ff_slit) * dtheta**2
yield_slit_ub = np.sum(I_ff_ub * mask_ff_slit) * dtheta**2
ratio_slit = yield_slit_b / yield_slit_ub if yield_slit_ub > 0 else np.nan

# Primary yield (slit, for backward compatibility)
yield_ap_ub = yield_slit_ub
ap_ratio = ratio_slit

# Total far-field yield (Parseval check)
yield_ff_total_b = np.sum(I_ff_b) * dtheta**2
yield_ff_total_ub = np.sum(I_ff_ub) * dtheta**2

print(f"\n  --- Far-Field HHG (Dual Geometry) ---")
print(f"  Circular: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m (half-angle = {circular_half_angle*1e3:.2f} mrad)")
print(f"    Yield ratio (b/ub): {ratio_circ:.4f}")
print(f"  Slit: {hhg_slit_width_mm}x{hhg_slit_height_mm} mm at {hhg_slit_distance} m (x: ±{slit_half_angle_x*1e3:.2f}, y: ±{slit_half_angle_y*1e3:.2f} mrad)")
print(f"    Yield ratio (b/ub): {ratio_slit:.4f}")
print(f"  On-axis ratio (b/ub):   {onaxis_ratio:.4f}")
print(f"  Total far-field ratio:  {yield_ff_total_b/yield_ff_total_ub:.4f}")

# Near-field 2D HHG intensity (from total-field integration)
I_q_2d_b = np.abs(E_q_2d_b)**2
I_q_2d_ub = np.abs(E_q_2d_ub)**2

# Total yield: integral of incoherent near-field sum
yield_b = np.sum(I_q_2d_b) * dx_hhg_m**2
yield_ub = np.sum(I_q_2d_ub) * dx_hhg_m**2
yield_ratio = yield_b / yield_ub if yield_ub > 0 else np.nan

# On-axis buildup vs z
center_2d = N_hhg_2d // 2
dz_gas_b = np.diff(z_gas_2d_b_m)
dz_gas_ub = np.diff(z_gas_2d_ub_m)
buildup_b = np.zeros(n_z_gas_b, dtype=np.complex128)
buildup_ub = np.zeros(n_z_gas_ub, dtype=np.complex128)
for j in range(1, n_z_gas_b):
    buildup_b[j] = buildup_b[j-1] + 0.5 * (integrand_3d_b[j-1, center_2d, center_2d] + integrand_3d_b[j, center_2d, center_2d]) * dz_gas_b[j-1]
for j in range(1, n_z_gas_ub):
    buildup_ub[j] = buildup_ub[j-1] + 0.5 * (integrand_3d_ub[j-1, center_2d, center_2d] + integrand_3d_ub[j, center_2d, center_2d]) * dz_gas_ub[j-1]

# Full xy-integrated yield buildup vs z (near-field + far-field)
buildup_2d_b = np.zeros((n_z_gas_b, N_hhg_2d, N_hhg_2d), dtype=np.complex128)
for j in range(1, n_z_gas_b):
    buildup_2d_b[j] = buildup_2d_b[j-1] + 0.5 * (integrand_3d_b[j-1] + integrand_3d_b[j]) * dz_gas_b[j-1]
yield_vs_z_b = np.zeros(n_z_gas_b)
yield_vs_z_slit_b = np.zeros(n_z_gas_b)
yield_vs_z_circ_b = np.zeros(n_z_gas_b)
for j in range(n_z_gas_b):
    yield_vs_z_b[j] = np.sum(np.abs(buildup_2d_b[j])**2) * dx_hhg_m**2
    if USE_SCIPY_FFT:
        E_ff_j = scipy_fft.fftshift(scipy_fft.fft2(buildup_2d_b[j], workers=-1)) * dx_hhg_m**2
    else:
        E_ff_j = np.fft.fftshift(np.fft.fft2(buildup_2d_b[j])) * dx_hhg_m**2
    I_ff_j = np.abs(E_ff_j)**2
    yield_vs_z_slit_b[j] = np.sum(I_ff_j * mask_ff_slit) * dtheta**2
    yield_vs_z_circ_b[j] = np.sum(I_ff_j * mask_ff_circular) * dtheta**2
del buildup_2d_b

buildup_2d_ub = np.zeros((n_z_gas_ub, N_hhg_2d, N_hhg_2d), dtype=np.complex128)
for j in range(1, n_z_gas_ub):
    buildup_2d_ub[j] = buildup_2d_ub[j-1] + 0.5 * (integrand_3d_ub[j-1] + integrand_3d_ub[j]) * dz_gas_ub[j-1]
yield_vs_z_ub = np.zeros(n_z_gas_ub)
yield_vs_z_slit_ub = np.zeros(n_z_gas_ub)
yield_vs_z_circ_ub = np.zeros(n_z_gas_ub)
for j in range(n_z_gas_ub):
    yield_vs_z_ub[j] = np.sum(np.abs(buildup_2d_ub[j])**2) * dx_hhg_m**2
    if USE_SCIPY_FFT:
        E_ff_j = scipy_fft.fftshift(scipy_fft.fft2(buildup_2d_ub[j], workers=-1)) * dx_hhg_m**2
    else:
        E_ff_j = np.fft.fftshift(np.fft.fft2(buildup_2d_ub[j])) * dx_hhg_m**2
    I_ff_j = np.abs(E_ff_j)**2
    yield_vs_z_slit_ub[j] = np.sum(I_ff_j * mask_ff_slit) * dtheta**2
    yield_vs_z_circ_ub[j] = np.sum(I_ff_j * mask_ff_circular) * dtheta**2
del buildup_2d_ub
gc.collect()

# x and y lineouts through center
I_q_x_b = I_q_2d_b[center_2d, :]    # y=0 slice
I_q_y_b = I_q_2d_b[:, center_2d]    # x=0 slice
I_q_x_ub = I_q_2d_ub[center_2d, :]
I_q_y_ub = I_q_2d_ub[:, center_2d]

print(f"\n  --- Macroscopic HHG Yield Results (Full 2D) ---")
print(f"  Total yield (blocked):   {yield_b:.4e}")
print(f"  Total yield (unblocked): {yield_ub:.4e}")
print(f"  Yield ratio (blocked/unblocked): {yield_ratio:.4f}")

# --- Incoherent yield: no phase matching (with absorption) ---
incoh_yield_b = np.trapz(
    np.sum(np.abs(dq_3d_b)**2 * (1.0 - nf_3d_b)**2 * abs_factor_b**2, axis=(1, 2)) * dx_hhg_m**2,
    z_gas_2d_b_m)
incoh_yield_ub = np.trapz(
    np.sum(np.abs(dq_3d_ub)**2 * (1.0 - nf_3d_ub)**2 * abs_factor_ub**2, axis=(1, 2)) * dx_hhg_m**2,
    z_gas_2d_ub_m)
incoh_ratio = incoh_yield_b / incoh_yield_ub if incoh_yield_ub > 0 else np.nan
print(f"\n  --- Incoherent HHG Yield (no phase matching) ---")
print(f"  Incoherent yield (blocked):   {incoh_yield_b:.4e}")
print(f"  Incoherent yield (unblocked): {incoh_yield_ub:.4e}")
print(f"  Incoherent ratio (b/ub):      {incoh_ratio:.4f}")

# Free integrand arrays (not needed for diagnostics)
del integrand_3d_b, integrand_3d_ub
# NOTE: dq, dk_prop, nf arrays kept for Figure HHG-5 diagnostics — deleted after that figure

# =============================================================================
# Figure HHG-4: Macroscopic HHG Yield (Full 2D)
# =============================================================================
x_hhg_um = x_hhg_2d * 1e3  # mm → μm
hhg_extent = [x_hhg_um[0], x_hhg_um[-1], x_hhg_um[0], x_hhg_um[-1]]

fig_hhg4, axes_hhg4 = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) 2D HHG far-field |E_q(x,y)|^2 — blocked (SELF-NORMALIZED)
ax = axes_hhg4[0, 0]
vmax_shared = max(I_q_2d_b.max(), I_q_2d_ub.max())
vmax_b = I_q_2d_b.max()
if vmax_b > 0:
    im = ax.imshow(I_q_2d_b.T / vmax_b, extent=hhg_extent, aspect='equal', origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Self-norm.')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title(f'Blocked HHG (H{hhg_harmonic_order}) [self-norm, peak={vmax_b/max(vmax_shared,1e-30):.1e} rel]')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])

# (0,1) 2D HHG far-field — unblocked
ax = axes_hhg4[0, 1]
if vmax_shared > 0:
    im = ax.imshow(I_q_2d_ub.T / vmax_shared, extent=hhg_extent, aspect='equal', origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalized')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title(f'Unblocked HHG (H{hhg_harmonic_order})')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])

# (0,2) Bar chart: total yield comparison
ax = axes_hhg4[0, 2]
bars = ax.bar(['Blocked', 'Unblocked'], [yield_b, yield_ub], color=['steelblue', 'salmon'], edgecolor='black')
ax.set_ylabel('Total HHG Yield (arb. units)')
ax.set_title(f'Integrated Yield (ratio = {yield_ratio:.3f})')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, [yield_b, yield_ub]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2e}',
            ha='center', va='bottom', fontsize=9)

# (1,0) x and y lineouts of HHG profile
ax = axes_hhg4[1, 0]
lineout_max = max(I_q_x_b.max(), I_q_y_b.max(), I_q_x_ub.max(), I_q_y_ub.max(), 1e-30)
ax.plot(x_hhg_um, I_q_x_b / lineout_max, 'b-', label='Blocked (x)', linewidth=2)
ax.plot(x_hhg_um, I_q_y_b / lineout_max, 'b--', label='Blocked (y)', linewidth=1.5)
ax.plot(x_hhg_um, I_q_x_ub / lineout_max, 'r-', label='Unblocked (x)', linewidth=2)
ax.plot(x_hhg_um, I_q_y_ub / lineout_max, 'r--', label='Unblocked (y)', linewidth=1.5)
ax.set_xlabel('Position (μm)')
ax.set_ylabel(r'$|E_q|^2$ (normalized)')
ax.set_title('HHG Lineouts (x and y)')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 50])

# (1,1) On-axis yield buildup vs z
ax = axes_hhg4[1, 1]
ax.plot(z_gas_2d_b, np.abs(buildup_b)**2 / max(np.abs(buildup_b[-1])**2, 1e-30), 'b-', label='Blocked', linewidth=2)
ax.plot(z_gas_2d_ub, np.abs(buildup_ub)**2 / max(np.abs(buildup_ub[-1])**2, 1e-30), 'r--', label='Unblocked', linewidth=2)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'On-axis $|E_q|^2$ (normalized)')
ax.set_title('On-axis HHG Buildup')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (1,2) Total yield buildup vs z (NF + slit + circ)
ax = axes_hhg4[1, 2]
yvz_norm_nf = max(yield_vs_z_ub[-1], 1e-30)
yvz_norm_slit = max(yield_vs_z_slit_ub[-1], 1e-30)
yvz_norm_circ = max(yield_vs_z_circ_ub[-1], 1e-30)
ax.plot(z_gas_2d_b, yield_vs_z_slit_b / yvz_norm_slit, 'b-', label='Blocked (slit)', linewidth=2)
ax.plot(z_gas_2d_ub, yield_vs_z_slit_ub / yvz_norm_slit, 'b--', label='Unblocked (slit)', linewidth=1.5, alpha=0.6)
ax.plot(z_gas_2d_b, yield_vs_z_circ_b / yvz_norm_circ, 'r-', label='Blocked (circ)', linewidth=2)
ax.plot(z_gas_2d_ub, yield_vs_z_circ_ub / yvz_norm_circ, 'r--', label='Unblocked (circ)', linewidth=1.5, alpha=0.6)
ax.plot(z_gas_2d_b, yield_vs_z_b / yvz_norm_nf, 'g-', label='Blocked (NF)', linewidth=2)
ax.plot(z_gas_2d_ub, yield_vs_z_ub / yvz_norm_nf, 'g--', label='Unblocked (NF)', linewidth=1.5, alpha=0.6)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Integrated yield (normalized)')
ax.set_title('Total HHG Yield Buildup')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig_hhg4.suptitle(f'Macroscopic HHG Yield (2D) - {hhg_gas_type.capitalize()}, P={hhg_gas_pressure} mbar, '
                   f'H{hhg_harmonic_order}, Gas={hhg_gas_length} mm, Grid={N_hhg_2d}x{N_hhg_2d}', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_yield_macroscopic_{_m2_tag}.png', dpi=300)
print(f"Macroscopic HHG yield figure saved to 'hhg_yield_macroscopic.png'")

# =============================================================================
# Figure HHG-7: Far-Field HHG with Aperture (2x2)
# =============================================================================
print("\nGenerating far-field HHG figure (HHG-7)...")

fig_ff, axes_ff = plt.subplots(2, 2, figsize=(12, 10))
theta_extent = [theta_mrad[0], theta_mrad[-1], theta_mrad[0], theta_mrad[-1]]
aperture_half_mrad = aperture_half_angle * 1e3

# (0,0) Far-field blocked (self-normalized, aperture circle)
ax = axes_ff[0, 0]
vmax_ff_b = I_ff_b.max()
vmax_ff_shared = max(I_ff_b.max(), I_ff_ub.max())
if vmax_ff_b > 0:
    im = ax.imshow(I_ff_b.T / vmax_ff_b, extent=theta_extent, aspect='equal',
                   origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Self-norm.')
circ = plt.Circle((0, 0), aperture_half_mrad, fill=False, color='white',
                   linestyle='--', linewidth=1.5)
ax.add_patch(circ)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$\theta_y$ (mrad)')
ax.set_title(f'Blocked Far-Field [self-norm, peak={vmax_ff_b/max(vmax_ff_shared,1e-30):.1e} rel]')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

# (0,1) Far-field unblocked (shared normalization, aperture circle)
ax = axes_ff[0, 1]
if vmax_ff_shared > 0:
    im = ax.imshow(I_ff_ub.T / vmax_ff_shared, extent=theta_extent, aspect='equal',
                   origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Shared norm.')
circ = plt.Circle((0, 0), aperture_half_mrad, fill=False, color='white',
                   linestyle='--', linewidth=1.5)
ax.add_patch(circ)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$\theta_y$ (mrad)')
ax.set_title(f'Unblocked Far-Field')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

# (1,0) Angular lineouts θx (both beams, log scale, aperture limits)
ax = axes_ff[1, 0]
lineout_ff_b = I_ff_b[c_ff, :]     # y=0 slice
lineout_ff_ub = I_ff_ub[c_ff, :]
ax.semilogy(theta_mrad, lineout_ff_b / max(lineout_ff_b.max(), 1e-30), 'b-',
            label='Blocked', linewidth=2)
ax.semilogy(theta_mrad, lineout_ff_ub / max(lineout_ff_ub.max(), 1e-30), 'r--',
            label='Unblocked', linewidth=2)
ax.axvline(aperture_half_mrad, color='green', linestyle='--', linewidth=1, label=f'Aperture ({aperture_half_mrad:.1f} mrad)')
ax.axvline(-aperture_half_mrad, color='green', linestyle='--', linewidth=1)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'Far-field $|E|^2$ (self-norm, log)')
ax.set_title('Angular Lineout')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (1,1) Cumulative yield vs acceptance half-angle
ax = axes_ff[1, 1]
n_angles = 200
max_theta_cum = min(theta_mrad.max(), 15.0)  # up to 15 mrad
angles_cum_mrad = np.linspace(0, max_theta_cum, n_angles)
cum_yield_b = np.zeros(n_angles)
cum_yield_ub = np.zeros(n_angles)
for ia, ang_mrad in enumerate(angles_cum_mrad):
    ang_rad = ang_mrad * 1e-3
    mask_cum = (r_theta <= ang_rad).astype(float)
    cum_yield_b[ia] = np.sum(I_ff_b * mask_cum) * dtheta**2
    cum_yield_ub[ia] = np.sum(I_ff_ub * mask_cum) * dtheta**2

ax.plot(angles_cum_mrad, cum_yield_b, 'b-', label='Blocked', linewidth=2)
ax.plot(angles_cum_mrad, cum_yield_ub, 'r--', label='Unblocked', linewidth=2)
ax.axvline(aperture_half_mrad, color='green', linestyle='--', linewidth=1,
           label=f'Aperture ({aperture_half_mrad:.1f} mrad)')
ax.set_xlabel('Acceptance half-angle (mrad)')
ax.set_ylabel('Cumulative yield (arb. units)')
ax.set_title('Cumulative Yield vs Acceptance Angle')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Mark the ratio at the aperture angle
if yield_ap_ub > 0:
    ax.text(aperture_half_mrad + 0.3, (cum_yield_b[np.searchsorted(angles_cum_mrad, aperture_half_mrad)] +
            cum_yield_ub[np.searchsorted(angles_cum_mrad, aperture_half_mrad)]) / 2,
            f'ratio={ap_ratio:.3f}', fontsize=9, color='green')

fig_ff.suptitle(f'Far-Field HHG (H{hhg_harmonic_order}) — Aperture: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m',
                fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_farfield_{_m2_tag}.png', dpi=300)
print(f"Far-field HHG figure saved to 'hhg_farfield.png'")

# Clean up far-field arrays not needed later
del r_theta
# NOTE: aperture_mask_ff kept for power-law exponent scan

# =============================================================================
# Figure HHG-5: Phase-Matching Diagnostics (2x3)
# =============================================================================
print("\nGenerating phase-matching diagnostic figure (HHG-5)...")

iz_foc_b = np.argmin(np.abs(z_gas_2d_b - focal_length))
iz_foc_ub = np.argmin(np.abs(z_gas_2d_ub - focal_length))

# On-axis dk terms vs z
dk_neut_onax_b = dk_neut_3d_b[:, center_2d, center_2d]
dk_plas_onax_b = dk_plas_3d_b[:, center_2d, center_2d]
dk_geom_onax_b = dk_geom_3d_b[:, center_2d, center_2d]
dk_tot_onax_b  = dk_prop_3d_b[:, center_2d, center_2d]

dk_neut_onax_ub = dk_neut_3d_ub[:, center_2d, center_2d]
dk_plas_onax_ub = dk_plas_3d_ub[:, center_2d, center_2d]
dk_geom_onax_ub = dk_geom_3d_ub[:, center_2d, center_2d]
dk_tot_onax_ub  = dk_prop_3d_ub[:, center_2d, center_2d]

# On-axis cumulative phase
Phi_onax_b  = Phi_3d_b[:, center_2d, center_2d]
Phi_onax_ub = Phi_3d_ub[:, center_2d, center_2d]

# 2D dk_total at focus z-plane
dk_tot_focus_b  = dk_prop_3d_b[iz_foc_b, :, :]
dk_tot_focus_ub = dk_prop_3d_ub[iz_foc_ub, :, :]

# x-lineout of coherence length at focus (y = center)
dk_xline_b  = dk_prop_3d_b[iz_foc_b, center_2d, :]
dk_xline_ub = dk_prop_3d_ub[iz_foc_ub, center_2d, :]
Lcoh_xline_b  = np.pi / (np.abs(dk_xline_b) + 1e-6)
Lcoh_xline_ub = np.pi / (np.abs(dk_xline_ub) + 1e-6)

fig_hhg5, axes_hhg5 = plt.subplots(2, 3, figsize=(18, 10))

# ---- (0,0) On-axis dk terms vs z — BLOCKED ----
ax = axes_hhg5[0, 0]
ax.plot(z_gas_2d_b, dk_neut_onax_b, 'g-',  label=r'$\Delta k_{\mathrm{neut}}$', linewidth=1.5)
ax.plot(z_gas_2d_b, dk_plas_onax_b, 'r-',  label=r'$\Delta k_{\mathrm{plas}}$', linewidth=1.5)
ax.plot(z_gas_2d_b, dk_geom_onax_b, 'b-',  label=r'$\Delta k_{\mathrm{geom}}$', linewidth=1.5)
ax.plot(z_gas_2d_b, dk_tot_onax_b,  'k-',  label=r'$\Delta k_{\mathrm{total}}$', linewidth=2)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
ax.axvline(focal_length, color='gray', linestyle='--', linewidth=0.5, label='Focus')
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Delta k$ (1/m)')
ax.set_title('Blocked: On-axis dk vs z')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# ---- (0,1) On-axis dk terms vs z — UNBLOCKED ----
ax = axes_hhg5[0, 1]
ax.plot(z_gas_2d_ub, dk_neut_onax_ub, 'g-',  label=r'$\Delta k_{\mathrm{neut}}$', linewidth=1.5)
ax.plot(z_gas_2d_ub, dk_plas_onax_ub, 'r-',  label=r'$\Delta k_{\mathrm{plas}}$', linewidth=1.5)
ax.plot(z_gas_2d_ub, dk_geom_onax_ub, 'b-',  label=r'$\Delta k_{\mathrm{geom}}$', linewidth=1.5)
ax.plot(z_gas_2d_ub, dk_tot_onax_ub,  'k-',  label=r'$\Delta k_{\mathrm{total}}$', linewidth=2)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
ax.axvline(focal_length, color='gray', linestyle='--', linewidth=0.5, label='Focus')
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Delta k$ (1/m)')
ax.set_title('Unblocked: On-axis dk vs z')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# ---- (0,2) On-axis cumulative phase Phi vs z — BOTH ----
ax = axes_hhg5[0, 2]
ax.plot(z_gas_2d_b,  Phi_onax_b,  'b-', label='Blocked', linewidth=2)
ax.plot(z_gas_2d_ub, Phi_onax_ub, 'r--', label='Unblocked', linewidth=2)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
ax.axhline(np.pi, color='orange', linestyle=':', linewidth=0.8, label=r'$\pm\pi$ (decoherence)')
ax.axhline(-np.pi, color='orange', linestyle=':', linewidth=0.8)
ax.axvline(focal_length, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$\Phi$ (rad)')
ax.set_title(r'On-axis cumulative phase $\Phi(z)$')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# ---- (1,0) 2D map dk_total at focus — BLOCKED ----
ax = axes_hhg5[1, 0]
_ext = [x_hhg_um[0], x_hhg_um[-1], x_hhg_um[0], x_hhg_um[-1]]
_vlim_b = max(np.abs(dk_tot_focus_b).max(), 1e-6)
im = ax.imshow(dk_tot_focus_b.T, extent=_ext, aspect='equal', origin='lower',
               cmap='RdBu_r', vmin=-_vlim_b, vmax=_vlim_b)
plt.colorbar(im, ax=ax, label=r'$\Delta k$ (1/m)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title(f'Blocked: dk_total at z={z_gas_2d_b[iz_foc_b]:.2f} mm')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])

# ---- (1,1) 2D map dk_total at focus — UNBLOCKED ----
ax = axes_hhg5[1, 1]
_vlim_ub = max(np.abs(dk_tot_focus_ub).max(), 1e-6)
im = ax.imshow(dk_tot_focus_ub.T, extent=_ext, aspect='equal', origin='lower',
               cmap='RdBu_r', vmin=-_vlim_ub, vmax=_vlim_ub)
plt.colorbar(im, ax=ax, label=r'$\Delta k$ (1/m)')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title(f'Unblocked: dk_total at z={z_gas_2d_ub[iz_foc_ub]:.2f} mm')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])

# ---- (1,2) x-lineout of L_coh at focus — BOTH ----
ax = axes_hhg5[1, 2]
ax.plot(x_hhg_um, Lcoh_xline_b * 1e3,  'b-', label='Blocked', linewidth=2)
ax.plot(x_hhg_um, Lcoh_xline_ub * 1e3, 'r--', label='Unblocked', linewidth=2)
ax.axhline(hhg_gas_length, color='gray', linestyle=':', linewidth=0.8, label=f'Gas length ({hhg_gas_length} mm)')
ax.set_xlabel('x (μm)')
ax.set_ylabel(r'$L_{\mathrm{coh}}$ (mm)')
ax.set_title('Coherence length at focus (y=0)')
ax.set_xlim([-50, 50])
ax.set_yscale('log')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

fig_hhg5.suptitle(f'Phase-Matching Diagnostics - {hhg_gas_type.capitalize()}, P={hhg_gas_pressure} mbar, '
                   f'H{hhg_harmonic_order}, Gas={hhg_gas_length} mm', fontsize=13)
plt.tight_layout()
plt.savefig(f'hhg_phase_matching_diagnostics_{_m2_tag}.png', dpi=300)
print(f"Phase-matching diagnostics figure saved to 'hhg_phase_matching_diagnostics.png'")

if False:  # DISABLED — Figure HHG-6: Pressure Scan
    # =============================================================================
    # Figure HHG-6: Pressure Scan
    # =============================================================================
    print("\nComputing pressure scan (HHG-6)...")

    # Precompute split cumulative phases for efficient pressure scan
    # dk_neut + dk_plas scale linearly with P; dk_geom is P-independent
    # Phi(P) = (P/P0)*Phi_np + Phi_geom
    Phi_np_b = np.zeros_like(dk_neut_3d_b)
    Phi_np_b[1:] = cumulative_trapezoid(dk_neut_3d_b + dk_plas_3d_b, z_gas_2d_b_m, axis=0)
    Phi_geom_b = np.zeros_like(dk_geom_3d_b)
    Phi_geom_b[1:] = cumulative_trapezoid(dk_geom_3d_b, z_gas_2d_b_m, axis=0)
    base_integ_b = dq_3d_b * (1.0 - nf_3d_b) * np.exp(1j * Phi_geom_b)

    Phi_np_ub = np.zeros_like(dk_neut_3d_ub)
    Phi_np_ub[1:] = cumulative_trapezoid(dk_neut_3d_ub + dk_plas_3d_ub, z_gas_2d_ub_m, axis=0)
    Phi_geom_ub = np.zeros_like(dk_geom_3d_ub)
    Phi_geom_ub[1:] = cumulative_trapezoid(dk_geom_3d_ub, z_gas_2d_ub_m, axis=0)
    base_integ_ub = dq_3d_ub * (1.0 - nf_3d_ub) * np.exp(1j * Phi_geom_ub)

    del Phi_geom_b, Phi_geom_ub

    # Precompute reference absorption optical depth for pressure scaling
    # tau scales linearly with P: tau(P) = (P/P0) * tau_ref
    tau_ref_b = tau_b.copy()   # tau at reference pressure P0
    tau_ref_ub = tau_ub.copy()

    # Free dk/dq arrays (no longer needed after precomputation)
    # NOTE: nf_3d_b/ub kept for power-law exponent scan
    del dq_3d_b, dq_3d_ub, dk_prop_3d_b, dk_prop_3d_ub
    del dk_neut_3d_b, dk_neut_3d_ub, dk_plas_3d_b, dk_plas_3d_ub, dk_geom_3d_b, dk_geom_3d_ub
    gc.collect()

    # Pressure scan loop
    P0 = hhg_gas_pressure
    pressures_scan = np.linspace(10, 500, 30)
    yields_scan_b = np.zeros(len(pressures_scan))
    yields_scan_ub = np.zeros(len(pressures_scan))

    for ip, P in enumerate(pressures_scan):
        scale = P / P0
        # Blocked: yield(P) = (P/P0)^2 × |∫ base × exp(i·scale·Phi_np) × exp(-scale·tau/2) dz|^2
        abs_p_b = np.exp(-scale * tau_ref_b / 2.0)
        integ = base_integ_b * np.exp(1j * scale * Phi_np_b) * abs_p_b
        E_q = np.trapz(integ, z_gas_2d_b_m, axis=0)
        yields_scan_b[ip] = scale**2 * np.sum(np.abs(E_q)**2) * dx_hhg_m**2
        # Unblocked
        abs_p_ub = np.exp(-scale * tau_ref_ub / 2.0)
        integ = base_integ_ub * np.exp(1j * scale * Phi_np_ub) * abs_p_ub
        E_q = np.trapz(integ, z_gas_2d_ub_m, axis=0)
        yields_scan_ub[ip] = scale**2 * np.sum(np.abs(E_q)**2) * dx_hhg_m**2
        if (ip + 1) % 10 == 0:
            print(f"  Pressure scan: {ip+1}/{len(pressures_scan)}")

    del base_integ_b, base_integ_ub, Phi_np_b, Phi_np_ub
    gc.collect()

    # Find optimal pressures
    P_opt_b = pressures_scan[np.argmax(yields_scan_b)]
    P_opt_ub = pressures_scan[np.argmax(yields_scan_ub)]
    print(f"  Optimal pressure (blocked):   {P_opt_b:.0f} mbar (yield={yields_scan_b.max():.4e})")
    print(f"  Optimal pressure (unblocked): {P_opt_ub:.0f} mbar (yield={yields_scan_ub.max():.4e})")

    # Figure HHG-6: Pressure Scan (1×2)
    fig_hhg6, axes_hhg6 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes_hhg6[0]
    ax.semilogy(pressures_scan, yields_scan_b, 'b-o', label='Blocked', markersize=4, linewidth=2)
    ax.semilogy(pressures_scan, yields_scan_ub, 'r-s', label='Unblocked', markersize=4, linewidth=2)
    ax.axvline(hhg_gas_pressure, color='gray', linestyle='--', linewidth=0.8, label=f'Current ({hhg_gas_pressure:.0f} mbar)')
    ax.axvline(P_opt_b, color='blue', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(P_opt_ub, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Pressure (mbar)')
    ax.set_ylabel('HHG Yield (arb. units)')
    ax.set_title(f'HHG Yield vs Pressure (H{hhg_harmonic_order})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes_hhg6[1]
    ratio_scan = yields_scan_b / np.maximum(yields_scan_ub, 1e-50)
    ax.plot(pressures_scan, ratio_scan, 'k-o', markersize=4, linewidth=2)
    ax.axhline(1.0, color='green', linestyle=':', linewidth=1, label='Equal yield')
    ax.axvline(hhg_gas_pressure, color='gray', linestyle='--', linewidth=0.8, label=f'Current ({hhg_gas_pressure:.0f} mbar)')
    ax.set_xlabel('Pressure (mbar)')
    ax.set_ylabel('Yield ratio (blocked / unblocked)')
    ax.set_title('Blocked/Unblocked Yield Ratio vs Pressure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_hhg6.suptitle(f'Pressure Optimization - {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                       f'Gas={hhg_gas_length} mm, PPT={"ON" if hhg_lut_include_ppt else "OFF"}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'hhg_pressure_scan_{_m2_tag}.png', dpi=300)
    print(f"Pressure scan figure saved to 'hhg_pressure_scan.png'")

if False:  # DISABLED — Figure HHG-9: Extrapolation Slope Scan
    # =============================================================================
    # Figure HHG-9: Extrapolation Slope Scan (experimental LUT only)
    # =============================================================================
    if hhg_lut_mode == 'experimental':
        print("\nComputing extrapolation slope scan (HHG-9)...")

        slope_values = np.linspace(0.0, 2.0, 11)
        scan_nf_ratios = np.zeros_like(slope_values)
        scan_ap_ratios = np.zeros_like(slope_values)
        scan_lut_curves = []

        for i_slope, s_val in enumerate(slope_values):
            # Rebuild experimental LUT with this slope
            dq_mag_scan = np.zeros_like(I_lut)
            for i, I_val in enumerate(I_lut):
                if I_val >= I_min_exp and I_val <= I_max_exp:
                    dq_mag_scan[i] = np.interp(I_val, exp_yield_I_Wcm2, exp_dq_scaled)
                elif I_val < I_min_exp:
                    dq_mag_scan[i] = exp_dq_scaled[0] * (I_val / I_min_exp)**s_val
                else:
                    dq_mag_scan[i] = exp_dq_scaled[-1]
            scan_lut_curves.append(dq_mag_scan.copy())

            dq_interp_scan = interp1d(I_lut, dq_mag_scan, kind='cubic',
                                       bounds_error=False, fill_value=0.0)

            # Recompute dq_3d with new LUT magnitude but original Lewenstein phase
            I_clip_b = np.clip(I_2d_gas_b_Wcm2, I_lut_min, I_lut_max)
            I_clip_ub = np.clip(I_2d_gas_ub_Wcm2, I_lut_min, I_lut_max)
            dq_b = dq_interp_scan(I_clip_b) * np.exp(1j * dq_phase_interp(I_clip_b))
            dq_b[I_2d_gas_b_Wcm2 < I_lut_min] = 0.0
            dq_ub = dq_interp_scan(I_clip_ub) * np.exp(1j * dq_phase_interp(I_clip_ub))
            dq_ub[I_2d_gas_ub_Wcm2 < I_lut_min] = 0.0

            # Macroscopic integration (with absorption)
            integ_b = dq_b * (1.0 - nf_3d_b) * np.exp(1j * Phi_3d_b) * abs_factor_b
            integ_ub = dq_ub * (1.0 - nf_3d_ub) * np.exp(1j * Phi_3d_ub) * abs_factor_ub
            E_q_b_s = np.trapz(integ_b, z_gas_2d_b_m, axis=0)
            E_q_ub_s = np.trapz(integ_ub, z_gas_2d_ub_m, axis=0)

            # Near-field ratio
            y_b = np.sum(np.abs(E_q_b_s)**2) * dx_hhg_m**2
            y_ub = np.sum(np.abs(E_q_ub_s)**2) * dx_hhg_m**2
            scan_nf_ratios[i_slope] = y_b / y_ub if y_ub > 0 else np.nan

            # Far-field aperture ratio
            E_ff_b_s = np.fft.fftshift(np.fft.fft2(E_q_b_s)) * dx_hhg_m**2
            E_ff_ub_s = np.fft.fftshift(np.fft.fft2(E_q_ub_s)) * dx_hhg_m**2
            ya_b = np.sum(np.abs(E_ff_b_s)**2 * aperture_mask_ff) * dtheta**2
            ya_ub = np.sum(np.abs(E_ff_ub_s)**2 * aperture_mask_ff) * dtheta**2
            scan_ap_ratios[i_slope] = ya_b / ya_ub if ya_ub > 0 else np.nan

            print(f"  slope={s_val:.2f}: NF ratio={scan_nf_ratios[i_slope]:.4f}, "
                  f"AP ratio={scan_ap_ratios[i_slope]:.4f}")

        # Experimental target at current peak intensity
        harmonic_key_scan = f'H{hhg_harmonic_order}'
        if harmonic_key_scan in exp_enhancement:
            exp_target = np.interp(hhg_peak_intensity_Wcm2 / 1e14, exp_intensities_1e14[::-1],
                                   exp_enhancement[harmonic_key_scan][::-1])
        else:
            exp_target = np.nan

        # Figure HHG-9 (1×2)
        fig_hhg9, axes_hhg9 = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes_hhg9[0]
        ax.plot(slope_values, scan_nf_ratios, 'b-o', label='Near-field', linewidth=2, markersize=5)
        ax.plot(slope_values, scan_ap_ratios, 'r-s', label='Aperture-filtered', linewidth=2, markersize=5)
        if not np.isnan(exp_target):
            ax.axhline(exp_target, color='green', linestyle='--', linewidth=1.5,
                       label=f'Exp. {harmonic_key_scan} target ({exp_target:.2f})')
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(low_slope, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Current slope ({low_slope:.2f})')
        ax.set_xlabel('Extrapolation slope')
        ax.set_ylabel('Yield ratio (blocked / unblocked)')
        ax.set_title('Ratio vs Extrapolation Slope')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes_hhg9[1]
        I_plot_scan = I_lut / 1e14
        for j in [0, 3, 5, 7, 10]:
            if j < len(slope_values):
                ax.semilogy(I_plot_scan, scan_lut_curves[j],
                            label=f'slope={slope_values[j]:.1f}', linewidth=1.5)
        ax.semilogy(I_plot_scan, dq_mag, 'k--', label='Lewenstein', linewidth=1.5, alpha=0.5)
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel(r'$|d_q|$ (a.u.)')
        ax.set_title('LUT Shape at Different Slopes')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig_hhg9.suptitle(f'Extrapolation Slope Scan — H{hhg_harmonic_order}, '
                           f'Mode: {exp_extrap_mode}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'hhg_slope_scan_{_m2_tag}.png', dpi=300)
        print(f"Slope scan figure saved to 'hhg_slope_scan.png'")

        del dq_b, dq_ub, integ_b, integ_ub, E_q_b_s, E_q_ub_s, E_ff_b_s, E_ff_ub_s
        gc.collect()

    elif hhg_lut_mode == 'deconvolved':
        print("\nComputing peaked response 2D scan (HHG-9)...")

        # Experimental target
        harmonic_key_scan = f'H{hhg_harmonic_order}'
        if harmonic_key_scan in exp_enhancement:
            exp_target = np.interp(hhg_peak_intensity_Wcm2 / 1e14, exp_intensities_1e14[::-1],
                                   exp_enhancement[harmonic_key_scan][::-1])
        else:
            exp_target = np.nan

        alpha_vals = np.array([0.5, 1.0, 1.5, 2.0])
        Is_vals = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 3.0]) * 1e14
        scan2d_nf = np.zeros((len(alpha_vals), len(Is_vals)))
        scan2d_ap = np.zeros((len(alpha_vals), len(Is_vals)))
        best_ratio = -1
        best_params = (0, 0)
        best_curve = None

        I_clip_b = np.clip(I_2d_gas_b_Wcm2, I_lut_min, I_lut_max)
        I_clip_ub = np.clip(I_2d_gas_ub_Wcm2, I_lut_min, I_lut_max)
        phase_b = np.exp(1j * dq_phase_interp(I_clip_b))
        phase_ub = np.exp(1j * dq_phase_interp(I_clip_ub))

        for ia, a_val in enumerate(alpha_vals):
            for js, Is_val in enumerate(Is_vals):
                # Build peaked |d_q| on LUT grid
                dq2_raw = I_lut**a_val * np.exp(-I_lut / Is_val)
                dq_raw = np.sqrt(dq2_raw)
                # Anchor at peak or I_max
                I_ref = min(a_val * Is_val, I_lut[-1])
                dq_lew_r = dq_mag_interp(I_ref)
                dq_raw_r = np.sqrt(I_ref**a_val * np.exp(-I_ref / Is_val))
                sc = dq_lew_r / (dq_raw_r + 1e-30)
                dq_curve = dq_raw * sc

                dq_interp_s = interp1d(I_lut, dq_curve, kind='cubic',
                                        bounds_error=False, fill_value=0.0)

                dq_b = dq_interp_s(I_clip_b) * phase_b
                dq_b[I_2d_gas_b_Wcm2 < I_lut_min] = 0.0
                dq_ub = dq_interp_s(I_clip_ub) * phase_ub
                dq_ub[I_2d_gas_ub_Wcm2 < I_lut_min] = 0.0

                integ_b = dq_b * (1.0 - nf_3d_b) * np.exp(1j * Phi_3d_b) * abs_factor_b
                integ_ub = dq_ub * (1.0 - nf_3d_ub) * np.exp(1j * Phi_3d_ub) * abs_factor_ub
                E_q_b_s = np.trapz(integ_b, z_gas_2d_b_m, axis=0)
                E_q_ub_s = np.trapz(integ_ub, z_gas_2d_ub_m, axis=0)

                y_b = np.sum(np.abs(E_q_b_s)**2) * dx_hhg_m**2
                y_ub = np.sum(np.abs(E_q_ub_s)**2) * dx_hhg_m**2
                scan2d_nf[ia, js] = y_b / y_ub if y_ub > 0 else np.nan

                E_ff_b_s = np.fft.fftshift(np.fft.fft2(E_q_b_s)) * dx_hhg_m**2
                E_ff_ub_s = np.fft.fftshift(np.fft.fft2(E_q_ub_s)) * dx_hhg_m**2
                ya_b = np.sum(np.abs(E_ff_b_s)**2 * aperture_mask_ff) * dtheta**2
                ya_ub = np.sum(np.abs(E_ff_ub_s)**2 * aperture_mask_ff) * dtheta**2
                scan2d_ap[ia, js] = ya_b / ya_ub if ya_ub > 0 else np.nan

                if scan2d_ap[ia, js] > best_ratio:
                    best_ratio = scan2d_ap[ia, js]
                    best_params = (a_val, Is_val)
                    best_curve = dq_curve.copy()

                print(f"  alpha={a_val:.1f}, I_s={Is_val/1e14:.1f}e14: "
                      f"NF={scan2d_nf[ia,js]:.4f}, AP={scan2d_ap[ia,js]:.4f}")

        print(f"\n  Best aperture ratio: {best_ratio:.4f} at alpha={best_params[0]:.1f}, "
              f"I_s={best_params[1]/1e14:.1f}e14")
        if not np.isnan(exp_target):
            print(f"  Experimental target: {exp_target:.2f}")

        # Figure HHG-9 (1×2)
        fig_hhg9, axes_hhg9 = plt.subplots(1, 2, figsize=(14, 5))

        # Left: 2D heatmap
        ax = axes_hhg9[0]
        Is_plot = Is_vals / 1e14
        im = ax.imshow(scan2d_ap, aspect='auto', origin='lower',
                        extent=[Is_plot[0], Is_plot[-1], alpha_vals[0], alpha_vals[-1]],
                        cmap='RdYlGn', vmin=0)
        plt.colorbar(im, ax=ax, label='Aperture ratio (b/ub)')
        if not np.isnan(exp_target):
            cs = ax.contour(Is_plot, alpha_vals, scan2d_ap,
                            levels=[1.0, exp_target], colors=['white', 'lime'], linewidths=2)
            ax.clabel(cs, fmt='%.1f', fontsize=9)
        ax.plot(best_params[1]/1e14, best_params[0], 'w*', markersize=15, label=f'Best ({best_ratio:.2f})')
        ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title('Aperture Ratio vs (α, I_s)')
        ax.legend(fontsize=8)

        # Right: Best |d_q| curve
        ax = axes_hhg9[1]
        I_plot_scan = I_lut / 1e14
        ax.semilogy(I_plot_scan, dq_mag, 'b-', label='Lewenstein SFA', linewidth=2)
        if best_curve is not None:
            ax.semilogy(I_plot_scan, best_curve, 'r--', linewidth=2,
                        label=f'Best peaked (α={best_params[0]:.1f}, I_s={best_params[1]/1e14:.1f})')
        if hhg_lut_mode == 'deconvolved':
            ax.semilogy(I_plot_scan, dq_mag_deconv, 'g:', linewidth=2, label='Current deconvolved')
        ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel(r'$|d_q|$ (a.u.)')
        ax.set_title('LUT Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig_hhg9.suptitle(f'Peaked Response Scan — H{hhg_harmonic_order}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'hhg_peaked_scan_{_m2_tag}.png', dpi=300)
        print(f"Peaked response scan figure saved to 'hhg_peaked_scan.png'")

        del dq_b, dq_ub, integ_b, integ_ub, E_q_b_s, E_q_ub_s, E_ff_b_s, E_ff_ub_s
        del phase_b, phase_ub
        gc.collect()

    # Free arrays no longer needed
    del nf_3d_b, nf_3d_ub, aperture_mask_ff, abs_factor_b, abs_factor_ub
    del tau_b, tau_ub, tau_ref_b, tau_ref_ub
    gc.collect()

if False:  # DISABLED — Figure HHG-8
    print("HHG-8 disabled")

# =============================================================================
# MULTI-HARMONIC LUT + JOINT FIT
# =============================================================================
TIMER.start_section("Multi-Harmonic LUT")
print(f"\n{'='*60}")
print("MULTI-HARMONIC LUT COMPUTATION")
print(f"{'='*60}")

multi_q_list = [11, 13, 15, 17, 19, 21]
q_ref = hhg_harmonic_order  # 21

# --- Recompute nf (needed by multi-harmonic section) ---
nf_mh_b = ionization_fraction(I_2d_gas_b_Wcm2, gas['Ip_eV'])
nf_mh_ub = ionization_fraction(I_2d_gas_ub_Wcm2, gas['Ip_eV'])

if not _LUT_LOADED:

    # --- Build Lewenstein LUT for each harmonic ---
    print(f"\nBuilding Lewenstein LUTs for {len(multi_q_list)} harmonics ({n_lut} intensity points each)...")
    multi_lut = {}
    lut_mh_start = time.time()

    for q in multi_q_list:
        if q == q_ref:
            # Reuse existing H21 LUT
            multi_lut[q] = {
                'mag': dq_mag.copy(),
                'phase': dq_phase.copy(),
            }
            print(f"  H{q}: reused existing LUT")
            continue

        sfa_idx_q_mh = np.argmin(np.abs(sfa_omegalist - q))
        dq_mh = np.zeros(n_lut, dtype=np.complex128)
        t0_q = time.time()

        for i_lut in range(n_lut):
            E0_i = np.sqrt(I_lut[i_lut] / sfa_I_to_E0)
            Ef_i = sfa_E_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
            Af_i = sfa_A_flist(sfa_ti, E0_i, sfa_omega, sfa_Thalf, sfa_Tcut, sfa_omega2, sfa_chirp)
            dm_i, _, _ = sfa_dipole_momion(Af_i, Ef_i, sfa_ti, sfa_Ip_au, sfa_omega, hhg_lut_include_ppt)
            HHG_i = np.fft.rfft(np.real(dm_i) * sfa_window)
            dq_mh[i_lut] = HHG_i[sfa_idx_q_mh]

        multi_lut[q] = {
            'mag': np.abs(dq_mh),
            'phase': np.unwrap(np.angle(dq_mh)),
        }
        print(f"  H{q}: LUT built in {time.time()-t0_q:.1f}s")

    print(f"All LUTs built in {time.time()-lut_mh_start:.1f}s ({(time.time()-lut_mh_start)/60:.1f} min)")

    # Build interpolators for each harmonic
    multi_lut_interp = {}
    for q in multi_q_list:
        mag_interp = interp1d(I_lut, multi_lut[q]['mag'], kind='cubic',
                              bounds_error=False, fill_value=0.0)
        phase_interp = interp1d(I_lut, multi_lut[q]['phase'], kind='cubic',
                                bounds_error=False, fill_value=0.0)
        multi_lut_interp[q] = {'mag': mag_interp, 'phase': phase_interp}

# --- Precompute per-harmonic: Phi (cumulative phase) and abs (XUV absorption) ---
# These depend on q but NOT on (alpha, Is), so compute once
print("\nPrecomputing phase matching and absorption per harmonic...")
I_clip_mh_b = np.clip(I_2d_gas_b_Wcm2, I_lut_min, I_lut_max)
I_clip_mh_ub = np.clip(I_2d_gas_ub_Wcm2, I_lut_min, I_lut_max)
below_min_b = I_2d_gas_b_Wcm2 < I_lut_min
below_min_ub = I_2d_gas_ub_Wcm2 < I_lut_min

# Lewenstein phase on 3D grid (per harmonic)
mh_lew_phase_b = {}
mh_lew_phase_ub = {}
for q in multi_q_list:
    mh_lew_phase_b[q] = np.exp(1j * multi_lut_interp[q]['phase'](I_clip_mh_b))
    mh_lew_phase_ub[q] = np.exp(1j * multi_lut_interp[q]['phase'](I_clip_mh_ub))

# --- Per-harmonic far-field angular grid + aperture mask ---
# dtheta depends on lambda_q = lambda_0 / q, so differs per harmonic
mh_dtheta = {}
mh_ap_mask = {}
for q in multi_q_list:
    lambda_q = lambda_0_m / q
    dt_q = lambda_q / (N_hhg_2d * dx_hhg_m)
    theta_ax_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q
    mh_dtheta[q] = dt_q
    if hhg_acceptance_type == 'slit':
        mh_ap_mask[q] = ((np.abs(theta_ax_q[None, :]) <= slit_half_angle_x) &
                         (np.abs(theta_ax_q[:, None]) <= slit_half_angle_y)).astype(float)
    else:
        r_th_q = np.sqrt(theta_ax_q[:, None]**2 + theta_ax_q[None, :]**2)
        mh_ap_mask[q] = (r_th_q <= aperture_half_angle).astype(float)
        del r_th_q

# --- Compute enhancement for current deconvolved parameters ---
print(f"\nComputing multi-harmonic enhancement (per-harmonic alpha, Is)...")
mh_results_current = {}

for q in multi_q_list:
    # Phase mismatch
    dk_neut_b = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_mh_b, gas['delta_n'])
    dk_neut_ub = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_mh_ub, gas['delta_n'])
    dk_plas_b = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_mh_b, gas['N_atm'])
    dk_plas_ub = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_mh_ub, gas['N_atm'])
    dk_geom_b = -(q - 1.0/q) * dphase_dz_3d_b
    dk_geom_ub = -(q - 1.0/q) * dphase_dz_3d_ub
    dk_total_b = dk_neut_b + dk_plas_b + dk_geom_b
    dk_total_ub = dk_neut_ub + dk_plas_ub + dk_geom_ub

    Phi_b = np.zeros_like(dk_total_b)
    Phi_b[1:] = cumulative_trapezoid(dk_total_b, z_gas_2d_b_m, axis=0)
    Phi_ub = np.zeros_like(dk_total_ub)
    Phi_ub[1:] = cumulative_trapezoid(dk_total_ub, z_gas_2d_ub_m, axis=0)

    # XUV absorption
    sigma_q = sigma_xuv_multi_Mb.get(q, 10.0) * 1e-22
    mu_b = sigma_q * n_gas_density * (1.0 - nf_mh_b)
    mu_ub = sigma_q * n_gas_density * (1.0 - nf_mh_ub)
    mu_cum_b = np.zeros_like(mu_b)
    mu_cum_b[1:] = cumulative_trapezoid(mu_b, z_gas_2d_b_m, axis=0)
    tau_b_q = mu_cum_b[-1:] - mu_cum_b
    mu_cum_ub = np.zeros_like(mu_ub)
    mu_cum_ub[1:] = cumulative_trapezoid(mu_ub, z_gas_2d_ub_m, axis=0)
    tau_ub_q = mu_cum_ub[-1:] - mu_cum_ub
    abs_b_q = np.exp(-tau_b_q / 2.0)
    abs_ub_q = np.exp(-tau_ub_q / 2.0)

    # Empirical |d_q| for this harmonic (per-harmonic alpha, Is, no Lewenstein anchoring)
    a_q = deconv_alpha_per_h.get(q, deconv_alpha)
    Is_q = deconv_Is_per_h.get(q, deconv_Is)
    dq_deconv_raw = np.sqrt(I_lut**a_q * np.exp(-I_lut / Is_q))
    dq_deconv_raw[I_lut < 1e13] = 0.0
    scale_q = np.nan  # no Lewenstein anchoring in empirical-magnitude mode
    dq_deconv_interp = interp1d(I_lut, dq_deconv_raw, kind='cubic',
                                 bounds_error=False, fill_value=0.0)

    # d_q on 3D grid
    dq_3d_b = dq_deconv_interp(I_clip_mh_b) * mh_lew_phase_b[q]
    dq_3d_b[below_min_b] = 0.0
    dq_3d_ub = dq_deconv_interp(I_clip_mh_ub) * mh_lew_phase_ub[q]
    dq_3d_ub[below_min_ub] = 0.0

    # Macroscopic integration
    integ_b = dq_3d_b * (1.0 - nf_mh_b) * np.exp(1j * Phi_b) * abs_b_q
    integ_ub = dq_3d_ub * (1.0 - nf_mh_ub) * np.exp(1j * Phi_ub) * abs_ub_q
    E_q_b = np.trapz(integ_b, z_gas_2d_b_m, axis=0)
    E_q_ub = np.trapz(integ_ub, z_gas_2d_ub_m, axis=0)

    # Near-field yield
    y_nf_b = np.sum(np.abs(E_q_b)**2) * dx_hhg_m**2
    y_nf_ub = np.sum(np.abs(E_q_ub)**2) * dx_hhg_m**2

    # Far-field + aperture (per-harmonic angular grid)
    E_ff_b = np.fft.fftshift(np.fft.fft2(E_q_b)) * dx_hhg_m**2
    E_ff_ub = np.fft.fftshift(np.fft.fft2(E_q_ub)) * dx_hhg_m**2
    dt_q = mh_dtheta[q]
    ya_b = np.sum(np.abs(E_ff_b)**2 * mh_ap_mask[q]) * dt_q**2
    ya_ub = np.sum(np.abs(E_ff_ub)**2 * mh_ap_mask[q]) * dt_q**2

    ratio_nf = y_nf_b / y_nf_ub if y_nf_ub > 0 else np.nan
    ratio_ap = ya_b / ya_ub if ya_ub > 0 else np.nan

    # On-axis phase mismatch at focus
    _iz = np.argmin(np.abs(z_gas_2d_b - focal_length))
    _c = I_2d_gas_b_Wcm2.shape[1] // 2
    dk_onaxis_b = dk_total_b[_iz, _c, _c]
    dk_onaxis_ub = dk_total_ub[np.argmin(np.abs(z_gas_2d_ub - focal_length)),
                               I_2d_gas_ub_Wcm2.shape[1]//2,
                               I_2d_gas_ub_Wcm2.shape[2]//2]

    mh_results_current[q] = {
        'nf_ratio': ratio_nf, 'ap_ratio': ratio_ap,
        'yield_nf_b': y_nf_b, 'yield_nf_ub': y_nf_ub,
        'yield_ap_b': ya_b, 'yield_ap_ub': ya_ub,
        'dk_onaxis_b': dk_onaxis_b, 'dk_onaxis_ub': dk_onaxis_ub,
        'scale_q': scale_q,
        'E_q_b': E_q_b.copy(), 'E_q_ub': E_q_ub.copy(),
        'E_ff_b': E_ff_b.copy(), 'E_ff_ub': E_ff_ub.copy(),
        'dtheta': dt_q,
    }

    # Experimental enhancement at current peak intensity
    hkey = f'H{q}'
    if hkey in exp_enhancement:
        exp_enh_q = np.interp(hhg_peak_intensity_Wcm2 / 1e14,
                              exp_intensities_1e14[::-1],
                              exp_enhancement[hkey][::-1])
    else:
        exp_enh_q = np.nan
    mh_results_current[q]['exp_enhancement'] = exp_enh_q

    print(f"  H{q}: NF ratio={ratio_nf:.3f}, AP ratio={ratio_ap:.3f}, "
          f"Exp={exp_enh_q:.2f}, dk_b={dk_onaxis_b:.1f} dk_ub={dk_onaxis_ub:.1f} (1/m)")

    del dk_neut_b, dk_neut_ub, dk_plas_b, dk_plas_ub, dk_geom_b, dk_geom_ub
    del dk_total_b, dk_total_ub, Phi_b, Phi_ub
    del mu_b, mu_ub, mu_cum_b, mu_cum_ub, tau_b_q, tau_ub_q, abs_b_q, abs_ub_q
    del dq_3d_b, dq_3d_ub, integ_b, integ_ub, E_q_b, E_q_ub, E_ff_b, E_ff_ub
gc.collect()

# --- Multi-harmonic (alpha, I_s) joint scan ---
print(f"\n{'='*60}")
print("MULTI-HARMONIC JOINT (alpha, I_s) SCAN")
print(f"{'='*60}")

mh_alpha_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
mh_Is_vals = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]) * 1e14
n_alpha = len(mh_alpha_vals)
n_Is = len(mh_Is_vals)

# Store scan results: enhancement per harmonic per (alpha, Is)
mh_scan_ap = {q: np.zeros((n_alpha, n_Is)) for q in multi_q_list}
mh_scan_nf = {q: np.zeros((n_alpha, n_Is)) for q in multi_q_list}
mh_scan_total_err = np.zeros((n_alpha, n_Is))  # RMS error across all harmonics

mh_scan_start = time.time()
scan_count = 0
total_scans = n_alpha * n_Is * len(multi_q_list)

# Loop: outer = harmonic (memory-efficient: one q's Phi/abs at a time)
for q in multi_q_list:
    print(f"\n  Scanning H{q}...")

    # Phase mismatch for this harmonic (compute once)
    dk_neut_b = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_mh_b, gas['delta_n'])
    dk_neut_ub = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_mh_ub, gas['delta_n'])
    dk_plas_b = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_mh_b, gas['N_atm'])
    dk_plas_ub = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_mh_ub, gas['N_atm'])
    dk_geom_b = -(q - 1.0/q) * dphase_dz_3d_b
    dk_geom_ub = -(q - 1.0/q) * dphase_dz_3d_ub
    dk_total_b = dk_neut_b + dk_plas_b + dk_geom_b
    dk_total_ub = dk_neut_ub + dk_plas_ub + dk_geom_ub
    Phi_b = np.zeros_like(dk_total_b)
    Phi_b[1:] = cumulative_trapezoid(dk_total_b, z_gas_2d_b_m, axis=0)
    Phi_ub = np.zeros_like(dk_total_ub)
    Phi_ub[1:] = cumulative_trapezoid(dk_total_ub, z_gas_2d_ub_m, axis=0)

    # XUV absorption
    sigma_q = sigma_xuv_multi_Mb.get(q, 10.0) * 1e-22
    mu_b = sigma_q * n_gas_density * (1.0 - nf_mh_b)
    mu_ub = sigma_q * n_gas_density * (1.0 - nf_mh_ub)
    mu_cum_b = np.zeros_like(mu_b)
    mu_cum_b[1:] = cumulative_trapezoid(mu_b, z_gas_2d_b_m, axis=0)
    tau_b_q = mu_cum_b[-1:] - mu_cum_b
    mu_cum_ub = np.zeros_like(mu_ub)
    mu_cum_ub[1:] = cumulative_trapezoid(mu_ub, z_gas_2d_ub_m, axis=0)
    tau_ub_q = mu_cum_ub[-1:] - mu_cum_ub
    abs_b_q = np.exp(-tau_b_q / 2.0)
    abs_ub_q = np.exp(-tau_ub_q / 2.0)

    # Base integrand: everything except |d_q| magnitude
    base_b = (1.0 - nf_mh_b) * np.exp(1j * Phi_b) * abs_b_q * mh_lew_phase_b[q]
    base_ub = (1.0 - nf_mh_ub) * np.exp(1j * Phi_ub) * abs_ub_q * mh_lew_phase_ub[q]

    del dk_neut_b, dk_neut_ub, dk_plas_b, dk_plas_ub, dk_geom_b, dk_geom_ub
    del dk_total_b, dk_total_ub, Phi_b, Phi_ub
    del mu_b, mu_ub, mu_cum_b, mu_cum_ub, tau_b_q, tau_ub_q, abs_b_q, abs_ub_q

    lew_mag_interp_q = multi_lut_interp[q]['mag']

    for ia, a_val in enumerate(mh_alpha_vals):
        for js, Is_val in enumerate(mh_Is_vals):
            # Deconvolved |d_q| on LUT grid
            dq_raw = np.sqrt(I_lut**a_val * np.exp(-I_lut / Is_val))
            I_ref = min(a_val * Is_val, I_lut[-1])
            lew_r = lew_mag_interp_q(I_ref)
            deconv_r = np.sqrt(I_ref**a_val * np.exp(-I_ref / Is_val))
            sc = lew_r / (deconv_r + 1e-30)
            dq_curve = dq_raw * sc
            dq_interp_s = interp1d(I_lut, dq_curve, kind='cubic',
                                    bounds_error=False, fill_value=0.0)

            dq_mag_b = dq_interp_s(I_clip_mh_b)
            dq_mag_b[below_min_b] = 0.0
            dq_mag_ub = dq_interp_s(I_clip_mh_ub)
            dq_mag_ub[below_min_ub] = 0.0

            integ_b = dq_mag_b * base_b
            integ_ub = dq_mag_ub * base_ub
            E_q_b = np.trapz(integ_b, z_gas_2d_b_m, axis=0)
            E_q_ub = np.trapz(integ_ub, z_gas_2d_ub_m, axis=0)

            y_nf_b = np.sum(np.abs(E_q_b)**2) * dx_hhg_m**2
            y_nf_ub = np.sum(np.abs(E_q_ub)**2) * dx_hhg_m**2
            mh_scan_nf[q][ia, js] = y_nf_b / y_nf_ub if y_nf_ub > 0 else np.nan

            E_ff_b = np.fft.fftshift(np.fft.fft2(E_q_b)) * dx_hhg_m**2
            E_ff_ub = np.fft.fftshift(np.fft.fft2(E_q_ub)) * dx_hhg_m**2
            dt_q_s = mh_dtheta[q]
            ya_b = np.sum(np.abs(E_ff_b)**2 * mh_ap_mask[q]) * dt_q_s**2
            ya_ub = np.sum(np.abs(E_ff_ub)**2 * mh_ap_mask[q]) * dt_q_s**2
            mh_scan_ap[q][ia, js] = ya_b / ya_ub if ya_ub > 0 else np.nan

            scan_count += 1

    print(f"    H{q} scan done ({scan_count}/{total_scans}), "
          f"elapsed={time.time()-mh_scan_start:.0f}s")

    del base_b, base_ub
    gc.collect()

# Compute total RMS error across all harmonics
for ia in range(n_alpha):
    for js in range(n_Is):
        err_sum = 0.0
        n_valid = 0
        for q in multi_q_list:
            hkey = f'H{q}'
            if hkey in exp_enhancement:
                exp_enh_q = np.interp(hhg_peak_intensity_Wcm2 / 1e14,
                                      exp_intensities_1e14[::-1],
                                      exp_enhancement[hkey][::-1])
                sim_enh_q = mh_scan_ap[q][ia, js]
                if not np.isnan(sim_enh_q) and not np.isnan(exp_enh_q):
                    err_sum += ((sim_enh_q - exp_enh_q) / exp_enh_q)**2
                    n_valid += 1
        mh_scan_total_err[ia, js] = np.sqrt(err_sum / n_valid) if n_valid > 0 else np.nan

# Find best parameters
best_idx = np.unravel_index(np.nanargmin(mh_scan_total_err), mh_scan_total_err.shape)
best_alpha = mh_alpha_vals[best_idx[0]]
best_Is = mh_Is_vals[best_idx[1]]
best_err = mh_scan_total_err[best_idx]

print(f"\n  === Joint scan results ===")
print(f"  Best: alpha={best_alpha:.1f}, I_s={best_Is/1e14:.1f}e14 (RMS rel. error={best_err:.3f})")
print(f"  Current: per-harmonic alpha/Is from fit")
print(f"\n  Per-harmonic enhancement at best params:")
for q in multi_q_list:
    hkey = f'H{q}'
    sim_best = mh_scan_ap[q][best_idx]
    exp_enh = mh_results_current[q]['exp_enhancement']
    sim_current = mh_results_current[q]['ap_ratio']
    print(f"    H{q}: Sim(best)={sim_best:.3f}, Sim(current)={sim_current:.3f}, Exp={exp_enh:.2f}")

print(f"\n  Total scan time: {time.time()-mh_scan_start:.0f}s ({(time.time()-mh_scan_start)/60:.1f} min)")

# Clean up Lewenstein phase arrays
del mh_lew_phase_b, mh_lew_phase_ub, nf_mh_b, nf_mh_ub
del I_clip_mh_b, I_clip_mh_ub, below_min_b, below_min_ub
gc.collect()

if False:  # DISABLED — Figure HHG-MH-1: Multi-Harmonic Results
    # =============================================================================
    # Figure HHG-MH-1: Multi-Harmonic Results (2×3)
    # =============================================================================
    print("\nGenerating multi-harmonic figures...")

    fig_mh1, axes_mh1 = plt.subplots(2, 3, figsize=(20, 12))

    # --- (0,0) Enhancement bar chart: sim vs exp ---
    ax = axes_mh1[0, 0]
    q_labels = [f'H{q}' for q in multi_q_list]
    x_bar = np.arange(len(multi_q_list))
    exp_enh_arr = [mh_results_current[q]['exp_enhancement'] for q in multi_q_list]
    sim_ap_arr = [mh_results_current[q]['ap_ratio'] for q in multi_q_list]
    sim_nf_arr = [mh_results_current[q]['nf_ratio'] for q in multi_q_list]
    w = 0.25
    ax.bar(x_bar - w, exp_enh_arr, w, label='Exp.', color='green', alpha=0.8)
    ax.bar(x_bar, sim_ap_arr, w, label='Sim (aperture)', color='red', alpha=0.8)
    ax.bar(x_bar + w, sim_nf_arr, w, label='Sim (near-field)', color='blue', alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(q_labels)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Enhancement (blocked / unblocked)')
    ax.set_title('Enhancement (per-harmonic $\\alpha$, $I_s$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Lewenstein |d_q| for all harmonics ---
    ax = axes_mh1[0, 1]
    colors_mh = plt.cm.viridis(np.linspace(0.1, 0.9, len(multi_q_list)))
    I_plot_mh = I_lut / 1e14
    for iq, q in enumerate(multi_q_list):
        ax.semilogy(I_plot_mh, multi_lut[q]['mag'], '-', color=colors_mh[iq],
                    label=f'H{q}', linewidth=1.5)
    ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
    ax.set_ylabel(r'$|d_q|$ (a.u.)')
    ax.set_title('Lewenstein SFA $|d_q|$ per harmonic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,2) Lewenstein vs deconvolved for each harmonic ---
    ax = axes_mh1[0, 2]
    for iq, q in enumerate(multi_q_list):
        ax.semilogy(I_plot_mh, multi_lut[q]['mag'], '-', color=colors_mh[iq],
                    linewidth=1.0, alpha=0.4)
        # Deconvolved curve (per-harmonic alpha, Is)
        a_q_dc = deconv_alpha_per_h.get(q, deconv_alpha)
        Is_q_dc = deconv_Is_per_h.get(q, deconv_Is)
        dq_dc = np.sqrt(I_lut**a_q_dc * np.exp(-I_lut / Is_q_dc))
        I_ref_dc = min(a_q_dc * Is_q_dc, I_lut[-1])
        sc_dc = multi_lut_interp[q]['mag'](I_ref_dc) / (np.sqrt(I_ref_dc**a_q_dc * np.exp(-I_ref_dc / Is_q_dc)) + 1e-30)
        dq_dc *= sc_dc
        ax.semilogy(I_plot_mh, dq_dc, '--', color=colors_mh[iq], linewidth=1.5,
                    label=f'H{q}')
    ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
    ax.set_ylabel(r'$|d_q|$ (a.u.)')
    ax.set_title(f'Deconvolved (dashed) vs SFA (solid)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Unblocked yield spectrum: sim vs exp ---
    ax = axes_mh1[1, 0]
    # Experimental yield at highest intensity (index -1)
    exp_yield_arr = [exp_yield_multi[q][-1] for q in multi_q_list]
    # Normalize to H21
    exp_yield_norm = np.array(exp_yield_arr) / exp_yield_arr[-1]
    sim_yield_ub = [mh_results_current[q]['yield_nf_ub'] for q in multi_q_list]
    sim_yield_norm = np.array(sim_yield_ub) / (sim_yield_ub[-1] + 1e-30)
    ax.bar(x_bar - 0.15, exp_yield_norm, 0.3, label='Exp. (norm to H21)', color='green', alpha=0.8)
    ax.bar(x_bar + 0.15, sim_yield_norm, 0.3, label='Sim. (norm to H21)', color='blue', alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(q_labels)
    ax.set_ylabel('Yield / H21 yield')
    ax.set_title('Unblocked yield spectrum (normalized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,1) On-axis dk for each harmonic ---
    ax = axes_mh1[1, 1]
    dk_b_arr = [mh_results_current[q]['dk_onaxis_b'] for q in multi_q_list]
    dk_ub_arr = [mh_results_current[q]['dk_onaxis_ub'] for q in multi_q_list]
    ax.plot(multi_q_list, dk_b_arr, 'rs-', label='Blocked', markersize=6, linewidth=1.5)
    ax.plot(multi_q_list, dk_ub_arr, 'bo-', label='Unblocked', markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Harmonic order q')
    ax.set_ylabel(r'$\Delta k$ on-axis at focus (1/m)')
    ax.set_title('Phase mismatch per harmonic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,2) Coherence length per harmonic ---
    ax = axes_mh1[1, 2]
    Lcoh_b_arr = [np.pi / (np.abs(mh_results_current[q]['dk_onaxis_b']) + 1e-6) * 1e3 for q in multi_q_list]
    Lcoh_ub_arr = [np.pi / (np.abs(mh_results_current[q]['dk_onaxis_ub']) + 1e-6) * 1e3 for q in multi_q_list]
    ax.plot(multi_q_list, Lcoh_b_arr, 'rs-', label='Blocked', markersize=6, linewidth=1.5)
    ax.plot(multi_q_list, Lcoh_ub_arr, 'bo-', label='Unblocked', markersize=6, linewidth=1.5)
    ax.set_xlabel('Harmonic order q')
    ax.set_ylabel('$L_{coh}$ on-axis (mm)')
    ax.set_title('Coherence length per harmonic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig_mh1.suptitle(f'Multi-Harmonic HHG Analysis — {hhg_gas_type.capitalize()}, '
                      f'{hhg_gas_pressure:.0f} mbar, {hhg_gas_length} mm gas', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'hhg_multi_harmonic_results_{_m2_tag}.png', dpi=300)
    print(f"Multi-harmonic results figure saved to 'hhg_multi_harmonic_results.png'")

# =============================================================================
# Figure HHG-MH-1b: Near-field 2D yield for ALL harmonics (2×6)
# =============================================================================
print("\nGenerating per-harmonic near-field yield figures...")
x_hhg_um = x_hhg_2d * 1e3  # mm → μm
hhg_extent_mh = [x_hhg_um[0], x_hhg_um[-1], x_hhg_um[0], x_hhg_um[-1]]
_c_mh = N_hhg_2d // 2

fig_mh1b, axes_mh1b = plt.subplots(2, 6, figsize=(30, 10))

for iq, q in enumerate(multi_q_list):
    r = mh_results_current[q]
    I_nf_b = np.abs(r['E_q_b'])**2
    I_nf_ub = np.abs(r['E_q_ub'])**2
    vmax_q = max(I_nf_b.max(), I_nf_ub.max(), 1e-30)

    # Row 0: blocked near-field
    ax = axes_mh1b[0, iq]
    if I_nf_b.max() > 0:
        im = ax.imshow(I_nf_b.T / vmax_q, extent=hhg_extent_mh, aspect='equal',
                       origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'H{q} blocked\nenh={r["nf_ratio"]:.2f}')
    ax.set_xlim([-50, 50]); ax.set_ylim([-50, 50])
    if iq == 0: ax.set_ylabel('Blocked\ny (μm)')

    # Row 1: unblocked near-field
    ax = axes_mh1b[1, iq]
    if I_nf_ub.max() > 0:
        im = ax.imshow(I_nf_ub.T / vmax_q, extent=hhg_extent_mh, aspect='equal',
                       origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'H{q} unblocked')
    ax.set_xlim([-50, 50]); ax.set_ylim([-50, 50])
    if iq == 0: ax.set_ylabel('Unblocked\ny (μm)')
    ax.set_xlabel('x (μm)')

fig_mh1b.suptitle(f'Near-Field HHG Yield (All Harmonics) — {hhg_gas_type.capitalize()}, '
                   f'{hhg_gas_pressure:.0f} mbar, M²=({M2x},{M2y})', fontsize=14)
plt.tight_layout()
plt.savefig(f'hhg_nearfield_all_harmonics_{_m2_tag}.png', dpi=300)
print(f"  Saved: hhg_nearfield_all_harmonics_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MH-1c: Far-field 2D yield for ALL harmonics (2×6)
# =============================================================================
print("Generating per-harmonic far-field yield figures...")

fig_mh1c, axes_mh1c = plt.subplots(2, 6, figsize=(30, 10))

for iq, q in enumerate(multi_q_list):
    r = mh_results_current[q]
    I_ff_b = np.abs(r['E_ff_b'])**2
    I_ff_ub = np.abs(r['E_ff_ub'])**2
    dt_q = r['dtheta']
    theta_ax_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q * 1e3  # mrad
    ff_ext = [theta_ax_q[0], theta_ax_q[-1], theta_ax_q[0], theta_ax_q[-1]]
    vmax_q = max(I_ff_b.max(), I_ff_ub.max(), 1e-30)

    # Row 0: blocked far-field
    ax = axes_mh1c[0, iq]
    if I_ff_b.max() > 0:
        im = ax.imshow(I_ff_b.T / vmax_q, extent=ff_ext, aspect='equal',
                       origin='lower', cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'H{q} blocked\nenh(ap)={r["ap_ratio"]:.2f}')
    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10])
    if iq == 0: ax.set_ylabel('Blocked\nθ_y (mrad)')

    # Row 1: unblocked far-field
    ax = axes_mh1c[1, iq]
    if I_ff_ub.max() > 0:
        im = ax.imshow(I_ff_ub.T / vmax_q, extent=ff_ext, aspect='equal',
                       origin='lower', cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'H{q} unblocked')
    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10])
    if iq == 0: ax.set_ylabel('Unblocked\nθ_y (mrad)')
    ax.set_xlabel('θ_x (mrad)')

fig_mh1c.suptitle(f'Far-Field HHG (All Harmonics) — {hhg_gas_type.capitalize()}, '
                   f'{hhg_gas_pressure:.0f} mbar, M²=({M2x},{M2y})', fontsize=14)
plt.tight_layout()
plt.savefig(f'hhg_farfield_all_harmonics_{_m2_tag}.png', dpi=300)
print(f"  Saved: hhg_farfield_all_harmonics_{_m2_tag}.png")

# =============================================================================
# Figure HHG-MH-1d: Far-field lineouts for ALL harmonics (1×2)
# =============================================================================
fig_mh1d, axes_mh1d = plt.subplots(1, 2, figsize=(14, 6))

ax = axes_mh1d[0]
for iq, q in enumerate(multi_q_list):
    r = mh_results_current[q]
    I_ff_b = np.abs(r['E_ff_b'])**2
    dt_q = r['dtheta']
    theta_ax_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q * 1e3
    lineout = I_ff_b[N_hhg_2d // 2, :]
    if lineout.max() > 0:
        ax.plot(theta_ax_q, lineout / lineout.max(), label=f'H{q}', color=COLORS_HQ[q])
ax.set_xlabel('θ_x (mrad)')
ax.set_ylabel('Normalized intensity')
ax.set_title('Blocked far-field lineouts')
ax.set_xlim([-10, 10])
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes_mh1d[1]
for iq, q in enumerate(multi_q_list):
    r = mh_results_current[q]
    I_ff_ub = np.abs(r['E_ff_ub'])**2
    dt_q = r['dtheta']
    theta_ax_q = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q * 1e3
    lineout = I_ff_ub[N_hhg_2d // 2, :]
    if lineout.max() > 0:
        ax.plot(theta_ax_q, lineout / lineout.max(), label=f'H{q}', color=COLORS_HQ[q])
ax.set_xlabel('θ_x (mrad)')
ax.set_ylabel('Normalized intensity')
ax.set_title('Unblocked far-field lineouts')
ax.set_xlim([-10, 10])
ax.legend()
ax.grid(True, alpha=0.3)

fig_mh1d.suptitle(f'Far-Field Lineouts (All Harmonics) — {hhg_gas_type.capitalize()}, '
                   f'{hhg_gas_pressure:.0f} mbar', fontsize=14)
plt.tight_layout()
plt.savefig(f'hhg_farfield_lineouts_all_{_m2_tag}.png', dpi=300)
print(f"  Saved: hhg_farfield_lineouts_all_{_m2_tag}.png")

if False:  # DISABLED
    # =============================================================================
    # Figure HHG-MH-2: Multi-Harmonic Joint Scan (2×3)
    # =============================================================================
    fig_mh2, axes_mh2 = plt.subplots(2, 3, figsize=(22, 12))
    Is_plot_mh = mh_Is_vals / 1e14

    # --- (0,0) Total RMS error heatmap ---
    ax = axes_mh2[0, 0]
    im = ax.imshow(mh_scan_total_err, aspect='auto', origin='lower',
                   extent=[Is_plot_mh[0], Is_plot_mh[-1],
                           mh_alpha_vals[0], mh_alpha_vals[-1]],
                   cmap='RdYlGn_r', vmin=0)
    plt.colorbar(im, ax=ax, label='RMS relative error')
    ax.plot(best_Is/1e14, best_alpha, 'w*', markersize=15,
            label=f'Best ({best_err:.3f})')
    for q_mk in multi_q_list:
        ax.plot(deconv_Is_per_h.get(q_mk, deconv_Is)/1e14,
                deconv_alpha_per_h.get(q_mk, deconv_alpha),
                'ko', markersize=6)
    ax.plot([], [], 'ko', markersize=6, label='Per-harmonic')
    ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')
    ax.set_ylabel(r'$\alpha$')
    ax.set_title('Total RMS error (all harmonics)')
    ax.legend(fontsize=8)

    # --- (0,1) Per-harmonic error at best params ---
    ax = axes_mh2[0, 1]
    best_errors = []
    for q in multi_q_list:
        sim_val = mh_scan_ap[q][best_idx]
        exp_val = mh_results_current[q]['exp_enhancement']
        rel_err = (sim_val - exp_val) / exp_val if exp_val > 0 else np.nan
        best_errors.append(rel_err)
    ax.bar(x_bar, best_errors, 0.5, color=colors_mh, alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(q_labels)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Relative error (sim - exp) / exp')
    ax.set_title(f'Per-harmonic error at best ($\\alpha$={best_alpha:.1f}, $I_s$={best_Is/1e14:.1f})')
    ax.grid(True, alpha=0.3)

    # --- (0,2) Enhancement spectrum: best vs current vs exp ---
    ax = axes_mh2[0, 2]
    best_enh = [mh_scan_ap[q][best_idx] for q in multi_q_list]
    ax.bar(x_bar - w, exp_enh_arr, w, label='Exp.', color='green', alpha=0.8)
    ax.bar(x_bar, best_enh, w, label=f'Best ($\\alpha$={best_alpha:.1f}, $I_s$={best_Is/1e14:.1f})',
           color='orange', alpha=0.8)
    ax.bar(x_bar + w, sim_ap_arr, w, label='Current (per-harmonic)',
           color='red', alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(q_labels)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Enhancement')
    ax.set_title('Best vs Current vs Experiment')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (1,0) to (1,2): Per-harmonic heatmaps (selected) ---
    for ip, q_show in enumerate([11, 17, 21]):
        ax = axes_mh2[1, ip]
        data_show = mh_scan_ap[q_show]
        exp_enh_show = mh_results_current[q_show]['exp_enhancement']

        im = ax.imshow(data_show, aspect='auto', origin='lower',
                       extent=[Is_plot_mh[0], Is_plot_mh[-1],
                               mh_alpha_vals[0], mh_alpha_vals[-1]],
                       cmap='RdYlGn', vmin=0,
                       vmax=max(np.nanmax(data_show), exp_enh_show * 1.5))
        plt.colorbar(im, ax=ax, label='AP ratio')
        if not np.isnan(exp_enh_show):
            cs = ax.contour(Is_plot_mh, mh_alpha_vals, data_show,
                            levels=[1.0, exp_enh_show], colors=['white', 'lime'], linewidths=2)
            ax.clabel(cs, fmt='%.1f', fontsize=9)
        ax.plot(best_Is/1e14, best_alpha, 'w*', markersize=12)
        ax.set_xlabel(r'$I_s$ ($\times 10^{14}$ W/cm$^2$)')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(f'H{q_show} aperture ratio (exp={exp_enh_show:.2f})')

    fig_mh2.suptitle(f'Multi-Harmonic Joint ($\\alpha$, $I_s$) Scan — '
                      f'{hhg_gas_type.capitalize()}, {hhg_gas_pressure:.0f} mbar', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'hhg_multi_harmonic_scan_{_m2_tag}.png', dpi=300)
    print(f"Multi-harmonic scan figure saved to 'hhg_multi_harmonic_scan.png'")

if False:  # DISABLED
    # =============================================================================
    # Figure HHG-MH-3: Experimental Yield vs Intensity (multi-harmonic)
    # =============================================================================
    fig_mh3, axes_mh3 = plt.subplots(1, 2, figsize=(14, 6))

    # (0) Absolute yield vs intensity per harmonic
    ax = axes_mh3[0]
    I_exp_plot = exp_yield_I_Wcm2 / 1e14
    for iq, q in enumerate(multi_q_list):
        ax.plot(I_exp_plot, exp_yield_multi[q], 'o-', color=colors_mh[iq],
                label=f'H{q}', markersize=5, linewidth=1.5)
    ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
    ax.set_ylabel('Yield (arb. units)')
    ax.set_title('Experimental unblocked yield')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1) Yield normalized to each harmonic's max
    ax = axes_mh3[1]
    for iq, q in enumerate(multi_q_list):
        y_norm = exp_yield_multi[q] / exp_yield_multi[q].max()
        ax.plot(I_exp_plot, y_norm, 'o-', color=colors_mh[iq],
                label=f'H{q}', markersize=5, linewidth=1.5)
    ax.set_xlabel(r'Intensity ($\times 10^{14}$ W/cm$^2$)')
    ax.set_ylabel('Yield / max(yield)')
    ax.set_title('Self-normalized yield per harmonic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig_mh3.suptitle('Multi-Harmonic Experimental Yield Data (Unblocked)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'hhg_multi_harmonic_yield_data_{_m2_tag}.png', dpi=300)
    print(f"Multi-harmonic yield data figure saved to 'hhg_multi_harmonic_yield_data.png'")

    TIMER.end_section()

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
        a_q = deconv_alpha_per_h.get(q, deconv_alpha)
        Is_q = deconv_Is_per_h.get(q, deconv_Is)
        dq_raw = np.sqrt(I_lut**a_q * np.exp(-I_lut / Is_q))
        dq_raw[I_lut < 1e13] = 0.0
        dq_interp_q = interp1d(I_lut, dq_raw, kind='cubic', bounds_error=False, fill_value=0.0)
        dq_3d = dq_interp_q(I_clip) * np.exp(1j * multi_lut_interp[q]['phase'](I_clip))
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
fig10, axes10 = plt.subplots(2, 2, figsize=(14, 10))
mask_colors = {'none': '#7F7F7F', 'circular': '#2F5597', 'twosided': '#B04A4A', 'diagonal': '#4F8B5B'}
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
            ha='center', fontsize=10)
ax.set_xticks(range(len(mnames)))
ax.set_xticklabels([mask_labels[m] for m in mnames])
ax.set_ylabel('Aperture Yield Ratio (mask / no-mask)')
ax.set_title('Far-Field Aperture Yield')
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax.grid(True, alpha=0.18, axis='y', linewidth=0.8)

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
    ax_sub.set_title(f'{mask_labels[mname]} ({nf_ratio:.2f}x)', fontsize=9)
    if idx >= 2:
        ax_sub.set_xlabel(r'x ($\mu$m)')
    if idx % 2 == 0:
        ax_sub.set_ylabel(r'y ($\mu$m)')

# (1,0) Far-field angular lineouts
ax = axes10[1, 0]
for mname in mnames:
    I_ff_line = mask_results[mname]['I_ff'][N_hhg_2d // 2, :]
    I_ff_line = I_ff_line / max(I_ff_line.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_line, color=mask_colors[mname],
                label=mask_labels[mname], linewidth=1.5)
ax.axvline(-slit_half_angle_x*1e3, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axvline(slit_half_angle_x*1e3, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.6, label='Slit')
ax.axvline(-aperture_half_angle*1e3, color='salmon', linestyle=':', linewidth=0.8, alpha=0.6)
ax.axvline(aperture_half_angle*1e3, color='salmon', linestyle=':', linewidth=0.8, alpha=0.6, label='Circ')
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'Far-field $|E|^2$ (self-norm, log)')
ax.set_title('Far-Field Angular Lineout')
ax.set_xlim([-10, 10])
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
ax.grid(True, which='both', alpha=0.18, linewidth=0.8)

# (1,1) On-axis Gouy phase gradient for each mask
ax = axes10[1, 1]
for mname in mnames:
    r = mask_results[mname]
    ax.plot(r['z_gas_mm'], r['gouy_grad'], color=mask_colors[mname],
            label=mask_labels[mname], linewidth=1.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'd$\phi$/dz (rad/m)')
ax.set_title('On-Axis Gouy Phase Gradient')
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
ax.grid(True, alpha=0.18, linewidth=0.8)

fig10.suptitle(f'Mask Shape Comparison -- H{hhg_harmonic_order}, '
               f'P={hhg_gas_pressure:.0f} mbar', fontsize=16, fontweight='bold')
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

    for im, mn in enumerate(_mnames):
        rq = mask_results[mn]['per_q'][q]
        enh_slit = rq['yield_slit'] / ref_slit if ref_slit > 0 else 0
        enh_circ = rq['yield_circ'] / ref_circ if ref_circ > 0 else 0
        enh_nf = rq['yield_nf'] / ref_nf_q if ref_nf_q > 0 else 0
        vals = [enh_slit, enh_circ, enh_nf]
        bars = ax.bar(x_group + offsets[im], vals, w, label=_mlabels[mn],
                      color=_mcolors[mn], alpha=0.92, edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.2f}', ha='center', fontsize=8.5, rotation=90)
    ax.set_xticks(x_group)
    ax.set_xticklabels(_det_names)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Enhancement (mask / no-mask)')
    ax.set_title(f'H{q} Enhancement')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    ax.grid(True, alpha=0.18, axis='y', linewidth=0.8)

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
        if idx >= 2: ax_sub.set_xlabel('x (μm)')
        if idx % 2 == 0: ax_sub.set_ylabel('y (μm)')

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
    ax.set_title('On-Axis Gouy Phase Gradient')
    ax.set_ylabel(r'd$\phi$/dz (rad/m)')
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
                    color=_mcolors[mn], linewidth=1.8, label=f'{_mlabels[mn]} (slit)')
        if yvz_c is not None:
            ax.plot(z_mm, yvz_c / yvz_norm_circ_ph,
                    color=_mcolors[mn], linewidth=1.3, linestyle='--')
        if yvz_n is not None:
            ax.plot(z_mm, yvz_n / yvz_norm_nf_ph,
                    color=_mcolors[mn], linewidth=1.3, linestyle=':')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Integrated yield (normalized)')
    ax.set_title('Yield buildup')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    fig_ph.suptitle(f'Mask Comparison — H{q}, {hhg_gas_type.capitalize()}, '
                     f'P={hhg_gas_pressure:.0f} mbar, M²=({M2x},{M2y})', fontsize=14)
    fig_ph.suptitle(f'Mask Comparison -- H{q}, {hhg_gas_type.capitalize()}, '
                     f'P={hhg_gas_pressure:.0f} mbar, M$^2$=({M2x},{M2y})',
                     fontsize=17, fontweight='bold')
    finalize_paper_figure(fig_ph)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'hhg_mask_H{q}_{_m2_tag}.png', dpi=400)
    print(f"    Saved: hhg_mask_H{q}_{_m2_tag}.png")

# Save compact mask-comparison arrays for standalone paper plotting.
try:
    _mask_npz_path = f'hhg_mask_comparison_data_{lavg_tag}_{pressure_tag}_{_m2_tag}.npz'
    print(f"\n  Saving mask-comparison plotting data to {_mask_npz_path} ...")

    _harmonics_save = np.array(multi_q_list, dtype=np.int32)
    _mask_names_save = np.array(_mnames)
    _mask_labels_save = np.array([_mlabels[mn] for mn in _mnames])
    _nq_save = len(_harmonics_save)
    _nm_save = len(_mnames)

    _yield_slit = np.zeros((_nq_save, _nm_save), dtype=np.float64)
    _yield_circ = np.zeros_like(_yield_slit)
    _yield_nf = np.zeros_like(_yield_slit)

    _nf_step = max(int(np.ceil(N_hhg_2d / 512)), 1)
    _nf_x_um = x_hhg_um_ph[::_nf_step].astype(np.float32)
    _nearfield_intensity = np.zeros(
        (_nq_save, _nm_save, len(_nf_x_um), len(_nf_x_um)), dtype=np.float32
    )
    _nearfield_peak_rel = np.zeros((_nq_save, _nm_save), dtype=np.float32)
    _nearfield_lineout_x = np.zeros((_nq_save, _nm_save, len(x_hhg_um_ph)), dtype=np.float32)
    _nearfield_lineout_y = np.zeros_like(_nearfield_lineout_x)

    _ff_theta_mrad = np.zeros((_nq_save, N_hhg_2d), dtype=np.float32)
    _ff_lineout_x = np.zeros((_nq_save, _nm_save, N_hhg_2d), dtype=np.float32)
    _ff_lineout_y = np.zeros_like(_ff_lineout_x)

    _z_len = len(mask_results[_mnames[0]]['z_gas_mm'])
    _z_gas_mm_masks = np.zeros((_nm_save, _z_len), dtype=np.float32)
    _ionization_onaxis = np.zeros((_nm_save, _z_len), dtype=np.float32)
    _I_onaxis_Wcm2 = np.zeros((_nm_save, _z_len), dtype=np.float32)
    _gouy_grad = np.zeros((_nm_save, _z_len), dtype=np.float32)
    _yield_vs_z = np.full((_nq_save, _nm_save, _z_len), np.nan, dtype=np.float64)
    _yield_vs_z_slit = np.full_like(_yield_vs_z, np.nan)
    _yield_vs_z_circ = np.full_like(_yield_vs_z, np.nan)

    _focus_peak_I_Wcm2 = np.array([mask_results[_mn]['peak_I'] for _mn in _mnames], dtype=np.float64)
    _focus_peak_rel = _focus_peak_I_Wcm2 / max(_focus_peak_I_Wcm2[0], 1e-30)
    _fwhm_x_um = np.full(_nm_save, np.nan, dtype=np.float64)
    _fwhm_y_um = np.full(_nm_save, np.nan, dtype=np.float64)
    _w0_x_um = np.full(_nm_save, np.nan, dtype=np.float64)
    _w0_y_um = np.full(_nm_save, np.nan, dtype=np.float64)
    _rayleigh_range_mm = np.full(_nm_save, np.nan, dtype=np.float64)
    _focus_z_mm = np.full(_nm_save, np.nan, dtype=np.float64)
    _focus_shift_um = np.full(_nm_save, np.nan, dtype=np.float64)
    _transmission_percent = np.array([mask_results[_mn]['transmission'] for _mn in _mnames], dtype=np.float64)
    _beam_metric_source = 'mask_results'

    if 'mc_data' in globals() and all(_mn in mc_data for _mn in _mnames):
        _beam_metric_source = 'mc_data_highres'
        _focus_peak_rel = np.array(
            [mc_data[_mn]['I_peak'] / max(mc_data['none']['I_peak'], 1e-30) for _mn in _mnames],
            dtype=np.float64,
        )
        _fwhm_x_um = np.array([mc_data[_mn]['fwhm_x'] * 1e3 for _mn in _mnames], dtype=np.float64)
        _fwhm_y_um = np.array([mc_data[_mn]['fwhm_y'] * 1e3 for _mn in _mnames], dtype=np.float64)
        _w0_x_um = np.array([mc_data[_mn]['w0_x'] * 1e3 for _mn in _mnames], dtype=np.float64)
        _w0_y_um = np.array([mc_data[_mn]['w0_y'] * 1e3 for _mn in _mnames], dtype=np.float64)
        _focus_z_mm = np.array([mc_data[_mn]['true_focus_z'] for _mn in _mnames], dtype=np.float64)
        _focus_shift_um = (_focus_z_mm - focal_length) * 1e3
        _transmission_percent = np.array([mc_data[_mn]['trans'] for _mn in _mnames], dtype=np.float64)

        if 'mc_onaxis' in globals() and 'z_focus_prop' in globals():
            _dz_focus = float(z_focus_prop[1] - z_focus_prop[0]) if len(z_focus_prop) > 1 else np.nan
            _rayleigh_range_mm = np.array([
                np.sum(np.asarray(mc_onaxis[_mn]['I']) > 0.5) * _dz_focus / 2.0
                for _mn in _mnames
            ], dtype=np.float64)

    _center_save = N_hhg_2d // 2
    for _iq, _q in enumerate(_harmonics_save):
        _nf_global_q = max(
            (np.abs(mask_results[_mn]['per_q'][int(_q)]['E_q'])**2).max()
            for _mn in _mnames
        )
        _nf_global_q = max(_nf_global_q, 1e-30)

        for _im, _mn in enumerate(_mnames):
            _rq = mask_results[_mn]['per_q'][int(_q)]
            _yield_slit[_iq, _im] = _rq['yield_slit']
            _yield_circ[_iq, _im] = _rq['yield_circ']
            _yield_nf[_iq, _im] = _rq['yield_nf']

            _I_nf = np.abs(_rq['E_q'])**2
            _I_nf_norm = _I_nf / _nf_global_q
            _nearfield_intensity[_iq, _im] = _I_nf_norm[::_nf_step, ::_nf_step].astype(np.float32)
            _nearfield_peak_rel[_iq, _im] = np.float32(np.nanmax(_I_nf_norm))
            _nearfield_lineout_x[_iq, _im] = _I_nf_norm[_center_save, :].astype(np.float32)
            _nearfield_lineout_y[_iq, _im] = _I_nf_norm[:, _center_save].astype(np.float32)

            _line_x = _rq['I_ff'][_center_save, :]
            _line_y = _rq['I_ff'][:, _center_save]
            _ff_lineout_x[_iq, _im] = (_line_x / max(np.nanmax(_line_x), 1e-30)).astype(np.float32)
            _ff_lineout_y[_iq, _im] = (_line_y / max(np.nanmax(_line_y), 1e-30)).astype(np.float32)
            _ff_theta_mrad[_iq] = (_rq['theta_axis'] * 1e3).astype(np.float32)

            _z_gas_mm_masks[_im] = mask_results[_mn]['z_gas_mm'].astype(np.float32)
            _ionization_onaxis[_im] = mask_results[_mn]['nf_onaxis'].astype(np.float32)
            _I_onaxis_Wcm2[_im] = mask_results[_mn]['I_onaxis_Wcm2'].astype(np.float32)
            _gouy_grad[_im] = mask_results[_mn]['gouy_grad'].astype(np.float32)
            for _key, _target in (
                ('yield_vs_z', _yield_vs_z),
                ('yield_vs_z_slit', _yield_vs_z_slit),
                ('yield_vs_z_circ', _yield_vs_z_circ),
            ):
                _arr = _rq.get(_key)
                if _arr is not None:
                    _target[_iq, _im, :len(_arr)] = _arr

    np.savez_compressed(
        _mask_npz_path,
        harmonic_orders=_harmonics_save,
        mask_names=_mask_names_save,
        mask_labels=_mask_labels_save,
        yield_slit=_yield_slit,
        yield_circ=_yield_circ,
        yield_nf=_yield_nf,
        nearfield_intensity_norm=_nearfield_intensity,
        nearfield_peak_rel=_nearfield_peak_rel,
        nearfield_x_um=_nf_x_um,
        nearfield_lineout_x_um=x_hhg_um_ph.astype(np.float32),
        nearfield_lineout_x_norm=_nearfield_lineout_x,
        nearfield_lineout_y_norm=_nearfield_lineout_y,
        nearfield_extent_um=np.array(ext_nf_ph, dtype=np.float32),
        ff_theta_mrad=_ff_theta_mrad,
        ff_lineout_norm=_ff_lineout_x,
        ff_lineout_x_norm=_ff_lineout_x,
        ff_lineout_y_norm=_ff_lineout_y,
        z_gas_mm=_z_gas_mm_masks,
        ionization_onaxis=_ionization_onaxis,
        I_onaxis_Wcm2=_I_onaxis_Wcm2,
        gouy_grad=_gouy_grad,
        yield_vs_z=_yield_vs_z,
        yield_vs_z_slit=_yield_vs_z_slit,
        yield_vs_z_circ=_yield_vs_z_circ,
        focus_peak_I_Wcm2=_focus_peak_I_Wcm2,
        focus_peak_rel=_focus_peak_rel,
        fwhm_x_um=_fwhm_x_um,
        fwhm_y_um=_fwhm_y_um,
        w0_x_um=_w0_x_um,
        w0_y_um=_w0_y_um,
        rayleigh_range_mm=_rayleigh_range_mm,
        focus_z_mm=_focus_z_mm,
        focus_shift_um=_focus_shift_um,
        transmission_percent=_transmission_percent,
        beam_metric_source=np.array(_beam_metric_source),
        slit_half_angle_mrad=np.float64(slit_half_angle_x * 1e3),
        aperture_half_angle_mrad=np.float64(aperture_half_angle * 1e3),
        hhg_gas_type=np.array(hhg_gas_type),
        hhg_gas_pressure=np.float64(hhg_gas_pressure),
        M2x=np.float64(M2x),
        M2y=np.float64(M2y),
        lavg_tag=np.array(lavg_tag),
        pressure_tag=np.array(pressure_tag),
    )
    print(f"  Saved: {_mask_npz_path} ({os.path.getsize(_mask_npz_path) / 1024 / 1024:.1f} MB)")
except Exception as _mask_npz_exc:
    print(f"  WARNING: could not save mask-comparison NPZ: {_mask_npz_exc}")

try:
    print("  Generating fitted peak-intensity-at-focus summary...")
    _beam_plot_names = list(_mnames)
    _beam_plot_labels = ['Unblocked' if _mn == 'none' else _mlabels[_mn] for _mn in _beam_plot_names]
    _beam_plot_colors = [_mcolors[_mn] for _mn in _beam_plot_names]

    if 'mc_data' in globals() and all(_mn in mc_data for _mn in _beam_plot_names):
        _beam_peak_rel_plot = np.array(
            [mc_data[_mn]['I_peak'] / max(mc_data['none']['I_peak'], 1e-30) for _mn in _beam_plot_names],
            dtype=float,
        )
    else:
        _beam_peak_rel_plot = np.array(
            [mask_results[_mn]['peak_I'] for _mn in _beam_plot_names],
            dtype=float,
        )
        _beam_peak_rel_plot = _beam_peak_rel_plot / max(_beam_peak_rel_plot[0], 1e-30)

    _fig_peak, _ax_peak = plt.subplots(figsize=(5.4, 5.1))
    _bar_x_peak = np.arange(len(_beam_plot_names))
    _bars_peak = _ax_peak.bar(
        _bar_x_peak,
        _beam_peak_rel_plot,
        color=_beam_plot_colors,
        alpha=0.88,
        edgecolor='black',
        linewidth=0.5,
    )
    for _bar_peak, _val_peak in zip(_bars_peak, _beam_peak_rel_plot):
        _ax_peak.text(
            _bar_peak.get_x() + _bar_peak.get_width() / 2,
            _bar_peak.get_height() + 0.018,
            f'{_val_peak:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    _ax_peak.set_xticks(_bar_x_peak)
    _ax_peak.set_xticklabels(_beam_plot_labels, fontsize=12)
    _ax_peak.set_ylabel('Peak I / I_unblocked', fontsize=14, fontweight='bold')
    _ax_peak.set_title('Peak Intensity at Focus', fontsize=15, fontweight='bold')
    _ax_peak.set_ylim(0, max(1.05, float(np.nanmax(_beam_peak_rel_plot)) * 1.18))
    style_paper_axis(_ax_peak, grid=True)
    _fig_peak.tight_layout()
    _beam_peak_png = f'mask_peak_intensity_at_focus_fitted_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
    _fig_peak.savefig(_beam_peak_png, dpi=400, bbox_inches='tight')
    plt.close(_fig_peak)
    print(f"  Saved: {_beam_peak_png}")
except Exception as _beam_peak_exc:
    print(f"  WARNING: could not save fitted peak-intensity summary: {_beam_peak_exc}")

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
    ax.plot(r['z_gas_mm'], r['dk_total_onaxis'], color=color, linewidth=1.5, label=label)
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
    ax.semilogy(r['z_gas_mm'], L_coh, color=color, linewidth=1.5, label=label)
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
    ax.plot(r['z_gas_mm'], r['Phi_onaxis'], color=color, linewidth=1.5, label=label)
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
    ax.plot(r['z_gas_mm'], r['I_onaxis_Wcm2'], color=color, linewidth=1.5, label=label)
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
    ax.plot(r['z_gas_mm'], r['nf_onaxis'], color=color, linewidth=1.5, label=label)
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
    ax.plot(r['z_gas_mm'], gouy_uw, color=color, linewidth=1.5, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Gouy phase (rad)')
ax.set_title('On-Axis Gouy Phase')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig_hmc1.suptitle(f'On-Axis HHG Physics — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc1_onaxis_physics_{_m2_tag}.png', dpi=300)
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
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc2_2d_phase_mismatch_{_m2_tag}.png', dpi=300)
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
    ax.plot(hmc3_x_um, r['dk_focus_x'], color=color, linewidth=1.5, label=label)
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
    ax.plot(hmc3_x_um, np.abs(r['dk_focus_x']), color=color, linewidth=1.5, label=label)
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
    ax.semilogy(hmc3_x_um, L_coh_f, color=color, linewidth=1.5, label=label)
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
    ax.plot(hmc3_x_um, r['I_focus_x_Wcm2'], color=color, linewidth=1.5, label=label)
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
    ax.plot(hmc3_x_um, r['nf_focus_x'], color=color, linewidth=1.5, label=label)
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
    ax.plot(r['z_gas_mm'], buildup, color=color, linewidth=1.5, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'On-axis $|E_q(z)|^2$ (normalized)')
ax.set_title('HHG Buildup Along Gas')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig_hmc3.suptitle(f'Phase Mismatch at Focus — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc3_focus_mismatch_{_m2_tag}.png', dpi=300)
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
    ax.plot(hmc3_x_um, I_nf_x_norm, color=color, linewidth=1.5, label=label)
ax.set_xlabel('x (um)')
ax.set_ylabel(r'$|E_q|^2$ (self-norm)')
ax.set_title('Near-Field x Lineout')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,1) Near-field y lineout
ax = axes_hmc4[0, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_nf = np.abs(r['E_q'])**2
    I_nf_y = I_nf[:, hmc4_nf_center]
    I_nf_y_norm = I_nf_y / max(I_nf_y.max(), 1e-30)
    ax.plot(hmc3_x_um, I_nf_y_norm, color=color, linewidth=1.5, label=label)
ax.set_xlabel('y (um)')
ax.set_ylabel(r'$|E_q|^2$ (self-norm)')
ax.set_title('Near-Field y Lineout')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (0,2) On-axis HHG buildup (absolute, not normalized)
ax = axes_hmc4[0, 2]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    buildup = np.abs(r['buildup_onaxis'])**2
    ax.plot(r['z_gas_mm'], buildup, color=color, linewidth=1.5, label=label)
ax.axvline(focal_length, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('z (mm)')
ax.set_ylabel(r'$|E_q(z)|^2$')
ax.set_title('On-Axis HHG Buildup (absolute)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,0) Far-field x lineout
ax = axes_hmc4[1, 0]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_ff_x = r['I_ff'][hmc4_nf_center, :]
    I_ff_x_norm = I_ff_x / max(I_ff_x.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_x_norm, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
ax.set_title('Far-Field x Lineout')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,1) Far-field y lineout
ax = axes_hmc4[1, 1]
for mn in hmc_masks:
    r = mask_results[mn]
    label, color = mask_disp[mn]
    I_ff_y = r['I_ff'][:, hmc4_nf_center]
    I_ff_y_norm = I_ff_y / max(I_ff_y.max(), 1e-30)
    ax.semilogy(theta_mrad, I_ff_y_norm, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
ax.set_title('Far-Field y Lineout')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

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
    ax.plot(angle_scan * 1e3, cum_yield_norm, color=color, linewidth=1.5, label=label)
ax.axvline(aperture_half_angle*1e3, color='gray', linestyle='--', linewidth=0.8, label=f'Aperture ({aperture_half_angle*1e3:.1f} mrad)')
ax.set_xlabel('Acceptance half-angle (mrad)')
ax.set_ylabel('Cumulative yield (normalized)')
ax.set_title('Cumulative Yield vs Acceptance')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig_hmc4.suptitle(f'HHG Yield & Far-Field — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc4_yield_farfield_{_m2_tag}.png', dpi=300)
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
                   cmap='hot', vmin=-4, vmax=0, interpolation='bicubic')
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
    ax.plot(x_hhg_um, I_x, color=color, linewidth=1.5, label='x')
    ax.plot(x_hhg_um, I_y, color=color, linewidth=1.5, linestyle='--', alpha=0.7, label='y')
    # Overlay unblocked for reference
    I_ref_x = hmc5_I_nf['none'][hmc5_nf_center, :] / max(hmc5_I_max_global, 1e-30)
    ax.plot(x_hhg_um, I_ref_x, color='gray', linewidth=0.8, alpha=0.5, label='No mask')
    ax.set_xlabel('Position (um)')
    if col_idx == 0:
        ax.set_ylabel(r'$|E_q|^2$ / $|E_q|^2_{max,all}$')
    ax.set_title(f'{label} lineouts (abs.)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-60, 60])

fig_hmc5.suptitle(f'Near-Field HHG Yield — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar, Grid={N_hhg_2d}x{N_hhg_2d}', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc5_nearfield_2d_{_m2_tag}.png', dpi=300)
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
                   cmap='hot', vmin=-4, vmax=0, interpolation='bicubic')
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
    ax.semilogy(theta_mrad, I_ff_x_n, color=color, linewidth=1.5, label=r'$\theta_x$')
    ax.semilogy(theta_mrad, I_ff_y_n, color=color, linewidth=1.5, linestyle='--', alpha=0.7, label=r'$\theta_y$')
    ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Angle (mrad)')
    if col_idx == 0:
        ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
    ax.set_title(f'{label} angular', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10, 10])
    ax.set_ylim([1e-6, 1.5])

fig_hmc6.suptitle(f'Far-Field HHG — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'Aperture: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'hhg_mc6_farfield_2d_{_m2_tag}.png', dpi=300)
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
    ax.semilogy(theta_mrad, I_ff_x_n, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm)')
ax.set_title(r'$\theta_x$ lineout — shape comparison')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (0,1) y-lineout overlay (self-normalized, log)
ax = axes_hmc7[0, 1]
for mn in hmc7_masks:
    I_ff_y = mask_results[mn]['I_ff'][:, hmc7_c]
    I_ff_y_n = I_ff_y / max(I_ff_y.max(), 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_y_n, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm)')
ax.set_title(r'$\theta_y$ lineout — shape comparison')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (1,0) x-lineout overlay (absolute, normalized to global max)
ax = axes_hmc7[1, 0]
for mn in hmc7_masks:
    I_ff_x = mask_results[mn]['I_ff'][hmc7_c, :]
    I_ff_x_abs = I_ff_x / max(hmc7_ff_peak_max, 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_x_abs, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_x$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ / $|E_{ff}|^2_{max,all}$')
ax.set_title(r'$\theta_x$ lineout — absolute intensity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

# (1,1) y-lineout overlay (absolute, normalized to global max)
ax = axes_hmc7[1, 1]
for mn in hmc7_masks:
    I_ff_y = mask_results[mn]['I_ff'][:, hmc7_c]
    I_ff_y_abs = I_ff_y / max(hmc7_ff_peak_max, 1e-30)
    label, color = mask_disp[mn]
    ax.semilogy(theta_mrad, I_ff_y_abs, color=color, linewidth=1.5, label=label)
ax.axvline(-aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(aperture_half_mrad_mc, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta_y$ (mrad)')
ax.set_ylabel(r'$|E_{ff}|^2$ / $|E_{ff}|^2_{max,all}$')
ax.set_title(r'$\theta_y$ lineout — absolute intensity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-10, 10])
ax.set_ylim([1e-6, 1.5])

fig_hmc7.suptitle(f'Far-Field Angular Comparison (±10 mrad) — {hhg_gas_type.capitalize()}, '
                   f'H{hhg_harmonic_order}, P={hhg_gas_pressure:.0f} mbar', fontsize=14, y=0.98)
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
                   origin='lower', cmap='hot', vmin=0, vmax=1)
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
                   aspect='equal', origin='lower', cmap='hot', vmin=0, vmax=1)
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
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [yield_m_mc, yield_ub_mc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)

    # (1,0) x and y lineouts
    ax = axes_mc8[1, 0]
    lineout_max_mc = max(I_q_x_m.max(), I_q_y_m.max(),
                         I_q_x_ub_mc.max(), I_q_y_ub_mc.max(), 1e-30)
    ax.plot(x_hhg_um, I_q_x_m / lineout_max_mc, color=mc8_color, linewidth=2,
            label=f'{mc8_label} (x)')
    ax.plot(x_hhg_um, I_q_y_m / lineout_max_mc, color=mc8_color, linewidth=1.5,
            linestyle='--', label=f'{mc8_label} (y)')
    ax.plot(x_hhg_um, I_q_x_ub_mc / lineout_max_mc, 'gray', linewidth=1.5,
            alpha=0.6, label='Unblocked (x)')
    ax.plot(x_hhg_um, I_q_y_ub_mc / lineout_max_mc, 'gray', linewidth=1,
            linestyle='--', alpha=0.4, label='Unblocked (y)')
    ax.set_xlabel('Position (um)')
    ax.set_ylabel(r'$|E_q|^2$ (normalized)')
    ax.set_title('HHG Lineouts (x and y)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-50, 50])

    # (1,1) On-axis buildup vs z
    ax = axes_mc8[1, 1]
    bu_m = r_m['buildup_onaxis']
    bu_ub = r_ub['buildup_onaxis']
    bu_m_norm = np.abs(bu_m)**2 / max(np.abs(bu_m[-1])**2, 1e-30)
    bu_ub_norm = np.abs(bu_ub)**2 / max(np.abs(bu_ub[-1])**2, 1e-30)
    ax.plot(r_m['z_gas_mm'], bu_m_norm, color=mc8_color, linewidth=2,
            label=mc8_label)
    ax.plot(r_ub['z_gas_mm'], bu_ub_norm, 'gray', linewidth=1.5, linestyle='--',
            alpha=0.7, label='Unblocked')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel(r'On-axis $|E_q|^2$ (normalized)')
    ax.set_title('On-axis HHG Buildup')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (1,2) Far-field 2D (log scale)
    ax = axes_mc8[1, 2]
    I_ff_m = mask_results[mc8_mname]['I_ff']
    I_ff_ub = mask_results['none']['I_ff']
    ff_max_mc = max(I_ff_m.max(), I_ff_ub.max())
    I_ff_log = np.log10(np.clip(I_ff_m / max(ff_max_mc, 1e-30), 1e-6, None))
    im = ax.imshow(I_ff_log, extent=theta_extent_mc, aspect='equal',
                   origin='lower', cmap='hot', vmin=-4, vmax=0,
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
    plt.tight_layout()
    fname_mc8 = f'hhg_yield_{mc8_mname}_{_m2_tag}.png'
    plt.savefig(fname_mc8, dpi=400)
    print(f"  Saved: {fname_mc8}")

# =============================================================================
# TIMING SUMMARY
# =============================================================================
TIMER.summary()

plt.show()
