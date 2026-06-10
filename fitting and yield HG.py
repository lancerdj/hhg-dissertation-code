"""
Fitting + HHG Yield: HG (Hermite-Gaussian) Beam Model
=====================================================
Merged pipeline: parametric fitting of dipole response followed by
macroscopic HHG yield computation + multi-mask comparison.
Uses HG mode decomposition for M² beam quality modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import os
import time
import gc

# Try to use scipy.fft (faster) if available, fallback to numpy
try:
    from scipy import fft as scipy_fft
    from scipy.signal import czt as scipy_czt
    USE_SCIPY_FFT = True
except ImportError:
    USE_SCIPY_FFT = False
    scipy_czt = None

import scipy.signal as signal
from scipy.interpolate import interp1d

# Try to use Numba for JIT compilation
try:
    from numba import njit, prange
    USE_NUMBA = True
    print("Numba available - using JIT-compiled functions for speedup")
except ImportError:
    USE_NUMBA = False
    print("Numba not available - using pure NumPy")

# =============================================================================
# SFA LEWENSTEIN MODEL FUNCTIONS (from unblocked beam lewenstein.py)
# =============================================================================

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
# PUBLICATION-QUALITY PLOT SETTINGS
# =============================================================================
plt.rcParams.update({
    # Font hierarchy
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    # Axes labels & titles
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'normal',
    'axes.labelpad': 6,
    # Tick formatting
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': True,
    'ytick.right': True,
    # Spines
    'axes.linewidth': 0.8,
    # Grid (off by default; enable per-plot)
    'axes.grid': False,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    # Lines
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
    # Legend
    'legend.fontsize': 9,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '0.7',
    'legend.fancybox': False,
    'legend.frameon': True,
    'legend.borderpad': 0.4,
    'legend.handlelength': 1.8,
    # Figure & saving
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    # Image
    'image.cmap': 'inferno',
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
        ax.grid(True, which='major', alpha=0.18, linewidth=0.8)


def finalize_paper_figure(fig):
    for ax in fig.axes:
        style_paper_axis(ax)


# Unified colorblind-friendly palette (Wong 2011)
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

# =============================================================================
# TIMING UTILITIES
# =============================================================================
class Timer:
    """Simple timer class to track execution time."""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.section_times = {}
        self.current_section = None
        self.section_start = None

    def start_section(self, name):
        if self.current_section is not None:
            self.end_section()
        self.current_section = name
        self.section_start = time.perf_counter()
        print(f"\n>>> Starting: {name}...")

    def end_section(self):
        if self.current_section is not None and self.section_start is not None:
            elapsed = time.perf_counter() - self.section_start
            self.section_times[self.current_section] = elapsed
            print(f"    Completed in {elapsed:.2f} s")
            self.current_section = None
            self.section_start = None

    def total_elapsed(self):
        return time.perf_counter() - self.start_time

    def summary(self):
        self.end_section()
        total = self.total_elapsed()
        print(f"\n{'='*60}")
        print("EXECUTION TIME SUMMARY")
        print(f"{'='*60}")
        sorted_sections = sorted(self.section_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in sorted_sections:
            percentage = (elapsed / total) * 100
            bar_len = int(percentage / 2)
            bar = '\u2588' * bar_len + '\u2591' * (50 - bar_len)
            print(f"  {name:<35} {elapsed:>8.2f}s ({percentage:>5.1f}%) {bar[:20]}")
        print(f"{'─'*60}")
        print(f"  {'TOTAL':<35} {total:>8.2f}s (100.0%)")
        print(f"{'='*60}")
        return total

TIMER = Timer()
print(f"Program started at {time.strftime('%H:%M:%S')}")

# =============================================================================
# BEAM PARAMETERS (all in mm)
# =============================================================================
wavelength = 790e-6     # mm, 790 nm
k = 2 * np.pi / wavelength

M2x = 1.6
M2y = 1.6
_m2_tag = f'M2x{M2x:.2f}_M2y{M2y:.2f}'
# PHASE_SCREEN_SEED = 42        # no longer used (replaced by HG mode decomposition)
SENSITIVITY_CHECK_MODE = False  # False = run full simulation
LAVG_MM_LIST = [1.0]                 # single pass (Wz = 1 since smoothing off)

# --- Temporal averaging (pulse envelope sampling) ---
USE_TEMPORAL_AVG = False              # disabled — all 3 temporal models tested, none produce plateau
N_TEMPORAL_PTS = 11                  # cumulative-nf temporal slices over full pulse ±1.5σ
PLOT_TEMPORAL_PM_DIAGNOSTIC = True   # plot Δk(z) for each time slice to show PM position clustering
w0x_measured = 6.0   # mm (D4σ measured beam waist)
w0y_measured = 6.0   # mm
w0x = w0x_measured / np.sqrt(M2x)  # fundamental (embedded Gaussian) waist
w0y = w0y_measured / np.sqrt(M2y)

zRx = np.pi * w0x**2 / wavelength  # fundamental Rayleigh range

focal_length = 400.0    # mm
lens_position = 5000.0  # mm

aperture_distance_before_lens = 2000.0
aperture_radius = 7.0
aperture_position = lens_position - aperture_distance_before_lens

mask_type = 'circular'
twosided_halfwidth = 4.3
diag_x_offset = 3.8
diag_block_size = 30.0
diag_y_shift = 13.8

w_at_lens_est = w0x_measured * np.sqrt(1 + (lens_position / zRx)**2)
focus_spot_size = M2x * wavelength * focal_length / (np.pi * w_at_lens_est)
print(f"Estimated focus spot size: {focus_spot_size*1e3:.2f} um")

# Spatial grid
N = 4096
L = 40.0    # mm
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2

# Frequency grid for FFT
fx = np.fft.fftfreq(N, dx)
fy = np.fft.fftfreq(N, dx)
FX, FY = np.meshgrid(fx, fy)
freq_term = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
propagating_mask = freq_term >= 0
evanescent_mask = ~propagating_mask
sqrt_freq_term_prop = np.sqrt(np.where(propagating_mask, freq_term, 0))
sqrt_freq_term_evan = np.sqrt(np.where(evanescent_mask, -freq_term, 0))


# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS
# =============================================================================
if USE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _compute_quadratic_phase_numba(X, Y, k_over_2z, N):
        result = np.zeros((N, N), dtype=np.complex128)
        for i in prange(N):
            for j in range(N):
                r2 = X[i, j]**2 + Y[i, j]**2
                phase = k_over_2z * r2
                result[i, j] = np.cos(phase) + 1j * np.sin(phase)
        return result

    @njit(parallel=True, fastmath=True, cache=True)
    def _thin_lens_phase_numba(field_real, field_imag, R2, k_over_2f, N):
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
        return np.exp(1j * k_over_2z * (X**2 + Y**2))

    def _thin_lens_phase_numba(field_real, field_imag, R2, k_over_2f, N):
        field = field_real + 1j * field_imag
        return field * np.exp(-1j * k_over_2f * R2)

    def _asm_transfer_function_numba(sqrt_prop, sqrt_evan, mask_prop, k_z, N):
        return np.where(mask_prop,
                       np.exp(1j * k_z * sqrt_prop),
                       np.exp(-k_z * sqrt_evan))


# =============================================================================
# CACHING UTILITIES
# =============================================================================
from collections import OrderedDict

_CZT_CACHE_MAX = 5
_QUAD_CACHE_MAX = 3
_czt_cache = OrderedDict()
_quad_phase_cache = OrderedDict()
_cache_stats = {'czt_hits': 0, 'czt_misses': 0, 'quad_hits': 0, 'quad_misses': 0}


def _get_cached_czt_params(L_in, L_out, N_in, N_out, scale):
    global _cache_stats
    cache_key = (L_in, L_out, N_in, N_out, round(scale, 10))
    if cache_key in _czt_cache:
        _cache_stats['czt_hits'] += 1
        _czt_cache.move_to_end(cache_key)
        return _czt_cache[cache_key]
    _cache_stats['czt_misses'] += 1
    while len(_czt_cache) >= _CZT_CACHE_MAX:
        _czt_cache.popitem(last=False)
    dx_in = L_in / (N_in - 1)
    dx_out = L_out / (N_out - 1)
    x_in_0 = -L_in / 2.0
    x_out_0 = -L_out / 2.0
    W = np.exp(-2j * np.pi * dx_out * dx_in / scale)
    A = np.exp(2j * np.pi * x_out_0 * dx_in / scale)
    const = np.exp(-2j * np.pi * x_out_0 * x_in_0 / scale)
    phase_k = np.exp(-2j * np.pi * np.arange(N_out) * dx_out * x_in_0 / scale)
    params = (W, A, const, phase_k)
    _czt_cache[cache_key] = params
    return params

def _get_cached_quad_phase(L, N, k_over_2z, cache_id):
    global _cache_stats
    cache_key = (L, N, round(k_over_2z, 10), cache_id)
    if cache_key in _quad_phase_cache:
        _cache_stats['quad_hits'] += 1
        _quad_phase_cache.move_to_end(cache_key)
        return _quad_phase_cache[cache_key]
    _cache_stats['quad_misses'] += 1
    while len(_quad_phase_cache) >= _QUAD_CACHE_MAX:
        _quad_phase_cache.popitem(last=False)
    coords = np.linspace(-L/2, L/2, N)
    X_grid, Y_grid = np.meshgrid(coords, coords)
    if USE_NUMBA:
        quad = _compute_quadratic_phase_numba(X_grid, Y_grid, k_over_2z, N)
    else:
        quad = np.exp(1j * k_over_2z * (X_grid**2 + Y_grid**2))
    _quad_phase_cache[cache_key] = quad
    return quad


# =============================================================================
# HG MODE DECOMPOSITION FUNCTIONS
# =============================================================================
def _solve_1d_thermal_weights(M2_target, weight_threshold=0.02):
    """
    Find thermal weights for one axis that exactly reconstruct M².

    Uses geometric distribution w_m ∝ r^m with r chosen so that
    sum(w_m * (2m+1)) = M2_target after truncation and renormalization.

    Returns list of (order, weight) tuples.
    """
    from scipy.optimize import brentq

    if M2_target <= 1.0 + 1e-10:
        return [(0, 1.0)]

    def m2_from_r(r, max_m):
        weights = [r**m for m in range(max_m + 1)]
        total = sum(weights)
        return sum((w / total) * (2 * m + 1) for m, w in enumerate(weights))

    # Increase max_m until the mode set can reach the target M²
    for max_m in range(1, 30):
        m2_max = m2_from_r(0.9999, max_m)
        if m2_max >= M2_target:
            r_opt = brentq(lambda r: m2_from_r(r, max_m) - M2_target, 1e-6, 0.9999)
            weights = [r_opt**m for m in range(max_m + 1)]
            total = sum(weights)
            return [(m, w / total) for m, w in enumerate(weights)]
    # Fallback (should not reach here for reasonable M²)
    return [(0, 1.0)]


def compute_hg_decomposition(M2x, M2y, weight_threshold=0.02):
    """
    Compute Hermite-Gaussian mode power fractions for target M² values
    using separable thermal (geometric) distribution with exact M² matching.
    """
    wx = _solve_1d_thermal_weights(M2x, weight_threshold)
    wy = _solve_1d_thermal_weights(M2y, weight_threshold)

    modes = []
    for m, am in wx:
        for n, bn in wy:
            p = am * bn
            if p >= weight_threshold:
                modes.append((m, n, p))

    if not modes:
        modes = [(0, 0, 1.0)]

    total = sum(p for _, _, p in modes)
    modes = [(m, n, p / total) for m, n, p in modes]
    modes.sort(key=lambda x: -x[2])
    return modes


def hermite_poly(order, x):
    """Physicist's Hermite polynomial H_n(x) via recurrence."""
    if order == 0:
        return np.ones_like(x)
    if order == 1:
        return 2.0 * x
    H_prev = np.ones_like(x)
    H_curr = 2.0 * x
    for k_h in range(2, order + 1):
        H_prev, H_curr = H_curr, 2.0 * x * H_curr - 2.0 * (k_h - 1) * H_prev
    return H_curr


def hermite_gaussian_field_z0(X, Y, m, n, w0x, w0y, dx):
    """
    Generate normalized Hermite-Gaussian HG_mn mode at z=0 (beam waist).
    Returns E : 2D complex array, normalized to unit power (sum |E|² * dx² = 1).
    """
    gauss = np.exp(-X**2 / w0x**2 - Y**2 / w0y**2)
    Hx = hermite_poly(m, np.sqrt(2.0) * X / w0x)
    Hy = hermite_poly(n, np.sqrt(2.0) * Y / w0y)
    E = Hx * Hy * gauss
    power = np.sum(np.abs(E)**2) * dx**2
    E = E / np.sqrt(power)
    return E


# =============================================================================
# PROPAGATION FUNCTIONS
# =============================================================================
# --- Old phase screen functions (commented out, kept for reference) ---
# def gaussian_beam_field_astigmatic(X, Y, z, w0x, w0y, wavelength, k, zRx, zRy, M2x, M2y):
#     wx_z = w0x * np.sqrt(1 + (z / zRx)**2)
#     wy_z = w0y * np.sqrt(1 + (z / zRy)**2)
#     if z == 0:
#         curvature_phase_x = 0
#         curvature_phase_y = 0
#     else:
#         Rx_z = z * (1 + (zRx / z)**2)
#         Ry_z = z * (1 + (zRy / z)**2)
#         curvature_phase_x = -k * X**2 / (2 * Rx_z)
#         curvature_phase_y = -k * Y**2 / (2 * Ry_z)
#     gouy_phase_x = 0.5 * np.arctan(z / zRx)
#     gouy_phase_y = 0.5 * np.arctan(z / zRy)
#     gouy_phase = gouy_phase_x + gouy_phase_y
#     prop_phase = -k * z
#     amplitude = np.sqrt(w0x * w0y / (wx_z * wy_z)) * np.exp(-X**2 / wx_z**2 - Y**2 / wy_z**2)
#     phase = prop_phase + curvature_phase_x + curvature_phase_y + gouy_phase
#     return amplitude * np.exp(1j * phase)


# def compute_beam_m2(field, x_1d, dx, verbose=False):
#     """Compute M²_x and M²_y using ISO 11146 second-moment method."""
#     intensity = np.abs(field)**2
#     Ix = intensity.sum(axis=0)
#     Iy = intensity.sum(axis=1)
#     total_x = Ix.sum()
#     x_bar = (x_1d * Ix).sum() / total_x
#     var_x = ((x_1d - x_bar)**2 * Ix).sum() / total_x
#     total_y = Iy.sum()
#     y_bar = (x_1d * Iy).sum() / total_y
#     var_y = ((x_1d - y_bar)**2 * Iy).sum() / total_y
#     field_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
#     int_ft = np.abs(field_ft)**2
#     Ifx = int_ft.sum(axis=0)
#     Ify = int_ft.sum(axis=1)
#     del field_ft, int_ft
#     fx = np.fft.fftshift(np.fft.fftfreq(len(x_1d), dx))
#     total_fx = Ifx.sum()
#     fx_bar = (fx * Ifx).sum() / total_fx
#     var_fx = ((fx - fx_bar)**2 * Ifx).sum() / total_fx
#     total_fy = Ify.sum()
#     fy_bar = (fx * Ify).sum() / total_fy
#     var_fy = ((fx - fy_bar)**2 * Ify).sum() / total_fy
#     M2x_val = 4 * np.pi * np.sqrt(var_x * var_fx)
#     M2y_val = 4 * np.pi * np.sqrt(var_y * var_fy)
#     if verbose:
#         print(f"    Spatial:   sigma_x={np.sqrt(var_x):.4f} mm, sigma_y={np.sqrt(var_y):.4f} mm")
#         print(f"    Frequency: sigma_fx={np.sqrt(var_fx):.4f} mm^-1, sigma_fy={np.sqrt(var_fy):.4f} mm^-1")
#         print(f"    M²x={M2x_val:.4f}, M²y={M2y_val:.4f}")
#     return M2x_val, M2y_val

# def apply_m2_phase_screen(field, x_1d, dx, M2x_target, M2y_target,
#                           w0_ref, seed=42, n_iter=25, tol=0.01):
#     """Apply random phase screen to achieve target M² (preserved for reference)."""
#     pass  # Full implementation commented out — replaced by HG mode decomposition


def focal_spot_diagnostics(field_focus_2d, x_focus, y_focus, label='',
                           save_prefix=None):
    """Compute and plot focal spot diagnostics to check for unrealistic speckle.

    Parameters
    ----------
    field_focus_2d : 2D complex array
        Field at focus from fresnel_propagate_zoom.
    x_focus, y_focus : 1D arrays
        Spatial coordinates at focus (mm).
    label : str
        Label for the beam (e.g., 'M²=2.37 beam').
    save_prefix : str or None
        If provided, save the figure to '{save_prefix}_focal_diag.png'.

    Returns
    -------
    metrics : dict
        Dictionary with peak_intensity, fwhm_x_um, fwhm_y_um,
        encircled_energy, strehl_ratio.
    """
    dx_f = x_focus[1] - x_focus[0]
    dy_f = y_focus[1] - y_focus[0]
    N_f = len(x_focus)
    intensity = np.abs(field_focus_2d)**2
    center = N_f // 2

    # Peak intensity and location
    peak_val = intensity.max()
    peak_idx = np.unravel_index(intensity.argmax(), intensity.shape)
    peak_x = x_focus[peak_idx[1]] * 1e3  # um
    peak_y = y_focus[peak_idx[0]] * 1e3  # um

    # Marginal profiles
    Ix_marginal = intensity[center, :]
    Iy_marginal = intensity[:, center]
    Ix_norm = Ix_marginal / Ix_marginal.max()
    Iy_norm = Iy_marginal / Iy_marginal.max()

    # FWHM from marginals
    def _fwhm(profile, coord_mm):
        above = profile >= 0.5
        if above.sum() < 2:
            return 0.0
        idx = np.where(above)[0]
        return (coord_mm[idx[-1]] - coord_mm[idx[0]]) * 1e3  # um

    fwhm_x = _fwhm(Ix_norm, x_focus)
    fwhm_y = _fwhm(Iy_norm, y_focus)

    # Encircled energy within 2x FWHM
    r_encircle = max(fwhm_x, fwhm_y) * 2 * 1e-3  # back to mm
    X_f, Y_f = np.meshgrid(x_focus, y_focus)
    R_f = np.sqrt((X_f - x_focus[peak_idx[1]])**2 + (Y_f - y_focus[peak_idx[0]])**2)
    total_power = intensity.sum()
    encircled = intensity[R_f <= r_encircle].sum() / total_power

    # Strehl ratio: peak / (ideal Gaussian peak with same total power)
    # For ideal Gaussian: I_peak = 2*P_total / (pi*w0²) where w0 ≈ FWHM/(2*sqrt(ln2))
    if fwhm_x > 0 and fwhm_y > 0:
        w0_ideal_x = fwhm_x * 1e-3 / (2 * np.sqrt(np.log(2)))  # mm
        w0_ideal_y = fwhm_y * 1e-3 / (2 * np.sqrt(np.log(2)))
        ideal_peak = 2 * total_power * dx_f * dy_f / (np.pi * w0_ideal_x * w0_ideal_y)
        strehl = peak_val / ideal_peak if ideal_peak > 0 else 0.0
    else:
        strehl = 0.0

    metrics = {
        'peak_intensity': peak_val,
        'fwhm_x_um': fwhm_x,
        'fwhm_y_um': fwhm_y,
        'encircled_energy': encircled,
        'strehl_ratio': strehl,
        'peak_x_um': peak_x,
        'peak_y_um': peak_y,
    }

    # --- Generate 2x2 diagnostic figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Focal Spot Diagnostics: {label}', fontsize=15)

    x_um = x_focus * 1e3
    y_um = y_focus * 1e3
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

    # Top-left: log10 intensity
    ax = axes[0, 0]
    I_log = np.log10(intensity / peak_val + 1e-20)
    im = ax.imshow(I_log, extent=extent, origin='lower', cmap='inferno',
                   vmin=-4, vmax=0, aspect='equal')
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    ax.set_title('log10(I/I_peak)')
    plt.colorbar(im, ax=ax)

    # Top-right: x and y marginal profiles with Gaussian overlay
    ax = axes[0, 1]
    ax.plot(x_um, Ix_norm, '-', color=COLOR_SIM, label='x-profile')
    ax.plot(y_um, Iy_norm, '-', color=COLOR_EXP, label='y-profile')
    if fwhm_x > 0:
        w0_fit = fwhm_x / (2 * np.sqrt(np.log(2)))
        gauss_ideal = np.exp(-x_um**2 / w0_fit**2)
        ax.plot(x_um, gauss_ideal, 'k--', alpha=0.5, label='Gaussian fit')
    ax.set_xlabel(r'Position ($\mu$m)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Marginal profiles')
    ax.legend()
    ax.set_xlim(-fwhm_x*3 if fwhm_x > 0 else x_um[0],
                fwhm_x*3 if fwhm_x > 0 else x_um[-1])

    # Bottom-left: 2D spatial power spectrum
    ax = axes[1, 0]
    I_ft = np.abs(np.fft.fftshift(np.fft.fft2(intensity)))**2
    I_ft_log = np.log10(I_ft / I_ft.max() + 1e-20)
    ax.imshow(I_ft_log, origin='lower', cmap='viridis', vmin=-6, vmax=0,
              aspect='equal')
    ax.set_title('Spatial power spectrum (log10)')
    ax.set_xlabel('fx')
    ax.set_ylabel('fy')

    # Bottom-right: azimuthally averaged radial profile
    ax = axes[1, 1]
    r_px = np.sqrt((np.arange(N_f) - center)**2)
    r_mm = r_px * dx_f
    # Azimuthal average
    r_grid = np.sqrt((X_f - x_focus[center])**2 + (Y_f - y_focus[center])**2)
    r_bins = np.linspace(0, r_mm.max() * 0.5, 100)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    radial_avg = np.zeros(len(r_centers))
    for ib in range(len(r_centers)):
        mask = (r_grid >= r_bins[ib]) & (r_grid < r_bins[ib+1])
        if mask.sum() > 0:
            radial_avg[ib] = intensity[mask].mean()
    radial_avg /= radial_avg.max() if radial_avg.max() > 0 else 1.0
    ax.semilogy(r_centers * 1e3, radial_avg, '-', color=COLOR_SIM)
    if fwhm_x > 0:
        w0_fit_mm = fwhm_x * 1e-3 / (2 * np.sqrt(np.log(2)))
        gauss_radial = np.exp(-2 * (r_centers / w0_fit_mm)**2)
        ax.semilogy(r_centers * 1e3, gauss_radial, 'k--', alpha=0.5, label='Gaussian')
    ax.set_xlabel(r'Radius ($\mu$m)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Azimuthal radial profile')
    ax.legend()
    ax.set_ylim(1e-4, 2)

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}_focal_diag.png')
        print(f"  Saved focal diagnostics to {save_prefix}_focal_diag.png")
    plt.show()

    return metrics


def angular_spectrum_propagate(field, z, k):
    if z == 0:
        return field.copy()
    if USE_NUMBA:
        H = _asm_transfer_function_numba(sqrt_freq_term_prop, sqrt_freq_term_evan,
                                         propagating_mask, k * z, N)
    else:
        H = np.where(propagating_mask,
                     np.exp(1j * k * z * sqrt_freq_term_prop),
                     np.exp(-k * z * sqrt_freq_term_evan))
    if USE_SCIPY_FFT:
        field_fft = scipy_fft.fft2(field, workers=-1)
        propagated_field = scipy_fft.ifft2(field_fft * H, workers=-1)
    else:
        field_fft = np.fft.fft2(field)
        propagated_field = np.fft.ifft2(field_fft * H)
    return propagated_field


def fresnel_propagate_zoom(field_in, z, k, L_in, L_out, N_out=None):
    N_in = field_in.shape[0]
    if N_out is None:
        N_out = N_in
    dx_in = L_in / N_in
    x_out = np.linspace(-L_out/2, L_out/2, N_out)
    y_out = np.linspace(-L_out/2, L_out/2, N_out)
    k_over_2z = k / (2 * z)
    scale = wavelength * z
    W, A, const, phase_k = _get_cached_czt_params(L_in, L_out, N_in, N_out, scale)
    quad_in = _get_cached_quad_phase(L_in, N_in, k_over_2z, 'in')
    quad_out = _get_cached_quad_phase(L_out, N_out, k_over_2z, 'out')
    field_quad = field_in * quad_in
    temp = scipy_czt(field_quad, m=N_out, w=W, a=A)
    temp *= const * phase_k[np.newaxis, :]
    field_out = scipy_czt(temp.T, m=N_out, w=W, a=A).T
    field_out *= const * phase_k[:, np.newaxis]
    prefactor = np.exp(1j * k * z) / (1j * wavelength * z)
    field_out = prefactor * quad_out * field_out * (dx_in ** 2)
    return field_out, x_out, y_out


def propagate_field(field, z, k, method=None):
    if z == 0:
        return field.copy()
    return angular_spectrum_propagate(field, z, k)


def thin_lens(field, R2, f, k):
    if USE_NUMBA:
        N_field = field.shape[0]
        k_over_2f = k / (2 * f)
        return _thin_lens_phase_numba(field.real, field.imag, R2, k_over_2f, N_field)
    else:
        return field * np.exp(-1j * k * R2 / (2 * f))


# =============================================================================
# BEAM INITIALIZATION AND PROPAGATION
# =============================================================================
TIMER.start_section("Grid and beam initialization")
center_idx = N // 2

# HG mode decomposition for M² beam
hg_modes = compute_hg_decomposition(M2x, M2y)
n_hg_modes = len(hg_modes)
print(f"HG decomposition: {n_hg_modes} modes for M²x={M2x:.2f}, M²y={M2y:.2f}")
for m_mode, n_mode, p_mode in hg_modes:
    print(f"  HG({m_mode},{n_mode}): weight = {p_mode:.4f}")

mode_fields_z0 = []
for (m_mode, n_mode, p_mn) in hg_modes:
    E_mn = hermite_gaussian_field_z0(X, Y, m_mode, n_mode, w0x, w0y, dx)
    mode_fields_z0.append((m_mode, n_mode, p_mn, E_mn))

TIMER.start_section("Beam propagation (waist -> lens)")

# Propagate each mode to aperture
print(f"Propagating {n_hg_modes} HG modes from waist to aperture (z = {aperture_position:.0f} mm)...")
mode_fields_at_aperture = []
for (m_mode, n_mode, p_mn, E_mn) in mode_fields_z0:
    f_at_ap = propagate_field(E_mn, aperture_position, k)
    mode_fields_at_aperture.append((m_mode, n_mode, p_mn, f_at_ap))

# Apply mask
def build_mask(X, Y, mtype, params):
    if mtype == 'circular':
        return np.where(np.sqrt(X**2 + Y**2) <= params['aperture_radius'], 1.0, 0.0)
    elif mtype == 'twosided':
        return np.where(np.abs(X) <= params['twosided_halfwidth'], 1.0, 0.0)
    elif mtype == 'diagonal':
        xo = params['diag_x_offset']
        bs = params['diag_block_size']
        ys = params['diag_y_shift']
        # Block 1: left side, shifted up (away from center)
        block1 = (X >= -xo - bs) & (X <= -xo) & (Y >= ys - bs/2) & (Y <= ys + bs/2)
        # Block 2: right side, shifted down (away from center)
        block2 = (X >= xo) & (X <= xo + bs) & (Y >= -ys - bs/2) & (Y <= -ys + bs/2)
        return np.where(block1 | block2, 0.0, 1.0)
    else:
        return np.ones_like(X)

from scipy.optimize import brentq

mask_params = {
    'aperture_radius': aperture_radius,
    'twosided_halfwidth': twosided_halfwidth,
    'diag_x_offset': diag_x_offset,
    'diag_block_size': diag_block_size,
    'diag_y_shift': diag_y_shift,
}
aperture_mask = build_mask(X, Y, mask_type, mask_params)

# Per-mode mask application, propagation to lens, and thin lens
# Blocked path: each mode goes through aperture + mask + lens
mode_fields_after_lens_b = []
power_before_aperture = 0.0
power_after_aperture = 0.0
for (m_mode, n_mode, p_mn, f_at_ap) in mode_fields_at_aperture:
    pwr_mode = np.sum(np.abs(f_at_ap)**2)
    f_masked = f_at_ap * aperture_mask
    pwr_masked = np.sum(np.abs(f_masked)**2)
    power_before_aperture += p_mn * pwr_mode
    power_after_aperture += p_mn * pwr_masked
    f_at_lens = propagate_field(f_masked, aperture_distance_before_lens, k)
    f_after_lens_mode = thin_lens(f_at_lens, R2, focal_length, k)
    mode_fields_after_lens_b.append((m_mode, n_mode, p_mn, f_after_lens_mode))

aperture_transmission = power_after_aperture / power_before_aperture * 100
print(f"Mask type: {mask_type}, Transmission: {aperture_transmission:.2f}%")

# Unblocked path: each mode goes directly to lens (no mask)
TIMER.start_section("Unblocked beam computation")
print(f"Computing unblocked beam ({n_hg_modes} modes, no aperture)...")
mode_fields_after_lens_ub = []
for (m_mode, n_mode, p_mn, E_mn) in mode_fields_z0:
    f_at_lens = propagate_field(E_mn, lens_position, k)
    f_after_lens_mode = thin_lens(f_at_lens, R2, focal_length, k)
    mode_fields_after_lens_ub.append((m_mode, n_mode, p_mn, f_after_lens_mode))

# Keep HG00 field_after_lens for backward compatibility (focus finding, Rayleigh range)
field_after_lens = mode_fields_after_lens_b[0][3]
field_after_lens_unblocked = mode_fields_after_lens_ub[0][3]

# Find true focus
def find_true_focus(field_after_lens, z_search, k, center_idx, dx):
    best_z = z_search[0]
    min_phase_std = np.inf
    for z in z_search:
        field_z = propagate_field(field_after_lens, z, k)
        intensity = np.abs(field_z[center_idx, :])**2
        phase = np.angle(field_z[center_idx, :])
        mask = intensity > 0.01 * intensity.max()
        if np.sum(mask) > 5:
            phase_masked = phase[mask]
            phase_unwrapped = np.unwrap(phase_masked)
            x_masked = np.arange(len(phase_unwrapped))
            if len(x_masked) > 2:
                coeffs = np.polyfit(x_masked, phase_unwrapped, 1)
                phase_detrended = phase_unwrapped - np.polyval(coeffs, x_masked)
                phase_std = np.std(phase_detrended)
                if phase_std < min_phase_std:
                    min_phase_std = phase_std
                    best_z = z
    return best_z, min_phase_std

print("Searching for true focus position...")
z_search = np.linspace(focal_length * 0.95, focal_length * 1.05, 50)
true_focus_z = find_true_focus(field_after_lens, z_search, k, center_idx, dx)[0]
print(f"Geometric focus (f): {focal_length:.2f} mm")
print(f"True focus found at: {true_focus_z:.2f} mm")

print("Searching for unblocked beam focus position...")
true_focus_z_unblocked, _ = find_true_focus(field_after_lens_unblocked, z_search, k, center_idx, dx)
print(f"Unblocked beam focus at: {true_focus_z_unblocked:.2f} mm")

# Rayleigh range calculation
TIMER.start_section("Rayleigh range calculation")

def calculate_rayleigh_range(field_after_lens, focus_z, k, L, N_calc=512, L_calc=1.0, z_range=10.0, n_z=50):
    dx_calc = L_calc / N_calc
    field_focus, x_focus, y_focus = fresnel_propagate_zoom(field_after_lens, focus_z, k, L, L_calc, N_calc)
    I_focus = np.abs(field_focus)**2
    I_focus_norm = I_focus / I_focus.max()
    center_calc = N_calc // 2
    I_x_focus = I_focus_norm[center_calc, :]
    above_e2 = I_x_focus > 1/np.e**2
    w0_focus = np.sum(above_e2) * dx_calc / 2
    I_peak_focus = I_focus.max()
    z_test_array = np.linspace(focus_z, focus_z + z_range, n_z)
    rayleigh_z = None
    for z_test in z_test_array[1:]:
        field_test, _, _ = fresnel_propagate_zoom(field_after_lens, z_test, k, L, L_calc, N_calc)
        I_test = np.abs(field_test)**2
        I_test_norm = I_test / I_test.max()
        I_x_test = I_test_norm[center_calc, :]
        above_e2_test = I_x_test > 1/np.e**2
        w_test = np.sum(above_e2_test) * dx_calc / 2
        if w_test >= np.sqrt(2) * w0_focus:
            rayleigh_z = z_test - focus_z
            break
    return w0_focus, rayleigh_z, I_peak_focus

print("  Computing blocked beam Rayleigh range...")
w0_blocked, zR_blocked, I_peak_blocked = calculate_rayleigh_range(
    field_after_lens, true_focus_z, k, L, N_calc=512, L_calc=1.0, z_range=20.0, n_z=100
)
print("  Computing unblocked beam Rayleigh range...")
w0_unblocked, zR_unblocked, I_peak_unblocked = calculate_rayleigh_range(
    field_after_lens_unblocked, true_focus_z_unblocked, k, L, N_calc=512, L_calc=1.0, z_range=20.0, n_z=100
)
print(f"  Blocked:   w0={w0_blocked*1e3:.2f} um, zR={zR_blocked:.2f} mm" if zR_blocked else f"  Blocked:   w0={w0_blocked*1e3:.2f} um, zR=N/A")
print(f"  Unblocked: w0={w0_unblocked*1e3:.2f} um, zR={zR_unblocked:.2f} mm" if zR_unblocked else f"  Unblocked: w0={w0_unblocked*1e3:.2f} um, zR=N/A")
print(f"  Intensity ratio (blocked/unblocked): {I_peak_blocked/I_peak_unblocked:.4f}")

# --- Focal spot diagnostics ---
TIMER.start_section("Focal spot diagnostics")
print("Running focal spot diagnostics...")
N_diag = 512
L_diag = 0.5  # mm

# Unblocked beam — multi-mode incoherent sum at focus
I_focus_ub = None
for idx_m, (m_m, n_m, p_mn, f_lens_m) in enumerate(mode_fields_after_lens_ub):
    ff, x_focus_ub, y_focus_ub = fresnel_propagate_zoom(f_lens_m, true_focus_z_unblocked, k, L, L_diag, N_diag)
    I_mode = p_mn * np.abs(ff)**2
    if I_focus_ub is None:
        I_focus_ub = I_mode
        field_focus_ub_hg00 = ff  # keep HG00 for phase
    else:
        I_focus_ub += I_mode
# Create pseudo-field for diagnostics (sqrt of incoherent intensity)
field_focus_ub_pseudo = np.sqrt(I_focus_ub)
metrics_ub = focal_spot_diagnostics(
    field_focus_ub_pseudo, x_focus_ub, y_focus_ub,
    label=f'Unblocked (M²x={M2x:.2f}, M²y={M2y:.2f})',
    save_prefix=f'unblocked_{_m2_tag}'
)
del I_focus_ub, field_focus_ub_pseudo, field_focus_ub_hg00

# Blocked beam — multi-mode incoherent sum at focus
I_focus_b = None
for idx_m, (m_m, n_m, p_mn, f_lens_m) in enumerate(mode_fields_after_lens_b):
    ff, x_focus_b, y_focus_b = fresnel_propagate_zoom(f_lens_m, true_focus_z, k, L, L_diag, N_diag)
    I_mode = p_mn * np.abs(ff)**2
    if I_focus_b is None:
        I_focus_b = I_mode
    else:
        I_focus_b += I_mode
field_focus_b_pseudo = np.sqrt(I_focus_b)
metrics_b = focal_spot_diagnostics(
    field_focus_b_pseudo, x_focus_b, y_focus_b,
    label=f'Blocked (M²x={M2x:.2f}, M²y={M2y:.2f})',
    save_prefix=f'blocked_{_m2_tag}'
)
del I_focus_b, field_focus_b_pseudo

# Print comparison table
print("\n  Focal Spot Diagnostics Comparison:")
print(f"  {'':25s} {'Unblocked':>14s} {'Blocked':>14s}")
print(f"  {'FWHM_x (um)':25s} {metrics_ub['fwhm_x_um']:14.1f} {metrics_b['fwhm_x_um']:14.1f}")
print(f"  {'FWHM_y (um)':25s} {metrics_ub['fwhm_y_um']:14.1f} {metrics_b['fwhm_y_um']:14.1f}")
print(f"  {'Encircled energy':25s} {metrics_ub['encircled_energy']:13.1%} {metrics_b['encircled_energy']:13.1%}")
print(f"  {'Strehl ratio':25s} {metrics_ub['strehl_ratio']:14.3f} {metrics_b['strehl_ratio']:14.3f}")
print()


# =============================================================================
# HIGH-RESOLUTION PROPAGATION + GAS REGION DATA EXTRACTION
# =============================================================================
TIMER.start_section("High-res propagation + gas region extraction")
print("Computing high-resolution x-z propagation near focus...")

n_z_steps_hr = 100
z_focus_prop = np.linspace(focal_length - 1.0, focal_length + 1.0, n_z_steps_hr)

L_xz_focus = 0.5   # mm
N_xz_focus = 1024
x_xz_hires = np.linspace(-L_xz_focus/2, L_xz_focus/2, N_xz_focus)

# 2D HHG grid (cropped from high-res)
N_hhg_2d = 512
hhg_gas_length_prop = 1.0  # mm
hhg_crop = slice(N_xz_focus // 2 - N_hhg_2d // 2, N_xz_focus // 2 + N_hhg_2d // 2)
x_hhg_2d = x_xz_hires[hhg_crop]
gas_z_start_prop = focal_length - hhg_gas_length_prop / 2.0
gas_z_end_prop = focal_length + hhg_gas_length_prop / 2.0

# Per-mode storage for gas region
gas_modes_b = [{'I_list': [], 'phase_list': [], 'z_list': []} for _ in range(n_hg_modes)]
gas_modes_ub = [{'I_list': [], 'phase_list': [], 'z_list': []} for _ in range(n_hg_modes)]

print(f"  2D HHG grid: {N_hhg_2d}x{N_hhg_2d}, gas region {gas_z_start_prop:.1f}-{gas_z_end_prop:.1f} mm")

# Arrays for on-axis diagnostics
xz_intensity_hires = np.zeros((n_z_steps_hr, N_xz_focus))
xz_gouy_phase_hires = np.zeros((n_z_steps_hr, N_xz_focus))
xz_intensity_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))
xz_gouy_phase_unblocked = np.zeros((n_z_steps_hr, N_xz_focus))

# Identify which z-steps fall in gas region (for higher-order mode optimization)
gas_z_indices_b = []
gas_z_indices_ub = []

# Pass 1: HG00 at ALL z-steps (diagnostics + gas region)
p0_b = mode_fields_after_lens_b[0][2]
p0_ub = mode_fields_after_lens_ub[0][2]
center_hr = N_xz_focus // 2

print(f"  Pass 1: HG00 mode at all {n_z_steps_hr} z-steps...")
for i, z in enumerate(z_focus_prop):
    if i % 40 == 0:
        print(f"    z = {z:.2f} mm ({i+1}/{n_z_steps_hr})")
    # Blocked HG00
    field_z_b, x_out, y_out = fresnel_propagate_zoom(mode_fields_after_lens_b[0][3], z, k, L, L_xz_focus, N_xz_focus)
    field_no_pw_b = field_z_b * np.exp(-1j * k * z)
    xz_intensity_hires[i, :] = p0_b * np.abs(field_z_b[center_hr, :])**2
    xz_gouy_phase_hires[i, :] = np.angle(field_no_pw_b[center_hr, :])
    # Unblocked HG00
    field_z_ub, _, _ = fresnel_propagate_zoom(mode_fields_after_lens_ub[0][3], z, k, L, L_xz_focus, N_xz_focus)
    field_no_pw_ub = field_z_ub * np.exp(-1j * k * z)
    xz_intensity_unblocked[i, :] = p0_ub * np.abs(field_z_ub[center_hr, :])**2
    xz_gouy_phase_unblocked[i, :] = np.angle(field_no_pw_ub[center_hr, :])

    if gas_z_start_prop <= z <= gas_z_end_prop:
        gas_z_indices_b.append(i)
        gas_z_indices_ub.append(i)
        gas_modes_b[0]['I_list'].append(p0_b * np.abs(field_z_b[hhg_crop, hhg_crop])**2)
        gas_modes_b[0]['phase_list'].append(np.angle(field_no_pw_b[hhg_crop, hhg_crop]))
        gas_modes_b[0]['z_list'].append(z)
        gas_modes_ub[0]['I_list'].append(p0_ub * np.abs(field_z_ub[hhg_crop, hhg_crop])**2)
        gas_modes_ub[0]['phase_list'].append(np.angle(field_no_pw_ub[hhg_crop, hhg_crop]))
        gas_modes_ub[0]['z_list'].append(z)

# Pass 2: Higher-order modes ONLY at gas z-positions
if n_hg_modes > 1:
    print(f"  Pass 2: {n_hg_modes - 1} higher-order modes at {len(gas_z_indices_b)} gas z-steps...")
    for mode_idx in range(1, n_hg_modes):
        m_m, n_m, p_mn, f_lens_b = mode_fields_after_lens_b[mode_idx]
        _, _, _, f_lens_ub = mode_fields_after_lens_ub[mode_idx]
        for gi, i in enumerate(gas_z_indices_b):
            z = z_focus_prop[i]
            # Blocked
            fz_b, _, _ = fresnel_propagate_zoom(f_lens_b, z, k, L, L_xz_focus, N_xz_focus)
            xz_intensity_hires[i, :] += p_mn * np.abs(fz_b[center_hr, :])**2
            gas_modes_b[mode_idx]['I_list'].append(p_mn * np.abs(fz_b[hhg_crop, hhg_crop])**2)
            field_no_pw_b_ho = fz_b * np.exp(-1j * k * z)
            gas_modes_b[mode_idx]['phase_list'].append(np.angle(field_no_pw_b_ho[hhg_crop, hhg_crop]))
            gas_modes_b[mode_idx]['z_list'].append(z)
            # Unblocked
            fz_ub, _, _ = fresnel_propagate_zoom(f_lens_ub, z, k, L, L_xz_focus, N_xz_focus)
            xz_intensity_unblocked[i, :] += p_mn * np.abs(fz_ub[center_hr, :])**2
            gas_modes_ub[mode_idx]['I_list'].append(p_mn * np.abs(fz_ub[hhg_crop, hhg_crop])**2)
            field_no_pw_ub_ho = fz_ub * np.exp(-1j * k * z)
            gas_modes_ub[mode_idx]['phase_list'].append(np.angle(field_no_pw_ub_ho[hhg_crop, hhg_crop]))
            gas_modes_ub[mode_idx]['z_list'].append(z)

# Per-mode data arrays (keep separate for per-mode HHG computation)
z_gas_2d_b = np.array(gas_modes_b[0]['z_list'])
z_gas_2d_ub = np.array(gas_modes_ub[0]['z_list'])

modes_I_phase_b = []
modes_I_phase_ub = []
for mode_idx in range(n_hg_modes):
    I_m_b = np.array(gas_modes_b[mode_idx]['I_list'])
    ph_m_b = np.array(gas_modes_b[mode_idx]['phase_list'])
    modes_I_phase_b.append((I_m_b, ph_m_b))
    I_m_ub = np.array(gas_modes_ub[mode_idx]['I_list'])
    ph_m_ub = np.array(gas_modes_ub[mode_idx]['phase_list'])
    modes_I_phase_ub.append((I_m_ub, ph_m_ub))

# Total intensity for diagnostics and thresholding
I_2d_gas_b = sum(I_m for I_m, _ in modes_I_phase_b)
I_2d_gas_ub = sum(I_m for I_m, _ in modes_I_phase_ub)

print(f"  Stored blocked beam data: {I_2d_gas_b.shape} ({I_2d_gas_b.nbytes/1e6:.0f} MB)")
print(f"  Stored unblocked beam data: {I_2d_gas_ub.shape} ({I_2d_gas_ub.nbytes/1e6:.0f} MB)")


# =============================================================================
# HHG PARAMETERS AND EXPERIMENTAL DATA
# =============================================================================
TIMER.start_section("HHG setup")

hhg_harmonic_order = 21
hhg_gas_pressure = 125.0        # mbar
hhg_gas_type = 'argon'
hhg_peak_intensity_Wcm2 = 2.0e14  # W/cm^2
pulse_fwhm_fs = 55.0              # FWHM of Gaussian pulse envelope (fs)

# Experimental H21 yield vs intensity
exp_yield_I_Wcm2 = np.array([1.74, 2.04, 2.31, 2.37, 2.55, 2.67, 2.85]) * 1e14

# Multi-harmonic unblocked yield (7 intensity points, low -> high)
exp_yield_multi = {
    11: np.array([1254.17, 1917.40, 2071.34, 2138.20, 2251.15, 2282.78, 2308.66]),
    13: np.array([1307.14, 2683.21, 3163.65, 3527.96, 4192.40, 4468.43, 4304.46]),
    15: np.array([918.94, 1413.24, 1586.04, 1668.70, 1741.57, 1808.46, 1887.21]),
    17: np.array([715.44, 1113.52, 1206.96, 1277.79, 1387.12, 1442.31, 1522.66]),
    19: np.array([433.74, 728.48, 796.15, 789.09, 932.30, 1007.91, 973.39]),
    21: np.array([255.50, 424.63, 464.11, 486.05, 448.98, 472.43, 506.14]),
}

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

APERTURE_SUBSAMPLE_STEP = 2  # use every 2nd aperture for fitting
ap_indices = list(range(0, len(exp_aperture_T), APERTURE_SUBSAMPLE_STEP))  # [0,2,4,6,8,10,12]
n_ap_sub = len(ap_indices)  # 7
ap_ref_idx = 0  # normalize to least-blocked aperture
lambda_ap = 1.0  # weight for aperture cost relative to power cost

# Joint optimization hyperparameters
lambda_smooth_alpha = 0.3   # 2nd-order difference penalty on alpha
lambda_smooth_Is = 0.5      # 2nd-order difference penalty on log(Is)
lambda_prior = 0.1          # soft Is prior weight
log_Is_prior_center = np.log(3e13)  # SFA deconv median
log_Is_prior_width = 1.5            # ~1 order of magnitude
# Weak alpha soft-box prior (range is free; only outside is penalized)
lambda_alpha_box = 0.2
alpha_box_min = 2.0
alpha_box_max = 6.0
sigma_alpha_box = 1.5
lambda_alpha_prior = 0.0    # deprecated center prior, disabled
alpha_prior_center = 4.0
alpha_prior_width = 2.0

def alpha_soft_box_penalty(alpha_val):
    below = np.maximum((alpha_box_min - alpha_val) / sigma_alpha_box, 0.0)
    above = np.maximum((alpha_val - alpha_box_max) / sigma_alpha_box, 0.0)
    return lambda_alpha_box * (below**2 + above**2)

# Parameter bounds (hard)
ALPHA_MIN, ALPHA_MAX = 1.0, 8.0     # α range (restored)
LOG_IS_MIN, LOG_IS_MAX = None, None  # set from Is_grid after parameter grid definition


# XUV absorption cross-sections (Mb) per harmonic in argon
sigma_xuv_multi_Mb = {
    11: 33.0,   # 73 nm, 17.1 eV
    13: 30.0,   # 62 nm, 20.2 eV
    15: 27.0,   # 53 nm, 23.3 eV
    17: 23.0,   # 47 nm, 26.4 eV (Cooper minimum is at ~48 eV, not here)
    19: 20.0,   # 42 nm, 29.5 eV
    21: 17.0,   # 38 nm, 32.6 eV
}

# Experimental enhancement data (blocked/unblocked yield ratio)
exp_intensities_1e14 = np.array([950, 890, 850, 790, 770, 680, 580]) * 3 / 1000
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
                     1.3884154432912577]),
}


# =============================================================================
# HHG HELPER FUNCTIONS
# =============================================================================
def get_gas_properties(gas_type):
    gas_data = {
        'argon':   {'delta_n': 2.8e-4, 'Ip_eV': 15.76, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
        'neon':    {'delta_n': 6.6e-5, 'Ip_eV': 21.56, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
        'helium':  {'delta_n': 3.5e-5, 'Ip_eV': 24.59, 'N_atm': 2.65e25, 'Z': 1, 'l': 0, 'm': 0, 'alpha_tl': 7.0},
        'krypton': {'delta_n': 4.2e-4, 'Ip_eV': 14.00, 'N_atm': 2.65e25, 'Z': 1, 'l': 1, 'm': 0, 'alpha_tl': 9.0},
    }
    if gas_type.lower() not in gas_data:
        raise ValueError(f"Unknown gas type: {gas_type}")
    return gas_data[gas_type.lower()]

def ionization_fraction_bsi(I_Wcm2, Ip_eV):
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
    # Time grid: ±5*FWHM/2, step = T_cycle/20
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

def calc_dk_neutral(q, P_mbar, lambda_0_m, n_f, delta_n_per_bar):
    P_bar = P_mbar / 1000.0
    return (2 * np.pi * q / lambda_0_m) * P_bar * (1 - n_f) * delta_n_per_bar

def calc_dk_plasma(q, P_mbar, lambda_0_m, n_f, N_atm):
    r_e = 2.8179403227e-15
    P_bar = P_mbar / 1000.0
    N_e = n_f * N_atm * P_bar
    return -(q - 1.0 / q) * N_e * r_e * lambda_0_m


# =============================================================================
# UNIT CONVERSIONS AND CALIBRATION
# =============================================================================
lambda_0_m = wavelength * 1e-3   # 790e-9 m

gas = get_gas_properties(hhg_gas_type)
print(f"Gas: {hhg_gas_type}, Ip = {gas['Ip_eV']:.2f} eV")
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
I_2d_gas_ub_Wcm2 = I_2d_gas_ub * I_scale_factor_2d

# Per-mode intensity in physical units
modes_I_Wcm2_b = [(I_m * I_scale_factor_2d) for I_m, _ in modes_I_phase_b]
modes_I_Wcm2_ub = [(I_m * I_scale_factor_2d) for I_m, _ in modes_I_phase_ub]

dx_hhg_m = (x_hhg_2d[1] - x_hhg_2d[0]) * 1e-3  # mm -> m

# Per-mode Gouy phase gradient (complex-domain method, robust against wrapping)
dz_arr_b = np.diff(z_gas_2d_b_m)
dz_arr_ub = np.diff(z_gas_2d_ub_m)

modes_dphase_b = []
modes_dphase_ub = []
for mode_idx in range(n_hg_modes):
    I_m_b, ph_m_b = modes_I_phase_b[mode_idx]
    I_m_ub, ph_m_ub = modes_I_phase_ub[mode_idx]

    # Blocked
    env_b = np.sqrt(I_m_b) * np.exp(1j * ph_m_b)
    dph_b = np.zeros_like(ph_m_b)
    for j in range(1, len(z_gas_2d_b_m)):
        dph_b[j] = np.angle(env_b[j] * np.conj(env_b[j-1])) / dz_arr_b[j-1]
    dph_b[0] = dph_b[1]
    I_thresh_b = 0.01 * I_m_b.max()
    dph_b[I_m_b < I_thresh_b] = 0.0
    modes_dphase_b.append(dph_b)

    # Unblocked
    env_ub = np.sqrt(I_m_ub) * np.exp(1j * ph_m_ub)
    dph_ub = np.zeros_like(ph_m_ub)
    for j in range(1, len(z_gas_2d_ub_m)):
        dph_ub[j] = np.angle(env_ub[j] * np.conj(env_ub[j-1])) / dz_arr_ub[j-1]
    dph_ub[0] = dph_ub[1]
    I_thresh_ub = 0.01 * I_m_ub.max()
    dph_ub[I_m_ub < I_thresh_ub] = 0.0
    modes_dphase_ub.append(dph_ub)

del env_b, env_ub
gc.collect()

# Keep HG00 phase gradient for aperture reuse (backward compat)
dphase_dz_3d_ub = modes_dphase_ub[0]

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

# Compute incoherent intensity at aperture (for transmission calculation)
I_at_aperture = sum(p_mn * np.abs(f)**2 for (_, _, p_mn, f) in mode_fields_at_aperture)
power_total_at_aperture = np.sum(I_at_aperture)
R_grid_ap = np.sqrt(X**2 + Y**2)
gas_data_ap = {}  # gas_data_ap[i_ap] = {modes_I_Wcm2, modes_dphase, z_gas_m, I_total_Wcm2}

for i_sub, i_ap in enumerate(ap_indices):
    T = exp_aperture_T[i_ap]
    print(f"\n  Aperture {i_sub+1}/{n_ap_sub}: iris={exp_aperture_x[i_ap]:.1f}, T={T:.3f}")

    if T > 0.995:
        print("    T > 0.995 — reusing unblocked beam data")
        gas_data_ap[i_ap] = {
            'modes_I_Wcm2': modes_I_Wcm2_ub,
            'modes_dphase': modes_dphase_ub,
            'z_gas_m': z_gas_2d_ub_m,
            'I_total_Wcm2': I_2d_gas_ub_Wcm2,
        }
        continue

    # Find aperture radius for this transmission (intensity-based for multi-mode)
    def _T_residual_ap(r):
        mask = np.where(R_grid_ap <= r, 1.0, 0.0)
        return np.sum(I_at_aperture * mask) / power_total_at_aperture - T
    r_ap = brentq(_T_residual_ap, 0.5, 19.0)
    print(f"    Aperture radius = {r_ap:.3f} mm")

    # Apply circular iris mask
    mask_ap = np.where(R_grid_ap <= r_ap, 1.0, 0.0)

    T_check = np.sum(I_at_aperture * mask_ap) / power_total_at_aperture
    print(f"    Transmission check: {T_check:.4f} (target {T:.4f})")

    # Per-mode propagation through iris → lens
    mode_fields_after_lens_ap = []
    for (m_mode, n_mode, p_mn, f_at_ap) in mode_fields_at_aperture:
        f_masked = f_at_ap * mask_ap
        f_at_lens_ap = propagate_field(f_masked, aperture_distance_before_lens, k)
        f_after_lens_ap = thin_lens(f_at_lens_ap, R2, focal_length, k)
        mode_fields_after_lens_ap.append((m_mode, n_mode, p_mn, f_after_lens_ap))
    del f_masked, f_at_lens_ap

    # High-res propagation through gas region — per-mode storage
    n_ap_modes = len(mode_fields_after_lens_ap)
    ap_modes_I_list = [[] for _ in range(n_ap_modes)]
    ap_modes_phase_list = [[] for _ in range(n_ap_modes)]
    z_gas_ap_list = []

    for i, z in enumerate(z_focus_prop):
        if i % 50 == 0:
            print(f"    z-step {i+1}/{n_z_steps_hr}")
        if gas_z_start_prop <= z <= gas_z_end_prop:
            z_gas_ap_list.append(z)
            for mode_idx, (m_mode, n_mode, p_mn, f_after_lens_mode) in enumerate(mode_fields_after_lens_ap):
                field_z, x_out, y_out = fresnel_propagate_zoom(
                    f_after_lens_mode, z, k, L, L_xz_focus, N_xz_focus)
                ap_modes_I_list[mode_idx].append(p_mn * np.abs(field_z[hhg_crop, hhg_crop])**2)
                field_no_pw = field_z * np.exp(-1j * k * z)
                ap_modes_phase_list[mode_idx].append(np.angle(field_no_pw[hhg_crop, hhg_crop]))

    del mode_fields_after_lens_ap

    z_gas_ap = np.array(z_gas_ap_list)
    z_gas_ap_m = z_gas_ap * 1e-3
    del z_gas_ap_list

    # Build per-mode intensity (Wcm2) and phase gradient arrays
    ap_modes_I_Wcm2 = []
    ap_modes_dphase = []
    dz_arr_ap = np.diff(z_gas_ap_m)
    for mode_idx in range(n_ap_modes):
        I_m = np.array(ap_modes_I_list[mode_idx]) * I_scale_factor_2d
        ph_m = np.array(ap_modes_phase_list[mode_idx])

        # Phase gradient (complex-domain)
        env_m = np.sqrt(np.array(ap_modes_I_list[mode_idx])) * np.exp(1j * ph_m)
        dph_m = np.zeros_like(ph_m)
        for j in range(1, len(z_gas_ap_m)):
            dph_m[j] = np.angle(env_m[j] * np.conj(env_m[j-1])) / dz_arr_ap[j-1]
        dph_m[0] = dph_m[1]
        I_thresh_m = 0.01 * np.array(ap_modes_I_list[mode_idx]).max()
        dph_m[np.array(ap_modes_I_list[mode_idx]) < I_thresh_m] = 0.0

        ap_modes_I_Wcm2.append(I_m)
        ap_modes_dphase.append(dph_m)

    del ap_modes_I_list, ap_modes_phase_list

    I_total_ap_Wcm2 = sum(ap_modes_I_Wcm2)

    gas_data_ap[i_ap] = {
        'modes_I_Wcm2': ap_modes_I_Wcm2,
        'modes_dphase': ap_modes_dphase,
        'z_gas_m': z_gas_ap_m,
        'I_total_Wcm2': I_total_ap_Wcm2,
    }

    print(f"    Gas region: {len(z_gas_ap_m)} z-points, I_peak = {I_total_ap_Wcm2.max():.3e} W/cm^2")
    gc.collect()

print(f"\n  Multi-aperture propagation complete: {len(gas_data_ap)} apertures stored")
mem_total = sum(sum(m.nbytes for m in gd['modes_I_Wcm2']) +
                sum(m.nbytes for m in gd['modes_dphase'])
                for gd in gas_data_ap.values()) / 1e9
print(f"  Total aperture data memory: {mem_total:.1f} GB")


# =============================================================================
# STEP 2: MULTI-POWER MULTI-HARMONIC GRID SCAN
# =============================================================================
TIMER.start_section("Step 2: Grid scan (power x harmonic x params)")

from scipy.integrate import cumulative_trapezoid

# Handle np.trapz deprecation in NumPy >= 2.0
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

multi_q_list = [11, 13, 15, 17, 19, 21]
n_q = len(multi_q_list)
n_P = len(exp_yield_I_Wcm2)

# Parameter grid
alpha_grid = np.arange(1.0, 8.5, 0.5)                              # 15 pts: α=1.0 to 8.0
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

        for iq, q in enumerate(multi_q_list):
            ff = ff_ap_masks[q]
            sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22

            # Accumulate yield across modes (incoherent sum of per-mode HHG)
            for mode_idx in range(n_hg_modes):
                I_3d_m = modes_I_Wcm2_ub[mode_idx] * power_scale
                log_I_3d_m = np.log(np.maximum(I_3d_m, 1e-30))
                nf_3d_m = ionization_fraction(I_3d_m, gas['Ip_eV'])

                dk_neut = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d_m, gas['delta_n'])
                dk_plas = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d_m, gas['N_atm'])
                dk_geom = -(q - 1.0 / q) * modes_dphase_ub[mode_idx]
                dk_total = dk_neut + dk_plas + dk_geom

                Phi_3d = np.zeros_like(dk_total)
                Phi_3d[1:] = cumulative_trapezoid(dk_total, z_gas_2d_ub_m, axis=0)

                mu_3d = sigma_q_m2 * n_gas_density * (1.0 - nf_3d_m)
                mu_cumfwd = np.zeros_like(mu_3d)
                mu_cumfwd[1:] = cumulative_trapezoid(mu_3d, z_gas_2d_ub_m, axis=0)
                tau = mu_cumfwd[-1:] - mu_cumfwd
                abs_factor = np.exp(-tau / 2.0)

                base = (1.0 - nf_3d_m) * np.exp(1j * Phi_3d) * abs_factor
                I_clipped = np.clip(I_3d_m, I_lut_min, I_lut_max)
                sfa_phase_3d = phase_interp_per_q[q](I_clipped)
                base_complex = base * np.exp(1j * sfa_phase_3d)
                base_complex[I_3d_m < I_lut_min] = 0.0

                for ia in range(n_alpha):
                    alpha = alpha_grid[ia]
                    for js in range(n_Is):
                        I_s = Is_grid[js]

                        dq_mag = np.exp(alpha / 2.0 * log_I_3d_m - I_3d_m / (2.0 * I_s))
                        dq_mag[I_3d_m < I_lut_min] = 0.0

                        E_q_2d = _trapz(dq_mag * base_complex, z_gas_2d_ub_m, axis=0)

                        if USE_SCIPY_FFT:
                            E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d, workers=-1)) * dx_hhg_m**2
                        else:
                            E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d)) * dx_hhg_m**2

                        yield_ub[iq, iP, ia, js] += np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2

                del dk_neut, dk_plas, dk_geom, dk_total, Phi_3d, mu_3d, mu_cumfwd, tau, abs_factor, base, base_complex
                del I_3d_m, log_I_3d_m, nf_3d_m

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
        z_m_ap = gd['z_gas_m']
        n_ap_modes_scan = len(gd['modes_I_Wcm2'])

        print(f"\n  Aperture {i_sub+1}/{n_ap_sub}: iris={exp_aperture_x[i_ap]:.1f}, T={exp_aperture_T[i_ap]:.3f}")

        for iq, q in enumerate(multi_q_list):
            ff = ff_ap_masks[q]
            sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22

            for mode_idx in range(n_ap_modes_scan):
                I_3d_m = gd['modes_I_Wcm2'][mode_idx] * power_scale_ap
                log_I_3d_m = np.log(np.maximum(I_3d_m, 1e-30))
                nf_3d_m = ionization_fraction(I_3d_m, gas['Ip_eV'])

                dk_neut = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_3d_m, gas['delta_n'])
                dk_plas = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_3d_m, gas['N_atm'])
                dk_geom = -(q - 1.0 / q) * gd['modes_dphase'][mode_idx]
                dk_total = dk_neut + dk_plas + dk_geom

                Phi_3d = np.zeros_like(dk_total)
                Phi_3d[1:] = cumulative_trapezoid(dk_total, z_m_ap, axis=0)

                mu_3d = sigma_q_m2 * n_gas_density * (1.0 - nf_3d_m)
                mu_cumfwd = np.zeros_like(mu_3d)
                mu_cumfwd[1:] = cumulative_trapezoid(mu_3d, z_m_ap, axis=0)
                tau = mu_cumfwd[-1:] - mu_cumfwd
                abs_factor = np.exp(-tau / 2.0)

                base = (1.0 - nf_3d_m) * np.exp(1j * Phi_3d) * abs_factor
                I_clipped_m = np.clip(I_3d_m, I_lut_min, I_lut_max)
                sfa_phase_3d_m = phase_interp_per_q[q](I_clipped_m)
                base_complex = base * np.exp(1j * sfa_phase_3d_m)
                base_complex[I_3d_m < I_lut_min] = 0.0

                for ia in range(n_alpha):
                    alpha = alpha_grid[ia]
                    for js in range(n_Is):
                        I_s = Is_grid[js]

                        dq_mag = np.exp(alpha / 2.0 * log_I_3d_m - I_3d_m / (2.0 * I_s))
                        dq_mag[I_3d_m < I_lut_min] = 0.0

                        E_q_2d = _trapz(dq_mag * base_complex, z_m_ap, axis=0)
                        if USE_SCIPY_FFT:
                            E_ff = scipy_fft.fftshift(scipy_fft.fft2(E_q_2d, workers=-1)) * dx_hhg_m**2
                        else:
                            E_ff = np.fft.fftshift(np.fft.fft2(E_q_2d)) * dx_hhg_m**2

                        yield_ap[iq, i_sub, ia, js] += np.sum(np.abs(E_ff)**2 * ff['mask']) * ff['dtheta']**2

                del dk_neut, dk_plas, dk_geom, dk_total, Phi_3d, mu_3d, mu_cumfwd, tau, abs_factor, base, base_complex
                del I_3d_m, log_I_3d_m, nf_3d_m

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
            print(f"  Beam model: HG mode decomposition ({n_hg_modes} modes)")
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

        from scipy.optimize import minimize
        from scipy.interpolate import RegularGridInterpolator

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
            for ih in range(n_h):
                a_h = params[2 * ih]
                li_h = params[2 * ih + 1]
                if a_h < ALPHA_MIN or a_h > ALPHA_MAX:
                    return 1e6
                if li_h < LOG_IS_MIN or li_h > LOG_IS_MAX:
                    return 1e6

            C_shape = 0.0
            for ih in range(n_h):
                c_h = cost_per_harmonic([params[2*ih], params[2*ih+1]], ih, multi_q_list[ih])
                if c_h >= 1e6:
                    return 1e6
                C_shape += c_h
            C_shape /= n_h

            C_smooth_alpha = 0.0
            C_smooth_Is = 0.0
            for ih in range(n_h - 2):
                d2_a = params[2*(ih+2)] - 2*params[2*(ih+1)] + params[2*ih]
                d2_li = params[2*(ih+2)+1] - 2*params[2*(ih+1)+1] + params[2*ih+1]
                C_smooth_alpha += d2_a**2
                C_smooth_Is += d2_li**2
            C_smooth_alpha /= max(n_h - 2, 1)
            C_smooth_Is /= max(n_h - 2, 1)

            C_prior = 0.0
            C_alpha_prior = 0.0
            for ih in range(n_h):
                C_prior += ((params[2*ih+1] - log_Is_prior_center) / log_Is_prior_width)**2
                C_alpha_prior += ((params[2*ih] - alpha_prior_center) / alpha_prior_width)**2
            C_prior /= n_h
            C_alpha_prior /= n_h

            return (C_shape
                    + lambda_smooth_alpha * C_smooth_alpha
                    + lambda_smooth_Is * C_smooth_Is
                    + lambda_prior * C_prior
                    + lambda_alpha_prior * C_alpha_prior)

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

        def save_cost_landscape_grid(cost_maps, colorbar_label, title, filename_token):
            fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
            axes_cm = np.asarray(axes_cm)
            for iq, q in enumerate(multi_q_list):
                ax = axes_cm[iq // 3, iq % 3]
                cost_map = cost_maps[iq]
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
                    shading='auto', cmap='viridis', vmin=vmin_cm, vmax=vmax_cm
                )
                ax.set_xscale('log')
                ax.set_xlim(cost_map_Is_range.min(), cost_map_Is_range.max())
                plt.colorbar(im, ax=ax, label=colorbar_label)
                ax.plot(best_Is_vals[q], best_alphas[q], 'r*', markersize=15, label='Joint opt')
                ax.plot(best_Is_grid, best_alpha_grid, 'wx', markersize=10, markeredgewidth=2, label='Grid best')
                ax.set_xlabel(r'$I_s$ (W/cm$^2$)')
                ax.set_ylabel(r'$\alpha$')
                ax.set_title(f'H{q}')
                ax.legend(fontsize=7)

            fig_cm.suptitle(f'{title} ({lavg_tag})', fontsize=14)
            fig_cm.tight_layout()
            if filename_token == 'combined':
                cost_map_path = f'hhg_cost_landscape_per_harmonic_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
            else:
                cost_map_path = f'hhg_cost_landscape_per_harmonic_{filename_token}_{lavg_tag}_{pressure_tag}_{_m2_tag}.png'
            fig_cm.savefig(cost_map_path, dpi=150, bbox_inches='tight')
            plt.close(fig_cm)
            print(f"  Saved {title.lower()}: {cost_map_path}")
            return cost_map_path

        cost_map_path = save_cost_landscape_grid(
            _per_harmonic_cost_maps, 'Cost', 'Per-Harmonic Cost Landscape', 'combined'
        )
        cost_map_power_path = save_cost_landscape_grid(
            _per_harmonic_cost_maps_power, r'$C_{P,q}^{norm}$', r'Per-Harmonic Normalized Power Cost $C_{P,q}^{norm}$', 'power'
        )
        cost_map_aperture_path = save_cost_landscape_grid(
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

            for iq, q in enumerate(multi_q_list):
                ff = ff_ap_masks[q]
                ap_mask_q = ff['mask']
                dtheta_q = ff['dtheta']

                sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22
                alpha_q = best_alphas[q]
                Is_q = best_Is_vals[q]

                for mode_idx in range(n_hg_modes):
                    # -- Unblocked --
                    I_3d_m_ub = modes_I_Wcm2_ub[mode_idx] * power_scale
                    nf_m_ub = ionization_fraction(I_3d_m_ub, gas['Ip_eV'])

                    dk_n_ub = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_m_ub, gas['delta_n'])
                    dk_p_ub = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_m_ub, gas['N_atm'])
                    dk_g_ub = -(q - 1.0 / q) * modes_dphase_ub[mode_idx]
                    dk_t_ub = dk_n_ub + dk_p_ub + dk_g_ub

                    Phi_ub = np.zeros_like(dk_t_ub)
                    Phi_ub[1:] = cumulative_trapezoid(dk_t_ub, z_gas_2d_ub_m, axis=0)

                    mu_ub = sigma_q_m2 * n_gas_density * (1.0 - nf_m_ub)
                    mu_cum_ub = np.zeros_like(mu_ub)
                    mu_cum_ub[1:] = cumulative_trapezoid(mu_ub, z_gas_2d_ub_m, axis=0)
                    tau_ub = mu_cum_ub[-1:] - mu_cum_ub
                    abs_ub = np.exp(-tau_ub / 2.0)

                    base_ub = (1.0 - nf_m_ub) * np.exp(1j * Phi_ub) * abs_ub
                    I_clipped_ub = np.clip(I_3d_m_ub, I_lut_min, I_lut_max)
                    sfa_phase_ub = phase_interp_per_q[q](I_clipped_ub)
                    base_complex_ub = base_ub * np.exp(1j * sfa_phase_ub)
                    base_complex_ub[I_3d_m_ub < I_lut_min] = 0.0

                    dq_mag_ub = I_3d_m_ub**(alpha_q / 2.0) * np.exp(-I_3d_m_ub / (2.0 * Is_q))
                    dq_mag_ub[I_3d_m_ub < I_lut_min] = 0.0

                    E_ub = _trapz(dq_mag_ub * base_complex_ub, z_gas_2d_ub_m, axis=0)
                    if USE_SCIPY_FFT:
                        E_ff_ub = scipy_fft.fftshift(scipy_fft.fft2(E_ub, workers=-1)) * dx_hhg_m**2
                    else:
                        E_ff_ub = np.fft.fftshift(np.fft.fft2(E_ub)) * dx_hhg_m**2
                    yield_ub_best[iq, iP] += np.sum(np.abs(E_ff_ub)**2 * ap_mask_q) * dtheta_q**2

                    # -- Blocked --
                    I_3d_m_b = modes_I_Wcm2_b[mode_idx] * power_scale
                    nf_m_b = ionization_fraction(I_3d_m_b, gas['Ip_eV'])

                    dk_n_b = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_m_b, gas['delta_n'])
                    dk_p_b = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_m_b, gas['N_atm'])
                    dk_g_b = -(q - 1.0 / q) * modes_dphase_b[mode_idx]
                    dk_t_b = dk_n_b + dk_p_b + dk_g_b

                    Phi_b = np.zeros_like(dk_t_b)
                    Phi_b[1:] = cumulative_trapezoid(dk_t_b, z_gas_2d_b_m, axis=0)

                    mu_b = sigma_q_m2 * n_gas_density * (1.0 - nf_m_b)
                    mu_cum_b = np.zeros_like(mu_b)
                    mu_cum_b[1:] = cumulative_trapezoid(mu_b, z_gas_2d_b_m, axis=0)
                    tau_b = mu_cum_b[-1:] - mu_cum_b
                    abs_b = np.exp(-tau_b / 2.0)

                    base_b = (1.0 - nf_m_b) * np.exp(1j * Phi_b) * abs_b
                    I_clipped_b = np.clip(I_3d_m_b, I_lut_min, I_lut_max)
                    sfa_phase_b = phase_interp_per_q[q](I_clipped_b)
                    base_complex_b = base_b * np.exp(1j * sfa_phase_b)
                    base_complex_b[I_3d_m_b < I_lut_min] = 0.0

                    dq_mag_b = I_3d_m_b**(alpha_q / 2.0) * np.exp(-I_3d_m_b / (2.0 * Is_q))
                    dq_mag_b[I_3d_m_b < I_lut_min] = 0.0

                    E_b = _trapz(dq_mag_b * base_complex_b, z_gas_2d_b_m, axis=0)
                    if USE_SCIPY_FFT:
                        E_ff_b = scipy_fft.fftshift(scipy_fft.fft2(E_b, workers=-1)) * dx_hhg_m**2
                    else:
                        E_ff_b = np.fft.fftshift(np.fft.fft2(E_b)) * dx_hhg_m**2
                    yield_b_best[iq, iP] += np.sum(np.abs(E_ff_b)**2 * ap_mask_q) * dtheta_q**2

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
            z_m_ap = gd['z_gas_m']
            n_ap_modes_s5 = len(gd['modes_I_Wcm2'])
            for iq, q in enumerate(multi_q_list):
                alpha_q = best_alphas[q]
                Is_q = best_Is_vals[q]
                for mode_idx in range(n_ap_modes_s5):
                    I_3d_m = gd['modes_I_Wcm2'][mode_idx] * power_scale_ap
                    nf_3d_m = ionization_fraction(I_3d_m, gas['Ip_eV'])
                    yield_ap_best[iq, i_sub] += compute_yield_single(
                        I_3d_m, nf_3d_m, gd['modes_dphase'][mode_idx], z_m_ap, q, alpha_q, Is_q)

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
        # ax.set_ylim(0.8, 3.5)  # auto-scale

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
        # ax.set_ylim(0.8, 3.5)  # auto-scale

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
# STEP 7: MASK SHAPE COMPARISON (using fitted alpha, Is) — HG multi-mode
# =============================================================================
TIMER.start_section("Step 7: Mask shape comparison")
print("\n" + "="*60)
print("MASK SHAPE COMPARISON (using fitted alpha, Is) — HG model")
print("="*60)

mask_configs_mc = ['none', 'circular', 'twosided', 'diagonal']
mask_labels_mc = {'none': 'No mask', 'circular': 'Circular', 'twosided': 'Two-side', 'diagonal': 'Diagonal'}
mask_colors_mc = {'none': '#7F7F7F', 'circular': '#2F5597', 'twosided': '#B04A4A', 'diagonal': '#4F8B5B'}

print("Using fitted dipole parameters:")
for q in multi_q_list:
    print(f"  H{q}: alpha={best_alphas[q]:.3f}, I_s={best_Is_vals[q]:.2e}")

mask_yield_results = {}

for mask_name in mask_configs_mc:
    print(f"\n--- Mask: {mask_name} ---")

    if mask_name == 'none':
        mc_modes_I_Wcm2 = modes_I_Wcm2_ub
        mc_modes_dphase = modes_dphase_ub
        mc_z_gas_m = z_gas_2d_ub_m
        mc_z_gas_mm = z_gas_2d_ub
        trans_mc = 100.0
        print(f"  Reusing unblocked data (transmission: 100%)")
    elif mask_name == 'circular':
        mc_modes_I_Wcm2 = modes_I_Wcm2_b
        mc_modes_dphase = modes_dphase_b
        mc_z_gas_m = z_gas_2d_b_m
        mc_z_gas_mm = z_gas_2d_b
        trans_mc = aperture_transmission
        print(f"  Reusing circular blocked data (transmission: {trans_mc:.1f}%)")
    else:
        msk = build_mask(X, Y, mask_name, mask_params)
        pwr_before_mc = sum(p_mn * np.sum(np.abs(f)**2) for (_, _, p_mn, f) in mode_fields_at_aperture)
        pwr_after_mc = sum(p_mn * np.sum(np.abs(f * msk)**2) for (_, _, p_mn, f) in mode_fields_at_aperture)
        trans_mc = pwr_after_mc / pwr_before_mc * 100
        print(f"  Transmission: {trans_mc:.1f}%")

        mc_mode_fields_after_lens = []
        for (m_mode, n_mode, p_mn, f_at_ap) in mode_fields_at_aperture:
            f_masked = f_at_ap * msk
            f_at_lens_mc = propagate_field(f_masked, aperture_distance_before_lens, k)
            f_after_lens_mc = thin_lens(f_at_lens_mc, R2, focal_length, k)
            mc_mode_fields_after_lens.append((m_mode, n_mode, p_mn, f_after_lens_mc))

        mc_n_modes = len(mc_mode_fields_after_lens)
        mc_modes_I_list = [[] for _ in range(mc_n_modes)]
        mc_modes_phase_list = [[] for _ in range(mc_n_modes)]
        mc_z_gas_list = []

        for i, z in enumerate(z_focus_prop):
            if gas_z_start_prop <= z <= gas_z_end_prop:
                mc_z_gas_list.append(z)
                for mode_idx, (m_mode, n_mode, p_mn, f_lens_mode) in enumerate(mc_mode_fields_after_lens):
                    fz, _, _ = fresnel_propagate_zoom(f_lens_mode, z, k, L, L_xz_focus, N_xz_focus)
                    mc_modes_I_list[mode_idx].append(p_mn * np.abs(fz[hhg_crop, hhg_crop])**2)
                    fz_nopw = fz * np.exp(-1j * k * z)
                    mc_modes_phase_list[mode_idx].append(np.angle(fz_nopw[hhg_crop, hhg_crop]))
            if i % 50 == 0:
                print(f"    z-step {i+1}/{n_z_steps_hr}")

        mc_z_gas_mm = np.array(mc_z_gas_list)
        mc_z_gas_m = mc_z_gas_mm * 1e-3

        mc_modes_I_Wcm2 = []
        mc_modes_dphase = []
        dz_arr_mc = np.diff(mc_z_gas_m)
        for mode_idx in range(mc_n_modes):
            I_m = np.array(mc_modes_I_list[mode_idx]) * I_scale_factor_2d
            ph_m = np.array(mc_modes_phase_list[mode_idx])
            env_m = np.sqrt(np.array(mc_modes_I_list[mode_idx])) * np.exp(1j * ph_m)
            dph_m = np.zeros_like(ph_m)
            for j in range(1, len(mc_z_gas_m)):
                dph_m[j] = np.angle(env_m[j] * np.conj(env_m[j-1])) / dz_arr_mc[j-1]
            dph_m[0] = dph_m[1]
            I_thresh_mc = 0.01 * np.array(mc_modes_I_list[mode_idx]).max()
            dph_m[np.array(mc_modes_I_list[mode_idx]) < I_thresh_mc] = 0.0
            mc_modes_I_Wcm2.append(I_m)
            mc_modes_dphase.append(dph_m)

        print(f"  Propagated {mc_n_modes} modes through gas ({len(mc_z_gas_list)} z-steps)")
        del mc_mode_fields_after_lens, mc_modes_I_list, mc_modes_phase_list, msk
        gc.collect()

    mask_yield_results[mask_name] = {'transmission': trans_mc}
    mc_n_modes_use = len(mc_modes_I_Wcm2)

    for q in multi_q_list:
        alpha_q = best_alphas[q]
        Is_q = best_Is_vals[q]
        sigma_q_m2 = sigma_xuv_multi_Mb[q] * 1e-22
        ff = ff_ap_masks[q]
        dtheta_q = ff['dtheta']

        yield_slit_total = 0.0
        yield_circ_total = 0.0
        yield_nf_total = 0.0
        E_q_main = None
        E_ff_main = None

        for mode_idx in range(mc_n_modes_use):
            I_3d_m = mc_modes_I_Wcm2[mode_idx]
            nf_m = ionization_fraction(I_3d_m, gas['Ip_eV'])

            dk_neut = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_m, gas['delta_n'])
            dk_plas = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_m, gas['N_atm'])
            dk_geom = -(q - 1.0 / q) * mc_modes_dphase[mode_idx]
            dk_total = dk_neut + dk_plas + dk_geom

            Phi = np.zeros_like(dk_total)
            Phi[1:] = cumulative_trapezoid(dk_total, mc_z_gas_m, axis=0)

            mu = sigma_q_m2 * n_gas_density * (1.0 - nf_m)
            mu_cum = np.zeros_like(mu)
            mu_cum[1:] = cumulative_trapezoid(mu, mc_z_gas_m, axis=0)
            tau = mu_cum[-1:] - mu_cum
            abs_factor = np.exp(-tau / 2.0)

            I_clipped = np.clip(I_3d_m, I_lut_min, I_lut_max)
            sfa_phase = phase_interp_per_q[q](I_clipped)
            dq_mag = I_3d_m**(alpha_q / 2.0) * np.exp(-I_3d_m / (2.0 * Is_q))
            dq_mag[I_3d_m < I_lut_min] = 0.0

            integrand = dq_mag * np.exp(1j * sfa_phase) * (1.0 - nf_m) * np.exp(1j * Phi) * abs_factor
            integrand[I_3d_m < I_lut_min] = 0.0
            E_q_mode = _trapz(integrand, mc_z_gas_m, axis=0)

            yield_nf_total += np.sum(np.abs(E_q_mode)**2) * dx_hhg_m**2

            if USE_SCIPY_FFT:
                E_ff_mode = scipy_fft.fftshift(scipy_fft.fft2(E_q_mode, workers=-1)) * dx_hhg_m**2
            else:
                E_ff_mode = np.fft.fftshift(np.fft.fft2(E_q_mode)) * dx_hhg_m**2
            I_ff_mode = np.abs(E_ff_mode)**2
            yield_slit_total += np.sum(I_ff_mode * ff['mask_slit']) * dtheta_q**2
            yield_circ_total += np.sum(I_ff_mode * ff['mask_circ']) * dtheta_q**2

            if mode_idx == 0:
                E_q_main = E_q_mode.copy()
                E_ff_main = E_ff_mode.copy()

            # Per-mode yield buildup (incoherent sum across modes): NF + slit + circ
            if len(mc_z_gas_m) > 1:
                n_z_mc = len(mc_z_gas_m)
                dz_mc = np.diff(mc_z_gas_m)
                buildup_mode = np.zeros((n_z_mc, N_hhg_2d, N_hhg_2d), dtype=complex)
                for j in range(1, n_z_mc):
                    buildup_mode[j] = buildup_mode[j-1] + 0.5 * (integrand[j-1] + integrand[j]) * dz_mc[j-1]
                nf_contrib = np.array([np.sum(np.abs(buildup_mode[j])**2) * dx_hhg_m**2 for j in range(n_z_mc)])
                slit_contrib = np.zeros(n_z_mc)
                circ_contrib = np.zeros(n_z_mc)
                for j in range(n_z_mc):
                    if USE_SCIPY_FFT:
                        E_ff_j = scipy_fft.fftshift(scipy_fft.fft2(buildup_mode[j], workers=-1)) * dx_hhg_m**2
                    else:
                        E_ff_j = np.fft.fftshift(np.fft.fft2(buildup_mode[j])) * dx_hhg_m**2
                    I_ff_j = np.abs(E_ff_j)**2
                    slit_contrib[j] = np.sum(I_ff_j * ff['mask_slit']) * dtheta_q**2
                    circ_contrib[j] = np.sum(I_ff_j * ff['mask_circ']) * dtheta_q**2
                if mode_idx == 0:
                    yield_vs_z_mc = nf_contrib
                    yield_vs_z_slit_mc = slit_contrib
                    yield_vs_z_circ_mc = circ_contrib
                else:
                    yield_vs_z_mc += nf_contrib
                    yield_vs_z_slit_mc += slit_contrib
                    yield_vs_z_circ_mc += circ_contrib
                del buildup_mode

        yield_ap = yield_slit_total if hhg_acceptance_type == 'slit' else yield_circ_total

        buildup_mc = None
        if q == hhg_harmonic_order:
            center_2d_mc = N_hhg_2d // 2
            I_dom = mc_modes_I_Wcm2[0]
            nf_dom = ionization_fraction(I_dom, gas['Ip_eV'])
            dk_n = calc_dk_neutral(q, hhg_gas_pressure, lambda_0_m, nf_dom, gas['delta_n'])
            dk_p = calc_dk_plasma(q, hhg_gas_pressure, lambda_0_m, nf_dom, gas['N_atm'])
            dk_g = -(q - 1.0 / q) * mc_modes_dphase[0]
            dk_t = dk_n + dk_p + dk_g
            Phi_dom = np.zeros_like(dk_t)
            Phi_dom[1:] = cumulative_trapezoid(dk_t, mc_z_gas_m, axis=0)
            mu_dom = sigma_q_m2 * n_gas_density * (1.0 - nf_dom)
            mu_cum_dom = np.zeros_like(mu_dom)
            mu_cum_dom[1:] = cumulative_trapezoid(mu_dom, mc_z_gas_m, axis=0)
            tau_dom = mu_cum_dom[-1:] - mu_cum_dom
            abs_dom = np.exp(-tau_dom / 2.0)
            I_cl = np.clip(I_dom, I_lut_min, I_lut_max)
            sfa_ph = phase_interp_per_q[q](I_cl)
            dq_m = I_dom**(alpha_q/2.0) * np.exp(-I_dom/(2.0*Is_q))
            dq_m[I_dom < I_lut_min] = 0.0
            integ_dom = dq_m * np.exp(1j*sfa_ph) * (1.0-nf_dom) * np.exp(1j*Phi_dom) * abs_dom
            integ_dom[I_dom < I_lut_min] = 0.0
            buildup_mc = np.zeros(len(mc_z_gas_m), dtype=complex)
            if len(mc_z_gas_m) > 1:
                buildup_mc[1:] = cumulative_trapezoid(integ_dom[:, center_2d_mc, center_2d_mc], mc_z_gas_m)

        mask_yield_results[mask_name][q] = {
            'yield_nf': yield_nf_total,
            'yield_ap': yield_ap,
            'yield_slit': yield_slit_total,
            'yield_circ': yield_circ_total,
            'E_q': E_q_main,
            'E_ff': E_ff_main,
            'I_ff': np.abs(E_ff_main)**2 if E_ff_main is not None else None,
            'dtheta': dtheta_q,
            'buildup': buildup_mc,
            'yield_vs_z': yield_vs_z_mc if len(mc_z_gas_m) > 1 else None,
            'yield_vs_z_slit': yield_vs_z_slit_mc if len(mc_z_gas_m) > 1 else None,
            'yield_vs_z_circ': yield_vs_z_circ_mc if len(mc_z_gas_m) > 1 else None,
        }

    I_total_mc = sum(mc_modes_I_Wcm2)
    focus_iz_mc = np.argmin(np.abs(mc_z_gas_mm - focal_length))
    peak_pos_mc = np.unravel_index(np.argmax(I_total_mc[focus_iz_mc]), I_total_mc[focus_iz_mc].shape)
    py_mc, px_mc = peak_pos_mc
    center_2d_mc = N_hhg_2d // 2

    focus_peak_pos_mc = np.unravel_index(np.nanargmax(I_total_mc), I_total_mc.shape)
    focus_peak_iz_mc, focus_peak_py_mc, focus_peak_px_mc = focus_peak_pos_mc
    focus_peak_I_mc = float(I_total_mc[focus_peak_pos_mc])

    mask_yield_results[mask_name]['beam'] = {
        'I_onaxis_Wcm2': I_total_mc[:, py_mc, px_mc].copy(),
        'nf_onaxis': ionization_fraction(I_total_mc[:, py_mc, px_mc], gas['Ip_eV']),
        'gouy_grad': mc_modes_dphase[0][:, py_mc, px_mc].copy(),
        'z_gas_mm': mc_z_gas_mm.copy(),
        'peak_I': I_total_mc.max(),
        'focus_peak_I': focus_peak_I_mc,
        'focus_peak_z_mm': float(mc_z_gas_mm[focus_peak_iz_mc]),
        'focus_peak_pos': np.array(focus_peak_pos_mc, dtype=np.int32),
        'I_focus_x': I_total_mc[focus_iz_mc, center_2d_mc, :].copy(),
        'I_focus_y': I_total_mc[focus_iz_mc, :, center_2d_mc].copy(),
        'I_2d_focus': I_total_mc[focus_iz_mc, :, :].copy(),
        'I_2d_peak_focus': I_total_mc[focus_peak_iz_mc, :, :].copy(),
        'nf_focus_x': ionization_fraction(I_total_mc[focus_iz_mc, center_2d_mc, :], gas['Ip_eV']),
    }

    print(f"  Peak I: {I_total_mc.max():.2e} W/cm²")
    for q in multi_q_list:
        r = mask_yield_results[mask_name][q]
        print(f"    H{q}: slit={r['yield_slit']:.3e}, circ={r['yield_circ']:.3e}")
    del I_total_mc
    gc.collect()

# --- Multi-mask beam parameter summary (focus metrics saved to NPZ) ---
print("Generating multi-mask beam parameter summary...")

def _threshold_width(coord, profile, frac):
    coord = np.asarray(coord, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if coord.size != profile.size or coord.size < 2:
        return np.nan
    peak = np.nanmax(profile)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    y = profile / peak
    above = y >= frac
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    left = int(idx[0])
    right = int(idx[-1])

    def _interp_edge(i0, i1):
        y0 = y[i0]
        y1 = y[i1]
        x0 = coord[i0]
        x1 = coord[i1]
        den = y1 - y0
        if not np.isfinite(den) or abs(den) < 1e-15:
            return x1
        return x0 + (frac - y0) * (x1 - x0) / den

    x_left = coord[left] if left == 0 else _interp_edge(left - 1, left)
    x_right = coord[right] if right == coord.size - 1 else _interp_edge(right, right + 1)
    return abs(x_right - x_left)


def _halfmax_halfwidth(z_mm, intensity):
    width = _threshold_width(z_mm, intensity, 0.5)
    return width / 2.0 if np.isfinite(width) else np.nan


_beam_masks = mask_configs_mc
_beam_x_um = np.asarray(x_hhg_2d, dtype=float) * 1e3
_beam_peak_I = np.zeros(len(_beam_masks), dtype=np.float64)
_beam_peak_rel = np.zeros_like(_beam_peak_I)
_beam_fwhm_x_um = np.zeros_like(_beam_peak_I)
_beam_fwhm_y_um = np.zeros_like(_beam_peak_I)
_beam_w0_x_um = np.zeros_like(_beam_peak_I)
_beam_w0_y_um = np.zeros_like(_beam_peak_I)
_beam_rayleigh_mm = np.zeros_like(_beam_peak_I)
_beam_focus_z_mm = np.zeros_like(_beam_peak_I)
_beam_focus_shift_um = np.zeros_like(_beam_peak_I)
_beam_trans_percent = np.zeros_like(_beam_peak_I)

for _im, _mn in enumerate(_beam_masks):
    _beam = mask_yield_results[_mn]['beam']
    _focus_pos = np.asarray(_beam.get('focus_peak_pos', [0, N_hhg_2d // 2, N_hhg_2d // 2]), dtype=int)
    _py = int(np.clip(_focus_pos[1], 0, N_hhg_2d - 1))
    _px = int(np.clip(_focus_pos[2], 0, N_hhg_2d - 1))
    _I2d_focus_peak = np.asarray(_beam.get('I_2d_peak_focus', _beam['I_2d_focus']), dtype=float)
    _line_x = _I2d_focus_peak[_py, :]
    _line_y = _I2d_focus_peak[:, _px]

    _beam_peak_I[_im] = float(_beam.get('focus_peak_I', _beam['peak_I']))
    _beam_fwhm_x_um[_im] = _threshold_width(_beam_x_um, _line_x, 0.5)
    _beam_fwhm_y_um[_im] = _threshold_width(_beam_x_um, _line_y, 0.5)
    _beam_w0_x_um[_im] = _threshold_width(_beam_x_um, _line_x, 1.0 / np.e**2) / 2.0
    _beam_w0_y_um[_im] = _threshold_width(_beam_x_um, _line_y, 1.0 / np.e**2) / 2.0
    _I_onaxis = np.asarray(_beam['I_onaxis_Wcm2'], dtype=float)
    _beam_rayleigh_mm[_im] = _halfmax_halfwidth(np.asarray(_beam['z_gas_mm'], dtype=float), _I_onaxis)
    _beam_focus_z_mm[_im] = float(_beam.get('focus_peak_z_mm', _beam['z_gas_mm'][np.nanargmax(_I_onaxis)]))
    _beam_focus_shift_um[_im] = (_beam_focus_z_mm[_im] - focal_length) * 1e3
    _beam_trans_percent[_im] = float(mask_yield_results[_mn]['transmission'])

_beam_peak_ref = max(_beam_peak_I[0], 1e-30)
_beam_peak_rel[:] = _beam_peak_I / _beam_peak_ref
_beam_summary = {
    'focus_peak_I_Wcm2': _beam_peak_I,
    'focus_peak_rel': _beam_peak_rel,
    'fwhm_x_um': _beam_fwhm_x_um,
    'fwhm_y_um': _beam_fwhm_y_um,
    'w0_x_um': _beam_w0_x_um,
    'w0_y_um': _beam_w0_y_um,
    'rayleigh_range_mm': _beam_rayleigh_mm,
    'focus_z_mm': _beam_focus_z_mm,
    'focus_shift_um': _beam_focus_shift_um,
    'transmission_percent': _beam_trans_percent,
}

for _im, _mn in enumerate(_beam_masks):
    _beam = mask_yield_results[_mn]['beam']
    for _key, _arr in _beam_summary.items():
        _beam[_key] = float(_arr[_im])

fig_mc5_hg, axes_mc5_hg = plt.subplots(2, 3, figsize=(16, 10))
_bp_title_fs = 16
_bp_label_fs = 15
_bp_tick_fs = 12
_bp_value_fs = 11
_bp_suptitle_fs = 19
_bar_x = np.arange(len(_beam_masks))
_bar_w = 0.35
_bar_colors = [mask_colors_mc[m] for m in _beam_masks]
_bar_labels = [mask_labels_mc[m] for m in _beam_masks]

ax = axes_mc5_hg[0, 0]
bars = ax.bar(_bar_x, _beam_peak_rel, color=_bar_colors, alpha=0.8)
for b, v in zip(bars, _beam_peak_rel):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=_bp_value_fs)
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('Peak I / I_unblocked'); ax.set_title('Peak Intensity at Focus')
ax.grid(True, alpha=0.3, axis='y')

ax = axes_mc5_hg[0, 1]
ax.bar(_bar_x - _bar_w / 2, _beam_fwhm_x_um, _bar_w, color=_bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.bar(_bar_x + _bar_w / 2, _beam_fwhm_y_um, _bar_w, color=_bar_colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('FWHM (um)'); ax.set_title('FWHM at Focus (solid=x, hatched=y)')
ax.grid(True, alpha=0.3, axis='y')

ax = axes_mc5_hg[0, 2]
ax.bar(_bar_x - _bar_w / 2, _beam_w0_x_um, _bar_w, color=_bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.bar(_bar_x + _bar_w / 2, _beam_w0_y_um, _bar_w, color=_bar_colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('w0 (um)'); ax.set_title('1/e^2 Radius at Focus (solid=x, hatched=y)')
ax.grid(True, alpha=0.3, axis='y')

ax = axes_mc5_hg[1, 0]
bars = ax.bar(_bar_x, _beam_rayleigh_mm, color=_bar_colors, alpha=0.8)
for b, v in zip(bars, _beam_rayleigh_mm):
    if np.isfinite(v):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{v:.3f}', ha='center', fontsize=_bp_value_fs)
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('Rayleigh range (mm)'); ax.set_title('Confocal Parameter (on-axis half-max)')
ax.grid(True, alpha=0.3, axis='y')

ax = axes_mc5_hg[1, 1]
bars = ax.bar(_bar_x, _beam_focus_shift_um, color=_bar_colors, alpha=0.8)
for b, v in zip(bars, _beam_focus_shift_um):
    va = 'bottom' if v >= 0 else 'top'
    offset = 2.0 if v >= 0 else -2.0
    ax.text(b.get_x() + b.get_width() / 2, v + offset, f'{v:.1f}', ha='center', va=va, fontsize=_bp_value_fs)
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('Focus shift (um)'); ax.set_title('Focus Shift from f')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

ax = axes_mc5_hg[1, 2]
bars = ax.bar(_bar_x, _beam_trans_percent, color=_bar_colors, alpha=0.8)
for b, v in zip(bars, _beam_trans_percent):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{v:.1f}%', ha='center', fontsize=_bp_value_fs)
ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=_bp_tick_fs)
ax.set_ylabel('Transmission (%)'); ax.set_title('Power Transmission')
ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis='y')

for _ax in axes_mc5_hg.flat:
    _ax.set_title(_ax.get_title(), fontsize=_bp_title_fs, fontweight='bold')
    _ax.xaxis.label.set_fontsize(_bp_label_fs)
    _ax.yaxis.label.set_fontsize(_bp_label_fs)
    _ax.xaxis.label.set_fontweight('bold')
    _ax.yaxis.label.set_fontweight('bold')
    _ax.tick_params(axis='both', which='major', labelsize=_bp_tick_fs)

fig_mc5_hg.suptitle('Multi-Mask Beam Parameters Summary', fontsize=_bp_suptitle_fs, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
_beam_summary_png = f'mask_comparison_beam_parameters_HG_fitted_{_m2_tag}.png'
fig_mc5_hg.savefig(_beam_summary_png, dpi=400, bbox_inches='tight')
plt.close(fig_mc5_hg)
print(f"  Saved: {_beam_summary_png}")

# --- Print mask comparison table ---
print("\n" + "="*60)
print("MASK COMPARISON RESULTS")
print("="*60)

ref_mask = 'none'
for det_name, det_key in [('SLIT', 'yield_slit'), ('CIRCULAR', 'yield_circ')]:
    print(f"\n  {det_name} detection:")
    print(f"  {'Mask':12s} {'Trans%':>7s}", end='')
    for q in multi_q_list:
        print(f"  H{q:2d}", end='')
    print()
    for mask_name in mask_configs_mc:
        r = mask_yield_results[mask_name]
        print(f"  {mask_name:12s} {r['transmission']:7.1f}", end='')
        for q in multi_q_list:
            ref_yield = mask_yield_results[ref_mask][q][det_key]
            if ref_yield > 0:
                ratio = r[q][det_key] / ref_yield
            else:
                ratio = float('nan')
            print(f"  {ratio:5.3f}", end='')
        print()

# --- Figure: Mask Enhancement (2x3: slit + circular) ---
TIMER.start_section("Step 7: Mask figures")

fig_mask, axes_mask = plt.subplots(2, 3, figsize=(18, 12))
x_bar = np.arange(len(multi_q_list))
bar_w = 0.18
exp_enh_at_max_power = {f'H{q}': exp_enhancement[f'H{q}'][0] for q in multi_q_list if f'H{q}' in exp_enhancement}

for row_idx, (det_label, det_key, det_desc) in enumerate([
    ('Slit', 'yield_slit', f'{hhg_slit_width_mm}x{hhg_slit_height_mm}mm @ {hhg_slit_distance}m'),
    ('Circular', 'yield_circ', f'{hhg_aperture_radius_mm}mm @ {hhg_aperture_distance}m'),
]):
    ax = axes_mask[row_idx, 0]
    for im, mn in enumerate(mask_configs_mc):
        ratios = []
        for q in multi_q_list:
            ref_y = mask_yield_results[ref_mask][q][det_key]
            ratios.append(mask_yield_results[mn][q][det_key] / ref_y if ref_y > 0 else 0)
        ax.bar(x_bar + im * bar_w, ratios, bar_w, label=mask_labels_mc[mn],
               color=mask_colors_mc[mn], alpha=0.8)
    ax.set_xticks(x_bar + 1.5 * bar_w)
    ax.set_xticklabels([f'H{q}' for q in multi_q_list])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel(f'{det_label} Yield Ratio')
    ax.set_title(f'{det_label.upper()} ({det_desc})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes_mask[row_idx, 1]
    avg_ratios = []
    for mn in mask_configs_mc:
        rr = []
        for q in multi_q_list:
            ref_y = mask_yield_results[ref_mask][q][det_key]
            if ref_y > 0:
                rr.append(mask_yield_results[mn][q][det_key] / ref_y)
        avg_ratios.append(np.mean(rr))
    bars = ax.bar(range(len(mask_configs_mc)), avg_ratios,
                  color=[mask_colors_mc[m] for m in mask_configs_mc], alpha=0.8)
    for b, v in zip(bars, avg_ratios):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_xticks(range(len(mask_configs_mc)))
    ax.set_xticklabels([mask_labels_mc[m] for m in mask_configs_mc])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel(f'Mean {det_label} Yield Ratio')
    ax.set_title(f'Average Enhancement ({det_label})')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes_mask[row_idx, 2]
    sim_r = []
    exp_r = []
    for q in multi_q_list:
        ref_y = mask_yield_results[ref_mask][q][det_key]
        sim_r.append(mask_yield_results['circular'][q][det_key] / ref_y if ref_y > 0 else 0)
        exp_r.append(exp_enh_at_max_power.get(f'H{q}', float('nan')))
    ax.bar(x_bar - 0.15, exp_r, 0.3, label='Exp.', color='orange', alpha=0.8)
    ax.bar(x_bar + 0.15, sim_r, 0.3, label=f'Sim ({det_label.lower()})', color='blue' if row_idx == 0 else 'red', alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels([f'H{q}' for q in multi_q_list])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Enhancement')
    ax.set_title(f'Circular Mask: {det_label} Sim vs Exp')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

fig_mask.suptitle(f'Mask Shape Comparison — HG Model, M²=({M2x:.1f}, {M2y:.1f}), '
                  f'P={hhg_gas_pressure:.0f} mbar', fontsize=13)
plt.tight_layout()
fig_mask.savefig(f'mask_comparison_HG_fitted_{_m2_tag}.png', dpi=300, bbox_inches='tight')
print(f"Mask comparison figure saved to 'mask_comparison_HG_fitted_{_m2_tag}.png'")

# --- Per-mask detail figures ---
TIMER.start_section("Step 7: Per-mask detail figures")
q_main = hhg_harmonic_order
x_hhg_um = x_hhg_2d * 1e3
hhg_extent_um = [x_hhg_um[0], x_hhg_um[-1], x_hhg_um[0], x_hhg_um[-1]]
r_ub_detail = mask_yield_results['none']
I_q_ub_detail = np.abs(r_ub_detail[q_main]['E_q'])**2
c_nf = N_hhg_2d // 2

for mc_mname in ['circular', 'twosided', 'diagonal']:
    r_m = mask_yield_results[mc_mname]
    if r_m[q_main]['E_q'] is None:
        continue
    mc_label = mask_labels_mc[mc_mname]
    mc_color = mask_colors_mc[mc_mname]

    I_q_m = np.abs(r_m[q_main]['E_q'])**2
    yield_m = r_m[q_main]['yield_nf']
    yield_ub = r_ub_detail[q_main]['yield_nf']
    yield_ratio_m = yield_m / max(yield_ub, 1e-30)
    I_ff_m = r_m[q_main]['I_ff']
    ap_ratio_slit = r_m[q_main]['yield_slit'] / max(r_ub_detail[q_main]['yield_slit'], 1e-30)
    ap_ratio_circ = r_m[q_main]['yield_circ'] / max(r_ub_detail[q_main]['yield_circ'], 1e-30)
    vmax_shared = max(I_q_m.max(), I_q_ub_detail.max())

    fig_d, axes_d = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes_d[0, 0]
    ax.imshow(I_q_m.T / max(I_q_m.max(), 1e-30), extent=hhg_extent_um, aspect='equal',
              origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'{mc_label} HHG (H{q_main}) [peak={I_q_m.max()/max(vmax_shared,1e-30):.2e} rel]')
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
    ax.set_xlim([-50, 50]); ax.set_ylim([-50, 50])

    ax = axes_d[0, 1]
    ax.imshow(I_q_ub_detail.T / max(vmax_shared, 1e-30), extent=hhg_extent_um,
              aspect='equal', origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Unblocked HHG (H{q_main})')
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
    ax.set_xlim([-50, 50]); ax.set_ylim([-50, 50])

    ax = axes_d[0, 2]
    bars = ax.bar([mc_label, 'Unblocked'], [yield_m, yield_ub], color=[mc_color, 'salmon'], edgecolor='black')
    ax.set_ylabel('Total HHG Yield')
    ax.set_title(f'Integrated Yield (ratio = {yield_ratio_m:.3f})')
    for b, v in zip(bars, [yield_m, yield_ub]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.2e}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes_d[1, 0]
    lmax = max(I_q_m[c_nf,:].max(), I_q_ub_detail[c_nf,:].max(), 1e-30)
    ax.plot(x_hhg_um, I_q_m[c_nf,:]/lmax, color=mc_color, lw=2, label=f'{mc_label} (x)')
    ax.plot(x_hhg_um, I_q_m[:,c_nf]/lmax, color=mc_color, lw=1.5, ls='--', label=f'{mc_label} (y)')
    ax.plot(x_hhg_um, I_q_ub_detail[c_nf,:]/lmax, 'gray', lw=1.5, alpha=0.6, label='Unblocked (x)')
    ax.set_xlabel('Position (um)'); ax.set_ylabel(r'$|E_q|^2$ (norm)')
    ax.set_title('HHG Lineouts'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim([-50, 50])

    ax = axes_d[1, 1]
    bu_m = r_m[q_main]['buildup']
    bu_ub = r_ub_detail[q_main]['buildup']
    if bu_m is not None and bu_ub is not None:
        ax.plot(r_m['beam']['z_gas_mm'], np.abs(bu_m)**2/max(np.abs(bu_m[-1])**2,1e-30),
                color=mc_color, lw=2, label=mc_label)
        ax.plot(r_ub_detail['beam']['z_gas_mm'], np.abs(bu_ub)**2/max(np.abs(bu_ub[-1])**2,1e-30),
                'gray', lw=1.5, ls='--', label='Unblocked')
    ax.set_xlabel('z (mm)'); ax.set_ylabel(r'On-axis $|E_q|^2$ (norm)')
    ax.set_title('On-axis HHG Buildup'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # (1,2) Total yield buildup vs z (slit + circ + NF)
    ax = axes_d[1, 2]
    z_mm_m = r_m['beam']['z_gas_mm']
    z_mm_ub = r_ub_detail['beam']['z_gas_mm']
    for key, ls, lbl in [('yield_vs_z_slit', '-', 'slit'), ('yield_vs_z_circ', '--', 'circ'), ('yield_vs_z', ':', 'NF')]:
        yvz_m = r_m[q_main].get(key)
        yvz_ub = r_ub_detail[q_main].get(key)
        if yvz_m is not None and yvz_ub is not None:
            yvz_norm = max(yvz_ub[-1], 1e-30)
            ax.plot(z_mm_m, yvz_m / yvz_norm, color=mc_color, linewidth=1.5, linestyle=ls, label=f'{mc_label} ({lbl})')
            ax.plot(z_mm_ub, yvz_ub / yvz_norm, color='gray', linewidth=1, linestyle=ls, alpha=0.5)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Integrated yield (normalized)')
    ax.set_title('Yield Buildup (solid=slit, dash=circ, dot=NF)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig_d.suptitle(f'Macroscopic HHG Yield — {mc_label} Mask, HG Model, '
                   f'P={hhg_gas_pressure:.0f} mbar, H{q_main}', fontsize=13)
    plt.tight_layout()
    fname_d = f'mask_detail_HG_{mc_mname}_fitted_{_m2_tag}.png'
    fig_d.savefig(fname_d, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname_d}")
    plt.close(fig_d)

# --- Beam Parameters Summary ---
print("Generating beam parameters summary...")
fig_bp, axes_bp = plt.subplots(2, 3, figsize=(16, 10))
bp_bar_x = np.arange(len(mask_configs_mc))
bp_colors = [mask_colors_mc[m] for m in mask_configs_mc]
bp_labels = [mask_labels_mc[m] for m in mask_configs_mc]

I_peak_ub_ref = mask_yield_results['none']['beam'].get('focus_peak_I', mask_yield_results['none']['beam']['peak_I'])
peaks_norm = [
    mask_yield_results[m]['beam'].get('focus_peak_I', mask_yield_results[m]['beam']['peak_I']) / I_peak_ub_ref
    for m in mask_configs_mc
]
ax = axes_bp[0, 0]
bars = ax.bar(bp_bar_x, peaks_norm, color=bp_colors, alpha=0.8)
for b, v in zip(bars, peaks_norm):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.3f}', ha='center', fontsize=9)
ax.set_xticks(bp_bar_x); ax.set_xticklabels(bp_labels, fontsize=9)
ax.set_ylabel('Peak I / I_unblocked'); ax.set_title('Peak Intensity at Focus')
ax.grid(True, alpha=0.3, axis='y')

ax = axes_bp[0, 1]
for mn in mask_configs_mc:
    r = mask_yield_results[mn]
    ax.plot(r['beam']['z_gas_mm'], r['beam']['I_onaxis_Wcm2'], color=mask_colors_mc[mn], lw=1.5, label=mask_labels_mc[mn])
ax.axvline(focal_length, color='gray', ls=':', lw=0.8)
ax.set_xlabel('z (mm)'); ax.set_ylabel('Intensity (W/cm²)'); ax.set_title('On-Axis Intensity')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes_bp[0, 2]
for mn in mask_configs_mc:
    r = mask_yield_results[mn]
    ax.plot(r['beam']['z_gas_mm'], r['beam']['nf_onaxis'], color=mask_colors_mc[mn], lw=1.5, label=mask_labels_mc[mn])
ax.axvline(focal_length, color='gray', ls=':', lw=0.8)
ax.set_xlabel('z (mm)'); ax.set_ylabel('Ionization fraction')
ax.set_title(f'Ionization ({hhg_gas_type.capitalize()}, Ip={gas["Ip_eV"]:.2f} eV)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes_bp[1, 0]
for mn in mask_configs_mc:
    r = mask_yield_results[mn]
    ax.plot(r['beam']['z_gas_mm'], r['beam']['gouy_grad'], color=mask_colors_mc[mn], lw=1.5, label=mask_labels_mc[mn])
ax.axvline(focal_length, color='gray', ls=':', lw=0.8)
ax.set_xlabel('z (mm)'); ax.set_ylabel(r'd$\phi$/dz (rad/m)')
ax.set_title('On-Axis Gouy Phase Gradient'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes_bp[1, 1]
for mn in mask_configs_mc:
    r = mask_yield_results[mn]
    ax.plot(x_hhg_um, r['beam']['I_focus_x'], color=mask_colors_mc[mn], lw=1.5, label=mask_labels_mc[mn])
ax.set_xlabel('x (um)'); ax.set_ylabel('Intensity (W/cm²)')
ax.set_title('Intensity at Focus (x lineout)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes_bp[1, 2]
trans_vals = [mask_yield_results[m]['transmission'] for m in mask_configs_mc]
bars = ax.bar(bp_bar_x, trans_vals, color=bp_colors, alpha=0.8)
for b, v in zip(bars, trans_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=9)
ax.set_xticks(bp_bar_x); ax.set_xticklabels(bp_labels, fontsize=9)
ax.set_ylabel('Transmission (%)'); ax.set_title('Power Transmission')
ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis='y')

fig_bp.suptitle(f'Multi-Mask Beam Parameters — HG Model, M²=({M2x:.1f}, {M2y:.1f})', fontsize=14)
plt.tight_layout()
fig_bp.savefig(f'mask_beam_params_HG_fitted_{_m2_tag}.png', dpi=300, bbox_inches='tight')
print(f"Beam parameters figure saved to 'mask_beam_params_HG_fitted_{_m2_tag}.png'")


# =============================================================================
# Multi-Mask Focus Intensity Comparison (2D + 3D) — HG Model
# =============================================================================
TIMER.start_section("Focus intensity comparison (2D + 3D)")
print("\nGenerating multi-mask focus intensity comparison...")

x_focus_um = x_hhg_2d * 1e3
I_peak_ub_focus = mask_yield_results['none']['beam'].get('focus_peak_I', mask_yield_results['none']['beam']['peak_I'])

fig_fi, axes_fi = plt.subplots(2, 4, figsize=(22, 10))
ext_focus = [x_focus_um[0], x_focus_um[-1], x_focus_um[0], x_focus_um[-1]]

for col, mn in enumerate(mask_configs_mc):
    I2d = mask_yield_results[mn]['beam']['I_2d_focus']
    peak_ratio = mask_yield_results[mn]['beam'].get('focus_peak_I', mask_yield_results[mn]['beam']['peak_I']) / max(I_peak_ub_focus, 1e-30)
    ax = axes_fi[0, col]
    I2d_norm = I2d / max(I2d.max(), 1e-30)
    I2d_log = np.log10(np.clip(I2d_norm, 1e-4, None))
    ax.imshow(I2d_log.T, extent=ext_focus, origin='lower', cmap='hot', vmin=-4, vmax=0)
    ax.set_title(f'{mask_labels_mc[mn]} ({peak_ratio:.2f}x)', fontsize=11, color=mask_colors_mc[mn])
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_xlabel('x (μm)')
    if col == 0: ax.set_ylabel('y (μm)')
    ax = axes_fi[1, col]
    center_fi = N_hhg_2d // 2
    I_x = I2d[center_fi, :] / max(I2d.max(), 1e-30)
    I_y = I2d[:, center_fi] / max(I2d.max(), 1e-30)
    ax.plot(x_focus_um, I_x, color=mask_colors_mc[mn], linewidth=1.5, label='x')
    ax.plot(x_focus_um, I_y, color=mask_colors_mc[mn], linewidth=1, linestyle='--', label='y')
    I_x_ub = mask_yield_results['none']['beam']['I_2d_focus'][center_fi, :]
    ax.plot(x_focus_um, I_x_ub / max(I_x_ub.max(), 1e-30), 'gray', linewidth=1, alpha=0.5, label='Unblocked')
    ax.set_xlim(-50, 50); ax.set_ylim(0, 1.1)
    ax.set_xlabel('Position (μm)')
    if col == 0: ax.set_ylabel('I / I_max')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig_fi.suptitle(f'Multi-Mask Focus Intensity — HG Model, M²=({M2x},{M2y})', fontsize=14)
plt.tight_layout()
fig_fi.savefig(f'mask_focus_intensity_HG_{_m2_tag}.png', dpi=300)
print(f"  Saved: mask_focus_intensity_HG_{_m2_tag}.png")

fig_3d, axes_3d = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': '3d'})
ds = max(1, N_hhg_2d // 128)
x_ds = x_focus_um[::ds]; y_ds = x_focus_um[::ds]
X_3d, Y_3d = np.meshgrid(x_ds, y_ds)
crop_x = (x_ds >= -50) & (x_ds <= 50); crop_y = (y_ds >= -50) & (y_ds <= 50)
X_crop = X_3d[np.ix_(crop_y, crop_x)]; Y_crop = Y_3d[np.ix_(crop_y, crop_x)]

for col, mn in enumerate(mask_configs_mc):
    ax = axes_3d[col]
    I2d = mask_yield_results[mn]['beam']['I_2d_focus']
    I_crop = I2d[::ds, ::ds][np.ix_(crop_y, crop_x)] / max(I_peak_ub_focus, 1e-30)
    ax.plot_surface(X_crop, Y_crop, I_crop, cmap='hot', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    peak_ratio = mask_yield_results[mn]['beam'].get('focus_peak_I', mask_yield_results[mn]['beam']['peak_I']) / max(I_peak_ub_focus, 1e-30)
    ax.set_title(f'{mask_labels_mc[mn]} ({peak_ratio:.2f}x)', fontsize=11, color=mask_colors_mc[mn])
    ax.set_xlabel('x (μm)', fontsize=8); ax.set_ylabel('y (μm)', fontsize=8)
    ax.set_zlabel('I / I_ub', fontsize=8)
    ax.view_init(elev=35, azim=-60); ax.tick_params(labelsize=7)

fig_3d.suptitle('3D Focus Intensity — HG Model', fontsize=14)
plt.tight_layout()
fig_3d.savefig(f'mask_focus_3d_HG_{_m2_tag}.png', dpi=300)
print(f"  Saved: mask_focus_3d_HG_{_m2_tag}.png")

# =============================================================================
# Near-Field HHG Yield 2D — All Masks (H21)
# =============================================================================
TIMER.start_section("Near-field HHG 2D maps")
print(f"\nGenerating near-field HHG 2D maps for H{hhg_harmonic_order}...")

fig_nf2d, axes_nf2d = plt.subplots(2, 4, figsize=(24, 11))
nf2d_center = N_hhg_2d // 2
x_nf_um = x_hhg_2d * 1e3
ext_nf2d = [x_nf_um[0], x_nf_um[-1], x_nf_um[0], x_nf_um[-1]]

nf2d_data = {}
nf2d_max_global = 0
for mn in mask_configs_mc:
    E_q_mn = mask_yield_results[mn][hhg_harmonic_order]['E_q']
    if E_q_mn is not None:
        I_nf = np.abs(E_q_mn)**2
        nf2d_data[mn] = I_nf
        nf2d_max_global = max(nf2d_max_global, I_nf.max())

for col, mn in enumerate(mask_configs_mc):
    ax = axes_nf2d[0, col]
    if mn in nf2d_data:
        I_nf = nf2d_data[mn]
        I_log = np.log10(np.clip(I_nf / max(nf2d_max_global, 1e-30), 1e-6, None))
        ax.imshow(I_log.T, extent=ext_nf2d, origin='lower', cmap='hot', vmin=-4, vmax=0)
        peak_ratio = I_nf.max() / max(nf2d_max_global, 1e-30)
        ax.set_title(f'{mask_labels_mc[mn]}\nyield={mask_yield_results[mn][hhg_harmonic_order]["yield_nf"]:.2e}, peak={peak_ratio:.2f}x',
                     fontsize=9, color=mask_colors_mc[mn])
    ax.set_xlim(-80, 80); ax.set_ylim(-80, 80)
    ax.set_xlabel('x (μm)')
    if col == 0: ax.set_ylabel('y (μm)')

for col, mn in enumerate(mask_configs_mc):
    ax = axes_nf2d[1, col]
    if mn in nf2d_data:
        I_nf = nf2d_data[mn]
        I_x = I_nf[nf2d_center, :] / max(nf2d_max_global, 1e-30)
        I_y = I_nf[:, nf2d_center] / max(nf2d_max_global, 1e-30)
        ax.plot(x_nf_um, I_x, color=mask_colors_mc[mn], linewidth=1.5, label='x')
        ax.plot(x_nf_um, I_y, color=mask_colors_mc[mn], linewidth=1, linestyle='--', label='y')
        I_ref = nf2d_data.get('none')
        if I_ref is not None:
            ax.plot(x_nf_um, I_ref[nf2d_center, :] / max(nf2d_max_global, 1e-30),
                    'gray', linewidth=1, alpha=0.5, label='No mask')
    ax.set_xlim(-80, 80)
    ax.set_xlabel('Position (μm)')
    if col == 0: ax.set_ylabel(r'$|E_q|^2$ / max (abs.)')
    ax.legend(fontsize=7)
    ax.set_title(f'{mask_labels_mc[mn]} lineouts (abs.)', fontsize=9)
    ax.grid(True, alpha=0.3)

fig_nf2d.suptitle(f'Near-Field HHG Yield — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'P={hhg_gas_pressure:.0f} mbar, HG M²=({M2x},{M2y})', fontsize=14)
plt.tight_layout()
fig_nf2d.savefig(f'hhg_nearfield_2d_HG_{_m2_tag}.png', dpi=300)
print(f"  Saved: hhg_nearfield_2d_HG_{_m2_tag}.png")

# =============================================================================
# Far-Field HHG 2D — All Masks (H21)
# =============================================================================
TIMER.start_section("Far-field HHG 2D maps")
print(f"\nGenerating far-field HHG 2D maps for H{hhg_harmonic_order}...")

fig_ff2d, axes_ff2d = plt.subplots(2, 4, figsize=(24, 11))
ff2d_center = N_hhg_2d // 2
ff2d_data = {}
ff2d_max_global = 0
for mn in mask_configs_mc:
    I_ff_mn = mask_yield_results[mn][hhg_harmonic_order]['I_ff']
    if I_ff_mn is not None:
        ff2d_data[mn] = I_ff_mn
        ff2d_max_global = max(ff2d_max_global, I_ff_mn.max())

dt_ff = mask_yield_results['none'][hhg_harmonic_order]['dtheta']
theta_mrad_ff = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_ff * 1e3
ext_ff2d = [theta_mrad_ff[0], theta_mrad_ff[-1], theta_mrad_ff[0], theta_mrad_ff[-1]]
circ_half_mrad = circ_half_angle * 1e3

for col, mn in enumerate(mask_configs_mc):
    ax = axes_ff2d[0, col]
    if mn in ff2d_data:
        I_ff = ff2d_data[mn]
        I_ff_log = np.log10(np.clip(I_ff / max(ff2d_max_global, 1e-30), 1e-6, None))
        im = ax.imshow(I_ff_log, extent=ext_ff2d, aspect='equal', origin='lower',
                       cmap='hot', vmin=-4, vmax=0, interpolation='bicubic')
        circ_patch = plt.Circle((0, 0), circ_half_mrad, fill=False, color='white',
                                linestyle='--', linewidth=1.5)
        ax.add_patch(circ_patch)
        ap_yield = mask_yield_results[mn][hhg_harmonic_order]['yield_slit']
        peak_ratio = I_ff.max() / max(ff2d_max_global, 1e-30)
        ax.set_title(f'{mask_labels_mc[mn]}\nap_yield={ap_yield:.2e}, peak={peak_ratio:.2f}x',
                     fontsize=9, color=mask_colors_mc[mn])
        if col == 3:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10(I/I_max)')
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel(r'$\theta_x$ (mrad)')
    if col == 0: ax.set_ylabel(r'$\theta_y$ (mrad)')

for col, mn in enumerate(mask_configs_mc):
    ax = axes_ff2d[1, col]
    if mn in ff2d_data:
        I_ff = ff2d_data[mn]
        I_ff_x = I_ff[ff2d_center, :] / max(I_ff[ff2d_center, :].max(), 1e-30)
        I_ff_y = I_ff[:, ff2d_center] / max(I_ff[:, ff2d_center].max(), 1e-30)
        ax.semilogy(theta_mrad_ff, I_ff_x, color=mask_colors_mc[mn], linewidth=1.5, label=r'$\theta_x$')
        ax.semilogy(theta_mrad_ff, I_ff_y, color=mask_colors_mc[mn], linewidth=1.5, linestyle='--', alpha=0.7, label=r'$\theta_y$')
        ax.axvline(-circ_half_mrad, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(circ_half_mrad, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Angle (mrad)')
    if col == 0: ax.set_ylabel(r'$|E_{ff}|^2$ (self-norm, log)')
    ax.set_title(f'{mask_labels_mc[mn]} angular', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 10); ax.set_ylim(1e-6, 1.5)

fig_ff2d.suptitle(f'Far-Field HHG — {hhg_gas_type.capitalize()}, H{hhg_harmonic_order}, '
                   f'Aperture: {hhg_aperture_radius_mm} mm at {hhg_aperture_distance} m, '
                   f'HG M²=({M2x},{M2y})', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_ff2d.savefig(f'hhg_farfield_2d_HG_{_m2_tag}.png', dpi=300)
print(f"  Saved: hhg_farfield_2d_HG_{_m2_tag}.png")


# =============================================================================
# Per-Harmonic Mask Comparison Figures (H11-H21) — HG Model
# =============================================================================
TIMER.start_section("Per-harmonic mask figures")
print("\n" + "="*60)
print("PER-HARMONIC MASK COMPARISON FIGURES")
print("="*60)

x_hhg_um_ph = x_hhg_2d * 1e3

for q in multi_q_list:
    print(f"  Generating per-harmonic figure for H{q}...")
    ref_slit = mask_yield_results[ref_mask][q]['yield_slit']
    ref_circ = mask_yield_results[ref_mask][q]['yield_circ']
    ref_nf_q = mask_yield_results[ref_mask][q]['yield_nf']

    fig_ph, axes_ph = plt.subplots(2, 3, figsize=(18.5, 9.6))

    # --- (0,0) Bar chart ---
    ax = axes_ph[0, 0]
    _det_names = ['Slit', 'Circular', 'Near-field']
    n_det = 3
    n_mask = len(mask_configs_mc)
    x_group = np.arange(n_det)
    w_ph = 0.18
    offsets_ph = np.arange(n_mask) * w_ph - (n_mask - 1) * w_ph / 2
    _ratio_max_ph = 1.0
    for im, mn in enumerate(mask_configs_mc):
        rq = mask_yield_results[mn][q]
        enh_slit = rq['yield_slit'] / ref_slit if ref_slit > 0 else 0
        enh_circ = rq['yield_circ'] / ref_circ if ref_circ > 0 else 0
        enh_nf = rq['yield_nf'] / ref_nf_q if ref_nf_q > 0 else 0
        vals = [enh_slit, enh_circ, enh_nf]
        _ratio_max_ph = max(_ratio_max_ph, float(np.nanmax(vals)))
        bars = ax.bar(x_group + offsets_ph[im], vals, w_ph, label=mask_labels_mc[mn],
                      color=mask_colors_mc[mn], alpha=0.92,
                      edgecolor='black', linewidth=0.55)
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

    # --- (0,1) Near-field 2×2 ---
    ax_main = axes_ph[0, 1]; ax_main.set_visible(False)
    nf_global = max((np.abs(mask_yield_results[m][q]['E_q'])**2).max()
                    for m in mask_configs_mc if mask_yield_results[m][q]['E_q'] is not None)
    ext_nf_ph = [x_hhg_um_ph[0], x_hhg_um_ph[-1], x_hhg_um_ph[0], x_hhg_um_ph[-1]]
    gs_nf = fig_ph.add_gridspec(2, 2, left=0.38, right=0.63, top=0.88, bottom=0.52,
                                 wspace=0.15, hspace=0.25)
    for idx, mn in enumerate(mask_configs_mc):
        ax_sub = fig_ph.add_subplot(gs_nf[idx // 2, idx % 2])
        E_q_mn = mask_yield_results[mn][q]['E_q']
        if E_q_mn is not None:
            I_nf = np.abs(E_q_mn)**2
            rel = I_nf.max() / max(nf_global, 1e-30)
            ax_sub.imshow(I_nf.T / max(nf_global, 1e-30), extent=ext_nf_ph,
                          origin='lower', cmap='magma', vmin=0, vmax=1)
            ax_sub.set_title(f'{mask_labels_mc[mn]} ({rel:.2f}x)', fontsize=11)
        else:
            ax_sub.set_title(f'{mask_labels_mc[mn]}', fontsize=11)
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
    for mn in mask_configs_mc:
        rq = mask_yield_results[mn][q]
        I_ff_mn = rq['I_ff']
        if I_ff_mn is not None:
            dt_q = rq['dtheta']
            th_ax = (np.arange(N_hhg_2d) - N_hhg_2d // 2) * dt_q * 1e3
            lineout = I_ff_mn[N_hhg_2d // 2, :]
            if lineout.max() > 0:
                ax.semilogy(th_ax, lineout / lineout.max(), color=mask_colors_mc[mn],
                            label=mask_labels_mc[mn], linewidth=2.0)
    ax.axvline(-slit_half_angle_x * 1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75)
    ax.axvline(slit_half_angle_x * 1e3, color='#5B7FA6', linestyle='--', linewidth=1.1, alpha=0.75, label='Slit')
    ax.axvline(-circ_half_angle * 1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75)
    ax.axvline(circ_half_angle * 1e3, color='#C78173', linestyle=':', linewidth=1.2, alpha=0.75, label='Circ')
    ax.set_xlabel(r'$\theta_x$ (mrad)')
    ax.set_ylabel(r'Far-field $|E|^2$ (self norm.)')
    ax.set_title('Far-Field Angular Lineout')
    ax.set_xlim([-10, 10])
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (1,1) On-axis Gouy phase gradient ---
    ax = axes_ph[1, 1]
    for mn in mask_configs_mc:
        r = mask_yield_results[mn]
        ax.plot(r['beam']['z_gas_mm'], r['beam']['gouy_grad'], color=mask_colors_mc[mn],
                label=mask_labels_mc[mn], linewidth=2.0)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel(r'd$\phi$/dz (rad/m)')
    ax.set_title('On-Axis Gouy Phase Gradient')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (0,2) HHG near-field lineouts ---
    ax = axes_ph[0, 2]
    center_ph = N_hhg_2d // 2
    for mn in mask_configs_mc:
        E_q_mn = mask_yield_results[mn][q]['E_q']
        if E_q_mn is not None:
            I_nf_mn = np.abs(E_q_mn)**2
            ax.plot(x_hhg_um_ph, I_nf_mn[center_ph, :] / max(nf_global, 1e-30),
                    color=mask_colors_mc[mn], linewidth=2.0, label=f'{mask_labels_mc[mn]} (x)')
            ax.plot(x_hhg_um_ph, I_nf_mn[:, center_ph] / max(nf_global, 1e-30),
                    color=mask_colors_mc[mn], linewidth=1.4, linestyle='--')
    ax.set_xlabel(r'Position ($\mu$m)')
    ax.set_ylabel(r'$|E_q|^2$ (normalized)')
    ax.set_title('HHG Near-Field Lineouts')
    ax.set_xlim([-50, 50])
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    # --- (1,2) Yield buildup vs z (slit + circ + NF) ---
    ax = axes_ph[1, 2]
    ref_yvz_slit = mask_yield_results[ref_mask][q].get('yield_vs_z_slit')
    ref_yvz_circ = mask_yield_results[ref_mask][q].get('yield_vs_z_circ')
    ref_yvz_nf = mask_yield_results[ref_mask][q].get('yield_vs_z')
    yvz_norm_slit_ph = max(ref_yvz_slit[-1], 1e-30) if ref_yvz_slit is not None else 1e-30
    yvz_norm_circ_ph = max(ref_yvz_circ[-1], 1e-30) if ref_yvz_circ is not None else 1e-30
    yvz_norm_nf_ph = max(ref_yvz_nf[-1], 1e-30) if ref_yvz_nf is not None else 1e-30
    for mn in mask_configs_mc:
        z_mm = mask_yield_results[mn]['beam']['z_gas_mm']
        yvz_s = mask_yield_results[mn][q].get('yield_vs_z_slit')
        yvz_c = mask_yield_results[mn][q].get('yield_vs_z_circ')
        yvz_n = mask_yield_results[mn][q].get('yield_vs_z')
        if yvz_s is not None:
            ax.plot(z_mm, yvz_s / yvz_norm_slit_ph,
                    color=mask_colors_mc[mn], linewidth=2.4, label=f'{mask_labels_mc[mn]} (slit)')
        if yvz_c is not None:
            ax.plot(z_mm, yvz_c / yvz_norm_circ_ph,
                    color=mask_colors_mc[mn], linewidth=1.8, linestyle='--')
        if yvz_n is not None:
            ax.plot(z_mm, yvz_n / yvz_norm_nf_ph,
                    color=mask_colors_mc[mn], linewidth=1.8, linestyle=':')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Integrated yield (normalized)')
    ax.set_title('Yield buildup')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='0.75')
    style_paper_axis(ax, grid=True)

    fig_ph.suptitle(f'Mask Comparison -- H{q}, {hhg_gas_type.capitalize()}, '
                     f'P={hhg_gas_pressure:.0f} mbar, HG M$^2$=({M2x},{M2y})',
                     fontsize=17, fontweight='bold')
    finalize_paper_figure(fig_ph)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'hhg_mask_HG_H{q}_{_m2_tag}.png', dpi=400)
    print(f"    Saved: hhg_mask_HG_H{q}_{_m2_tag}.png")
    plt.close(fig_ph)

# Save compact mask-comparison arrays for standalone plotting.
try:
    _mask_npz_path = f'hhg_mask_comparison_data_HG_{lavg_tag}_{pressure_tag}_{_m2_tag}.npz'
    print(f"\n  Saving HG mask-comparison plotting data to {_mask_npz_path} ...")

    _mnames_save = mask_configs_mc
    _harmonics_save = np.array(multi_q_list, dtype=np.int32)
    _mask_names_save = np.array(_mnames_save)
    _mask_labels_save = np.array([mask_labels_mc[mn] for mn in _mnames_save])
    _nq_save = len(_harmonics_save)
    _nm_save = len(_mnames_save)

    _x_hhg_um_save = (x_hhg_2d * 1e3).astype(np.float32)
    _nf_step = max(int(np.ceil(N_hhg_2d / 512)), 1)
    _nf_x_um = _x_hhg_um_save[::_nf_step]
    _center_save = N_hhg_2d // 2

    _yield_slit = np.zeros((_nq_save, _nm_save), dtype=np.float64)
    _yield_circ = np.zeros_like(_yield_slit)
    _yield_nf = np.zeros_like(_yield_slit)
    _nearfield_intensity = np.zeros(
        (_nq_save, _nm_save, len(_nf_x_um), len(_nf_x_um)), dtype=np.float32
    )
    _nearfield_peak_rel = np.zeros((_nq_save, _nm_save), dtype=np.float32)
    _nearfield_lineout_x = np.zeros((_nq_save, _nm_save, len(_x_hhg_um_save)), dtype=np.float32)
    _nearfield_lineout_y = np.zeros_like(_nearfield_lineout_x)
    _ff_theta_mrad = np.zeros((_nq_save, N_hhg_2d), dtype=np.float32)
    _ff_lineout_x = np.zeros((_nq_save, _nm_save, N_hhg_2d), dtype=np.float32)
    _ff_lineout_y = np.zeros_like(_ff_lineout_x)

    _z_len = max(len(mask_yield_results[mn]['beam']['z_gas_mm']) for mn in _mnames_save)
    _z_gas_mm_masks = np.full((_nm_save, _z_len), np.nan, dtype=np.float32)
    _ionization_onaxis = np.full((_nm_save, _z_len), np.nan, dtype=np.float32)
    _I_onaxis_Wcm2 = np.full((_nm_save, _z_len), np.nan, dtype=np.float32)
    _gouy_grad = np.full((_nm_save, _z_len), np.nan, dtype=np.float32)
    _yield_vs_z = np.full((_nq_save, _nm_save, _z_len), np.nan, dtype=np.float64)
    _yield_vs_z_slit = np.full_like(_yield_vs_z, np.nan)
    _yield_vs_z_circ = np.full_like(_yield_vs_z, np.nan)

    for _iq, _q in enumerate(_harmonics_save):
        _nf_global_q = max(
            (np.abs(mask_yield_results[_mn][int(_q)]['E_q'])**2).max()
            for _mn in _mnames_save
        )
        _nf_global_q = max(_nf_global_q, 1e-30)

        for _im, _mn in enumerate(_mnames_save):
            _rq = mask_yield_results[_mn][int(_q)]
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
            _ff_theta_mrad[_iq] = (
                (np.arange(N_hhg_2d) - _center_save) * float(_rq['dtheta']) * 1e3
            ).astype(np.float32)

            _beam = mask_yield_results[_mn]['beam']
            _len_z = len(_beam['z_gas_mm'])
            _z_gas_mm_masks[_im, :_len_z] = _beam['z_gas_mm'].astype(np.float32)
            _ionization_onaxis[_im, :_len_z] = _beam['nf_onaxis'].astype(np.float32)
            _I_onaxis_Wcm2[_im, :_len_z] = _beam['I_onaxis_Wcm2'].astype(np.float32)
            _gouy_grad[_im, :_len_z] = _beam['gouy_grad'].astype(np.float32)

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
        nearfield_lineout_x_um=_x_hhg_um_save,
        nearfield_lineout_x_norm=_nearfield_lineout_x,
        nearfield_lineout_y_norm=_nearfield_lineout_y,
        nearfield_extent_um=np.array([_x_hhg_um_save[0], _x_hhg_um_save[-1],
                                      _x_hhg_um_save[0], _x_hhg_um_save[-1]], dtype=np.float32),
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
        focus_peak_I_Wcm2=_beam_summary['focus_peak_I_Wcm2'].astype(np.float64),
        focus_peak_rel=_beam_summary['focus_peak_rel'].astype(np.float64),
        fwhm_x_um=_beam_summary['fwhm_x_um'].astype(np.float64),
        fwhm_y_um=_beam_summary['fwhm_y_um'].astype(np.float64),
        w0_x_um=_beam_summary['w0_x_um'].astype(np.float64),
        w0_y_um=_beam_summary['w0_y_um'].astype(np.float64),
        rayleigh_range_mm=_beam_summary['rayleigh_range_mm'].astype(np.float64),
        focus_z_mm=_beam_summary['focus_z_mm'].astype(np.float64),
        focus_shift_um=_beam_summary['focus_shift_um'].astype(np.float64),
        transmission_percent=_beam_summary['transmission_percent'].astype(np.float64),
        slit_half_angle_mrad=np.float64(slit_half_angle_x * 1e3),
        aperture_half_angle_mrad=np.float64(circ_half_angle * 1e3),
        hhg_gas_type=np.array(hhg_gas_type),
        hhg_gas_pressure=np.float64(hhg_gas_pressure),
        M2x=np.float64(M2x),
        M2y=np.float64(M2y),
        lavg_tag=np.array(lavg_tag),
        pressure_tag=np.array(pressure_tag),
    )
    print(f"  Saved: {_mask_npz_path} ({os.path.getsize(_mask_npz_path) / 1024 / 1024:.1f} MB)")
except Exception as _mask_npz_exc:
    print(f"  WARNING: could not save HG mask-comparison NPZ: {_mask_npz_exc}")


TIMER.summary()
plt.show()
