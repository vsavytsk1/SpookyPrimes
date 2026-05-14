#!/usr/bin/env python3
"""
Three-well biased GaAs/AlGaAs active region, Khalatpour-style direct-phonon
THz QCL, target ~3.4 THz.

Compute the interface-roughness contribution to pure dephasing of the
upper-to-lower laser transition, per interface and total, with no fitted
parameters.  Compare to experimental T2 ~ 1.0-1.5 ps in cool THz QCLs.

The geometry is a biased three-well stage:

    | barrier | well_1 | barrier | well_2 | barrier | well_3 | barrier |
                 ULS                LLS                injector

The bias provides the THz transition between ULS and LLS, and the third
(narrowest) well plus a thin LO-phonon-assisted barrier extracts to the
next injector.  Layer thicknesses chosen to match the design family in
Khalatpour et al., Nat. Photonics 15, 16 (2021).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
hbar = 1.054_571_817e-34
m_e  = 9.109_383_7015e-31
e    = 1.602_176_634e-19
kB   = 1.380_649e-23
meV  = 1e-3 * e
nm   = 1e-9

# GaAs / Al_x Ga_{1-x} As at x = 0.15 (cool THz QCL barrier composition)
m_star = 0.067 * m_e
V0     = 0.135 * e          # 135 meV conduction band offset

# Ando interface-roughness parameters (Gaussian autocorrelation)
Delta_rms = 0.2825 * nm     # 1 monolayer GaAs
Lambda_c  = 5.0    * nm     # typical MBE correlation length
T_K       = 50.0            # cool-QCL operation

# ---------------------------------------------------------------------
# Layer structure (one period of a three-well direct-phonon design)
# Layer thicknesses in nm; barriers shown in barrier_layers list,
# wells in well_layers list, alternating starting with a barrier.
# Values chosen in the design family of Khalatpour et al. 2021, slightly
# adjusted so the 1->2 transition lands near 3.4 THz at the design bias.
# ---------------------------------------------------------------------
# Thicknesses (nm), alternating: barrier, well, barrier, well, ...
layers = [
    ('B', 4.3),   # injection barrier
    ('W', 8.5),   # well 1 (upper laser state)
    ('B', 2.4),   # barrier
    ('W', 8.5),   # well 2 (lower laser state, ULS-LLS pair)
    ('B', 4.1),   # barrier
    ('W', 16.1),  # well 3 (extractor / injector)
    ('B', 4.3),   # closing barrier of stage
]

# Applied bias (V/cm) — direct-phonon designs typically run at ~10–12 kV/cm
F_bias_kVcm = 11.0
F = F_bias_kVcm * 1e5  # V/m

# ---------------------------------------------------------------------
# Build the potential profile on a fine grid
# ---------------------------------------------------------------------
def build_profile(layers, dx_nm=0.05, pad_nm=5.0):
    total = sum(t for _, t in layers) + 2 * pad_nm
    N = int(total / dx_nm) + 1
    z = np.linspace(0, total * nm, N)
    dz = z[1] - z[0]
    V_band = np.zeros_like(z)
    # pad region: barrier material
    V_band[:] = V0
    zc = pad_nm * nm
    interfaces = []
    for kind, t in layers:
        z_end = zc + t * nm
        mask = (z >= zc) & (z < z_end)
        V_band[mask] = 0.0 if kind == 'W' else V0
        interfaces.append((zc, kind))     # left edge of this layer
        zc = z_end
    interfaces.append((zc, 'end'))
    V_total = V_band - e * F * (z - z[0])  # tilt by applied bias
    return z, dz, V_total, V_band, interfaces

z, dz, V, V_band, interfaces = build_profile(layers)

# ---------------------------------------------------------------------
# Solve Schrödinger equation
# ---------------------------------------------------------------------
diag = hbar**2 / (m_star * dz**2) + V
off  = -hbar**2 / (2 * m_star * dz**2) * np.ones(len(z) - 1)
E, psi = eigh_tridiagonal(diag, off, select='i', select_range=(0, 7))
for i in range(psi.shape[1]):
    psi[:, i] /= np.sqrt(np.sum(psi[:, i] ** 2) * dz)

# ---------------------------------------------------------------------
# Identify ULS and LLS:  the pair of states whose splitting is in the
# THz band (10–20 meV) AND whose mean position is in the first two wells.
# ---------------------------------------------------------------------
pad_nm = 5.0
z_pair_left  = pad_nm * nm
z_pair_right = (pad_nm + layers[0][1] + layers[1][1] + layers[2][1] + layers[3][1]) * nm

def in_pair_region(p):
    return np.sum(p**2 * (z >= z_pair_left) * (z <= z_pair_right)) * dz

weights = np.array([in_pair_region(psi[:, i]) for i in range(psi.shape[1])])
order = np.argsort(-weights)

# Find ULS, LLS as the two highest-weighted states with splitting in 8-20 meV
pair = None
for i in range(len(order)):
    for j in range(i + 1, len(order)):
        a, b = order[i], order[j]
        gap = abs(E[a] - E[b]) / meV
        if 8.0 < gap < 22.0 and weights[a] > 0.4 and weights[b] > 0.4:
            ULS, LLS = (a, b) if E[a] > E[b] else (b, a)
            pair = (ULS, LLS)
            break
    if pair:
        break

if pair is None:
    # fallback: just pick the two most localized states in the pair region
    ULS, LLS = order[0], order[1]
    if E[LLS] > E[ULS]:
        ULS, LLS = LLS, ULS

E_ULS, E_LLS = E[ULS], E[LLS]
psi_ULS, psi_LLS = psi[:, ULS], psi[:, LLS]
gap_meV = (E_ULS - E_LLS) / meV
f_THz = gap_meV * 1e-3 * e / (2 * np.pi * hbar) / 1e12

print('=' * 72)
print(' Three-well biased active region, direct-phonon design')
print(' Khalatpour-style, x = 0.15, F = {:.1f} kV/cm'.format(F_bias_kVcm))
print('=' * 72)
print(f' Layers (nm): {[(k, t) for k, t in layers]}')
print(f' ULS energy : {E_ULS / meV:8.3f} meV')
print(f' LLS energy : {E_LLS / meV:8.3f} meV')
print(f' Transition : {gap_meV:8.3f} meV  ({f_THz:.2f} THz)')
print()

# ---------------------------------------------------------------------
# Per-interface roughness contribution to pure dephasing
#
#   γ_pure = (m* ΔE_c² π Δ² Λ²) / (2 ℏ³)
#            Σ_i [|ψ_u(z_i)|² − |ψ_l(z_i)|²]²  · F(kΛ)
#
# Compute the contribution OF EACH INTERFACE separately so we can see
# which interface dominates the coherence budget — useful design info.
# ---------------------------------------------------------------------
k2_thermal = m_star * kB * T_K / hbar**2
F_factor = np.exp(-k2_thermal * Lambda_c**2 / 4.0)

prefactor = (m_star * V0**2 * np.pi * Delta_rms**2 * Lambda_c**2) \
            / (2.0 * hbar**3)

# Each layer boundary in `interfaces` (except the synthetic 'end' marker)
# is an interface where the material changes; collect those between
# different materials (B→W or W→B).
real_interfaces = []
for k in range(len(interfaces) - 1):
    z_i, kind = interfaces[k]
    next_kind = interfaces[k + 1][1] if k + 1 < len(interfaces) else None
    if kind != 'end' and next_kind != 'end':
        # the boundary at z_i separates the previous layer from this one
        if k == 0:
            # boundary between pad (barrier) and first layer
            pass
        else:
            prev_kind = interfaces[k - 1][1]
            if prev_kind != kind:
                real_interfaces.append(z_i)

# Simpler: every layer boundary is an interface (material always changes
# in a barrier/well stack).  Drop the first and last (pad boundaries).
all_boundaries = [zi for zi, k in interfaces if k != 'end']
real_interfaces = all_boundaries  # includes pad-to-stage on each side

print(' Per-interface contribution to γ_pure  (Δ = {:.4f} nm, Λ = {:.1f} nm)'
      .format(Delta_rms / nm, Lambda_c / nm))
print(' ' + '-' * 70)
print('  {:>3s}  {:>8s}    {:>10s}   {:>10s}   {:>10s}'
      .format('#', 'z (nm)', '|ψU|² (1/nm)', '|ψL|² (1/nm)', 'γ_i (ps⁻¹)'))
print(' ' + '-' * 70)

gamma_total = 0.0
contributions = []
for k, zi in enumerate(real_interfaces):
    i = int(np.argmin(np.abs(z - zi)))
    pU = psi_ULS[i] ** 2
    pL = psi_LLS[i] ** 2
    g_i = prefactor * (pU - pL) ** 2 * F_factor
    g_i_invps = g_i * 1e-12
    gamma_total += g_i_invps
    contributions.append((zi / nm, pU * nm, pL * nm, g_i_invps))
    print('  {:3d}   {:8.2f}    {:10.4f}    {:10.4f}    {:10.4f}'
          .format(k + 1, zi / nm, pU * nm, pL * nm, g_i_invps))

print(' ' + '-' * 70)
print(f'  TOTAL γ_pure (single stage)             = {gamma_total:.4f} ps⁻¹')
print(f'  Corresponding T2*                       = {1.0 / gamma_total:.3f} ps')
print()
print(' Experimental γ_pure in cool THz QCLs:  ≈ 0.65 – 1.0 ps⁻¹')
print(' (Khalatpour et al., Nat. Photonics 15, 16 (2021); Nat. Comm. 2024)')

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                               gridspec_kw={'width_ratios': [1.4, 1.0]})

# Panel A: potential + ULS, LLS wavefunctions
ax1.plot(z / nm, V / meV, color='black', lw=1.0)
ax1.fill_between(z / nm, V / meV, V.min() / meV - 50,
                 color='#cccccc', alpha=0.25)

scale = 350.0
ax1.plot(z / nm, E_ULS / meV + scale * psi_ULS ** 2 * nm,
         color='#d62728', lw=2.0, label='ULS')
ax1.plot(z / nm, E_LLS / meV + scale * psi_LLS ** 2 * nm,
         color='#1f77b4', lw=2.0, label='LLS')
ax1.axhline(E_ULS / meV, color='#d62728', ls=':', lw=0.7)
ax1.axhline(E_LLS / meV, color='#1f77b4', ls=':', lw=0.7)

for zi in real_interfaces:
    ax1.axvline(zi / nm, color='black', ls='--', lw=0.4, alpha=0.4)

ax1.set_xlabel('z  (nm)')
ax1.set_ylabel('Energy  (meV)')
ax1.set_title(f'Three-well biased stage, F = {F_bias_kVcm:.0f} kV/cm\n'
              f'ULS–LLS gap = {gap_meV:.2f} meV   ({f_THz:.2f} THz)',
              fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(alpha=0.25)
ax1.set_xlim(z[0] / nm, z[-1] / nm)
ax1.set_ylim(V.min() / meV - 20, V.max() / meV + 30)

# Panel B: per-interface contribution as a bar chart
zs, pUs, pLs, gs = zip(*contributions)
indices = np.arange(len(gs))
colors = ['#d62728' if g == max(gs) else '#1f77b4' for g in gs]
ax2.bar(indices + 1, gs, color=colors, alpha=0.85, edgecolor='black', lw=0.6)
ax2.axhline(gamma_total, color='black', ls='--', lw=1.0,
            label=f'sum = {gamma_total:.3f} ps⁻¹')
ax2.axhspan(0.65, 1.0, color='#ffd966', alpha=0.35,
            label='experimental γ_pure\n(cool THz QCLs)')

ax2.set_xticks(indices + 1)
ax2.set_xticklabels([f'{i+1}' for i in indices])
ax2.set_xlabel('interface #')
ax2.set_ylabel(r'$\gamma_i$  (ps$^{-1}$)')
ax2.set_title('Per-interface dephasing contribution\n'
              '(no fitting, Δ = 1 ML, Λ = 5 nm, T = 50 K)',
              fontsize=11)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax2.grid(alpha=0.25, axis='y')
y_top = max(max(gs) * 1.4, 1.1)
ax2.set_ylim(0, y_top)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
out_png = os.path.join(script_dir, 'thz_threewell_dephasing.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f'\nFigure saved to: {out_png}')
