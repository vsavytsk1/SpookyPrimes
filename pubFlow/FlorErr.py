"""
Pipe Flow Friction Factor Analysis
Version 4.1 - Final

This script analyses publicly available DNS data for turbulent pipe flow
and compares the resulting friction factors against established correlations.

Intended for educational and research use in Chemical Engineering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

# ============================================================
# ESTABLISHED CORRELATIONS
# ============================================================

def prandtl_von_karman_friction(Re):
    """Classical Prandtl–von Kármán smooth-pipe correlation."""
    def equation(f):
        return 1 / np.sqrt(f) - 2.0 * np.log10(Re * np.sqrt(f)) + 0.8
    return fsolve(equation, 0.015)[0]


def mckeon_friction(Re):
    """McKeon et al. (2004/2005) correlation fitted to Princeton Superpipe data."""
    def equation(f):
        return 1 / np.sqrt(f) - 1.930 * np.log10(Re * np.sqrt(f)) + 0.537
    return fsolve(equation, 0.015)[0]


# ============================================================
# DATA READING
# ============================================================

def read_dns_header(filename):
    """
    Extracts Re_tau and u_tau/U_b from the file header.
    Returns friction factor calculated as f = 8 * (u_tau / U_b)^2.
    """
    Re_tau = None
    u_tau_over_Ub = None

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('Re_tau'):
                Re_tau = float(line.split('=')[1].strip())
            if line.startswith('u_tau/U_b'):
                u_tau_over_Ub = float(line.split('=')[1].strip())
            if Re_tau is not None and u_tau_over_Ub is not None:
                break

    if Re_tau is None or u_tau_over_Ub is None:
        raise ValueError(f"Required header values not found in {filename}")

    friction_factor = 8 * (u_tau_over_Ub ** 2)
    return Re_tau, friction_factor


# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == "__main__":
    
    output_dir = "florres"
    os.makedirs(output_dir, exist_ok=True)

    # DNS cases (El Khoury et al., publicly available)
    data_files = {
        180:  "PIPE/180_Re_1.dat",
        360:  "PIPE/360_Re_1.dat",
        550:  "PIPE/550_Re_1.dat",
        1000: "PIPE/1000_Re_1.dat",
    }

    results = []

    print("Pipe Flow Friction Factor Analysis\n")

    for nominal_Re, filepath in data_files.items():
        Re_tau, f_dns = read_dns_header(filepath)
        f_mckeon = mckeon_friction(Re_tau)
        relative_error = (f_dns - f_mckeon) / f_mckeon * 100

        results.append({
            "Re_tau": Re_tau,
            "f_dns": f_dns,
            "f_mckeon": f_mckeon,
            "relative_error_%": relative_error
        })

        print(f"Re_τ ≈ {Re_tau:.1f}   f_DNS = {f_dns:.5f}   "
              f"Error vs McKeon = {relative_error:+.2f}%")

    # ============================================================
    # PLOT
    # ============================================================
    Re_range = np.logspace(3, 8, 400)
    f_classical = np.array([prandtl_von_karman_friction(r) for r in Re_range])
    f_mckeon_curve = np.array([mckeon_friction(r) for r in Re_range])

    plt.figure(figsize=(11, 7.5))
    plt.loglog(Re_range, f_classical, 'b-', linewidth=2.0, label='Prandtl–von Kármán')
    plt.loglog(Re_range, f_mckeon_curve, 'r-', linewidth=2.0, label='McKeon et al.')

    for res in results:
        plt.loglog(res["Re_tau"], res["f_dns"], 'ko', markersize=8)
        plt.annotate(
            f'Re_τ ≈ {res["Re_tau"]:.0f}\n{res["relative_error_%"]:+.1f}%',
            xy=(res["Re_tau"], res["f_dns"]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color='0.5', lw=0.6)
        )

    plt.xlabel('Reynolds number Re')
    plt.ylabel('Darcy friction factor f')
    plt.title('Friction Factor from DNS Data vs Established Correlations')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "friction_factor_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {output_path}")
    print("Analysis complete.")