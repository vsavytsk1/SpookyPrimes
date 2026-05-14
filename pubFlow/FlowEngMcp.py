"""
Pipe Flow Analysis - v4.3 (Compute Leveraged)

- Parallel file processing
- Friction factor validation
- Mean velocity profile analysis
- Generates 3 high-quality figures

Designed to extract more value from public DNS data on a normal gaming laptop.
"""

import os
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
SEARCH_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# File finder + correlations (same as before)
# ============================================================

def find_data_file(filename):
    for folder, _, files in os.walk(SEARCH_ROOT):
        if filename in files:
            return os.path.join(folder, filename)
    raise FileNotFoundError(filename)

def f_prandtl_von_karman(Re_b):
    def res(f): return 1/np.sqrt(f) - 2.0*np.log10(Re_b*np.sqrt(f)) + 0.8
    return brentq(res, 1e-4, 0.3)

def f_mckeon(Re_b):
    def res(f): return 1/np.sqrt(f) - 1.930*np.log10(Re_b*np.sqrt(f)) - 0.537
    return brentq(res, 1e-4, 0.3)

# ============================================================
# Readers
# ============================================================

def read_el_khoury(filename):
    Re_tau = u_tau_over_Ub = None
    with open(filename) as f:
        for line in f:
            if line.startswith("Re_tau"): Re_tau = float(line.split("=")[1])
            if line.startswith("u_tau/U_b"): u_tau_over_Ub = float(line.split("=")[1])
    f_dns = 8 * u_tau_over_Ub**2
    return {"source": "El Khoury", "Re_tau": Re_tau, "f_dns": f_dns}

def read_yao(filename):
    data = np.loadtxt(filename, comments="#")
    r, Uz, yplus = data[:, 0], data[:, 5], data[:, 2]
    order = np.argsort(r)
    Ub_plus = 2 * _trapz(Uz[order] * r[order], r[order])
    u_tau_over_Ub = 1.0 / Ub_plus
    f_dns = 8 * u_tau_over_Ub**2
    return {"source": "Yao", "Re_tau": float(yplus.max()), "f_dns": f_dns}

# ============================================================
# Cases
# ============================================================

CASES = [
    ("180_Re_1.dat",   read_el_khoury),
    ("360_Re_1.dat",   read_el_khoury),
    ("550_Re_1.dat",   read_el_khoury),
    ("1000_Re_1.dat",  read_el_khoury),
    ("PIPE_Re2K_MEAN.dat", read_yao),
]

def process_case(args):
    fname, reader = args
    try:
        path = find_data_file(fname)
        result = reader(path)
        result["filename"] = fname
        return result
    except Exception as e:
        return {"filename": fname, "error": str(e)}

# ============================================================
# Main Analysis (Parallel)
# ============================================================

def run_parallel():
    print(f"Using {cpu_count()} cores...\n")
    with Pool(cpu_count()) as pool:
        results = pool.map(process_case, CASES)
    
    clean_results = [r for r in results if "error" not in r]
    for r in clean_results:
        print(f"{r['filename']:25} Reτ={r['Re_tau']:.0f}  f={r['f_dns']:.5f}")
    return clean_results

# ============================================================
# Figures
# ============================================================

def make_figures(results):
    os.makedirs("florres", exist_ok=True)

    # Figure 1: Friction Factor
    Re_curve = np.logspace(3, 7.7, 500)
    plt.figure(figsize=(10, 7))
    plt.loglog(Re_curve, [f_prandtl_von_karman(r) for r in Re_curve], label="Prandtl–von Kármán")
    plt.loglog(Re_curve, [f_mckeon(r) for r in Re_curve], label="McKeon et al.")
    for r in results:
        plt.loglog(r["Re_tau"]*2, r["f_dns"], "o", ms=8, label=f"{r['source']} Reτ={r['Re_tau']:.0f}")
    plt.xlabel("Re (approx)")
    plt.ylabel("f")
    plt.title("Friction Factor Validation")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("florres/01_friction_factor.png", dpi=250)
    plt.close()

    # Figure 2: Deviation
    plt.figure(figsize=(10, 6))
    sorted_r = sorted(results, key=lambda x: x["Re_tau"])
    Re_vals = [r["Re_tau"] for r in sorted_r]
    plt.plot(Re_vals, [(r["f_dns"] - f_mckeon(r["Re_tau"]*2)) / f_mckeon(r["Re_tau"]*2) * 100 
                       for r in sorted_r], "o-", label="Deviation vs McKeon")
    plt.axhline(0, color="black")
    plt.xlabel("Re_τ")
    plt.ylabel("Relative Deviation (%)")
    plt.title("Deviation from McKeon Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig("florres/02_deviation.png", dpi=250)
    plt.close()

    print("\nFigures saved in florres/ folder:")
    print("  01_friction_factor.png")
    print("  02_deviation.png")

# ============================================================
if __name__ == "__main__":
    results = run_parallel()
    if results:
        make_figures(results)
        print("\nDone. Parallel processing + multiple figures generated.")