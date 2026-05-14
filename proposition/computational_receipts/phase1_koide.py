"""
Phase 1: Honest verification of the Koide ratio against current PDG values.

The claim in strange_idea.pdf (Sec 8):
    K = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2
      = 0.666661 +/- 0.000007
    deviation from 2/3 at the 1e-5 level

We compute K from current PDG charged lepton masses, propagate uncertainties
rigorously via Monte Carlo, and report:
  (a) the central value
  (b) standard uncertainty
  (c) the deviation from 2/3 in sigmas
  (d) the deviation from 2/3 in absolute units
"""

import numpy as np
from numpy.random import default_rng

# PDG 2024 charged lepton masses, in MeV.
# Source: pdg.lbl.gov 2024 review.
# Electron mass: derived from m_e c^2 = 0.51099895069(16) MeV  (CODATA 2022)
# Muon mass:     m_mu c^2 = 105.6583755(23) MeV                  (CODATA 2022, PDG)
# Tau mass:      m_tau c^2 = 1776.93(9) MeV                      (PDG 2024;
#                CODATA 2022 had 1776.86(12) but PDG 2024 updated to 1776.93(9))

# We use the standard one-sigma uncertainties. The masses we plug in are pole-mass
# equivalents from PDG; the Koide relation is famously sensitive to which mass
# scheme is used (it works best for pole masses, drifts under RG).

m_e   = 0.51099895069
s_e   = 0.00000000016

m_mu  = 105.6583755
s_mu  = 0.0000023

m_tau = 1776.86          # using CODATA 2022 value as in strange_idea.pdf
s_tau = 0.12             # so we can compare directly to its quoted K = 0.666661(7)

# Also evaluate with PDG 2024 tau mass for completeness
m_tau_pdg2024 = 1776.93
s_tau_pdg2024 = 0.09

def koide(me, mmu, mtau):
    num = me + mmu + mtau
    den = (np.sqrt(me) + np.sqrt(mmu) + np.sqrt(mtau))**2
    return num / den

def mc_koide(me, sme, mmu, smmu, mtau, smtau, N=2_000_000, seed=0):
    rng = default_rng(seed)
    # Truncated Gaussian: sigmas are ~10^-10 relative for electron, ~10^-8 for muon,
    # ~7e-5 for tau. Plain Gaussian sampling is fine; nowhere near zero.
    me_s   = rng.normal(me,   sme,   N)
    mmu_s  = rng.normal(mmu,  smmu,  N)
    mtau_s = rng.normal(mtau, smtau, N)
    K = koide(me_s, mmu_s, mtau_s)
    return K.mean(), K.std(ddof=1)

# ---- Evaluation with CODATA 2022 tau value ----
K_central = koide(m_e, m_mu, m_tau)
K_mc, K_sigma = mc_koide(m_e, s_e, m_mu, s_mu, m_tau, s_tau)

print("=" * 70)
print("Koide ratio with CODATA 2022 charged lepton masses (matches strange_idea)")
print("=" * 70)
print(f"  m_e   = {m_e}  +/- {s_e}  MeV")
print(f"  m_mu  = {m_mu}  +/- {s_mu}  MeV")
print(f"  m_tau = {m_tau} +/- {s_tau}  MeV")
print()
print(f"  K (central, exact propagation): {K_central:.10f}")
print(f"  K (Monte Carlo mean, 2e6 sam): {K_mc:.10f}")
print(f"  sigma(K) from MC:               {K_sigma:.3e}")
print()
two_thirds = 2.0/3.0
delta = K_central - two_thirds
n_sigma = delta / K_sigma
print(f"  2/3 = {two_thirds:.10f}")
print(f"  K - 2/3 = {delta:+.3e}")
print(f"  |K - 2/3| in sigmas: {abs(n_sigma):.2f}")
print()

# ---- Evaluation with PDG 2024 tau value ----
K_central_24 = koide(m_e, m_mu, m_tau_pdg2024)
K_mc_24, K_sigma_24 = mc_koide(m_e, s_e, m_mu, s_mu, m_tau_pdg2024, s_tau_pdg2024)
print("=" * 70)
print("Koide ratio with PDG 2024 tau mass (1776.93 +/- 0.09)")
print("=" * 70)
print(f"  K (central): {K_central_24:.10f}")
print(f"  sigma(K):    {K_sigma_24:.3e}")
delta_24 = K_central_24 - two_thirds
n_sigma_24 = delta_24 / K_sigma_24
print(f"  K - 2/3 = {delta_24:+.3e}")
print(f"  |K - 2/3| in sigmas: {abs(n_sigma_24):.2f}")
print()

# ---- Sensitivity analysis: how must tau move for K to hit 2/3 exactly? ----
# K(m_e, m_mu, m_tau) = 2/3, solve for m_tau given m_e, m_mu fixed.
from scipy.optimize import brentq
def K_minus_target(mtau):
    return koide(m_e, m_mu, mtau) - two_thirds
mtau_for_exact_2_3 = brentq(K_minus_target, 1700.0, 1850.0)
print("=" * 70)
print("Sensitivity: what tau mass gives K = 2/3 exactly?")
print("=" * 70)
print(f"  m_tau for K=2/3 exactly: {mtau_for_exact_2_3:.6f} MeV")
print(f"  vs CODATA 2022:          {m_tau} +/- {s_tau} MeV")
print(f"  diff:                    {mtau_for_exact_2_3 - m_tau:+.4f} MeV "
      f"({(mtau_for_exact_2_3 - m_tau)/s_tau:+.2f} sigma)")
print(f"  vs PDG 2024:             {m_tau_pdg2024} +/- {s_tau_pdg2024} MeV")
print(f"  diff:                    {mtau_for_exact_2_3 - m_tau_pdg2024:+.4f} MeV "
      f"({(mtau_for_exact_2_3 - m_tau_pdg2024)/s_tau_pdg2024:+.2f} sigma)")
print()

# ---- One-loop RG running of the Koide ratio ----
# The charged-lepton mass ratios run under QED. Above QED threshold, the relevant
# anomalous dimensions are tiny but nonzero. The famous-ish observation:
# K is *not* RG-invariant. Let's estimate the running from M_Z (tau pole mass scale)
# down to common pole-mass scale.
#
# At one loop in QED with N_f light fermions, the running mass satisfies
#    m(mu)/m(mu_0) = [alpha(mu)/alpha(mu_0)]^(gamma_m / b)
# where gamma_m = 6/(4 pi) * Q^2 and b is QED beta-function. For all three leptons
# Q^2 = 1, so the ratios m_mu/m_e and m_tau/m_e are RG-invariant at one-loop QED:
# they all scale by the SAME factor. Hence K at one-loop is RG-INVARIANT in pure QED.
#
# It becomes scheme-dependent only when including:
#   (i) different pole-vs-MSbar choices,
#   (ii) electroweak corrections that distinguish charged leptons by mass
#        (e.g. running through their own thresholds).
# These shift K at the 10^-4 level or smaller. The famous tightness of K = 2/3 to
# 10^-5 only holds in a specific (pole-mass) scheme.
print("=" * 70)
print("Scheme note:")
print("=" * 70)
print("  Koide's K is pole-mass-scheme dependent. In pure QED at one loop, ")
print("  pole-mass ratios are RG-invariant (all Q^2=1), so K is also invariant.")
print("  Beyond that, EW threshold corrections shift K at the ~10^-4 level.")
print("  The reported precision K = 0.666661(7) is a pole-mass statement only.")
