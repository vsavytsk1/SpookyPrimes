"""
Phase 2c: The SM finite spectral triple and the A_F -> A_PS obstruction.

We build:
    H_F = C^32 (per-generation fermion Hilbert space)
    pi_F : A_F = C (+) H (+) M_3(C) -> End(H_F)   (canonical SM action)
    gamma_F                                        (chirality grading, KO-dim 6)
    J_F                                            (real structure, J^2 = +1)
    D_F                                            (SM Dirac with Yukawa)

and verify:
    All KO-dim 6 conditions
    pi_F preserves gamma     <-- this is where PS extension fails
    D_F anticommutes with gamma
    D_F commutes with J
    Order-one condition for A_F: [[D_F, pi_F(a)], J pi_F(b) J^{-1}] = 0  for all a, b in A_F

Then we extend the algebra to A_PS = M_2(H) (+) M_4(C) via a natural inclusion
(NOT a unital *-subalgebra inclusion in our embedding; the canonical extension)
and observe two things:
    (1) [gamma, pi_PS(X)] != 0 for X involving the off-diagonal quaternion
        blocks of M_2(H) (the L-R mixing blocks).
    (2) Even if we project pi_PS to its [gamma, .]=0 part, the order-one
        condition fails for the larger algebra (this needs the full
        Yukawa-Dirac structure to make manifest).

Basis convention as before:
    |a, s, w, c>, indexed by a in {+1,-1}, s in {+1,-1}, w in {+1,-1}, c in 1..4
    s = +1 means LEFT, s = -1 means RIGHT
    w = +1 means UP component of weak doublet, w = -1 means DOWN
    c = 1,2,3 means quark colors, c = 4 means lepton "color"
    a = +1 means particle, a = -1 means antiparticle
"""

import numpy as np
rng = np.random.default_rng(0)

# ---- Basis indexing ----
def idx(a, s, w, c):
    a_idx = 0 if a == +1 else 1
    s_idx = 0 if s == +1 else 1
    w_idx = 0 if w == +1 else 1
    return a_idx * 16 + s_idx * 8 + w_idx * 4 + (c - 1)

def unidx(i):
    a_idx, rem = divmod(i, 16)
    s_idx, rem = divmod(rem, 8)
    w_idx, c_minus_1 = divmod(rem, 4)
    a = +1 if a_idx == 0 else -1
    s = +1 if s_idx == 0 else -1
    w = +1 if w_idx == 0 else -1
    c = c_minus_1 + 1
    return a, s, w, c

N = 32

# ---- gamma and J as before (KO-dim 6) ----
gamma = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i)
    gamma[i, i] = a * s

P_J = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i)
    j = idx(-a, s, w, c)
    P_J[j, i] = 1.0

def J_apply(psi):
    return P_J @ np.conj(psi)

def pi_op_of(pi_a):
    """pi_op(a) = J pi(a)^dag J^{-1}, as a matrix.
    Derivation: J pi(a)^dag J^{-1} v = J(pi(a)^dag P_J conj(v))
              = P_J conj(pi(a)^dag P_J conj(v))
              = P_J conj(pi(a)^dag) P_J^{-1} conj(conj(v))
              = P_J conj(pi(a))^T P_J v          (P_J real, P_J^{-1}=P_J)
              ... pi(a)^dag = conj(pi(a))^T, so conj(pi(a)^dag) = pi(a)^T? No:
              conj(conj(M)^T) = M^T -- yes, conj of (conjugate-then-transpose) = transpose.
              So conj(pi(a)^dag) = pi(a)^T.
    Therefore: pi_op(a) = P_J @ pi(a)^T @ P_J.
    """
    return P_J @ pi_a.T @ P_J

# ---- Build A_F's canonical action on H_F ----
#
# Element of A_F: (lambda, q, m) with lambda in C, q in H, m in M_3(C).
# The SM action [Connes-Marcolli convention]:
#
# On particle sector (a = +1):
#   L doublet (s = +1, w = +1, c), (s = +1, w = -1, c):
#     q acts as 2x2 quaternion-matrix-image on (w = +1, w = -1), c trivial.
#     I.e., pi(lambda, q, m)|+1, +1, +1, c> = q_{00} |+1,+1,+1,c> + q_{01} |+1,+1,-1,c>
#           pi(lambda, q, m)|+1, +1, -1, c> = q_{10} |+1,+1,+1,c> + q_{11} |+1,+1,-1,c>
#     where q is realized as q -> M_2(C) matrix [[q0+i q1, q2+i q3],[-q2+i q3, q0-i q1]].
#   R singlets (s = -1):
#     up sector (w = +1): scalar lambda
#     down sector (w = -1): scalar lambda-bar (complex conjugate)
#     color: m acts on c = 1,2,3 (quark); identity on c = 4 (lepton).
#
# On antiparticle sector (a = -1):
#   Convention: pi acts as the "color" representation, i.e., color algebra of A_F.
#   - quarks (c = 1,2,3): color rotation by m_bar^T (or appropriate)
#   - leptons (c = 4): scalar lambda or lambda-bar depending on s, w
#   - The L/R structure is preserved.
#
# This convention is somewhat fiddly. We code it carefully and then verify
# [gamma, pi_F] = 0 (algebra preserves chirality).

def quat_to_M2C(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0 + 1j*q1,  q2 + 1j*q3],
        [-q2 + 1j*q3, q0 - 1j*q1],
    ], dtype=complex)

def pi_F(lam, q, m):
    """Canonical SM action of (lam, q, m) in A_F = C (+) H (+) M_3(C).
    Returns 32x32 complex matrix.
    """
    op = np.zeros((N, N), dtype=complex)
    Q = quat_to_M2C(q)  # 2x2 complex matrix realizing q

    for c in range(1, 5):
        # Particle L sector (a=+1, s=+1): q acts on w doublet, color trivial
        for w1_idx, w1 in enumerate([+1, -1]):
            for w2_idx, w2 in enumerate([+1, -1]):
                op[idx(+1, +1, w1, c), idx(+1, +1, w2, c)] = Q[w1_idx, w2_idx]
        # Particle R sector (a=+1, s=-1):
        #   up (w=+1): scalar lambda
        op[idx(+1, -1, +1, c), idx(+1, -1, +1, c)] = lam
        #   down (w=-1): scalar lambda-bar
        op[idx(+1, -1, -1, c), idx(+1, -1, -1, c)] = np.conj(lam)
        # Color action of m on c = 1,2,3 (quarks): applied to particles ?
        # In the CC07 convention, m acts on particles' COLOR index ONLY when
        # c is a quark color. The convention I'll use here is:
        # m acts on antiparticles' color index, with the lepton "color" (c=4)
        # being fixed by the C action.
        pass

    # Antiparticle sector (a=-1): pi acts as
    #   m on quark colors (c=1,2,3)
    #   IDENTITY on lepton (c=4)
    #   does NOT touch the (s, w) indices for antiparticles.
    # This is the standard convention that makes pi_op = J pi(.)^dag J^{-1}
    # commute with pi (because pi acts on (s,w) of particles + color of
    # antiparticles, pi_op acts on color of particles + (s,w) of antiparticles,
    # so they act on disjoint indices).
    for s in [+1, -1]:
        for w in [+1, -1]:
            # quark color block: m in M_3(C) acts on c index
            for c1 in range(1, 4):  # quarks
                for c2 in range(1, 4):
                    op[idx(-1, s, w, c1), idx(-1, s, w, c2)] = m[c1 - 1, c2 - 1]
            # lepton (c=4): IDENTITY (no lambda, no lambda-bar)
            op[idx(-1, s, w, 4), idx(-1, s, w, 4)] = 1.0
    return op

# ---- Verify pi_F is multiplicative and *-preserving ----

def random_lam():
    return complex(rng.standard_normal(), rng.standard_normal())
def random_q():
    return tuple(rng.standard_normal(4))
def random_m():
    return rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))

def AF_mult(t1, t2):
    lam1, q1, m1 = t1
    lam2, q2, m2 = t2
    lam = lam1 * lam2
    # quaternion mult
    a, b, c, d = q1
    e, f, g, h = q2
    q = (a*e - b*f - c*g - d*h,
         a*f + b*e + c*h - d*g,
         a*g - b*h + c*e + d*f,
         a*h + b*g - c*f + d*e)
    m = m1 @ m2
    return (lam, q, m)

def AF_star(t):
    lam, q, m = t
    return (np.conj(lam), (q[0], -q[1], -q[2], -q[3]), m.conj().T)

def AF_one():
    return (1.0+0j, (1.0, 0, 0, 0), np.eye(3, dtype=complex))

# Unital
unit_err = np.max(np.abs(pi_F(*AF_one()) - np.eye(N)))
print(f"pi_F(1) = I: max err = {unit_err:.3e}")

# Multiplicativity
mult_errs = []
for _ in range(20):
    t1 = (random_lam(), random_q(), random_m())
    t2 = (random_lam(), random_q(), random_m())
    LHS = pi_F(*AF_mult(t1, t2))
    RHS = pi_F(*t1) @ pi_F(*t2)
    mult_errs.append(np.max(np.abs(LHS - RHS)))
print(f"pi_F multiplicative: max err over 20 random pairs = {max(mult_errs):.3e}")

# *-preservation
star_errs = []
for _ in range(20):
    t = (random_lam(), random_q(), random_m())
    LHS = pi_F(*AF_star(t))
    RHS = pi_F(*t).conj().T
    star_errs.append(np.max(np.abs(LHS - RHS)))
print(f"pi_F *-preserving: max err over 20 random elts = {max(star_errs):.3e}")

# [gamma, pi_F] = 0
gamma_errs = []
for _ in range(20):
    t = (random_lam(), random_q(), random_m())
    A = pi_F(*t)
    gamma_errs.append(np.max(np.abs(gamma @ A - A @ gamma)))
print(f"[gamma, pi_F] = 0 (even, preserves chirality): max err = {max(gamma_errs):.3e}")

# Bimodule [pi(a), pi_op(b)] = 0
bimod_errs = []
for _ in range(20):
    t1 = (random_lam(), random_q(), random_m())
    t2 = (random_lam(), random_q(), random_m())
    A = pi_F(*t1)
    Bop = pi_op_of(pi_F(*t2))
    bimod_errs.append(np.max(np.abs(A @ Bop - Bop @ A)))
print(f"[pi_F(a), pi_F^op(b)] = 0 (bimodule): max err = {max(bimod_errs):.3e}")

print()

# ---- Build the SM Yukawa Dirac D_F ----
#
# D_F mixes L and R fermions via Yukawa couplings:
#   D_F|+1, +1, w, c>  = Y_w |+1, -1, w', c>   for some Yukawa Y
#   D_F|+1, -1, w, c>  = (Y_w)^* |+1, +1, w', c>     (self-adjoint)
#
# For the SM, the up-quark Yukawa Y_u and down-quark Yukawa Y_d are different,
# and there's a CKM-like rotation between them. For simplicity here we use
# diagonal Y, choosing arbitrary positive Yukawa values:
#   Y(quark, up) = m_u,  Y(quark, down) = m_d,  Y(lepton, up) = m_nu (Dirac),
#   Y(lepton, down) = m_e
# So Y depends on (w, c=quark vs lepton): 4 Yukawa values.
#
# The Dirac also has terms in the antiparticle sector (mirror of particle, by J).
#
# Additionally, for KO-dim 6, the right-handed neutrino has a Majorana mass term
# coupling particle nu_R to antiparticle nu_R-bar. We omit this for simplicity
# (it's important for see-saw, but doesn't affect the order-one check.)

Y = {  # (w, "type") -> Yukawa value
    (+1, 'quark'):  0.005,   # u-quark Yukawa
    (-1, 'quark'):  0.003,   # d-quark Yukawa
    (+1, 'lepton'): 0.0001,  # nu Dirac Yukawa
    (-1, 'lepton'): 0.002,   # e Yukawa
}

def fermion_type(c):
    return 'quark' if c in (1, 2, 3) else 'lepton'

def build_D_F():
    D = np.zeros((N, N), dtype=complex)
    # Particle Yukawa: L <-> R, same w, same c
    for c in range(1, 5):
        ftype = fermion_type(c)
        for w in [+1, -1]:
            y = Y[(w, ftype)]
            # L -> R
            D[idx(+1, -1, w, c), idx(+1, +1, w, c)] = y
            # R -> L (Hermitian conjugate)
            D[idx(+1, +1, w, c), idx(+1, -1, w, c)] = np.conj(y)
    # Antiparticle Yukawa: mirror via J. The condition D J = J D gives the
    # antiparticle Yukawa = particle Yukawa (with appropriate index swap).
    # Specifically, J|+1,s,w,c> = |-1,s,w,c> (with conj), so the L-R mixing
    # on particles is mirrored on antiparticles.
    for c in range(1, 5):
        ftype = fermion_type(c)
        for w in [+1, -1]:
            y = Y[(w, ftype)]
            D[idx(-1, -1, w, c), idx(-1, +1, w, c)] = np.conj(y)  # for J-commutativity
            D[idx(-1, +1, w, c), idx(-1, -1, w, c)] = y
    return D

D = build_D_F()
print(f"D is Hermitian: {np.allclose(D, D.conj().T)}")
print(f"D anticommutes with gamma: max |Dgamma + gammaD| = "
      f"{np.max(np.abs(D @ gamma + gamma @ D)):.3e}")

# D commutes with J: D(J v) = J(D v).  Since J v = P_J conj(v):
#   D P_J conj(v)  ==  P_J conj(D v) = P_J conj(D) conj(v)
# So we need D P_J = P_J conj(D), equivalently D P_J = P_J D-bar.
print(f"D J = J D (real Dirac, KO-dim 6 sign): max |D P_J - P_J D-bar| = "
      f"{np.max(np.abs(D @ P_J - P_J @ np.conj(D))):.3e}")

# ---- Order-one condition for A_F ----
print()
print("Order-one condition for A_F: [[D, pi(a)], pi_op(b)] = 0 for all a, b in A_F?")
order1_errs = []
for _ in range(50):
    t1 = (random_lam(), random_q(), random_m())
    t2 = (random_lam(), random_q(), random_m())
    A = pi_F(*t1)
    Bop = pi_op_of(pi_F(*t2))
    err = np.max(np.abs((D @ A - A @ D) @ Bop - Bop @ (D @ A - A @ D)))
    order1_errs.append(err)
print(f"  Order-one for A_F: max err over 50 random pairs = {max(order1_errs):.3e}")
print(f"  (zero -> order-one holds for A_F; non-zero -> it fails)")

# ---- Now: extend the algebra to A_PS = M_2(H) (+) M_4(C) ----
#
# Canonical PS action:
#   M_2(H) acts on the (s, w) 4-block of PARTICLES, treating (L_u, L_d, R_u, R_d)
#     as an H^2 module. Color trivial.
#   M_4(C) acts on the full 4-color index of ANTIPARTICLES (treating c=4 lepton
#     same as quark colors). The (s, w) structure trivial on antiparticles.
#
# This action specializes to A_F's action via the embedding iota that we
# verified above, but extends it to the full M_2(H) (+) M_4(C). Crucially,
# the off-diagonal quaternion blocks of M_2(H) mix s = +1 with s = -1.

def pi_PS(Q, Z):
    """Q is a 2x2 matrix of quaternions; Z is a 4x4 complex matrix.
    Returns 32x32 op acting:
        Q on particle (s, w) sector  (treating (L_u, L_d, R_u, R_d) as H^2)
        Z on antiparticle c sector"""
    op = np.zeros((N, N), dtype=complex)
    # particles: Q acts on (s, w) -> use H^2 = C^4 realization
    # (s, w) order in H^2: (L_u, L_d, R_u, R_d) <-> (0,1,2,3)
    M_sw = np.zeros((4, 4), dtype=complex)
    for I in range(2):
        for J in range(2):
            M_sw[2*I:2*I+2, 2*J:2*J+2] = quat_to_M2C(Q[I][J])

    def sw_pair_to_block(s, w):
        return (0 if s == +1 else 2) + (0 if w == +1 else 1)

    for s1 in [+1, -1]:
        for w1 in [+1, -1]:
            for s2 in [+1, -1]:
                for w2 in [+1, -1]:
                    z = M_sw[sw_pair_to_block(s1, w1), sw_pair_to_block(s2, w2)]
                    for c in range(1, 5):
                        op[idx(+1, s1, w1, c), idx(+1, s2, w2, c)] = z
    # antiparticles: Z acts on c
    for s in [+1, -1]:
        for w in [+1, -1]:
            for c1 in range(1, 5):
                for c2 in range(1, 5):
                    op[idx(-1, s, w, c1), idx(-1, s, w, c2)] = Z[c1-1, c2-1]
    return op

# Check [gamma, pi_PS] for a few random elements
print()
print("Extension to A_PS = M_2(H) (+) M_4(C):")
gamma_PS_errs = []
for _ in range(20):
    Q = [[tuple(rng.standard_normal(4)) for _ in range(2)] for _ in range(2)]
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = pi_PS(Q, Z)
    gamma_PS_errs.append(np.max(np.abs(gamma @ A - A @ gamma)))
print(f"  [gamma, pi_PS] = 0?: max err = {max(gamma_PS_errs):.3e}")
print(f"  (NONZERO: the off-diagonal H-blocks of M_2(H) mix L and R, breaking gamma)")

# What if we restrict to the BLOCK-DIAGONAL part of M_2(H), i.e., q_LL (+) q_RR?
# This is H (+) H, not M_2(H). Does this preserve gamma?
def pi_block_diag_LR(qL, qR, Z):
    """Diagonal-only quaternion: qL on L block, qR on R block. Color action Z."""
    Q = [[qL, (0,0,0,0)], [(0,0,0,0), qR]]
    return pi_PS(Q, Z)

block_gamma_errs = []
for _ in range(20):
    qL = tuple(rng.standard_normal(4))
    qR = tuple(rng.standard_normal(4))
    Z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = pi_block_diag_LR(qL, qR, Z)
    block_gamma_errs.append(np.max(np.abs(gamma @ A - A @ gamma)))
print(f"  Restricted to L-block + R-block (H + H + M_4(C)):")
print(f"  [gamma, pi] = 0?: max err = {max(block_gamma_errs):.3e}")

# Now check order-one for the L-R-block-diagonal subalgebra (= H + H + M_4(C))
# with the SM Dirac D.
print()
print(f"Order-one for H (L) + H (R) + M_4(C) (with SM Dirac D)?")
o1_blockdiag_errs = []
for _ in range(50):
    qL1 = tuple(rng.standard_normal(4)); qR1 = tuple(rng.standard_normal(4))
    Z1 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    qL2 = tuple(rng.standard_normal(4)); qR2 = tuple(rng.standard_normal(4))
    Z2 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = pi_block_diag_LR(qL1, qR1, Z1)
    Bop = pi_op_of(pi_block_diag_LR(qL2, qR2, Z2))
    err = np.max(np.abs((D @ A - A @ D) @ Bop - Bop @ (D @ A - A @ D)))
    o1_blockdiag_errs.append(err)
print(f"  Order-one for H_L + H_R + M_4(C): max err = {max(o1_blockdiag_errs):.3e}")
print(f"  (positive value -> order-one FAILS for this larger algebra)")
