"""
Phase 6: The 8-dim H_L+H_R+M_4(C) Dirac space, as a subspace of A_F's 16-dim space.

Key insight: H_L+H_R+M_4(C) imposes a SUPERSET of constraints relative to A_F
(it's a bigger algebra acting on the same bimodule, modulo bimodule-rep issues).
So its Dirac space is a SUBSPACE of A_F's 16-dim Dirac space.

To find it efficiently: stay within A_F's 16-dim null space coords (instead of
going back to the 272-dim ambient base-null space), and project H+H+M_4's
additional constraints onto these coords.

For each (a, b) pair from H+H+M_4 x H+H+M_4 basis:
    1. Build order-one constraint matrix in base-null coords: C ∈ R^{2 N^2 × d_base}
    2. Project to A_F coords: C_AF = C @ V_AF  ∈  R^{2 N^2 × 16}
    3. Accumulate.

Within 16-dim, we only need 8 linearly independent constraints to cut to 8-dim
Dirac space. So early termination kicks in fast.
"""

import numpy as np
import scipy.linalg
import time

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ---- Reload setup ----
def idx(a, s, w, c):
    return (0 if a==+1 else 1)*16 + (0 if s==+1 else 1)*8 + (0 if w==+1 else 1)*4 + (c-1)
def unidx(i):
    a_idx, rem = divmod(i, 16); s_idx, rem = divmod(rem, 8)
    w_idx, c_minus_1 = divmod(rem, 4)
    return (+1 if a_idx==0 else -1, +1 if s_idx==0 else -1,
            +1 if w_idx==0 else -1, c_minus_1 + 1)
N = 32
gamma = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i); gamma[i, i] = a*s
P_J = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i); P_J[idx(-a, s, w, c), i] = 1.0
def quat_to_M2C(q):
    q0, q1, q2, q3 = q
    return np.array([[q0+1j*q1, q2+1j*q3], [-q2+1j*q3, q0-1j*q1]], dtype=complex)
def pi_F(lam, q, m):
    op = np.zeros((N, N), dtype=complex); Q = quat_to_M2C(q)
    for c in range(1, 5):
        for w1_idx, w1 in enumerate([+1, -1]):
            for w2_idx, w2 in enumerate([+1, -1]):
                op[idx(+1, +1, w1, c), idx(+1, +1, w2, c)] = Q[w1_idx, w2_idx]
        op[idx(+1, -1, +1, c), idx(+1, -1, +1, c)] = lam
        op[idx(+1, -1, -1, c), idx(+1, -1, -1, c)] = np.conj(lam)
    for s in [+1, -1]:
        for w in [+1, -1]:
            for c1 in range(1, 4):
                for c2 in range(1, 4):
                    op[idx(-1, s, w, c1), idx(-1, s, w, c2)] = m[c1-1, c2-1]
            op[idx(-1, s, w, 4), idx(-1, s, w, 4)] = 1.0
    return op
def pi_op_of(M): return P_J @ M.T @ P_J
def pi_PS_full(Q, Z):
    op = np.zeros((N, N), dtype=complex)
    M_sw = np.zeros((4, 4), dtype=complex)
    for I in range(2):
        for J in range(2):
            M_sw[2*I:2*I+2, 2*J:2*J+2] = quat_to_M2C(Q[I][J])
    def swb(s, w): return (0 if s==+1 else 2) + (0 if w==+1 else 1)
    for s1 in [+1, -1]:
        for w1 in [+1, -1]:
            for s2 in [+1, -1]:
                for w2 in [+1, -1]:
                    z = M_sw[swb(s1,w1), swb(s2,w2)]
                    for c in range(1, 5):
                        op[idx(+1, s1, w1, c), idx(+1, s2, w2, c)] = z
    for s in [+1, -1]:
        for w in [+1, -1]:
            for c1 in range(1, 5):
                for c2 in range(1, 5):
                    op[idx(-1, s, w, c1), idx(-1, s, w, c2)] = Z[c1-1, c2-1]
    return op

# Base null space
def hermitian_rows():
    rows = []
    for i in range(N):
        for j in range(i+1, N):
            r = np.zeros(2*N*N); r[i*N+j] = 1; r[j*N+i] = -1; rows.append(r)
            r = np.zeros(2*N*N); r[N*N+i*N+j] = 1; r[N*N+j*N+i] = 1; rows.append(r)
    for i in range(N):
        r = np.zeros(2*N*N); r[N*N+i*N+i] = 1; rows.append(r)
    return np.array(rows)
def antigamma_rows():
    g = np.diag(gamma).real.astype(int); rows = []
    for i in range(N):
        for j in range(N):
            if g[i] == g[j]:
                r = np.zeros(2*N*N); r[i*N+j] = 1; rows.append(r)
                r = np.zeros(2*N*N); r[N*N+i*N+j] = 1; rows.append(r)
    return np.array(rows)
def Jcommute_rows():
    pi_J = np.array([int(np.argmax(np.abs(P_J[:, j]))) for j in range(N)])
    rows = []
    for i in range(N):
        for j in range(N):
            ip, jp = pi_J[i], pi_J[j]
            r = np.zeros(2*N*N); r[i*N+j] = 1; r[ip*N+jp] -= 1
            if not np.allclose(r, 0): rows.append(r)
            r = np.zeros(2*N*N); r[N*N+i*N+j] = 1; r[N*N+ip*N+jp] += 1
            if not np.allclose(r, 0): rows.append(r)
    return np.array(rows)

print("Setup...")
t0 = time.time()
base_C = np.vstack([hermitian_rows(), antigamma_rows(), Jcommute_rows()])
U, S, Vt = scipy.linalg.svd(base_C, full_matrices=True)
rank = int(np.sum(S > S.max() * 1e-10))
B = Vt[rank:].T
d_base = B.shape[1]
D_stack = np.zeros((d_base, N, N), dtype=complex)
for k in range(d_base):
    v = B[:, k]
    D_stack[k] = v[:N*N].reshape(N, N) + 1j * v[N*N:].reshape(N, N)
print(f"  d_base = {d_base}  ({time.time()-t0:.1f}s)")

# ---- Compute A_F null space V_AF in base coords ----
def order1_pair_fast(M, N_op):
    DMN = (D_stack @ M) @ N_op
    MDN = (M @ D_stack) @ N_op
    NDM = (N_op @ D_stack) @ M
    NMD = (N_op @ M) @ D_stack
    T = DMN - MDN - NDM + NMD
    T_flat = T.reshape(d_base, N*N)
    return np.vstack([T_flat.real.T, T_flat.imag.T])

def compress_rows(M):
    if M.shape[0] == 0: return M
    Q, R, _ = scipy.linalg.qr(M.T, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    if len(d) == 0: return np.zeros((0, M.shape[1]))
    rnk = int(np.sum(d > d.max() * 1e-10))
    return Q[:, :rnk].T

def AF_basis_elts():
    out = []
    for lr, li in [(1,0),(0,1)]:
        out.append((complex(lr,li), (0,0,0,0), np.zeros((3,3), dtype=complex)))
    for q in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        out.append((0+0j, q, np.zeros((3,3), dtype=complex)))
    for i in range(3):
        for j in range(3):
            for kind in ['re','im']:
                m = np.zeros((3,3), dtype=complex)
                m[i,j] = 1.0 if kind=='re' else 1j
                out.append((0+0j, (0,0,0,0), m))
    out.append((1+0j, (1,0,0,0), np.eye(3, dtype=complex)))
    return out

print("Computing A_F's 16-dim Dirac space basis...")
t0 = time.time()
AF_elts = AF_basis_elts()
AF_pis = [pi_F(*t) for t in AF_elts]
AF_pi_ops = [pi_op_of(M) for M in AF_pis]
acc = np.zeros((0, d_base))
for i in range(len(AF_pis)):
    for j in range(len(AF_pis)):
        acc = np.vstack([acc, order1_pair_fast(AF_pis[i], AF_pi_ops[j])])
        if acc.shape[0] > 4 * d_base:
            acc = compress_rows(acc)
U, S, Vt = scipy.linalg.svd(acc, full_matrices=True)
rnk = int(np.sum(S > S.max() * 1e-10))
V_AF = Vt[rnk:].T   # (d_base, 16)
print(f"  V_AF shape: {V_AF.shape}  ({time.time()-t0:.1f}s)")

# ---- Build H_L+H_R+M_4(C) basis (in canonical PS rep) ----
def HHM4_pis():
    out = []
    for qL in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        out.append(pi_PS_full([[qL,(0,0,0,0)],[(0,0,0,0),(0,0,0,0)]],
                              np.zeros((4,4), dtype=complex)))
    for qR in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        out.append(pi_PS_full([[(0,0,0,0),(0,0,0,0)],[(0,0,0,0),qR]],
                              np.zeros((4,4), dtype=complex)))
    for i in range(4):
        for j in range(4):
            for kind in ['re','im']:
                Z = np.zeros((4,4), dtype=complex)
                Z[i,j] = 1.0 if kind=='re' else 1j
                out.append(pi_PS_full([[(0,0,0,0),(0,0,0,0)],
                                       [(0,0,0,0),(0,0,0,0)]], Z))
    out.append(pi_PS_full([[(1,0,0,0),(0,0,0,0)],[(0,0,0,0),(1,0,0,0)]],
                          np.eye(4, dtype=complex)))
    return out

print("Building H+H+M_4(C) basis (PS rep, 40 elements)...")
HHM4 = HHM4_pis()
HHM4_op = [pi_op_of(M) for M in HHM4]
print(f"  {len(HHM4)} basis elements")

# ---- Efficient: accumulate constraints projected onto V_AF (16-dim) ----
# For each (a, b) pair, the constraint is C @ alpha = 0 where alpha lives in
# d_base. Project to V_AF: alpha = V_AF @ beta, beta lives in R^16.
# Then the constraint becomes (C @ V_AF) @ beta = 0.
# This is a (2 N^2, 16) matrix. We accumulate and find rank.

# With early termination: rank 16 - 8 = 8 expected. So we can stop after rank
# hits 8 (if it stabilizes).

print()
print("Projecting H+H+M_4 order-one constraints onto V_AF (16-dim)...")
print("Looking for additional constraints beyond A_F's...")

t0 = time.time()
acc_AF = np.zeros((0, 16))
prev_rank = 0
plateau = 0
pair_count = 0
max_pairs = 1600
for i in range(len(HHM4)):
    for j in range(len(HHM4)):
        pair_count += 1
        C_full = order1_pair_fast(HHM4[i], HHM4_op[j])  # (2 N^2, d_base)
        C_AF = C_full @ V_AF  # (2 N^2, 16)
        # Reduce to row space (rows that are non-zero modulo numerical noise)
        rownorms = np.linalg.norm(C_AF, axis=1)
        keep = rownorms > 1e-12
        if not keep.any():
            continue
        C_AF = C_AF[keep]
        acc_AF = np.vstack([acc_AF, C_AF])
        if acc_AF.shape[0] > 50:
            # Compress
            U, S, Vt = scipy.linalg.svd(acc_AF, full_matrices=False)
            rnk = int(np.sum(S > S.max() * 1e-10)) if len(S) else 0
            acc_AF = (np.diag(S[:rnk]) @ Vt[:rnk])
        # Check rank
        if acc_AF.shape[0] > 0:
            current_rank = np.linalg.matrix_rank(acc_AF, tol=1e-8)
        else:
            current_rank = 0
        if current_rank == prev_rank:
            plateau += 1
            if plateau > 50:
                break
        else:
            plateau = 0
        prev_rank = current_rank
    if plateau > 50:
        break

# Final rank
if acc_AF.shape[0] > 0:
    U, S, Vt = scipy.linalg.svd(acc_AF, full_matrices=True)
    rnk = int(np.sum(S > S.max() * 1e-10))
    extra_constraints = Vt[:rnk]   # (rnk, 16): each row is a real constraint on beta
    null_in_AF = Vt[rnk:]          # (16-rnk, 16): the surviving Dirac directions
else:
    rnk = 0
    extra_constraints = np.zeros((0, 16))
    null_in_AF = np.eye(16)

print(f"  pairs scanned: {pair_count}/{max_pairs}")
print(f"  additional order-one rank from H+H+M_4 \\ A_F constraints: {rnk}")
print(f"  resulting Dirac dim = 16 - {rnk} = {16 - rnk}")
print(f"  ({time.time()-t0:.1f}s)")

# ---- Identify which physical Yukawa params survive ----
print()
print("=" * 75)
print("PHYSICAL CONTENT OF THE 8-DIM SURVIVING DIRAC SPACE")
print("=" * 75)

# Build 16 physical Yukawa basis Diracs (same as Phase 5)
def physical_basis_dirac(ftype, wL, wR, kind):
    D = np.zeros((N, N), dtype=complex)
    cs = [1, 2, 3] if ftype == 'quark' else [4]
    val = 1.0 if kind == 're' else 1j
    for c in cs:
        D[idx(+1, -1, wR, c), idx(+1, +1, wL, c)] += val
        D[idx(+1, +1, wL, c), idx(+1, -1, wR, c)] += np.conj(val)
        D[idx(-1, -1, wR, c), idx(-1, +1, wL, c)] += np.conj(val)
        D[idx(-1, +1, wL, c), idx(-1, -1, wR, c)] += val
    return D

phys_labels = []
phys_in_AF_coords = []  # in V_AF's 16-dim coord
for ftype in ['quark', 'lepton']:
    for wL in [+1, -1]:
        for wR in [+1, -1]:
            for kind in ['re', 'im']:
                D = physical_basis_dirac(ftype, wL, wR, kind)
                v = np.concatenate([D.real.flatten(), D.imag.flatten()])
                # Project onto V_AF coords
                alpha = B.T @ v          # to d_base coords
                beta = V_AF.T @ alpha    # to 16-dim A_F coords
                phys_labels.append(f"{ftype:6s} L_{'u' if wL==+1 else 'd'} <-> R_{'u' if wR==+1 else 'd'} ({kind})")
                phys_in_AF_coords.append(beta)
phys_M = np.array(phys_in_AF_coords)  # (16, 16) in V_AF coords

# Express physical basis in A_F coords; then apply the extra constraints from H+H+M_4
# A "physical Yukawa direction" survives the additional constraints iff
# extra_constraints @ beta = 0.
print("Each physical Yukawa parameter, checked against H+H+M_4 extra constraints:")
print(f"{'Yukawa direction':45s}  {'|constraint . beta|':>20s}  survives?")
for k, lbl in enumerate(phys_labels):
    beta = phys_in_AF_coords[k]
    if extra_constraints.shape[0] > 0:
        violation = np.max(np.abs(extra_constraints @ beta))
    else:
        violation = 0.0
    survives = violation < 1e-6
    print(f"  {lbl:45s}  {violation:>20.3e}  {'YES' if survives else 'no'}")

# What linear combinations of the 16 Yukawa directions form the 8-dim surviving
# subspace?
print()
print("=" * 75)
print("BASIS FOR THE SURVIVING 8-DIM DIRAC SPACE  (in physical Yukawa coords)")
print("=" * 75)
# null_in_AF is (8, 16) in V_AF coords. Each row is a basis vector beta in
# V_AF's 16-dim coord. Express in terms of the 16 physical Yukawa parameters:
# the matrix M = (16 physical Yukawa)_to_(V_AF coords) is phys_M, shape (16, 16).
# We have beta = phys_M.T @ phys_coeffs, so phys_coeffs = (phys_M.T)^{-1} beta.

# Solve phys_M.T @ phys_coeffs = beta for each row of null_in_AF
phys_M_inv_T = np.linalg.inv(phys_M.T)  # if phys_M is invertible
for k, beta in enumerate(null_in_AF):
    coeffs = phys_M_inv_T @ beta
    # Clean small entries
    coeffs[np.abs(coeffs) < 1e-8] = 0
    print(f"\nSurviving direction {k+1}:")
    for j, c in enumerate(coeffs):
        if abs(c) > 1e-8:
            sign = '+' if c.real >= 0 else '-'
            print(f"  {sign} {abs(c.real):.4f} * ({phys_labels[j]})")
