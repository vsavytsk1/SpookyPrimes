"""
Phase 4 (fast): vectorized chain of subalgebras + structure of the A_F Dirac space.

Key optimization: instead of a Python loop over the 272 base-null basis vectors
per (M, N_op) pair, we batch all 272 D_k matrices as a (272, 32, 32) tensor and
apply the order-one operator T via batched numpy matmul:

    T(D_k) = D_k M N_op  -  M D_k N_op  -  N_op D_k M  +  N_op M D_k

This is 4 batched (272, 32, 32) @ (32, 32) matmuls per (M, N_op) pair, executed
in BLAS, ~50x faster than the Python loop version.
"""

import numpy as np
import scipy.linalg
import time

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True, linewidth=140)

# ---- Setup ----
def idx(a, s, w, c):
    return (0 if a==+1 else 1)*16 + (0 if s==+1 else 1)*8 + (0 if w==+1 else 1)*4 + (c-1)

def unidx(i):
    a_idx, rem = divmod(i, 16)
    s_idx, rem = divmod(rem, 8)
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

def pi_op_of(M):
    return P_J @ M.T @ P_J

# ---- Base null space ----
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

print("Building base null space...")
t0 = time.time()
base_C = np.vstack([hermitian_rows(), antigamma_rows(), Jcommute_rows()])
U, S, Vt = scipy.linalg.svd(base_C, full_matrices=True)
rank = int(np.sum(S > S.max() * 1e-10))
B = Vt[rank:].T  # 2*N^2 x d_base
d_base = B.shape[1]
# Pre-build the (d_base, N, N) complex D-tensor of base-null basis Diracs
D_stack = np.zeros((d_base, N, N), dtype=complex)
for k in range(d_base):
    v = B[:, k]
    D_stack[k] = v[:N*N].reshape(N, N) + 1j * v[N*N:].reshape(N, N)
print(f"  d_base = {d_base}  (in {time.time()-t0:.1f}s)")

# ---- Fast vectorized order-one constraint for one (M, N_op) pair ----
def order1_pair_fast(M, N_op):
    """Returns (2 N^2, d_base) real constraint matrix C such that C @ alpha = 0
    iff [[B @ alpha (as D), M], N_op] = 0."""
    # T_stack[k] = D_k M N_op - M D_k N_op - N_op D_k M + N_op M D_k
    DM = D_stack @ M             # (d_base, N, N)
    DMN = DM @ N_op              # (d_base, N, N)
    MDN = (M @ D_stack) @ N_op   # (d_base, N, N)
    NDM = (N_op @ D_stack) @ M   # (d_base, N, N)
    NMD = (N_op @ M) @ D_stack   # (d_base, N, N)
    T = DMN - MDN - NDM + NMD    # (d_base, N, N)
    T_flat = T.reshape(d_base, N*N)  # complex (d_base, N^2)
    # constraint matrix: rows = 2 N^2 entries (real, then imag), cols = alpha index
    C = np.vstack([T_flat.real.T, T_flat.imag.T])  # (2 N^2, d_base)
    return C

def compress_rows(M, tol_factor=1e-10):
    if M.shape[0] == 0: return M
    Q, R, _ = scipy.linalg.qr(M.T, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    if len(d) == 0: return np.zeros((0, M.shape[1]))
    rnk = int(np.sum(d > d.max() * tol_factor))
    return Q[:, :rnk].T

def stable_rank(M, tol_factor=1e-10):
    if M.shape[0] == 0: return 0
    _, R, _ = scipy.linalg.qr(M.T, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    return int(np.sum(d > d.max() * tol_factor)) if len(d) else 0

def dirac_dim(pi_basis, pi_op_basis, label, verbose=True):
    t0 = time.time()
    acc = np.zeros((0, d_base))
    n_pairs = len(pi_basis) * len(pi_op_basis)
    saturated = False
    for i in range(len(pi_basis)):
        for j in range(len(pi_op_basis)):
            C = order1_pair_fast(pi_basis[i], pi_op_basis[j])
            acc = np.vstack([acc, C])
            if acc.shape[0] > 4 * d_base:
                acc = compress_rows(acc)
                if acc.shape[0] >= d_base:
                    saturated = True
                    break
        if saturated:
            break
    rnk = stable_rank(acc) if acc.shape[0] > 0 else 0
    nullity = d_base - rnk
    if verbose:
        print(f"  {label:45s}  basis={len(pi_basis):3d}  pairs={n_pairs:5d}  "
              f"rank={rnk:3d}  Dirac_dim={nullity:3d}  ({time.time()-t0:.1f}s)")
    return nullity

def dirac_null_basis(pi_basis, pi_op_basis):
    """Returns (dim, V_null) where V_null has shape (d_base, dim) and its columns
    span the order-one Dirac null space in base-null coords."""
    acc = np.zeros((0, d_base))
    for i in range(len(pi_basis)):
        for j in range(len(pi_op_basis)):
            acc = np.vstack([acc, order1_pair_fast(pi_basis[i], pi_op_basis[j])])
            if acc.shape[0] > 4 * d_base:
                acc = compress_rows(acc)
    # Null space via SVD
    if acc.shape[0] == 0:
        return d_base, np.eye(d_base)
    U, S, Vt = scipy.linalg.svd(acc, full_matrices=True)
    rnk = int(np.sum(S > S.max() * 1e-10)) if len(S) else 0
    null = Vt[rnk:].T   # (d_base, nullity)
    return d_base - rnk, null

# ---- Subalgebra basis builders ----
def AF_unit():
    return (1+0j, (1,0,0,0), np.eye(3, dtype=complex))

def AF_subalg(C_yes, H_yes, M3_yes):
    out = []
    if C_yes:
        for lr, li in [(1,0),(0,1)]:
            out.append((complex(lr,li), (0,0,0,0), np.zeros((3,3), dtype=complex)))
    if H_yes:
        for q in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
            out.append((0+0j, q, np.zeros((3,3), dtype=complex)))
    if M3_yes:
        for i in range(3):
            for j in range(3):
                for kind in ['re','im']:
                    m = np.zeros((3,3), dtype=complex)
                    m[i,j] = 1.0 if kind=='re' else 1j
                    out.append((0+0j, (0,0,0,0), m))
    out.append(AF_unit())  # always include unit
    return out

# ---- Larger algebras (preserve gamma) ----
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

def HL_HR_M4_pis():
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

def HL_HR_M3_pis():
    out = []
    for qL in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        out.append(pi_PS_full([[qL,(0,0,0,0)],[(0,0,0,0),(0,0,0,0)]],
                              np.zeros((4,4), dtype=complex)))
    for qR in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
        out.append(pi_PS_full([[(0,0,0,0),(0,0,0,0)],[(0,0,0,0),qR]],
                              np.zeros((4,4), dtype=complex)))
    for i in range(3):
        for j in range(3):
            for kind in ['re','im']:
                Z = np.zeros((4,4), dtype=complex)
                Z[i,j] = 1.0 if kind=='re' else 1j
                out.append(pi_PS_full([[(0,0,0,0),(0,0,0,0)],
                                       [(0,0,0,0),(0,0,0,0)]], Z))
    out.append(pi_PS_full([[(1,0,0,0),(0,0,0,0)],[(0,0,0,0),(1,0,0,0)]],
                          np.eye(4, dtype=complex)))
    return out

# ---- Run chain ----
print()
print("=" * 80)
print("CHAIN OF SUBALGEBRAS:  Dirac space dim as a functional on A")
print("=" * 80)
print(f"  baseline (no order-one):                  d_base = {d_base}")

results = {}
# Subalgebras of A_F
for label, args in [
    ("trivial (unit only)",          (False, False, False)),
    ("C",                             (True,  False, False)),
    ("H",                             (False, True,  False)),
    ("M_3(C)",                        (False, False, True)),
    ("C + H",                         (True,  True,  False)),
    ("C + M_3(C)",                    (True,  False, True)),
    ("H + M_3(C)",                    (False, True,  True)),
    ("A_F = C + H + M_3(C)",          (True,  True,  True)),
]:
    elts = AF_subalg(*args)
    pis = [pi_F(*t) for t in elts]
    pi_ops = [pi_op_of(M) for M in pis]
    results[label] = dirac_dim(pis, pi_ops, label)

# Larger algebras: skip (we have H_L+H_R+M_4(C) -> 8 from Phase 3 already)
print()
print(f"  --- (skipping recomputation of H_L+H_R+M_4(C); from Phase 3: dim 8) ---")
results["H_L + H_R + M_4(C)"] = 8

# ---- Structure of the A_F 16-dim Dirac space ----
print()
print("=" * 80)
print("STRUCTURE OF THE 16-DIM A_F DIRAC NULL SPACE")
print("=" * 80)

AF_elts = AF_subalg(True, True, True)
AF_pis = [pi_F(*t) for t in AF_elts]
AF_pi_ops = [pi_op_of(M) for M in AF_pis]
dim_AF, V_null = dirac_null_basis(AF_pis, AF_pi_ops)
print(f"Confirmed: dim = {dim_AF}")

# For each basis vector, reconstruct D and identify nonzero entries
from collections import defaultdict
transitions_with_count = defaultdict(int)  # (from, to) -> count of basis vecs touching it
for k in range(dim_AF):
    alpha = V_null[:, k]
    v_full = B @ alpha
    D_k = v_full[:N*N].reshape(N, N) + 1j * v_full[N*N:].reshape(N, N)
    for i in range(N):
        for j in range(N):
            if abs(D_k[i, j]) > 1e-6:
                ai, si, wi, ci = unidx(i)
                aj, sj, wj, cj = unidx(j)
                ftype_i = 'q' if ci in (1,2,3) else 'l'
                ftype_j = 'q' if cj in (1,2,3) else 'l'
                key = ((ai, si, wi, ftype_i), (aj, sj, wj, ftype_j))
                transitions_with_count[key] += 1

# Sort by descending count and display
print()
print(f"{'destination |i>':>35s}  <==  {'source |j>':35s}    in_basis")
print("-" * 95)
def fmt_state(s):
    a, sg, w, ty = s
    a_lbl = 'p' if a == +1 else 'a'  # particle / antiparticle
    sg_lbl = 'L' if sg == +1 else 'R'
    w_lbl = 'u' if w == +1 else 'd'
    return f"{a_lbl}-{sg_lbl}-{w_lbl}-{ty}"
for (frm, to), cnt in sorted(transitions_with_count.items(),
                              key=lambda x: (-x[1], x[0])):
    print(f"   {fmt_state(frm):>30s}      <==  {fmt_state(to):30s}      {cnt:>3d}")

print()
print("Legend: p=particle, a=antiparticle; L=left, R=right; u=up, d=down (weak doublet); q=quark, l=lepton")
print()
print("=" * 80)
print("FINAL CHAIN  (sorted by real algebra dim)")
print("=" * 80)
chain = [
    ("trivial (unit only)", 1),
    ("C", 3),
    ("H", 5),
    ("C + H", 7),
    ("M_3(C)", 19),
    ("C + M_3(C)", 21),
    ("H + M_3(C)", 23),
    ("A_F = C + H + M_3(C)", 25),
    ("H_L + H_R + M_4(C)", 41),
]
print(f"{'algebra':40s} {'~real dim':>10s} {'Dirac dim':>12s}")
for name, dim in chain:
    if name in results:
        print(f"{name:40s} {dim:>10d} {results[name]:>12d}")
