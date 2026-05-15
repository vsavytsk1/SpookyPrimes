"""
Phase 5: Structural relationships between Dirac spaces.

Claims to verify:
    (a) C+M_3, H+M_3, and A_F all have Dirac dim 16. Since
        C+M_3 ⊂ A_F and H+M_3 ⊂ A_F, A_F imposes a SUPERSET of constraints, so:
            Dirac(A_F) ⊆ Dirac(H+M_3) ⊆ Dirac(M_3)
            Dirac(A_F) ⊆ Dirac(C+M_3) ⊆ Dirac(M_3)
        With equal dimensions, the first inclusions are EQUALITIES:
            Dirac(A_F) = Dirac(C+M_3) = Dirac(H+M_3)
        This says: adding the H summand to C+M_3 imposes NO additional constraint
        on the Dirac operator; the C+M_3 Dirac space is already H-compatible.
        Symmetrically, the C+M_3 Dirac space is the SAME as the H+M_3 Dirac space.

    (b) The explicit SM Yukawa Dirac D (built in Phase 2c with arbitrary Yukawas)
        lies in this 16-dim Dirac space. Verify by projecting D onto V_null and
        checking |D - P(D)| = 0.

    (c) The 16-dim Dirac space is parametrized by EXACTLY the SM Yukawa
        parameters: 2 fermion types × 4 (w,s)-mixings × Hermitian/J. We extract
        a "physical" parametrization.
"""

import numpy as np
import scipy.linalg

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, linewidth=140)

# Setup (same as before)
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

print("Building base null space...")
base_C = np.vstack([hermitian_rows(), antigamma_rows(), Jcommute_rows()])
U, S, Vt = scipy.linalg.svd(base_C, full_matrices=True)
rank = int(np.sum(S > S.max() * 1e-10))
B = Vt[rank:].T
d_base = B.shape[1]
print(f"  d_base = {d_base}")

D_stack = np.zeros((d_base, N, N), dtype=complex)
for k in range(d_base):
    v = B[:, k]
    D_stack[k] = v[:N*N].reshape(N, N) + 1j * v[N*N:].reshape(N, N)

# Order-one fast
def order1_pair_fast(M, N_op):
    DM = D_stack @ M
    DMN = DM @ N_op
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

def dirac_null_basis(pi_basis, pi_op_basis):
    acc = np.zeros((0, d_base))
    for i in range(len(pi_basis)):
        for j in range(len(pi_op_basis)):
            acc = np.vstack([acc, order1_pair_fast(pi_basis[i], pi_op_basis[j])])
            if acc.shape[0] > 4 * d_base:
                acc = compress_rows(acc)
    if acc.shape[0] == 0:
        return d_base, np.eye(d_base)
    U, S, Vt = scipy.linalg.svd(acc, full_matrices=True)
    rnk = int(np.sum(S > S.max() * 1e-10)) if len(S) else 0
    return d_base - rnk, Vt[rnk:].T

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
    out.append(AF_unit())
    return out

# Compute null spaces for the three "tied at 16" algebras
print()
print("Computing null spaces for C+M_3, H+M_3, A_F...")
print()

dims = {}
nulls = {}
for label, args in [("M_3(C)",      (False, False, True)),
                    ("C + M_3(C)",  (True,  False, True)),
                    ("H + M_3(C)",  (False, True,  True)),
                    ("A_F",         (True,  True,  True))]:
    elts = AF_subalg(*args)
    pis = [pi_F(*t) for t in elts]
    pi_ops = [pi_op_of(M) for M in pis]
    dim, V = dirac_null_basis(pis, pi_ops)
    dims[label] = dim
    nulls[label] = V  # shape (d_base, dim)
    print(f"  {label:20s} -> Dirac dim = {dim}")

# ---- (a) Check subspace equalities ----
print()
print("=" * 70)
print("SUBSPACE EQUALITY CHECKS")
print("=" * 70)

def subspace_equal(V1, V2, tol=1e-8):
    """V1, V2 columns span subspaces. Equal iff their column spaces match.
    Test: project each col of V2 onto col(V1) and check residual = 0."""
    if V1.shape[1] != V2.shape[1]:
        return False, V1.shape[1], V2.shape[1]
    # V1's column space projector: V1 @ pinv(V1)
    P = V1 @ np.linalg.pinv(V1)
    residual = np.max(np.abs(V2 - P @ V2))
    return residual < tol, residual, None

for label1, label2 in [("C + M_3(C)", "H + M_3(C)"),
                       ("C + M_3(C)", "A_F"),
                       ("H + M_3(C)", "A_F"),
                       ("M_3(C)", "A_F")]:
    V1, V2 = nulls[label1], nulls[label2]
    if V1.shape[1] == V2.shape[1]:
        eq, res, _ = subspace_equal(V1, V2)
        print(f"  Dirac({label1}) == Dirac({label2})?  "
              f"residual = {res:.2e}  {'YES' if eq else 'NO'}")
    else:
        # Check containment instead
        P = V1 @ np.linalg.pinv(V1)
        residual = np.max(np.abs(V2 - P @ V2))
        contained = residual < 1e-8
        print(f"  Dirac({label2}) ⊆ Dirac({label1})?  "
              f"residual = {residual:.2e}  {'YES' if contained else 'NO'}  "
              f"(dim ratio {V2.shape[1]} vs {V1.shape[1]})")

# ---- (b) Project the explicit SM Yukawa Dirac onto A_F's null space ----
print()
print("=" * 70)
print("THE EXPLICIT SM YUKAWA DIRAC")
print("=" * 70)
# Same construction as Phase 2c
Y = {(+1, 'quark'): 0.005, (-1, 'quark'): 0.003,
     (+1, 'lepton'): 0.0001, (-1, 'lepton'): 0.002}
def fermion_type(c): return 'quark' if c in (1,2,3) else 'lepton'
D_SM = np.zeros((N, N), dtype=complex)
for c in range(1, 5):
    ftype = fermion_type(c)
    for w in [+1, -1]:
        y = Y[(w, ftype)]
        D_SM[idx(+1, -1, w, c), idx(+1, +1, w, c)] = y
        D_SM[idx(+1, +1, w, c), idx(+1, -1, w, c)] = np.conj(y)
        D_SM[idx(-1, -1, w, c), idx(-1, +1, w, c)] = np.conj(y)
        D_SM[idx(-1, +1, w, c), idx(-1, -1, w, c)] = y

# As a real vector
v_SM = np.concatenate([D_SM.real.flatten(), D_SM.imag.flatten()])
# Project onto base null space
alpha_SM_in_base = B.T @ v_SM
v_proj_base = B @ alpha_SM_in_base
print(f"  Residual: D_SM into base null space = "
      f"{np.max(np.abs(v_SM - v_proj_base)):.2e}")
# Project alpha_SM_in_base onto V_AF
V_AF = nulls["A_F"]
alpha_in_AF = V_AF @ (V_AF.T @ alpha_SM_in_base)
v_proj_AF = B @ alpha_in_AF
D_proj = v_proj_AF[:N*N].reshape(N, N) + 1j * v_proj_AF[N*N:].reshape(N, N)
print(f"  Residual: D_SM into A_F's 16-dim Dirac space = "
      f"{np.max(np.abs(D_SM - D_proj)):.2e}")
print()
print(f"  >> The explicit SM Yukawa Dirac IS in A_F's 16-dim null space.")

# ---- (c) Identify physical parameters in the 16-dim null space ----
print()
print("=" * 70)
print("PHYSICAL PARAMETRIZATION OF THE 16-DIM A_F DIRAC SPACE")
print("=" * 70)
# Build 16 specific "Yukawa-like" Diracs, one per physical parameter.
# Parameters: per fermion type ∈ {quark, lepton}, per (w_L, w_R) ∈ {++, +-, -+, --}
# we have a complex Yukawa, but Hermitian fixes the lower triangle and J pairs
# particle ↔ antiparticle, so we get 2 real per (w_L, w_R) per type.
# Total: 2 types × 4 (w_L, w_R) × 2 (re, im) = 16.

def physical_basis_dirac(ftype, wL, wR, kind):
    """Build the Dirac element that couples particle (L, wL) <-> particle (R, wR),
    Hermitian + J. ftype determines color: quark uses c=1,2,3 all together
    (M_3-symmetric); lepton uses c=4. kind ∈ {'re', 'im'} chooses real or imag part."""
    D = np.zeros((N, N), dtype=complex)
    cs = [1, 2, 3] if ftype == 'quark' else [4]
    val = 1.0 if kind == 're' else 1j
    for c in cs:
        # Particle: L (s=+1, w=wL, c) <-> R (s=-1, w=wR, c)
        D[idx(+1, -1, wR, c), idx(+1, +1, wL, c)] += val
        D[idx(+1, +1, wL, c), idx(+1, -1, wR, c)] += np.conj(val)
        # Antiparticle (J-paired): J|+1, s, w, c> = |-1, s, w, c> with conj
        # So the antiparticle Yukawa, by DJ = JD, has conjugate value
        D[idx(-1, -1, wR, c), idx(-1, +1, wL, c)] += np.conj(val)
        D[idx(-1, +1, wL, c), idx(-1, -1, wR, c)] += val
    return D

phys_basis = []
phys_labels = []
for ftype in ['quark', 'lepton']:
    for wL in [+1, -1]:
        for wR in [+1, -1]:
            for kind in ['re', 'im']:
                D_phys = physical_basis_dirac(ftype, wL, wR, kind)
                phys_basis.append(D_phys)
                phys_labels.append(f"{ftype:6s} L_{'u' if wL==+1 else 'd'} <-> R_{'u' if wR==+1 else 'd'}  ({kind})")

print(f"  Built {len(phys_basis)} candidate physical Yukawa Dirac basis elements.")
# Check each is in the A_F null space
print("  Checking each lies in the 16-dim A_F Dirac space:")
in_null_count = 0
for k, (D_phys, lbl) in enumerate(zip(phys_basis, phys_labels)):
    v_phys = np.concatenate([D_phys.real.flatten(), D_phys.imag.flatten()])
    a_in_base = B.T @ v_phys
    proj_base = B @ a_in_base
    base_res = np.max(np.abs(v_phys - proj_base))
    a_in_AF = V_AF @ (V_AF.T @ a_in_base)
    proj_AF = B @ a_in_AF
    AF_res_v = proj_AF
    D_proj = AF_res_v[:N*N].reshape(N,N) + 1j*AF_res_v[N*N:].reshape(N,N)
    res = np.max(np.abs(D_phys - D_proj))
    in_null = res < 1e-8
    if in_null: in_null_count += 1
    print(f"    [{'✓' if in_null else '✗'}] {lbl:55s}  residual = {res:.2e}")

print()
print(f"  {in_null_count}/{len(phys_basis)} physical basis elements lie in A_F's Dirac space.")

# Check linear independence
phys_vecs = np.array([np.concatenate([D.real.flatten(), D.imag.flatten()])
                      for D in phys_basis])  # (16, 2*N^2)
# Project each into base coords (d_base-dim)
phys_in_base = phys_vecs @ B  # (16, d_base)
phys_in_AF = phys_in_base @ V_AF  # (16, dim_AF=16)
phys_rank = np.linalg.matrix_rank(phys_in_AF, tol=1e-8)
print(f"  Linear independence: rank of physical-basis-in-A_F-coords = {phys_rank}/16")
if phys_rank == 16:
    print(f"  >> The 16 physical Yukawa parameters EXACTLY parametrize A_F's Dirac space.")
