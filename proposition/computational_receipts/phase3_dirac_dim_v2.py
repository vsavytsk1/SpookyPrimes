"""
Phase 3 (memory-efficient): Dirac space dimension as a functional on subalgebras.

Strategy:
    1. Compute the BASE null space  Null = {D : Hermitian, anti-gamma, commute-J}
       This is a real subspace of dimension d_base << 2*N^2 = 2048.
    2. Parametrize D in terms of an orthonormal basis B (2*N^2 x d_base) of Null.
    3. For each algebra A, sample (a, b) pairs from A x A, compute order-one
       constraint as a linear map on this d_base-dim parameter space, stack and
       take rank.
    4. Dim of Dirac space = d_base - rank(order-one constraints in d_base coords).
"""

import numpy as np
import scipy.linalg
rng = np.random.default_rng(0)

# ---- Setup (copy from phase2c) ----
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
gamma = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i)
    gamma[i, i] = a * s
P_J = np.zeros((N, N), dtype=complex)
for i in range(N):
    a, s, w, c = unidx(i)
    j = idx(-a, s, w, c)
    P_J[j, i] = 1.0

def quat_to_M2C(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0 + 1j*q1,  q2 + 1j*q3],
        [-q2 + 1j*q3, q0 - 1j*q1],
    ], dtype=complex)

def pi_F(lam, q, m):
    op = np.zeros((N, N), dtype=complex)
    Q = quat_to_M2C(q)
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

def pi_op_of(pi_a):
    return P_J @ pi_a.T @ P_J

# ---- Parametrization: D <-> v in R^{2 N^2} ----
def D_to_v(D):
    return np.concatenate([D.real.flatten(), D.imag.flatten()])

def v_to_D(v):
    Dr = v[:N*N].reshape(N, N)
    Di = v[N*N:].reshape(N, N)
    return Dr + 1j * Di

# ---- Base constraints (Hermitian + anti-gamma + commute J) ----
# Strategy: build constraints as linear functionals on v in R^{2 N^2}.

def hermitian_rows():
    rows = []
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(2*N*N); row[i*N + j] = 1; row[j*N + i] = -1
            rows.append(row)
            row = np.zeros(2*N*N); row[N*N + i*N + j] = 1; row[N*N + j*N + i] = 1
            rows.append(row)
    for i in range(N):
        row = np.zeros(2*N*N); row[N*N + i*N + i] = 1
        rows.append(row)
    return np.array(rows)

def antigamma_rows():
    rows = []
    g = np.diag(gamma).real.astype(int)
    for i in range(N):
        for j in range(N):
            if g[i] == g[j]:
                row = np.zeros(2*N*N); row[i*N + j] = 1
                rows.append(row)
                row = np.zeros(2*N*N); row[N*N + i*N + j] = 1
                rows.append(row)
    return np.array(rows)

def Jcommute_rows():
    pi_J = np.zeros(N, dtype=int)
    for j in range(N):
        pi_J[j] = int(np.argmax(np.abs(P_J[:, j])))
    rows = []
    for i in range(N):
        for j in range(N):
            ip = pi_J[i]; jp = pi_J[j]
            row = np.zeros(2*N*N); row[i*N + j] = 1; row[ip*N + jp] -= 1
            if not np.allclose(row, 0): rows.append(row)
            row = np.zeros(2*N*N); row[N*N + i*N + j] = 1; row[N*N + ip*N + jp] += 1
            if not np.allclose(row, 0): rows.append(row)
    return np.array(rows)

print("Building base constraint matrices...")
H_rows = hermitian_rows()
G_rows = antigamma_rows()
J_rows = Jcommute_rows()
print(f"  Hermitian: {H_rows.shape[0]} rows")
print(f"  Anti-gamma: {G_rows.shape[0]} rows")
print(f"  J-commute: {J_rows.shape[0]} rows")
base_C = np.vstack([H_rows, G_rows, J_rows])
print(f"  Stacked: {base_C.shape}")

# Compute null space (in R^2048) of base_C
print("Computing null space of base constraints...")
U, S, Vt = scipy.linalg.svd(base_C, full_matrices=True)
tol = max(S) * 1e-10
rank = int(np.sum(S > tol))
print(f"  Base rank: {rank}, base null dim: {2*N*N - rank}")
B = Vt[rank:].T  # 2*N^2 x d_base; orthonormal columns span null space
d_base = B.shape[1]
print(f"  Base null space dim d_base = {d_base}")

# Sanity-check: a random base-null vector should satisfy all base constraints.
alpha_test = rng.standard_normal(d_base)
v_test = B @ alpha_test
D_test = v_to_D(v_test)
print(f"  Verify D Hermitian: max |D - D*| = {np.max(np.abs(D_test - D_test.conj().T)):.2e}")
print(f"  Verify D anti-gamma: max |D gamma + gamma D| = "
      f"{np.max(np.abs(D_test @ gamma + gamma @ D_test)):.2e}")
# J commute: D P_J = P_J conj(D)
print(f"  Verify D commutes J: max |D P_J - P_J conj(D)| = "
      f"{np.max(np.abs(D_test @ P_J - P_J @ np.conj(D_test))):.2e}")

# ---- Order-one constraint in base-null coordinates ----
# For fixed M = pi(a), N_op = pi_op(b), the operator
#     T(D) := [[D, M], N_op]
# is a linear function of D. We compute T(B @ e_k) for each basis element
# e_k of the base-null space, getting a vector in C^{N^2}. Stacking gives
# a matrix C_pair (2 N^2 x d_base) [real after taking re/im parts]. We want
# vec(T) = 0, so we get 2 N^2 = 2048 real constraints per pair on alpha.

def order1_constraints(M, N_op):
    """Returns (2 N^2) x d_base real matrix C such that C @ alpha = 0 iff
    [[B @ alpha, M], N_op] = 0."""
    C = np.zeros((2*N*N, d_base))
    for k in range(d_base):
        v = B[:, k]
        D = v_to_D(v)
        T = (D @ M - M @ D) @ N_op - N_op @ (D @ M - M @ D)
        C[:N*N, k] = T.real.flatten()
        C[N*N:, k] = T.imag.flatten()
    return C

# ---- Algebras ----

# A_F basis (24 real basis elements)
def af_basis():
    basis = []
    for lam_re, lam_im in [(1,0), (0,1)]:
        basis.append((complex(lam_re, lam_im), (0,0,0,0), np.zeros((3,3), dtype=complex)))
    for q in [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]:
        basis.append((0+0j, q, np.zeros((3,3), dtype=complex)))
    for i in range(3):
        for j in range(3):
            for kind in ['re', 'im']:
                m = np.zeros((3,3), dtype=complex)
                m[i, j] = 1.0 if kind == 're' else 1j
                basis.append((0+0j, (0,0,0,0), m))
    return basis

AF_basis_elts = af_basis()
print(f"\nA_F basis size: {len(AF_basis_elts)}")
AF_pis = [pi_F(*t) for t in AF_basis_elts]
AF_pi_ops = [pi_op_of(M) for M in AF_pis]

# Block-diag PS basis: H_L (+) H_R (+) M_4(C), 40 real dim
def pi_PS_full(Q, Z):
    op = np.zeros((N, N), dtype=complex)
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
    for s in [+1, -1]:
        for w in [+1, -1]:
            for c1 in range(1, 5):
                for c2 in range(1, 5):
                    op[idx(-1, s, w, c1), idx(-1, s, w, c2)] = Z[c1-1, c2-1]
    return op

def blockdiag_basis():
    basis = []
    for qL in [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]:
        Q = [[qL, (0,0,0,0)], [(0,0,0,0), (0,0,0,0)]]
        basis.append(pi_PS_full(Q, np.zeros((4,4), dtype=complex)))
    for qR in [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]:
        Q = [[(0,0,0,0), (0,0,0,0)], [(0,0,0,0), qR]]
        basis.append(pi_PS_full(Q, np.zeros((4,4), dtype=complex)))
    for i in range(4):
        for j in range(4):
            for kind in ['re', 'im']:
                Z = np.zeros((4,4), dtype=complex)
                Z[i, j] = 1.0 if kind == 're' else 1j
                Q = [[(0,0,0,0), (0,0,0,0)], [(0,0,0,0), (0,0,0,0)]]
                basis.append(pi_PS_full(Q, Z))
    return basis

bd_pis = blockdiag_basis()
bd_pi_ops = [pi_op_of(M) for M in bd_pis]
print(f"H_L + H_R + M_4(C) basis size: {len(bd_pis)}")

# ---- Compute Dirac dim per algebra ----
def stable_rank(M, tol_factor=1e-10):
    """Compute numerical rank of M using QR with column pivoting (robust)."""
    if M.shape[0] == 0:
        return 0
    Q, R, piv = scipy.linalg.qr(M.T, pivoting=True)
    diag = np.abs(np.diag(R))
    if len(diag) == 0:
        return 0
    tol = diag.max() * tol_factor
    return int(np.sum(diag > tol))

def compress_rows(M, tol_factor=1e-10):
    """Compress M's rows to its row-rank's worth of independent rows via QR."""
    if M.shape[0] == 0:
        return M
    # QR of M^T: Q is N x k, R is k x rows
    Q, R, piv = scipy.linalg.qr(M.T, mode='economic', pivoting=True)
    diag = np.abs(np.diag(R))
    if len(diag) == 0:
        return np.zeros((0, M.shape[1]))
    tol = diag.max() * tol_factor
    rank = int(np.sum(diag > tol))
    # The row space basis = first `rank` columns of Q (transposed back)
    return Q[:, :rank].T

def dirac_dim(pi_basis, pi_op_basis, label):
    print(f"\n--- Algebra: {label} (basis size {len(pi_basis)}) ---")
    accumulated = np.zeros((0, d_base))
    n_pairs = len(pi_basis) * len(pi_op_basis)
    print(f"  Total pairs: {n_pairs}")
    count = 0
    for i in range(len(pi_basis)):
        for j in range(len(pi_op_basis)):
            C_pair = order1_constraints(pi_basis[i], pi_op_basis[j])
            accumulated = np.vstack([accumulated, C_pair])
            count += 1
            if accumulated.shape[0] > 4 * d_base:
                accumulated = compress_rows(accumulated)
    if accumulated.shape[0] > 0:
        rank = stable_rank(accumulated)
    else:
        rank = 0
    nullity = d_base - rank
    print(f"  Order-one rank in d_base={d_base} coords: {rank}")
    print(f"  Dim of Dirac space subject to order-one: {nullity}")
    return nullity

# Baseline (no order-one)
print(f"\nBaseline Dirac space dim (no order-one): {d_base}")

# A_F
dim_AF = dirac_dim(AF_pis, AF_pi_ops, "A_F = C (+) H (+) M_3(C)")

# Block-diag PS
dim_BD = dirac_dim(bd_pis, bd_pi_ops, "H_L (+) H_R (+) M_4(C)")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  d_base (KO-dim 6 + Hermitian, no algebra):     {d_base}")
print(f"  Dirac dim with A_F order-one:                  {dim_AF}")
print(f"  Dirac dim with H_L+H_R+M_4(C) order-one:       {dim_BD}")
print()
if dim_AF > dim_BD:
    print("  >> A_F gives STRICTLY MORE Dirac freedom than H_L+H_R+M_4(C).")
    print("  >> This is the carve-out at the Dirac-dimension level.")
elif dim_AF == dim_BD:
    print("  >> A_F and H_L+H_R+M_4(C) give the same Dirac freedom.")
    print("     The extension is order-one trivial (extends pi_op trivially).")
else:
    print("  >> H_L+H_R+M_4(C) gives MORE Dirac freedom than A_F.")
    print("     This would be surprising and warrants investigation.")
