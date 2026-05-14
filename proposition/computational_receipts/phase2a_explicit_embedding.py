"""
Phase 2a v3: Construct an explicit unital *-embedding

        iota:  A_F = C (+) H (+) M_3(C)  -->  PS = M_2(H) (+) M_4(C)

and verify it preserves identity, addition, multiplication, and the *-involution
at the matrix level.

The candidate embedding:

  iota(z, q, X) = ( diag(z, q) in M_2(H), diag(z, X) in M_4(C) )

where
    z in C    -> embedded in H via the standard C <= H, AND in C as itself
    q in H    -> embedded in H trivially
    X in M_3(C) -> embedded as the upper-left 3x3 corner of M_4(C)

We represent everything as real matrices, then check homomorphism properties.

Concrete real-matrix realizations:
    C  -> M_2(R) via   a + bi  ->  [[a, -b], [b, a]]
    H  -> M_4(R) via   a + bi + cj + dk -> regular representation:
                       4x4 real matrix giving left multiplication on H ~ R^4
    M_n(C) -> M_{2n}(R) via the standard real form
    M_n(H) -> M_{4n}(R) via the standard real form

So:
    M_2(H) becomes an 8x8 real matrix algebra
    M_4(C) becomes an 8x8 real matrix algebra
    PS becomes a 16x16 real matrix algebra (block-diagonal)
    A_F becomes a 24-dim real subalgebra of the 16x16 ambient via iota.
"""

import numpy as np

# --- Real matrix realizations ---

def C_to_R2(z):
    """Complex number a + bi -> 2x2 real matrix."""
    a, b = z.real, z.imag
    return np.array([[a, -b], [b, a]], dtype=float)

def H_to_R4(q):
    """Quaternion q = a + bi + cj + dk represented as a numpy array (a,b,c,d).
    Returns 4x4 real matrix = left multiplication by q on R^4 in basis (1,i,j,k)."""
    a, b, c, d = q
    return np.array([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a],
    ], dtype=float)

def Mn_C_to_R(X):
    """M_n(C) represented as 2n x 2n real matrix via Kronecker."""
    n = X.shape[0]
    R = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            R[2*i:2*i+2, 2*j:2*j+2] = C_to_R2(X[i, j])
    return R

def Mn_H_to_R(Q):
    """M_n(H) represented as 4n x 4n real matrix. Q is an n x n array of
    quaternion 4-tuples."""
    n = len(Q)
    R = np.zeros((4 * n, 4 * n))
    for i in range(n):
        for j in range(n):
            R[4*i:4*i+4, 4*j:4*j+4] = H_to_R4(Q[i][j])
    return R

# --- Build the embedding ---

def iota(z, q, X):
    """Given (z in C, q in H as 4-tuple, X in M_3(C) as 3x3 complex array),
    return (iota_1 in M_2(H) realized as 8x8 real, iota_2 in M_4(C) as 8x8 real).
    """
    # First component: diag(z embedded in H, q) in M_2(H).
    # z = a + bi embeds in H as a + bi + 0 j + 0 k.
    z_as_q = (z.real, z.imag, 0.0, 0.0)
    Q_block = [[z_as_q, (0,0,0,0)],
               [(0,0,0,0), q]]
    iota1 = Mn_H_to_R(Q_block)  # 8x8 real

    # Second component: diag(z, X) in M_4(C).
    # z occupies the (0,0) entry; X occupies the (1:4, 1:4) block.
    M4C = np.zeros((4, 4), dtype=complex)
    M4C[0, 0] = z
    M4C[1:4, 1:4] = X
    iota2 = Mn_C_to_R(M4C)  # 8x8 real

    return iota1, iota2

def full_iota(z, q, X):
    """Block-diagonal 16x16 real matrix."""
    a, b = iota(z, q, X)
    R = np.zeros((16, 16))
    R[:8, :8] = a
    R[8:, 8:] = b
    return R

# --- Verify homomorphism properties ---

rng = np.random.default_rng(42)

def random_C():
    return complex(rng.standard_normal(), rng.standard_normal())

def random_H():
    return tuple(rng.standard_normal(4))

def random_M3C():
    return rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))

def AF_multiply(t1, t2):
    z1, q1, X1 = t1
    z2, q2, X2 = t2
    z = z1 * z2
    # Quaternion multiplication
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    q = (a1*a2 - b1*b2 - c1*c2 - d1*d2,
         a1*b2 + b1*a2 + c1*d2 - d1*c2,
         a1*c2 - b1*d2 + c1*a2 + d1*b2,
         a1*d2 + b1*c2 - c1*b2 + d1*a2)
    X = X1 @ X2
    return (z, q, X)

def AF_add(t1, t2):
    return (t1[0] + t2[0], tuple(a+b for a, b in zip(t1[1], t2[1])), t1[2] + t2[2])

def AF_scalar(r, t):
    return (r * t[0], tuple(r*a for a in t[1]), r * t[2])

def AF_star(t):
    z, q, X = t
    z_star = z.conjugate()
    a, b, c, d = q
    q_star = (a, -b, -c, -d)
    X_star = X.conjugate().T
    return (z_star, q_star, X_star)

def AF_one():
    return (1.0+0j, (1.0, 0.0, 0.0, 0.0), np.eye(3, dtype=complex))

# --- Check 1: iota(1) = identity_{16x16} ---
I16 = full_iota(*AF_one())
err_identity = np.max(np.abs(I16 - np.eye(16)))
print(f"Check unitality: ||iota(1) - I_16||_inf = {err_identity:.3e}")
assert err_identity < 1e-12

# --- Check 2: multiplicativity iota(a * b) = iota(a) * iota(b) for random a, b ---
errs_mult = []
for _ in range(50):
    t1 = (random_C(), random_H(), random_M3C())
    t2 = (random_C(), random_H(), random_M3C())
    LHS = full_iota(*AF_multiply(t1, t2))
    RHS = full_iota(*t1) @ full_iota(*t2)
    errs_mult.append(np.max(np.abs(LHS - RHS)))
print(f"Check multiplicativity over 50 random pairs: max ||iota(ab) - iota(a)iota(b)||_inf = "
      f"{max(errs_mult):.3e}")

# --- Check 3: additivity ---
errs_add = []
for _ in range(50):
    t1 = (random_C(), random_H(), random_M3C())
    t2 = (random_C(), random_H(), random_M3C())
    LHS = full_iota(*AF_add(t1, t2))
    RHS = full_iota(*t1) + full_iota(*t2)
    errs_add.append(np.max(np.abs(LHS - RHS)))
print(f"Check additivity over 50 random pairs: max error = {max(errs_add):.3e}")

# --- Check 4: real linearity (scalar) ---
errs_scal = []
for _ in range(50):
    t = (random_C(), random_H(), random_M3C())
    r = float(rng.standard_normal())
    LHS = full_iota(*AF_scalar(r, t))
    RHS = r * full_iota(*t)
    errs_scal.append(np.max(np.abs(LHS - RHS)))
print(f"Check R-linearity over 50 random pairs: max error = {max(errs_scal):.3e}")

# --- Check 5: *-preservation iota(t^*) = iota(t)^T (since real-matrix * is transpose) ---
errs_star = []
for _ in range(50):
    t = (random_C(), random_H(), random_M3C())
    LHS = full_iota(*AF_star(t))
    RHS = full_iota(*t).T
    errs_star.append(np.max(np.abs(LHS - RHS)))
print(f"Check *-preservation over 50 random elements: max error = {max(errs_star):.3e}")

# --- Check 6: faithfulness (kernel = 0) ---
# Random non-zero element of A_F should map to non-zero matrix.
nz_kernel_count = 0
for _ in range(100):
    t = (random_C(), random_H(), random_M3C())
    M = full_iota(*t)
    if np.max(np.abs(M)) < 1e-14:
        # would be zero in image
        # only if input was tiny
        if max(abs(t[0]), max(abs(x) for x in t[1]), np.max(np.abs(t[2]))) > 1e-10:
            nz_kernel_count += 1
print(f"Check faithfulness: nonzero elements mapped to zero (out of 100): {nz_kernel_count}")

print()
print("CONCLUSION:")
print(f"  iota: A_F -> PS is a valid unital *-algebra homomorphism with zero kernel.")
print(f"  Therefore A_F embeds as a unital *-subalgebra of M_2(H) (+) M_4(C).")
print(f"  This DISAGREES with strange_idea.pdf Observation 9's parenthetical that")
print(f"  'M_2(H) (+) M_4(C) does not [contain A_F as a unital *-subalgebra].'")
print()
print("  The honest reading is: A_F embeds in PS at the level of unital *-")
print("  homomorphisms (this is a multiplicity-2 / corner-3 embedding), but")
print("  does NOT embed via a 'canonical' multiplicity-1 inclusion where each")
print("  irrep of A_F appears in the natural module of PS exactly once. The")
print("  bimodule statement in the NCG literature is at the level of THE SM's")
print("  specific (J, gamma)-decorated representation, not abstract subalgebra")
print("  containment.")
