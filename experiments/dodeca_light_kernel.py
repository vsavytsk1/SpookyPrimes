#!/usr/bin/env python3
"""
dodeca_light_kernel.py -- proof by kernel for GENESIS v8.3 (the SpookyPrimes sim).

Vlad's idea: animate the net's LINES using the LIGHT MATRIX, and let that be the
path to hearing how the net SOUNDS. The honest bridge (Thea status grammar --
fake NEVER): M_light acts on shell-data (k^2,kl,l^2,P), NOT on 3D points. The net's
TRUE sound is its GRAPH-LAPLACIAN spectrum L = deg*I - A. Its eigenvectors are the
standing-wave modes (the honest discrete Fourier basis on the graph); its
eigenvalues are the squared frequencies.

THE GOLDEN THREAD (exact, must reproduce): the dodecahedron (C20, the pure P=12
seed, T=1) has Fiedler value
    lambda_2(L) = 3 - sqrt(5) = 2 * phi^-2 = 0.7639320225...
That number is ALREADY in Thea's certificate (Part VI, stability-line-D, row 0:
V=20 T=1 lambda2=0.7639320225). And phi^-2 is EXACTLY the contracting eigenvalue of
M_light = {phi^2, 1, -1, phi^-2}. So the net's fundamental tone squared IS twice the
light matrix's decay mode. Not narrated -- proven here, then matched live in-browser.

This kernel builds the SAME dodecahedron the sim builds (20 verts from phi, 30 edges
= nearest neighbors), forms A and L, factors the exact characteristic polynomial,
and emits the numeric spectrum + Fiedler vector to JSON for the browser to match.

Run:
    py -3 dodeca_light_kernel.py            # human proof
    py -3 dodeca_light_kernel.py --json     # machine receipt for the sim

spini. P=12. chi=2. The pentagons hold; the hexes pay; the net sings its Laplacian.
"""
import json
import sys
from itertools import combinations

import sympy as sp


PHI = (1 + sp.sqrt(5)) / 2


def dodecahedron_vertices():
    """20 vertices -- the SAME construction the sim uses (index.html Geometry):
    8 cube corners + 12 golden-ratio points. Exact (symbolic) coordinates."""
    invphi = 1 / PHI
    verts = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                verts.append((sp.Integer(x), sp.Integer(y), sp.Integer(z)))
    for a in (-1, 1):
        for b in (-1, 1):
            verts.append((sp.Integer(0), a * invphi, b * PHI))
            verts.append((a * invphi, b * PHI, sp.Integer(0)))
            verts.append((a * PHI, sp.Integer(0), b * invphi))
    return verts


def dist2(u, v):
    return sum((u[i] - v[i]) ** 2 for i in range(3))


def adjacency(verts):
    """30 edges = the pairs at minimum distance (exactly the sim's rule)."""
    n = len(verts)
    d2s = [sp.nsimplify(sp.simplify(dist2(verts[i], verts[j])))
           for i, j in combinations(range(n), 2)]
    min_d2 = min(sp.nsimplify(d) for d in d2s)
    A = sp.zeros(n, n)
    edges = []
    for (i, j) in combinations(range(n), 2):
        if sp.simplify(dist2(verts[i], verts[j]) - min_d2) == 0:
            A[i, j] = 1
            A[j, i] = 1
            edges.append((i, j))
    return A, edges, min_d2


def main():
    verts = dodecahedron_vertices()
    A, edges, min_d2 = adjacency(verts)
    n = len(verts)
    deg = [sum(A[i, j] for j in range(n)) for i in range(n)]
    assert all(d == 3 for d in deg), "dodecahedron must be 3-regular"
    assert len(edges) == 30, "must have 30 edges"

    L = 3 * sp.eye(n) - A  # cubic graph -> deg = 3 for every vertex

    # exact eigenvalues of A and L
    A_eig = A.eigenvals()      # {eigenvalue: multiplicity}
    L_eig = L.eigenvals()

    # numeric, sorted ascending, with multiplicity expanded
    L_num = []
    for val, mult in L_eig.items():
        for _ in range(mult):
            L_num.append(complex(sp.N(val, 20)).real)
    L_num.sort()

    lam2_exact = sp.nsimplify(sorted(L_eig.keys(), key=lambda e: sp.N(e))[1])
    lam2_num = float(sp.N(lam2_exact, 20))

    phi_m2 = float(sp.N(1 / PHI ** 2, 20))
    thea_row0 = 0.7639320225  # Thea cert stability-line-D, level 0 (V=20, T=1)

    if "--json" in sys.argv:
        # Fiedler vector (first nonzero mode) for the sim to match live.
        # Use numeric eigensystem for a stable, exportable vector.
        import mpmath as mp
        mp.mp.dps = 30
        An = mp.matrix([[int(A[i, j]) for j in range(n)] for i in range(n)])
        Ln = mp.eye(n) * 3 - An
        E, V = mp.eigsy(Ln)  # symmetric -> real
        pairs = sorted(range(n), key=lambda k: float(E[k]))
        fiedler_idx = pairs[1]
        fiedler = [float(V[i, fiedler_idx]) for i in range(n)]
        receipt = {
            "schema": "dodeca_light.v1",
            "n": n, "edges": len(edges), "degree": 3, "P": 12, "chi": 2,
            "phi": float(sp.N(PHI, 20)),
            "phi_minus_2": phi_m2,
            "lambda2_exact": "3 - sqrt(5) = 2*phi^-2",
            "lambda2_numeric": lam2_num,
            "thea_cert_row0": thea_row0,
            "matches_thea": abs(lam2_num - thea_row0) < 1e-9,
            "laplacian_spectrum": [round(x, 12) for x in L_num],
            "vertices": [[float(sp.N(c, 20)) for c in v] for v in verts],
            "edge_list": edges,
            "fiedler_vector": fiedler,
        }
        print(json.dumps(receipt, indent=2))
        return

    print("DODECAHEDRON OF OPEN QUESTIONS -- the light kernel (proof by kernel)")
    print("  n=%d  edges=%d  degree=3  P=12  chi=%d" % (n, len(edges), n - len(edges) + 12))
    print("  A char poly (factored):")
    print("    ", sp.factor(A.charpoly().as_expr()))
    print("  L char poly (factored):")
    print("    ", sp.factor(L.charpoly().as_expr()))
    print()
    print("  Laplacian spectrum (ascending, with multiplicity):")
    # group for readability
    grouped = {}
    for val, mult in L_eig.items():
        grouped[sp.nsimplify(val)] = mult
    for val in sorted(grouped, key=lambda e: sp.N(e)):
        print("    %-22s  x%d   (~ %.10f)" % (sp.sstr(val), grouped[val], float(sp.N(val, 20))))
    print()
    print("  THE GOLDEN THREAD:")
    print("    lambda_2 (Fiedler) exact = %s" % sp.sstr(lam2_exact))
    print("    lambda_2 numeric        = %.10f" % lam2_num)
    print("    2 * phi^-2              = %.10f" % (2 * phi_m2))
    print("    Thea cert row 0         = %.10f" % thea_row0)
    ok = abs(lam2_num - 2 * phi_m2) < 1e-12 and abs(lam2_num - thea_row0) < 1e-9
    print("    lambda_2 == 2*phi^-2 == Thea row0 : %s" % ("PASS" if ok else "FAIL"))
    print()
    print("  M_light spectrum = {phi^2, 1, -1, phi^-2}; the net's fundamental tone^2")
    print("  is exactly TWICE the light matrix's contracting eigenvalue phi^-2.")
    print()
    print("PROOF: " + ("PASS -- the net sings phi." if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
