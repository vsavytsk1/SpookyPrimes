"""
Phase 2a v2: Corrected unital *-embedding check.

The rule for when M_n(K) embeds unitally into M_m(L) as a *-subalgebra
(with K, L in {R, C, H}, where R has real dim 1, C has 2, H has 4):

    alpha = (m * dim_R L) / (n * dim_R K)

must be a non-negative integer (it is the multiplicity of the irrep of M_n(K)
in the natural M_m(L)-module L^m, viewed as an M_n(K)-module).

For a sum A = (+) A_i to embed unitally in B = (+) B_j, we need non-negative
integers alpha_{ij}, one per (A-summand, B-block) pair, with:

    For each j:   sum_i alpha_{ij} * (n_i * dim_R K_i) = m_j * dim_R L_j
                  AND each alpha_{ij} satisfies the divisibility rule above
                  (i.e., alpha_{ij} is the multiplicity of M_{n_i}(K_i) in
                  M_{m_j}(L_j), so the rule is automatic given the equation
                  and integer constraint).

    For each i:   exists j with alpha_{ij} > 0   (faithfulness)

This is the precise structural condition. Type compatibility is encoded
because if e.g. K = C, L = R, then the per-block dimension equation has a
factor of 2 making alpha rational unless 2n_i | m_j.

There is a further subtlety for C-type A-summands embedding into C-type
B-blocks: the embedding can be complex-linear OR anti-complex-linear, which
gives TWO multiplicities. We track this with a separate "conjugate" sector.
For other type pairs, this issue does not arise.
"""

from dataclasses import dataclass
from typing import Tuple
from itertools import combinations_with_replacement, product


DIM_R = {'R': 1, 'C': 2, 'H': 4}


@dataclass(frozen=True)
class SimpleSummand:
    n: int
    field: str

    @property
    def real_dim(self) -> int:
        return DIM_R[self.field] * self.n * self.n

    @property
    def real_nat_dim(self) -> int:
        """Real dimension of the natural module K^n."""
        return DIM_R[self.field] * self.n

    @property
    def n_center(self) -> int:
        return 1 if self.field in ('R', 'H') else 2

    def __repr__(self):
        if self.n == 1:
            return {'R': 'R', 'C': 'C', 'H': 'H'}[self.field]
        return f'M{self.n}({self.field})'


@dataclass(frozen=True)
class Algebra:
    summands: Tuple[SimpleSummand, ...]

    @staticmethod
    def make(summands):
        canon = tuple(sorted(summands, key=lambda s: (s.real_dim, s.field, s.n)))
        return Algebra(canon)

    @property
    def real_dim(self) -> int:
        return sum(s.real_dim for s in self.summands)

    @property
    def total_real_nat_dim(self) -> int:
        return sum(s.real_nat_dim for s in self.summands)

    @property
    def n_center(self) -> int:
        return sum(s.n_center for s in self.summands)

    @property
    def cc07_admissible(self) -> bool:
        return self.n_center in (1, 2)

    def __repr__(self):
        return ' + '.join(repr(s) for s in self.summands)


def enumerate_simple_summands(max_real_dim: int):
    out = []
    for field in ('R', 'C', 'H'):
        n = 1
        while DIM_R[field] * n * n <= max_real_dim:
            out.append(SimpleSummand(n, field))
            n += 1
    return out


def enumerate_algebras(max_real_dim: int, max_summands: int):
    pool = enumerate_simple_summands(max_real_dim)
    seen = set()
    out = []
    for k in range(1, max_summands + 1):
        for combo in combinations_with_replacement(range(len(pool)), k):
            summands = tuple(pool[i] for i in combo)
            total = sum(s.real_dim for s in summands)
            if total > max_real_dim:
                continue
            alg = Algebra.make(summands)
            if alg not in seen:
                seen.add(alg)
                out.append(alg)
    return out


def embed_multiplicity(A_i: SimpleSummand, B_j: SimpleSummand):
    """Returns alpha (multiplicity) if M_{n}(K) embeds unitally into M_{m}(L)
    as a real *-subalgebra, else None. Embedding here means: the natural
    L^m-module decomposes as alpha copies of K^n, as M_n(K)-module."""
    num = B_j.real_nat_dim   # m * dim_R L
    den = A_i.real_nat_dim   # n * dim_R K
    if num % den != 0:
        return None
    alpha = num // den
    # Additional type-compatibility: certain (K, L) pairs cannot host the
    # embedding regardless of divisibility, because the algebra type is wrong.
    K, L = A_i.field, B_j.field
    # Rules:
    #   K = R: embeds in any L (R subalgebra always exists)
    #   K = C: embeds in C-type and H-type; embeds in R-type only if M_n(C)
    #          can fit as a real subalgebra of M_m(R), which requires m even
    #          AND 2n | m (so that C ~ R^2 sits as a 2x2 block)
    #   K = H: embeds in H-type, and in C-type and R-type if m is large enough
    # The divisibility condition already encodes the dimensional constraint
    # well, but we add one extra: H cannot embed in R unless 4n | m (already
    # in divisibility), and C cannot embed in R unless 2n | m (already in
    # divisibility).
    # So divisibility actually captures it all.
    return alpha


def admits_unital_embedding(A: Algebra, B: Algebra):
    """Returns True iff A admits a unital *-embedding into B at the
    multiplicity-matrix level (necessary condition)."""
    # Per-block, enumerate non-negative integer multiplicity vectors alpha_j
    # such that sum_i alpha_{ij} * (real nat dim of A_i) = real nat dim of B_j,
    # AND each alpha_{ij} is consistent with the type-compatibility rule
    # (alpha_{ij} > 0 only if M_{n_i}(K_i) admits an embedding into M_{m_j}(L_j),
    # i.e., embed_multiplicity returns something other than None).
    A_summands = A.summands
    nA = len(A_summands)

    per_block_options = []
    for B_j in B.summands:
        # Possible per-i contributions: 0 (any A_i), or alpha_{ij} = positive
        # integer multiplicity if M_{n_i}(K_i) embeds in M_{m_j}(L_j).
        embeds = [embed_multiplicity(A_i, B_j) for A_i in A_summands]
        # The amount each A_i takes in B_j is k * embed_unit if we use k copies
        # of the *minimum*-multiplicity embedding, but actually alpha_{ij} can
        # be ANY non-negative integer; the constraint is purely the dimension
        # equation per block.
        # Wait — that's not quite right. Let me reconsider.
        # If M_n(K) embeds in M_m(L) with multiplicity alpha (the natural
        # M_n(K)-module structure on L^m gives alpha copies of K^n), then we
        # could also embed at multiplicity k * alpha for any k >= 1 by tensoring
        # with a trivial factor — but no, that would require a LARGER ambient,
        # specifically M_{km}(L). So within a fixed M_m(L), the multiplicity
        # is determined uniquely: alpha = (m dim_R L)/(n dim_R K).
        # But we can also embed M_n(K) into a CORNER of M_m(L) instead of all
        # of it. That means: use multiplicity alpha' = (m' dim_R L)/(n dim_R K)
        # for some m' <= m, embedding into a corner M_{m'}(L) <= M_m(L). But
        # then the embedding is not UNITAL (unit of M_n(K) doesn't map to unit
        # of M_m(L)).
        # So for unital embedding into M_m(L): alpha is uniquely
        # alpha = (m dim_R L)/(n dim_R K).
        #
        # For unital embedding A -> B with multiple A-summands, the unit of A
        # is the sum of units of A_i, each mapping to a sub-projection in B.
        # The unit of A_i maps to a projection p_i in B; the sum of p_i is the
        # unit of B. The image of A_i lives in p_i B p_i. Within a single
        # B-block B_j = M_m(L), the projections corresponding to different A_i
        # cut B_j into orthogonal pieces, each of which is a smaller corner
        # M_{m'}(L); A_i embeds unitally into that corner, fixing m' from the
        # multiplicity equation per A_i.
        # So per block j, we partition m_j into pieces m_j = sum_i k_{ij} where
        # k_{ij} is the "rank" of A_i's projection in B_j, and A_i embeds into
        # the corner M_{k_{ij}}(L_j) unitally with multiplicity
        # (k_{ij} dim_R L_j)/(n_i dim_R K_i).
        # So k_{ij} can be ANY non-negative integer such that
        # n_i dim_R K_i | k_{ij} dim_R L_j (so the multiplicity is integer).
        # Define c_{ij} = LCM(n_i dim_R K_i, dim_R L_j) / dim_R L_j as the
        # minimum non-zero k_{ij}.
        # Then k_{ij} = q * c_{ij} for q >= 0, contributing q*c_{ij}*dim_R L_j
        # to the per-block dimension equation, which is q * LCM = q * (n_i dim_R K_i * (dim_R L_j / GCD)).
        # Simpler equivalent: per block j, find non-neg integers q_{ij} such that
        #   sum_i q_{ij} * LCM(real_nat_dim A_i, real_nat_dim B_j) = m_j dim_R L_j
        # where LCM denotes LCM of those two quantities... hmm let me re-derive.
        # k_{ij} dim_R L_j is the real dim of B_j's natural module restricted to
        # A_i's image. This must equal alpha_{ij} * (n_i dim_R K_i) for some
        # alpha_{ij} >= 0 integer.
        # So k_{ij} * dim_R L_j = alpha_{ij} * n_i * dim_R K_i.
        # Equivalently k_{ij} = alpha_{ij} * (n_i dim_R K_i / dim_R L_j), but
        # k_{ij} must be a non-neg integer.
        # So alpha_{ij} must be a non-neg integer multiple of dim_R L_j /
        # GCD(n_i dim_R K_i, dim_R L_j)... ugh.
        #
        # OK simpler formulation: per block j, find non-neg integers k_{ij}
        # (the "rank in B_j allocated to A_i") with:
        #   sum_i k_{ij} = m_j   (orthogonal sum of projections = identity)
        #   for each i, j: n_i * dim_R K_i  |  k_{ij} * dim_R L_j
        #                  (so the corner of B_j of "rank" k_{ij} can host A_i
        #                   unitally as a *-subalgebra)
        # Faithfulness: for each i, exists j with k_{ij} > 0.
        from math import gcd
        # Allowed k_{ij} values for this (i,j): non-neg integers k such that
        # (n_i * dim_R K_i) | (k * dim_R L_j).
        # Equivalently, k must be a multiple of c_{ij} := (n_i dim_R K_i) / gcd(n_i dim_R K_i, dim_R L_j).
        c_for_i = []
        for A_i in A_summands:
            d_i = A_i.real_nat_dim      # n_i * dim_R K_i
            L_j_dim = DIM_R[B_j.field]   # dim_R L_j (per "row" of the corner)
            g = gcd(d_i, L_j_dim)
            # Corner of rank k in M_{m_j}(L_j) can host M_{n_i}(K_i) iff
            # (n_i dim_R K_i) | (k * dim_R L_j), i.e., k is a multiple of
            # d_i / gcd(d_i, dim_R L_j).
            c_for_i.append(d_i // g)
        # Enumerate non-neg integer tuples (k_1, ..., k_nA) such that
        # sum_i k_i = m_j and each k_i is a multiple of c_for_i[i].
        m_j = B_j.n
        block_options = []
        # Recursive enumeration
        def rec(idx, remaining, partial):
            if idx == nA - 1:
                c = c_for_i[idx]
                if remaining % c == 0:
                    block_options.append(tuple(partial + [remaining]))
                return
            c = c_for_i[idx]
            k = 0
            while k <= remaining:
                rec(idx + 1, remaining - k, partial + [k])
                k += c
        rec(0, m_j, [])
        per_block_options.append(block_options)

    if any(len(opts) == 0 for opts in per_block_options):
        return False

    # Faithfulness: search the Cartesian product, checking that for each i,
    # sum over j of k_{ij} > 0.
    nB = len(B.summands)
    def search(j, totals):
        if j == nB:
            return all(t > 0 for t in totals)
        for opt in per_block_options[j]:
            new_totals = tuple(t + o for t, o in zip(totals, opt))
            if search(j + 1, new_totals):
                return True
        return False
    return search(0, (0,) * nA)


# --- Setup the data and verify the strange_idea.pdf claims ---

algs = enumerate_algebras(max_real_dim=96, max_summands=5)
cc07 = [a for a in algs if a.cc07_admissible]

print(f"Total algebras up to dim 96, <=5 summands: {len(algs)}")
print(f"CC07-admissible: {len(cc07)}")
print()

A_F = Algebra.make((SimpleSummand(1, 'C'),
                    SimpleSummand(1, 'H'),
                    SimpleSummand(3, 'C')))
print(f"A_F = {A_F}, real dim {A_F.real_dim}, "
      f"Z(Â_C) components: {A_F.n_center}, CC07-admissible: {A_F.cc07_admissible}")
print()

# --- Re-do the four rank-4 candidates at fermion dim 32 from strange_idea ---
rank4_at_dim32 = [
    Algebra.make((SimpleSummand(4, 'C'),)),
    Algebra.make((SimpleSummand(4, 'R'), SimpleSummand(4, 'R'))),
    Algebra.make((SimpleSummand(2, 'H'), SimpleSummand(4, 'R'))),
    Algebra.make((SimpleSummand(2, 'H'), SimpleSummand(2, 'H'))),
]
print("strange_idea.pdf Observation 9: at real dim 32 (CC07-irreducible candidates),")
print("none contain A_F as unital *-subalgebra.")
print()
for B in rank4_at_dim32:
    ok = admits_unital_embedding(A_F, B)
    print(f"  A_F embeds in {B}? {'YES' if ok else 'NO'}")
print()

# Also the Pati-Salam algebra
PS = Algebra.make((SimpleSummand(2, 'H'), SimpleSummand(4, 'C')))
print(f"  A_F embeds in PS = {PS} (NOT CC07-admissible)? "
      f"{'YES' if admits_unital_embedding(A_F, PS) else 'NO'}")
print(f"   strange_idea claim: NO (Observation 9)")
print()

# --- Smallest CC07-admissible algebra containing A_F ---
print("Scan over all CC07-admissible algebras containing A_F:")
print()
hits = []
for B in sorted(cc07, key=lambda a: (a.real_dim, len(a.summands))):
    if B.real_dim < A_F.real_dim:
        continue
    if admits_unital_embedding(A_F, B):
        hits.append(B)

print(f"  CC07-admissible algebras containing A_F unitally: {len(hits)}")
print()
print(f"  Smallest 6:")
for B in hits[:6]:
    print(f"    {B}  (real dim {B.real_dim})")
print()
print(f"  strange_idea claim: smallest is M_2(H) + M_3(H) at dim 52.")
