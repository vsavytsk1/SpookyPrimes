# From Dirac Operators to Dodecahedra
## A Computational Narrative of Accidental Convergence
### V. Savytskyy · Buenos Aires · May 2026

> **Status:** Working document. Not a paper. Not a claim.
> A timestamped record of what was built, what was observed,
> and why it kept working. All referenced theorems belong to
> their original authors. We show. We do not claim.

---

## Abstract

This document traces a sequence of computational experiments
conducted between January and May 2026, beginning with a
machine-verified study of Dirac operator spaces in noncommutative
geometry (the SpookyPrimes project) and ending with a browser-based
fractal geometry engine (goldberg_kernel.js, 634 lines, zero
external dependencies) that independently reproduces known results
in topology, differential geometry, and discrete curvature theory.

The convergence was not planned. The narrative is presented
chronologically, with version-controlled timestamps from Git,
to document the order in which observations were made and the
absence of post-hoc rationalization.

All code is public. All results are reproducible.

---

## 1. The Starting Point: dim D(A_F) = 16

### 1.1 The Plateau

The SpookyPrimes project (proposition/computational_receipts/)
established through machine verification that the space of Dirac
operators compatible with the Standard Model algebra
A_F = C + H + M_3(C) has real dimension 16, matching the
Yukawa parameter count per generation.

Three algebras share this dimension:

| Algebra | Real dim | dim D |
|---------|----------|-------|
| C + M_3(C) | 21 | **16** |
| H + M_3(C) | 23 | **16** |
| C + H + M_3(C) | 25 | **16** |

This is a plateau in algebraic parameter space.

### 1.2 The Open Problem

No known variational functional on finite real spectral triples
(KO-dimension 6) uniquely selects A_F as its critical point
using dimension alone. The plateau makes this explicit.

### 1.3 The Pin

At this point, the algebraic investigation was paused. An email
was sent to Prof. van Suijlekom. The computational receipts were
archived. A pin was placed. The question left open: What additional
structure, beyond dimension counting, forces the Standard Model algebra?

---

## 2. The Pivot: From Algebra to Geometry

### 2.1 The Fractal Geometry Builder (January 2026)

Repository: vsavytsk1/Mnet

While the algebraic work was paused, a separate project was
initiated: an interactive browser-based tool for exploring
Goldberg-Coxeter polyhedra. The initial motivation was pedagogical.

### 2.2 The Kernel Extraction (May 2026)

By v4.4, the mathematical core had been extracted into
goldberg_kernel.js with the following properties:

- 634 lines of JavaScript (ES5 compatible)
- Zero external dependencies (no DOM, no WebGL, no framework)
- 43 pure functions (input to output, no side effects)
- 4 trigonometric calls total
- Universal export (Node.js, browser, Deno, globalThis)

### 2.3 The Architectural Decision

The separation of kernel (math) from shell (rendering) was
deliberate but its consequences were not anticipated. The kernel
was designed to be portable. It was not designed to be physics.

---

## 3. The Seven Primitives

Seven primitive graph operations were identified:

| Primitive | Operation | Graph-theoretic name |
|-----------|-----------|---------------------|
| P1 | CREATE NODE | Vertex insertion |
| P2 | CREATE EDGE | Edge insertion |
| P3 | COMPOSE | Path composition |
| P4 | TRANSFORM | Morphism application |
| P5 | ITERATE | Recursive application |
| P6 | AGGREGATE | Reduction / folding |
| P7 | COMPARE | Invariant checking |

Three constraints (C1: determinism, C2: irreversibility, C3: consistency)
were identified as necessary for topologically consistent structures.

---

## 4. The Invariants (Verified at Every Level)

| Level | Faces | P | V | E | chi | E/V |
|-------|-------|---|---|---|-----|-----|
| 0 | 32 | 12 | 60 | 90 | 2 | 1.5000 |
| 1 | 212 | 12 | 420 | 630 | 2 | 1.5000 |
| 2 | 1,472 | 12 | 2,940 | 4,410 | 2 | 1.5000 |
| 3 | 10,292 | 12 | 20,580 | 30,870 | 2 | 1.5000 |
| 4 | 72,032 | 12 | 144,060 | 216,090 | 2 | 1.5000 |
| 5 | 504,212 | 12 | 1,008,420 | 1,512,630 | 2 | 1.5000 |

Theoretical basis:
- P = 12: Euler (1758) + discrete Gauss-Bonnet
- chi = 2: Gauss-Bonnet theorem (1827)
- E/V = 3/2: Handshaking lemma, Euler (1736)
- Growth 7x: Asymptotic refinement ratio

---

## 5. The Observations

### 5.1 Fractal Neural Patterns

At refinement level 5, viewed at high zoom, the projected surface
displays patterns structurally similar to Golgi-stained neural tissue.
This is a consequence of shared optimization structure: both solve
maximizing surface/connectivity under topological constraints.

Attribution: Ramon y Cajal (1906), Tallinen et al. (2014).
We make no claim of novelty.

### 5.2 Kernel-Physics Mapping

Every kernel function maps to a known mathematical result:

| Kernel function | Attribution |
|----------------|-------------|
| vadd, vsub, vscale | Linear algebra |
| vdot, vcross | Grassmann (1844) |
| centroid | Archimedes (~250 BC) |
| projectToSphere | Stereographic projection |
| refineFace | Goldberg (1937) / Caspar-Klug (1962) |
| chi = 2 | Gauss-Bonnet (1827) |
| E/V = 3/2 | Euler (1736) |
| P = 12 | Euler (1758) |
| sphereToMobius | Mobius (1858) |

---

## 6. The Connection to dim D = 16

The connection between dim D = 16 (algebraic) and P = 12 (geometric)
is currently OBSERVATIONAL, NOT PROVEN. Both are instances of:

"A topological invariant constrains a combinatorial count
to a fixed value regardless of the scale or complexity."

Whether this analogy is deep or superficial is an open question.
We record it without resolution.

---

## 7. Implementation Timeline (Git-verified)

| Date | Event |
|------|-------|
| Jan 2026 | First Goldberg explorer (Three.js) |
| Jan-May 2026 | Progressive kernel extraction (v3.2 to v5.1) |
| May 22, 2026 | goldberg_kernel.js standalone |
| May 25, 2026 | Genesis v1.5: 7-primitive automaton |
| May 25, 2026 | Genesis v1.6: E/V convergence discovered |
| May 26, 2026 | v7.1-v7.5.1: 3D explorer, bug fixes, dodecahedron seed |
| May 26, 2026 | Kernel dependency scan: 0 browser dependencies |
| May 26, 2026 | Kernel-to-physics mapping completed |

---

## 8. What This Is and What This Is Not

### This IS:
- A chronological record of computational experiments
- Reproducible, version-controlled code
- Observations connecting known mathematical results
- A demonstration that minimal code produces known physics

### This is NOT:
- A claim of discovery (all theorems attributed)
- A unification of physics (the analogy is unproven)
- A replacement for rigorous proof
- A theory of everything

---

## 9. Open Questions

1. Is there a formal functor mapping dim D = 16 to P = 12?
2. Is {P1,...,P7} a complete basis for finite graph construction?
3. Is there a variational principle whose minimizer is Goldberg refinement?
4. Can the Mobius deformation (chi: 2 to 0) lift to spectral triples?
5. What is the significance of the 7x growth rate?

These questions are recorded without answers.

---

## References

- Euler, L. (1758). Elementa doctrinae solidorum.
- Gauss, C.F. (1827). Disquisitiones generales circa superficies curvas.
- Goldberg, M. (1937). A class of multi-symmetric polyhedra.
- Caspar & Klug (1962). Physical principles in regular viruses.
- Connes, A. (1994). Noncommutative Geometry.
- Chamseddine & Connes (1997). The spectral action principle.
- Ramon y Cajal (1906). Nobel Lecture.
- Tallinen et al. (2014). Gyrification from constrained cortical expansion.
- van Suijlekom (2015). NCG and Particle Physics.

---

All code: github.com/vsavytsk1
All results: reproducible
All claims: attributed
All questions: open

Buenos Aires, May 26, 2026
"There is no shame in not understanding reality yet."
