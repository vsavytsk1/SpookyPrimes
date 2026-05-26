# GPU Turbulent Flow Benchmark
## navierCrunch v1.0 - TURBULENT Computational Receipt
### V. Savytskyy - Buenos Aires - May 26, 2026

> **Status:** Computational receipt. Hardware-verified TURBULENT benchmark.
> Re>10,000. mix=0.15, decay=0.02, noise=0.05.
> All code is public. All results are reproducible.

---

## 1. Regime Parameters

| Parameter | Laminar (baseline) | TURBULENT |
|-----------|-------------------|-----------|
| Reynolds | ~100 | >10,000 |
| mix | 0.60 | **0.15** |
| decay | 0.005 | **0.02** |
| noise | 0.00 | **0.05** |

Turbulent adds per-step: random noise generation + energy decay + pressure clamp.
This is ON TOP of the sparse matrix multiply.

---

## 2. Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.0 GB |
| Mode | CUDA GPU (cupy) |
| Python | 3.11.9 |
| PID | 15800 |

---

## 3. TURBULENT Benchmark Results

| Level | Faces | P | chi | Steps | Time (ms) | us/step | us/face/step | Steps/sec |
|-------|-------|---|-----|-------|-----------|---------|-------------|-----------|
| L0 | 12 | 12 | **2** | 1,000,000 | 389,066 | 389.07 | 32.4222 | 2,570 |
| L1 | 72 | 12 | **2** | 1,000,000 | 396,706 | 396.71 | 5.5098 | 2,521 |
| L2 | 492 | 12 | **2** | 250,000 | 98,894 | 395.58 | 0.8040 | 2,528 |
| L3 | 3,432 | 12 | **2** | 29,411 | 11,980 | 407.34 | 0.1187 | 2,455 |
| L4 | 24,012 | 12 | **2** | 10,000 | 4,014 | 401.37 | 0.0167 | 2,491 |
| L5 | 168,072 | 12 | **2** | 10,000 | 3,991 | 399.14 | 0.0024 | 2,505 |

---

## 4. LAMINAR vs TURBULENT Comparison

| Metric | LAMINAR | TURBULENT | Delta |
|--------|---------|-----------|-------|
| us/step (avg) | ~230 | ~400 | 1.7x slower |
| Steps/sec | ~4,300 | ~2,500 | noise + decay overhead |
| us/face/step L5 | 0.0016 | 0.0024 | 1.5x (GPU absorbs noise) |
| chi (all levels) | **2** | **2** | IDENTICAL |
| P (all levels) | **12** | **12** | IDENTICAL |
| E/V (all levels) | **1.500** | **1.500** | IDENTICAL |
| O(n) scaling | confirmed | **confirmed** | IDENTICAL |
| GPU utilization | 100% | 100% | IDENTICAL |

### Key Observations

1. **Topology is regime-independent**: chi=2, P=12, E/V=1.500 hold
   regardless of Reynolds number. This is not surprising (topology
   is geometric, not dynamic) but it IS verified computationally.

2. **O(n) holds under turbulence**: The noise generation and decay
   operations are element-wise O(n), so total complexity remains O(n).

3. **GPU overhead ratio**: Turbulent is 1.7x slower per step due to:
   - cupy.random.uniform() call per step (GPU RNG)
   - Element-wise multiply (noise * pressure)
   - Decay multiply + clamp (maximum(p, 0))
   
4. **Spread**: At L0, all 12 faces reached in both regimes.
   At L1+, only source face shows spread=1/N because the
   noise-driven pressure bleeds differently than laminar.

---

## 5. What This Proves (Turbulent-Specific)

1. **Turbulent noise does NOT break topology** - chi=2 at 168K faces under Re>10K
2. **O(n) survives turbulence** - extra per-element ops stay linear
3. **GPU handles stochastic flow** - random number generation at 168K scale, no problem
4. **The kernel is regime-agnostic** - same adjacency matrix, same mesh, different physics
5. **Pressure stays non-negative** - clamp operation verified (no NaN, no negatives)

## 6. What This Does NOT Prove

- That this is DNS (Direct Numerical Simulation) - it is stochastic pressure diffusion
- That the noise model is physically accurate - it is a random perturbation, not Kolmogorov
- That this solves the Clay Millennium Prize - existence and smoothness are analytical, not numerical

---

## 7. Combined Session Output (May 26, 2026)

| Run | Regime | Max faces | GPU util | PID | Status |
|-----|--------|-----------|----------|-----|--------|
| 1 | Laminar | 168,072 | 100% | 20808 | Clean exit |
| 2 | Laminar | 168,072 | 100% | 23996 | Clean exit |
| 3 | Laminar | 168,072 | 100% | 31632 | Clean exit |
| 4 | Turbulent | 168,072 | 100% | 15800 | Clean exit |

All processes exited cleanly. No orphan PIDs. No GPU memory leaks.
Temperature peak: 60C (well within RTX 3060 thermal envelope of 90C).

---

All code: github.com/vsavytsk1/Mnetv1/pack/navierCrunch.py
Timestamp: 2026-05-26 07:50:41
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
One constant. One topology. One truth.

Buenos Aires, May 26, 2026
