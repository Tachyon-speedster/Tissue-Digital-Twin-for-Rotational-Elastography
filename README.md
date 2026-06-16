# Tissue Digital Twin: Translational vs Rotational Stiffness Estimation

## Overview

This project investigates whether rotational deformation measurements can improve local tissue property estimation compared to translational measurements alone.

A spring-network tissue digital twin was developed to simulate force propagation through anisotropic soft tissues and estimate local stiffness properties from measured deformations.

**Research Question**

> Can rotational deformation measurements improve local tissue property estimation compared to translational measurements alone?

---

## Motivation

Many tissue characterization methods rely solely on translational displacement measurements.

This project explores whether adding rotational deformation information can provide additional constraints for estimating local mechanical properties, particularly in anisotropic tissues such as muscle and tendon.

---

## System Architecture

### Mode A — Translational Estimation

Uses displacement measurements only:

F = Ku

where:

- F = applied force
- u = measured displacement
- K = local stiffness

A least-squares estimator recovers directional stiffness values from multiple loading directions.

---

### Mode B — Translational + Rotational Estimation

Uses:

- Translational displacement
- Local rotational response

The hypothesis was that rotational measurements contain additional information about local anisotropy and could improve stiffness estimation accuracy.

---

## Key Features

- Spring-network tissue simulator
- Anisotropic tissue modeling
- Multi-direction force loading
- Least-squares stiffness estimation
- Rotational deformation analysis
- Uncertainty estimation
- Comparative Mode A vs Mode B evaluation
- Noise robustness experiments

---

## Tissue Models

The simulator includes:

### Fat
- Near isotropic behavior

### Muscle
- Moderately anisotropic
- Fiber orientation effects

### Tendon
- Strong anisotropy
- High directional stiffness ratio

---

## Major Debugging and Research Findings

### Bug 1: False Isotropic Recovery

Original implementation initialized displacement using scalar stiffness:

```python
disp = force / k
```

This caused:

```text
kx_est = ky_est = k_scalar
```

for all measurements.

#### Fix

Component-wise directional stiffness:

```python
dx = Fx / k(0°)
dy = Fy / k(90°)
```

Result:

- Correct directional recovery
- ky estimates no longer collapse to kx

---

### Bug 2: Single-Load Anisotropy Estimation

A single force direction cannot recover two unknown stiffness components.

#### Fix

Measurement accumulation across:

- 7 force directions
- 6 force magnitudes

Total:

```text
42 measurements per cell
```

followed by least-squares estimation.

---

### Bug 3: Rotational Model Calibration

The original rotational coupling model:

```text
θ ≈ C · (Δk/kmean) · sin(2φ)
```

was investigated and calibrated.

Result:

```text
C = 0.196 ± 0.243
```

Coefficient of variation > 100%.

Conclusion:

The rotational response depends strongly on local topology and boundary conditions and cannot be accurately described by a single global coupling constant.

---

## Experimental Setup

### Loading Directions

```text
0°
30°
45°
60°
90°
120°
150°
```

### Force Magnitudes

```text
1 N
2 N
4 N
6 N
8 N
10 N
```

### Measurements

```text
42 measurements per cell
18 representative cells
3 tissue classes
```

---

## Results

### Mode A Performance

| Tissue | Relative Error |
|----------|----------|
| Fat | < 0.5% |
| Tendon | 2–6% |
| Muscle | 26–42% |

### Mode B Performance

No statistically significant improvement over Mode A.

| Noise Level | Improvement |
|------------|------------|
| 1% | 0.0% |
| 3% | 0.0% |
| 5% | 0.0% |
| 10% | 0.0% |

Paired t-test:

```text
p = 0.36
```

Not significant.

---

## Conclusions

### Supported Findings

✓ Translational least-squares estimation successfully recovers directional stiffness after correction of the isotropic-displacement bug.

✓ Strongly anisotropic tendon tissue can be estimated with low error.

✓ Rotational deformation contains information related to anisotropy.

### Current Limitation

The analytical rotational inversion model is not sufficiently accurate for this spring-network geometry.

As implemented, rotational measurements do not improve estimation accuracy.

Therefore:

> The hypothesis that Mode B outperforms Mode A is currently not supported by the present model.

---

## Future Work

Potential next steps include:

- Full stiffness tensor estimation
- Fiber-angle recovery
- Graph Neural Networks (GNNs)
- Neural inverse models for rotational fields
- Gaussian Process regression
- FEM-based validation
- Experimental tissue phantom studies

---

## Disclaimer

This is a research prototype.

- Not clinically validated
- Uses a spring-network model rather than finite element analysis (FEM)
- Intended for exploratory research and algorithm development

Do not use for clinical decision making.

---

## Author

Independent research project exploring digital-twin-based tissue characterization and rotational deformation sensing.
