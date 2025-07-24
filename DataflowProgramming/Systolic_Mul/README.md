# Systolic Array for Blocked Matrix Multiplication in CSL

This program implements a systolic array for blocked matrix multiplication in CSL, designed to compute **C = A * B** for large matrices `A` and `B`.

---

## Problem Statement

We aim to compute:
**C = A × B**

### Given Matrices
- **Matrix A dimensions**:  
  - `M`: rows  
  - `K`: columns  
  \( A_{M \times K} \)

- **Matrix B dimensions**:  
  - `K`: rows  
  - `N`: columns  
  \( B_{K \times N} \)

- **Matrix C dimensions**:  
  \( C_{M \times N} \)

---

## Blocked Matrix Dimensions
To handle large matrices, we divide them into smaller **block matrices**:

- **Block Matrix A dimensions**:  
  - `Mt`: block matrix rows  
  - `Kt`: block matrix columns  

- **Block Matrix B dimensions**:  
  - `Kt`: block matrix rows  
  - `Nt`: block matrix columns  

### Requirements
- \( M \mod Mt = 0 \)  
- \( K \mod Kt = 0 \)  
- \( N \mod Nt = 0 \)

### Derived Parameters
- \( P_r = M / Mt \) (Number of block rows in A)  
- \( P_c = N / Nt \) (Number of block columns in B)  
- \( \text{beat} = K / Kt \) (Number of steps for systolic execution)

---

## Implementation Stages
1. **H2D Wavelet Stream**:  
   - West artificial halo  
   - North artificial halo  

2. **Systolic Execution**

3. **D2H Wavelet Stream**:  
   - East artificial halo  

---

## Hardware Grid Details
### Grid Dimensions
- **Physical Grid**:  
  - 757 columns  
  - 996 rows  

- **ROI Resources**:  
  - 750 columns  
  - 994 rows  

### Artificial Halos
- **West Halo**:  
  - 1 column  
  - Start point: \( P(0, 1) \)  
  - Height: \( P_r \)  
  - Width: 1  

- **North Halo**:  
  - 1 row  
  - Start point: \( P(1, 0) \)  
  - Height: 1  
  - Width: \( P_c \)  

- **East Halo**:  
  - 1 column  
  - Start point: \( P(P_c+1, 1) \)  
  - Height: \( P_r \)  
  - Width: \( P_c \)  

---

## PE Grid Details
- **Compute PE Grid**:  
  - Number of rows: \( P_r \)  
  - Number of columns: \( P_c \)  
  - Start point: \( P(1, 1) \)  
  - Height: \( P_r \)  
  - Width: \( P_c \)  

---

## ROI Grid
The ROI (Region of Interest) grid is laid out as:
```text
layout(P_c + 2, P_r + 1)
```

### Constraints
- \( P_c + 2 \leq 750 \)  
- \( P_r + 1 \leq 994 \)  

---

## Parameter Settings
- `M`: Total rows in matrix A  
- `K`: Total columns in matrix A / rows in matrix B  
- `N`: Total columns in matrix B  
- `Mt`: Block rows in matrix A  
- `Kt`: Block columns in matrix A / rows in matrix B  
- `Nt`: Block columns in matrix B  
- \( P_r = M / Mt \): Number of block rows  
- \( P_c = N / Nt \): Number of block columns  
- \( \text{beat} = K / Kt \): Number of systolic beats  

---

## Example
Given:
- \( M = 1200 \), \( K = 800 \), \( N = 1600 \)  
- \( Mt = 150 \), \( Kt = 200 \), \( Nt = 400 \)  

Derived parameters:
- \( P_r = M / Mt = 8 \)  
- \( P_c = N / Nt = 4 \)  
- \( \text{beat} = K / Kt = 4 \)

Ensure:
- \( P_c + 2 \leq 750 \)  
- \( P_r + 1 \leq 994 \)  

---

## Authors
- [Lingzhi Yang](https://github.com/lingzhi227)

---

This structured format is ready for GitHub and easy to read, with clear sections for problem description, requirements, and implementation details.
