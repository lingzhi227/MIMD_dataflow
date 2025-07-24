#!/usr/bin/env python

import numpy as np

# Given parameters
M = 150
K = 120
N = 160
Mt = 10
Kt = 10
Nt = 10

# Calculate block matrix dimensions
Pr = M // Mt  
Cycle = K // Kt  
Pc = N // Nt  

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

A1 = A.reshape(Pr, Mt, Cycle, Kt)
A2 = A1.transpose(0, 2, 1, 3)
A3 = A2.reshape(Pr, Cycle, Mt*Kt)

# Reshape and transpose to create block matrices
B1 = B.reshape(Cycle, Kt, Pc, Nt)
B2 = B1.transpose(0, 2, 1, 3)
B3 = B2.reshape(Cycle, Pc, Kt * Nt)

# Group A3 blocks by row + column index
A3_groups = {}

for i in range(Pr):
    for j in range(Cycle):
        group_idx = i + j
        if group_idx not in A3_groups:
            A3_groups[group_idx] = []
        A3_groups[group_idx].append(A3[i, j].tolist())

# Sort each group by row index (i) from smallest to largest
for group_idx in A3_groups:
    A3_groups[group_idx] = sorted(A3_groups[group_idx], key=lambda x: x[0])

# Convert A3_groups lists to numpy arrays and ravel them
ravel_A3_groups = {}
for group_idx in A3_groups:
    ravel_A3_groups[group_idx] = np.array(A3_groups[group_idx], dtype=np.float32).ravel()

# Group B3 blocks by row + column index
B3_groups = {}

for i in range(Cycle):
    for j in range(Pc):
        group_idx = i + j
        if group_idx not in B3_groups:
            B3_groups[group_idx] = []
        B3_groups[group_idx].append(B3[i, j])

# Print the number of elements in each B3_groups[i]
print("Number of elements in each A3_groups[i]:")
for group_idx in sorted(A3_groups.keys()):
    num_blocks = len(A3_groups[group_idx])
    num_elements = num_blocks * Mt * Kt
    print(f"Group {group_idx}: {num_elements} elements ({num_blocks} blocks)")


# Print the number of elements in each B3_groups[i]
print("Number of elements in each B3_groups[i]:")
for group_idx in sorted(B3_groups.keys()):
    num_blocks = len(B3_groups[group_idx])
    num_elements = num_blocks * Kt * Nt
    print(f"Group {group_idx}: {num_elements} elements ({num_blocks} blocks)")
