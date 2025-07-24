import numpy as np

# Given parameters
M = 15
K = 12
N = 16
Mt = 3
Kt = 3
Nt = 4

# Calculate block matrix dimensions
Pr = M // Mt  # Pr = 5
Cycle = K // Kt  # Cycle = 4
Pc = N // Nt  # Pc = 4

# Generate matrices A (MxK) and B (KxN)
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# Calculate expected C
C_expected = np.dot(A, B)

# Construct block matrix A_blocks[i,k], and block matrices B_blocks[k,j] both in row major
A1 = A.reshape(Pr, Mt, Cycle, Kt)
A2 = A1.transpose(0, 2, 1, 3)
A3 = A2.reshape(Pr, Cycle, Mt * Kt)

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
        B3_groups[group_idx].append(B3[i, j].tolist())

# Sort each group by column index (j) from smallest to largest
for group_idx in B3_groups:
    B3_groups[group_idx] = sorted(B3_groups[group_idx], key=lambda x: x[0])

# Convert B3_groups lists to numpy arrays and ravel them
ravel_B3_groups = {}
for group_idx in B3_groups:
    ravel_B3_groups[group_idx] = np.array(B3_groups[group_idx], dtype=np.float32).ravel()

# Print all matrices
print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nExpected Matrix C (A dot B):")
print(C_expected)

print("\nA3_groups (Grouped A Blocks):")
for group_idx, group in A3_groups.items():
    print(f"Group {group_idx}:")
    print(np.array(group))

print("\nRaveled A3_groups:")
for group_idx, ravel_group in ravel_A3_groups.items():
    print(f"Group {group_idx}:")
    print(ravel_group)

print("\nB3_groups (Grouped B Blocks):")
for group_idx, group in B3_groups.items():
    print(f"Group {group_idx}:")
    print(np.array(group))

print("\nRaveled B3_groups:")
for group_idx, ravel_group in ravel_B3_groups.items():
    print(f"Group {group_idx}:")
    print(ravel_group)
