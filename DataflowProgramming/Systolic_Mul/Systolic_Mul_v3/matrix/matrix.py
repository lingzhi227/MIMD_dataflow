#!/usr/bin/env cs_python

import numpy as np

Pr = 2
Pc = 2
Cycle = 2

Mt = 5
Kt = 5
Nt = 5

M = Pr * Mt
K = Cycle * Kt
N = Pc * Nt

# Generate matrices A and B with specified integer sequences
A = np.arange(1, M * K + 1, dtype=np.float32).reshape(M, K)
B = np.arange(1, K * N + 1, dtype=np.float32).reshape(K, N)

print("A matrix: \n")
print(A)

print("B matrix: \n")
print(B)

# Construct block matrix A_blocks[i,k], and block matrices B_blocks[k,j] in row major
A1 = A.reshape(Pr, Mt, Cycle, Kt)  # Reshape A into (Pr, Mt, Cycle, Kt)
A2 = A1.transpose(2, 0, 1, 3)  # Transpose to bring Cycle to the first axis: (Cycle, Pr, Mt, Kt)
A_groups = A2.reshape(Cycle, Pr, Mt * Kt)  # Reshape into (Cycle, Pr, Mt*Kt) to get block matrices in row-major order

B1 = B.reshape(Cycle, Kt, Pc, Nt)  # Reshape B into (Cycle, Kt, Pc, Nt)
B2 = B1.transpose(0, 2, 1, 3)  # Transpose to bring Pc to the second axis: (Cycle, Pc, Kt, Nt)
B_groups = B2.reshape(Cycle, Pc, Kt * Nt)  # Reshape into (Cycle, Pc, Kt*Nt) to get block matrices in row-major order

# Ravel each group of A and B
raveled_A_groups = [A_groups[i].ravel() for i in range(Cycle)]
raveled_B_groups = [B_groups[i].ravel() for i in range(Cycle)]

# Print raveled A_groups and B_groups to confirm correctness
for i, raveled_A in enumerate(raveled_A_groups):
    print(f"raveled_A_group[{i}]: \n{raveled_A}\n")

for i, raveled_B in enumerate(raveled_B_groups):
    print(f"raveled_B_group[{i}]: \n{raveled_B}\n")
