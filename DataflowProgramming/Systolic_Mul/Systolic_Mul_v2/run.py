#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
import sys
import struct

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
M = int(compile_data['params']['M'])
K = int(compile_data['params']['K'])
N = int(compile_data['params']['N'])

# block matrix dimensions
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Size of N dimension on each PE
# block matrix dimensions
Pr = int(compile_data['params']['Pr'])
Cycle = int(compile_data['params']['Cycle'])
Pc = int(compile_data['params']['Pc'])

# Check if requirements are met
if not (Pc + 1 <= 750 and Pr + 1 <= 994):
    sys.stderr.write("Error: Conditions Pc + 2 <= 750 and Pr + 1 <= 994 must be met.\n")
    sys.exit(1)

# Colors used for memcpy streaming
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])
MEMCPYH2D_DATA_2 = int(compile_data['params']['MEMCPYH2D_DATA_2_ID'])


# # Generate matrices A and B with specified integer sequences
# A = np.arange(1, M * K + 1, dtype=np.float32).reshape(M, K)
# B = np.arange(1, K * N + 1, dtype=np.float32).reshape(K, N)

# # Generate matrices A and B with specified integer sequences in descending order
# A = np.arange(M * K, 0, -1, dtype=np.float32).reshape(M, K)
# B = np.arange(K * N, 0, -1, dtype=np.float32).reshape(K, N)

# Generate matrices A and B with random values, rounded to 4 decimal places
A = np.round(np.random.rand(M, K).astype(np.float32), 4)
B = np.round(np.random.rand(K, N).astype(np.float32), 4)

# print("A matrix: \n")
# print(A)

# print("B matrix: \n")
# print(B)

# Calculate expected C
C_expected = np.matmul(A, B)
# C_expected = np.dot(A, B)

# Construct block matrix A_blocks[i,k], and block matrices B_blocks[k,j]
# both in row major
A1 = A.reshape(Pr, Mt, Cycle, Kt)
A2 = A1.transpose(0, 2, 1, 3)
A3 = A2.reshape(Pr, Cycle, Mt*Kt)

B1 = B.reshape(Cycle, Kt, Pc, Nt)
B2 = B1.transpose(0, 2, 1, 3)
B3 = B2.reshape(Cycle, Pc, Kt*Nt)

# Group A3 blocks by row + column index
A3_groups = {}

for i in range(Pr):
    for j in range(Cycle):
        group_idx = i + j
        if group_idx not in A3_groups:
            A3_groups[group_idx] = []
        A3_groups[group_idx].append(A3[i, j].tolist())

# Sort each group by row index (i) from smallest to largest
# for group_idx in A3_groups:
#     A3_groups[group_idx] = sorted(A3_groups[group_idx], key=lambda x: x[0])
# for group_idx in A3_groups:
#     A3_groups[group_idx] = sorted(A3_groups[group_idx], key=lambda block: block[0])
# for group_idx in A3_groups:
#     A3_groups[group_idx] = [item[1] for item in sorted(A3_groups[group_idx], key=lambda x: x[0])]

# Convert A3_groups lists to numpy arrays and ravel them
ravel_A3_groups = {}
for group_idx in A3_groups:
    ravel_A3_groups[group_idx] = np.array(A3_groups[group_idx], dtype=np.float32).ravel()

# # Print ravel_A3_groups and ravel_B3_groups
# print("\nravel_A3_groups:")
# for group_idx, group in ravel_A3_groups.items():
#     print(f"Group {group_idx}: {group}")

# Group B3 blocks by row + column index
B3_groups = {}

for i in range(Cycle):
    for j in range(Pc):
        group_idx = i + j
        if group_idx not in B3_groups:
            B3_groups[group_idx] = []
        B3_groups[group_idx].append((j, B3[i, j].tolist()))

# Sort B3_groups by column index for each group (using j as the key)
for group_idx in B3_groups:
    B3_groups[group_idx] = [item[1] for item in sorted(B3_groups[group_idx], key=lambda x: x[0])]

# Convert B3_groups lists to numpy arrays and ravel them
ravel_B3_groups = {}
for group_idx in B3_groups:
    ravel_B3_groups[group_idx] = np.array(B3_groups[group_idx], dtype=np.float32).ravel()

# print("\nravel_B3_groups:")
# for group_idx, group in ravel_B3_groups.items():
#     print(f"Group {group_idx}: {group}")

# Construct a runner using SdkRuntime
# suppress_simfab_trace=True will stop recording for GUI
# simfab_numthreads=16 will use 16 CPU cores
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True, simfab_numthreads=16)

# Load and run the program
runner.load()
runner.run()

C_symbol = runner.get_id('C')
timestamps_symbol = runner.get_id('time_buf_f32')

# Stream matrix A and B to device
for beat in range(max(Pr + Cycle - 1, Pc + Cycle - 1)):
    if beat < Cycle:
        # print(f"Beat {beat}: ")
        start = 1
        w = beat + 1 if beat < min(Cycle, Pc) else min(Cycle, Pc) if beat <= max(Cycle, Pc) - 1 else Cycle + Pc - 1 - beat
        # Stream B to north halos
        # print(f"Size of B3 group data = {ravel_B3_groups[beat].size}")
        runner.memcpy_h2d(MEMCPYH2D_DATA_2, ravel_B3_groups[beat], start, 0, w, 1, Kt * Nt, streaming=True,
                          data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=True)
        h = beat + 1 if beat < min(Cycle, Pr) else min(Cycle, Pr) if beat <= max(Cycle, Pr) - 1 else Cycle + Pr - 1 - beat
        # Stream A to west halo
        # print(f"Size of A3 group data = {ravel_A3_groups[beat].size}")
        runner.memcpy_h2d(MEMCPYH2D_DATA_1, ravel_A3_groups[beat], 0, start, 1, h, Mt * Kt, streaming=True,
                          data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    else:
        # print(f"Beat {beat}: ")
        if beat < Pc + Cycle - 1:
            start = beat - Cycle + 2
            w = beat + 1 if beat < min(Cycle, Pc) else min(Cycle, Pc) if beat <= max(Cycle, Pc) - 1 else Cycle + Pc - 1 - beat
            # Stream B to north halo
            # print(f"Size of B3 group data = {ravel_B3_groups[beat].size}")
            runner.memcpy_h2d(MEMCPYH2D_DATA_2, ravel_B3_groups[beat], start, 0, w, 1, Kt * Nt, streaming=True,
                              data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=True)
        if beat < Pr + Cycle - 1:
            start = beat - Cycle + 2
            h = beat + 1 if beat < min(Cycle, Pr) else min(Cycle, Pr) if beat <= max(Cycle, Pr) - 1 else Cycle + Pr - 1 - beat
            # Stream A to west halo
            # print(f"Size of A3 group data = {ravel_A3_groups[beat].size}")
            runner.memcpy_h2d(MEMCPYH2D_DATA_1, ravel_A3_groups[beat], 0, start, 1, h, Mt * Kt, streaming=True,
                              data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

# Barrier
runner.launch('rpc_sync', nonblock=False)

print(f"RPC_SYNC DONE")

# memcpy result C[i,j] back to host
C_temp = np.zeros(Pc * Pr * Mt * Nt, np.float32)
runner.memcpy_d2h(C_temp, C_symbol, 1, 1, Pc, Pr, Mt * Nt, streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print(f"C transfer done")

timestamps = np.zeros(Pc * Pr * 3, np.float32)
runner.memcpy_d2h(timestamps, timestamps_symbol, 1, 1, Pc, Pr, 3, streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print(f"Timestamps transfer done")

# reset kernel PE C_dsd to zero
runner.launch('init', nonblock=False)

C3 = C_temp.reshape((Pr, Pc, Mt, Nt))
# C2 is of the form (h, Mt, w, Nt)
C2 = C3.transpose(0, 2, 1, 3)
# C1 is of the form (M, N)
C = C2.reshape(M, N)

runner.stop()

print(C)
print(C_expected)

# compare result
# np.testing.assert_allclose(C_expected, C, rtol=1e-02, atol=1e-03)
np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

time_start = np.zeros((Pr, Pc)).astype(int)
time_end = np.zeros((Pr, Pc)).astype(int)
word = np.zeros(3).astype(np.uint16)

timestamps = timestamps.reshape(Pr, Pc, 3);

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

for w in range(Pc):
  for h in range(Pr):
    hex_t0 = int(float_to_hex(timestamps[(h, w, 0)]), base=16)
    hex_t1 = int(float_to_hex(timestamps[(h, w, 1)]), base=16)
    hex_t2 = int(float_to_hex(timestamps[(h, w, 2)]), base=16)
    word[0] = hex_t0 & 0x0000ffff 
    word[1] = (hex_t0 >> 16) & 0x0000ffff
    word[2] = hex_t1 & 0x0000ffff
    time_start[(h, w)] = make_u48(word)
    word[0] = (hex_t1 >> 16) & 0x0000ffff
    word[1] = hex_t2 & 0x0000ffff
    word[2] = (hex_t2 >> 16) & 0x0000ffff
    time_end[(h, w)] = make_u48(word)


timings = time_end - time_start

cycles_send = timings.max()
time_mm = (cycles_send / 0.85) *1.e-3
print(time_mm)

# feedback report
print("SUCCESS!")