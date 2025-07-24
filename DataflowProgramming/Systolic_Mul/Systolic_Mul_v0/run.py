#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Full matrix dimensions
# A is M x K, B is K x N, C is M x N
M = int(compile_data['params']['M'])
K = int(compile_data['params']['K'])
N = int(compile_data['params']['N'])

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)



# Mt number of rows of block matrix Aij
# Kt number of colums of block matrix Aij, number of rows of block matrix Bij
# Nt number of colums of block matrix Bijf
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])


# Kernel rectangle and per-PE matrix dimensions
# Pr number of rows of Kernel PE
# Pc number of columns of Kernel PE
# M mod Mt = 0
# K mod Kt = 0
# N mod Nt = 0

def check_divisibility(value, divisor, name):
    if value % divisor != 0:
        raise ValueError(f"{name} ({value}) must be divisible by {name}t ({divisor})")
    return value // divisor

def validate_inputs(M, Mt, K, Kt, N, Nt):
    try:
        M_result = check_divisibility(M, Mt, "M")
        K_result = check_divisibility(K, Kt, "K")
        N_result = check_divisibility(N, Nt, "N")
        return M_result, K_result, N_result
    except ValueError as e:
        print(f"Error: {e}")
        return None

try:
    results = validate_inputs(M, Mt, K, Kt, N, Nt)
    
    if results:
        Pr, Pk, Pc = results
        print(f"All inputs are valid.")
        print(f"M / Mt = {Pr}")
        print(f"K / Kt = {Pk}")
        print(f"N / Nt = {Pc}")
    else:
        print("Input validation failed. Please check your inputs and try again.")

except ValueError:
    print("Error: Please enter valid integers for all inputs.")


Pr = M / Mt # number of row PEs in the core rectangle
Pk = K / Kt
Pc = N / Nt # number of columns PEs in the core rectangle


# How to transform a 2-D tensor into a cliff distribution with
# column-major local tensor
#
# Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
# A = |  0  1  2  3 |
#     |  4  5  6  7 |
#     |  8  9 10 11 |
#     | 12 13 14 15 |
# A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
# A1 = | | 0  1|  | 4  5| |
#      | | 2  3|, | 6  7| |
#      |                  |
#      | | 8  9|  |12 13| |
#      | |10 11|, |14 15| |
# A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
# so the local tensor lh-by-lw is col-major
# A2 = | | 0  4|  | 2  6| |
#      | | 1  5|, | 3  7| |
#      |                  |
#      | | 8 12|  |10 14| |
#      | | 9 13|, |11 15| |
# A3 = A2.reshape(2,2,4)
# A3 = |  0  4  1  5 |
#      |  2  6  3  7 |
#      |  8 12  9 13 |
#      | 10 14 11 15 |
# A3 is h-w-l
# split matrix A and B into blocked matrices
A1 = A.reshape(Pr, Mt, Pc, Kt)
A2 = A1.transpose(0, 2, 3, 1)
A3 = A2.reshape(Pr, Pc, Mt*Kt)

B1 = B.reshape(Pr, Kt, Pc, Nt)
B2 = B1.transpose(0, 2, 3, 1)
B3 = B2.reshape(Pr, Pc, Kt*Nt)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

runner.load()
runner.run()

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

print("step 1: streaming block matrix A H2D to West Halo FIFO")
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 1, 1, Pr, Mt*Kt, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)


print("step 2: streaming block matrix B H2D to North Halo FIFO")
runner.memcpy_h2d(sym_B, B3.ravel(), 1, 0, Pc, 1, Kt*Nt, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)


print("step 3: kernel runs")
runner.launch("main", nonblock=False)
print("step 4: streaming block matrix C D2H from East Halo FIFO")


C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# C3 is h-by-w-l or
# C3 is of the form (h, w, Nt, Mt) where local tensor Mt-by-Nt is column-major
C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
# C2 is of the form (h, Mt, w, Nt)
C2 = C3.transpose(0, 3, 1, 2)
# C1 is of the form (M, N)
C1 = C2.reshape(M, N)
# C has the correct data type
C = C1.view(np.float32)

runner.stop()

# Check the result
C_expected = np.dot(A, B)

# absolute(a - b) <= (atol + rtol * absolute(b))
np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

print("SUCCESS")
