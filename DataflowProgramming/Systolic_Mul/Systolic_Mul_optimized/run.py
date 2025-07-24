#!/usr/bin/env cs_python

"""
Optimized Systolic Array Runner
Demonstrates proper usage of the streaming systolic array implementation
"""

import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

def run_systolic_array(args):
    """Run the optimized systolic array implementation"""
    
    # Parse dimensions
    M, K, N = args.M, args.K, args.N
    Mt, Kt, Nt = args.Mt, args.Kt, args.Nt
    
    # Calculate PE grid dimensions
    Pr = M // Mt  # Number of PE rows
    Pc = N // Nt  # Number of PE columns  
    Cycle = K // Kt  # Number of systolic beats
    
    print(f"Matrix dimensions: M={M}, K={K}, N={N}")
    print(f"Block dimensions: Mt={Mt}, Kt={Kt}, Nt={Nt}")
    print(f"PE grid: {Pr}x{Pc}, Cycles: {Cycle}")
    
    # Validate dimensions
    assert M % Mt == 0, f"M ({M}) must be divisible by Mt ({Mt})"
    assert K % Kt == 0, f"K ({K}) must be divisible by Kt ({Kt})"
    assert N % Nt == 0, f"N ({N}) must be divisible by Nt ({Nt})"
    assert Pc + 2 <= 750, f"PE grid width {Pc + 2} exceeds WSE limit"
    assert Pr + 1 <= 994, f"PE grid height {Pr + 1} exceeds WSE limit"
    
    # Generate test matrices
    print("Generating test matrices...")
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_expected = A @ B
    
    # Initialize SDK runtime
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    
    # Load compiled program
    runner.load()
    
    # Start runtime
    runner.run()
    
    # Initialize all PEs
    print("Initializing PEs...")
    runner.launch("init", nonblock=False)
    
    # Run systolic computation - test without actual computation
    print("Testing basic functionality...")
    
    # Skip computation for now - just test if we can read initialized data
    print("Skipping computation, reading initialized results...")
    
    # Collect results from compute PEs (using working pattern)
    print("Collecting results...")
    
    # Get symbol IDs
    C_symbol = runner.get_id('C')
    time_symbol = runner.get_id('time_buf_f32')
    
    # Copy result back to host (like the working examples)
    C_temp = np.zeros(Pc * Pr * Mt * Nt, np.float32)
    runner.memcpy_d2h(C_temp, C_symbol, 1, 1, Pc, Pr, Mt * Nt, streaming=False,
                     data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    
    # Get timing information
    timing_data = np.zeros(3, np.float32)
    runner.memcpy_d2h(timing_data, time_symbol, 1, 1, 1, 1, 3, streaming=False,
                     data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    
    # Reshape result matrix
    C_result_temp = C_temp.reshape(Pr, Pc, Mt, Nt)
    C_result = np.zeros((M, N), dtype=np.float32)
    
    for pr in range(Pr):
        for pc in range(Pc):
            m_start = pr * Mt
            m_end = m_start + Mt
            n_start = pc * Nt  
            n_end = n_start + Nt
            C_result[m_start:m_end, n_start:n_end] = C_result_temp[pr, pc]
    
    print(C_result)
    print(C_expected)

    # Verify correctness
    print("Verifying results...")
    max_error = np.max(np.abs(C_result - C_expected))
    rel_error = max_error / np.max(np.abs(C_expected))
    
    print(f"Maximum absolute error: {max_error:.2e}")
    print(f"Maximum relative error: {rel_error:.2e}")
    
    if rel_error < 1e-4:
        print("✓ Results are correct!")
    else:
        print("✗ Results do not match expected values")
    
    # Calculate performance metrics
    total_ops = 2 * M * K * N  # GEMM FLOPs
    
    # Extract timing (implementation-specific)
    start_time = timing_data[0]
    end_time = timing_data[2] + (timing_data[1] << 16)
    elapsed_cycles = end_time - start_time
    
    # Assume 850 MHz clock (implementation-specific)
    elapsed_time_us = elapsed_cycles / 850.0
    gflops = (total_ops / 1e9) / (elapsed_time_us / 1e6)
    
    print(f"Elapsed time: {elapsed_time_us:.2f} μs")
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    runner.stop()
    
    return {
        'elapsed_time_us': elapsed_time_us,
        'gflops': gflops,
        'max_error': max_error,
        'rel_error': rel_error
    }

def main():
    parser = argparse.ArgumentParser(description="Optimized Systolic Array Runner")
    
    # Matrix dimensions
    parser.add_argument("--M", type=int, default=128, help="Matrix A rows")
    parser.add_argument("--K", type=int, default=128, help="Matrix A cols / B rows")
    parser.add_argument("--N", type=int, default=128, help="Matrix B cols")
    
    # Block dimensions
    parser.add_argument("--Mt", type=int, default=32, help="Block A rows")
    parser.add_argument("--Kt", type=int, default=32, help="Block A cols / B rows")
    parser.add_argument("--Nt", type=int, default=32, help="Block B cols")
    
    # Runtime parameters
    parser.add_argument("--name", default="out", help="Compile output name")
    parser.add_argument("--cmaddr", help="CM address")
    
    args = parser.parse_args()
    
    results = run_systolic_array(args)
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Time: {results['elapsed_time_us']:.2f} μs")
    print(f"Performance: {results['gflops']:.2f} GFLOPS") 
    print(f"Accuracy: {results['rel_error']:.2e} relative error")
    print("="*50)

if __name__ == "__main__":
    main()