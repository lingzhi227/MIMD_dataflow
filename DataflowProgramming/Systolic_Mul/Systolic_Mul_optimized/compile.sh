#!/bin/bash

# Optimized Systolic Array Compilation Script

# Default parameters
M=${M:-128}
K=${K:-128}  
N=${N:-128}
Mt=${Mt:-32}
Kt=${Kt:-32}
Nt=${Nt:-32}

# Calculate derived parameters
Pr=$((M / Mt))
Pc=$((N / Nt))
Cycle=$((K / Kt))

echo "Compiling Optimized Systolic Array"
echo "Matrix dimensions: M=${M}, K=${K}, N=${N}"
echo "Block dimensions: Mt=${Mt}, Kt=${Kt}, Nt=${Nt}"
echo "PE grid: ${Pr}x${Pc}, Cycles: ${Cycle}"

# Validate constraints
if [ $((Pc + 2)) -gt 750 ]; then
    echo "Error: PE grid width $(($Pc + 2)) exceeds WSE limit of 750"
    exit 1
fi

if [ $((Pr + 1)) -gt 994 ]; then
    echo "Error: PE grid height $(($Pr + 1)) exceeds WSE limit of 994"
    exit 1
fi

# Compile with cslc (using smaller fabric for simulation)
cslc layout.csl \
    --fabric-dims=50,50 \
    --fabric-offsets=4,1 \
    --params=M:${M},K:${K},N:${N} \
    --params=Mt:${Mt},Kt:${Kt},Nt:${Nt} \
    --params=Pc:${Pc},Pr:${Pr},Cycle:${Cycle} \
    --params=MEMCPYH2D_DATA_1_ID:0 \
    --params=MEMCPYH2D_DATA_2_ID:1 \
    -o out \
    --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

# Check for compilation success
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running test with run.py..."

    # Run the test
    cs_python run.py --name=out --M=${M} --K=${K} --N=${N} --Mt=${Mt} --Kt=${Kt} --Nt=${Nt}

    if [ $? -eq 0 ]; then
        echo "Test run completed successfully!"
    else
        echo "Test run failed!"
        exit 2
    fi
else
    echo "Compilation failed!"
    exit 1
fi