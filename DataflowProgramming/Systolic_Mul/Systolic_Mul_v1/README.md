# Systolic Array Development - Program Run Status

In this version, we focus on making the program run successfully.

---

## Upgrade Overview

### Successful Run
- We successfully ran the program on a **980 x 740 grid** with a **5 x 5 block size**.

---

## Known Issues

### Bugs
- **Host Streaming (run.py)**:  
  - The H2D streaming in `run.py` is not fully general. When dealing with random floating-point numbers, the program does not always produce correct results.

- **Kernel Debugging**:  
  - Debugging prints have not yet been added to `kernel.csl`.  
  - We did not trace the instructions to `sim.log` for more detailed debugging.

---

## Next Steps
- Generalize the H2D host streaming for random data.
- Add debugging prints to `kernel.csl`.
- Enable detailed instruction tracing in `sim.log`.