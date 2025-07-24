# Systolic Array Development - Scaling Up the Design

In this version, we focus on scaling up the design to a **990 x 740 grid** with a **74 x 74 block size**.

---

## Upgrade Overview

### Enhancements
- **H2D Streaming Issue Resolved**: The host-to-device streaming problem has been successfully fixed.
- **Extensive Debugging**: Added numerous debugging pragmas to help trace issues effectively.
- **Configuration Testing**: Tested multiple grid and block size configurations to evaluate performance.
- **Timing Added**: Implemented timing to analyze performance metrics.

---

## Known Issues

### Bugs
- **Kernel Performance**: The `kernel.csl` is currently showing poor performance.
- **Stalling Issue**: When the block size is increased, the program stalls.

---

## Next Steps
- Optimize `kernel.csl` for better performance.
- Investigate and resolve the stalling issue when increasing block size.
- Continue testing with different configurations to ensure scalability.