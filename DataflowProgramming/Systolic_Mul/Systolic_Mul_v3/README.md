# Systolic Array Development - Scaling the Design

In this version, we are scaling the design further.

---

## Upgrade Overview

### Enhancements
- **Kernel Replacement**: Replaced the kernel with the same implementation as the SUMMA sample code to improve scalability and performance.
- **Simplified H2D Streaming**: Changed the host-to-device (H2D) streaming to a simpler design that achieves the same functionality.

---

## Known Issues

### Bugs
- **Timer Issue**: The timer is not set correctly, causing inaccurate performance measurements.

---

## Next Steps
- Correct the timer setup to ensure accurate performance metrics.
- Monitor the performance of the new kernel implementation and validate against expected outcomes.
- Further simplify and optimize the H2D streaming mechanism.