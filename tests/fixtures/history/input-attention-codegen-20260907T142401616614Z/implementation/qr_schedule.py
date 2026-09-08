"""Symbolic enumeration of the pinned blocked QR control schedule.

No numeric inputs or device observations are used. Tags identify a local pair,
upper-only neighbor update, or lower-only neighbor update. This mirrors the
original function-level loops/branches, independently of CSL instrumentation.
"""


def rotations(rows, cols, nt, x, y):
    if not (1 <= cols <= rows and 0 <= x < cols and 0 <= y < rows and nt >= 2):
        raise ValueError("QR schedule dimensions")
    events = []
    count = 0
    down = y == rows - 1

    def local_triangle():
        for col in range(nt - 1):
            events.extend([1] * (nt - col - 1))

    def lower():
        nonlocal count, down
        local_triangle()
        if x == y:
            if y != rows - 1:
                events.extend([2] * nt)
            return
        while count < nt:
            if down:
                events.append(3)
                count += 1
                events.extend([1] * (nt - count))
            else:
                events.append(2)
            if y != rows - 1:
                down = not down

    if x == 0:
        lower()
    else:
        while True:
            local_triangle()
            if count == y and x > y:
                events.extend([2] * nt)
                break
            for col in range(nt):
                if not down:
                    events.append(2)
                events.append(3)
                events.extend([1] * (nt - col - 1))
            count += 1
            if count == x:
                count = 0
                lower()
                break
    return events


def sampled_slots(total):
    slots = {}
    for serial in range(total):
        if serial < 7 or serial % 16 == 0:
            slots[serial if serial < 7 else 7 + (serial // 16) % 9] = serial
    return slots
