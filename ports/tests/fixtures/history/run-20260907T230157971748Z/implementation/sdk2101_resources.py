"""Default memcpy reservations of the pinned SDK2.10.1 core rectangle.

A collision check for layouts importing default <memcpy/get_params>, not a
universal SDK allocator or a claim to enumerate compiler temporary registers.
"""

import copy
from frontend import check

CONTRACT = dict(
    sdk_image_sha256="fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d",
    source="<memcpy/sys_params>",
    source_sha256="e70159b986fd92a4f4a83a0a46abe209ce64ed45903ac8ccda9ca0c7a4ef47df",
    colors=[20, 21, 22, 23],
    input_queues=[0, 1],
    output_queues=[0, 1],
    local_tasks=[21, 24, 27, 28, 30],
    local_task_sources=dict(
        memcpy_data=dict(
            identifier=21,
            source="memcpy/wse3/memcpyd2h.csl",
            source_sha256="352c56bbf4f3b5c0ea7a3851128270e9175dd1023d997cffaf9314a89f785b43",
            derivation="LOCAL_MEMCPYD2H_DATA derives from default MEMCPYD2H_DATA=21 and binds f_send_data",
        ),
        persistent_context=dict(identifier=24, conservative_reservation=True),
    ),
    control_tasks=[33, 34, 35, 36, 37, 40],
    scope="WSE3 default core-rectangle memcpy implementation bindings plus context declarations; custom overrides require another contract. Hardware identifier ranges remain compiler-checked.",
)


def check_default_memcpy(application):
    for kind in (
        "colors",
        "input_queues",
        "output_queues",
        "local_tasks",
        "control_tasks",
    ):
        values = application.get(kind, [])
        check(
            all(type(v) is int for v in values) and len(values) == len(set(values)),
            "distinct typed resource identifiers: " + kind,
        )
        check(
            set(values).isdisjoint(CONTRACT[kind]),
            "SDK default memcpy reserved " + kind,
        )
    return copy.deepcopy(CONTRACT)
