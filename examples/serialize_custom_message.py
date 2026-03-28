"""Minimal end-to-end xcdrjit serialization example."""

from __future__ import annotations

import numpy as np

from xcdrjit.idl import CythonFieldType, load_cython_serialize_function


SCHEMA = {
    "header": {
        "stamp": {
            "sec": CythonFieldType.INT32,
            "nanosec": CythonFieldType.UINT32,
        },
        "frame_id": CythonFieldType.STRING,
    },
    "labels": CythonFieldType.TEXT_SEQUENCE,
    "position": CythonFieldType.FLOAT64_SEQUENCE,
}


def main() -> None:
    serialize = load_cython_serialize_function(SCHEMA)

    values = {
        "header": {
            "stamp": {
                "sec": np.int32(10),
                "nanosec": np.uint32(123),
            },
            "frame_id": b"map",
        },
        "labels": [b"joint_a", b"joint_b"],
        "position": np.array([1.0, 2.5], dtype=np.float64),
    }

    payload = serialize(values)

    print(f"serialized {len(payload)} bytes")
    print(bytes(payload).hex(" "))


if __name__ == "__main__":
    main()
