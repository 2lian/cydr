"""Minimal end-to-end xcdrjit roundtrip example."""

import os
import time
from pprint import pprint

import numpy as np

from xcdrjit.idl import (
    CYTHON_CACHE_DIR,
    array,
    boolean,
    float64,
    get_codec_for,
    int32,
    sequence,
    string,
    uint32,
    warmup_codec,
)

ARR_SIZE = 100_000

SCHEMA = {
    "header": {
        "stamp": {
            "sec": int32,
            "nanosec": uint32,
        },
        "frame_id": string,
    },
    "labels": sequence(string),
    "position": sequence(float64),
    "bools": array(boolean, ARR_SIZE),
}

DEFAULT_VALUES = {
    "header": {
        "stamp": {
            "sec": np.int32(0),
            "nanosec": np.uint32(0),
        },
        "frame_id": b"",
    },
    "labels": [],
    "position": np.array([], dtype=np.float64),
    "bools": np.zeros(ARR_SIZE, dtype=np.bool_),
}


def main() -> None:
    print("Compiling or loading schema...")
    print(f"Cache dir: {CYTHON_CACHE_DIR}")
    print("Override at process start with XCDRJIT_CACHE_DIR=/path/to/cache")
    codec = get_codec_for(SCHEMA)
    serialize = codec.serialize
    deserialize = codec.deserialize
    print("Done")

    print("Creating data...")
    values = {
        "header": {
            "stamp": {
                "sec": np.int32(170000),
                "nanosec": np.uint32(123),
            },
            "frame_id": b"map",
        },
        "labels": [b"joint_a", b"joint_b"],
        "position": np.array([1.0, 2.5], dtype=np.float64),
        "bools": np.arange(ARR_SIZE, dtype=np.int64) % 2 == 0,
    }
    print("Done")

    print("Warming one full roundtrip...")
    _ = warmup_codec(values, SCHEMA)
    print("Done")

    print("Serializing...")
    started = time.perf_counter()
    payload = serialize(values)
    serialize_elapsed_us = (time.perf_counter() - started) * 1e6
    print("Done")

    print("Deserializing...")
    started = time.perf_counter()
    decoded = deserialize(payload)
    deserialize_elapsed_us = (time.perf_counter() - started) * 1e6
    print("Done")

    print("Verifying...")
    is_stable = serialize(decoded) == payload
    print(f"Roundtrip stable: {is_stable}")
    if "XCDRJIT_CACHE_DIR" in os.environ:
        print(f"XCDRJIT_CACHE_DIR={os.environ['XCDRJIT_CACHE_DIR']}")
    else:
        print("XCDRJIT_CACHE_DIR is not set")

    print(f"Payload size: {len(payload):_} bytes")
    print(
        f"serialize: {serialize_elapsed_us:.2f} us "
        f"({len(payload)/serialize_elapsed_us:_.1f} MB/s)"
    )
    print(
        f"deserialize: {deserialize_elapsed_us:.2f} us "
        f"({len(payload)/deserialize_elapsed_us:_.1f} MB/s)"
    )
    print()
    print("decoded values:")
    pprint(decoded, sort_dicts=False)


if __name__ == "__main__":
    main()
