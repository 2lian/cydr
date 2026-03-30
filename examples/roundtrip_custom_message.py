"""Minimal end-to-end cydr roundtrip example."""

import os
import time
from pprint import pprint
from typing import Any

import numpy as np
from nptyping import Bool, Bytes, Float64, NDArray

from cydr.idl import (
    CYDR_CACHE_DIR,
    get_codec_for,
    int32,
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
    "labels": NDArray[Any, Bytes],
    "position": NDArray[Any, Float64],
    "bools": NDArray[Any, Bool],
}

DEFAULT_VALUES = {
    "header": {
        "stamp": {
            "sec": np.int32(0),
            "nanosec": np.uint32(0),
        },
        "frame_id": b"",
    },
    "labels": np.array([b"hello", b"world"], dtype=np.bytes_),
    "position": np.array([1, 2, 3, 4], dtype=np.float64),
    "bools": np.zeros(ARR_SIZE, dtype=np.bool_),
}


def main() -> None:
    print("Compiling or loading schema...")
    print(f"Cache dir: {CYDR_CACHE_DIR}")
    print("Override at process start with CYDR_CACHE_DIR=/path/to/cache")
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
        "labels": np.array([b"joint_a", b"joint_with_some_other_name"], dtype=np.bytes_),
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
    if "CYDR_CACHE_DIR" in os.environ:
        print(f"CYDR_CACHE_DIR={os.environ['CYDR_CACHE_DIR']}")
    else:
        print("CYDR_CACHE_DIR is not set")

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
