"""End-to-end cydr example using XcdrStruct."""

import os
import time
from pprint import pprint
from typing import Any

import msgspec
import numpy as np
from nptyping import Bytes, Float64, NDArray, Shape

from cydr.idl import (
    CYDR_CACHE_DIR,
    XcdrStruct,
    int32,
    string,
    uint32,
    warmup_codec,
)


class Stamp(XcdrStruct):
    sec: int32 = np.int32(0)
    nanosec: uint32 = np.uint32(0)


class Header(XcdrStruct):
    stamp: Stamp = msgspec.field(default_factory=Stamp)
    frame_id: string = b""


class JointStateLite(XcdrStruct):
    header: Header = msgspec.field(default_factory=Header)
    name: NDArray[Any, Bytes] = msgspec.field(
        default_factory=lambda: np.empty(0, Bytes)
    )
    position: NDArray[Any, Float64] = msgspec.field(
        default_factory=lambda: np.array([], dtype=Float64)
    )
    effort: NDArray[Shape["30"], Float64] = msgspec.field(
        default_factory=lambda: np.zeros(shape=(30,), dtype=Float64)
    )


def main() -> None:
    print("Compiling or loading schema...")
    print(f"Cache dir: {CYDR_CACHE_DIR}")
    print("Override at process start with CYDR_CACHE_DIR=/path/to/cache")
    _ = JointStateLite._get_codec()
    print("Done")

    print("Creating data...")
    message = JointStateLite(
        header=Header(
            stamp=Stamp(sec=np.int32(17000), nanosec=np.uint32(1234)),
            frame_id=b"base_link",
        ),
        name=np.array([b"joint_a", b"joint_b", b"joint_c"]),
        position=np.array([0.5, 1.5, 2.5], dtype=np.float64),
        effort=np.array([3.5, 4.5, 5.5]*10, dtype=np.float64),
    )
    print("Done")

    print("Warming one full roundtrip...")
    _ = warmup_codec(message)
    print("Done")

    print("Serializing...")
    started = time.perf_counter()
    payload = message.serialize()
    serialize_elapsed_us = (time.perf_counter() - started) * 1e6
    print("Done")

    print("Deserializing...")
    started = time.perf_counter()
    decoded_message = JointStateLite.deserialize(payload)
    deserialize_elapsed_us = (time.perf_counter() - started) * 1e6
    print("Done")

    print("Verifying...")
    roundtrip_stable = bytes(decoded_message.serialize()) == bytes(payload)
    print(f"Roundtrip stable: {roundtrip_stable}")
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
    pprint(decoded_message)


if __name__ == "__main__":
    main()
