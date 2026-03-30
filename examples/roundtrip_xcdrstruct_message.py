"""End-to-end xcdrjit example using XcdrStruct."""

import os
import time
from pprint import pprint

import msgspec
import numpy as np

from xcdrjit.idl import (
    CYTHON_CACHE_DIR,
    XcdrStruct,
    array,
    float64,
    int32,
    sequence,
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
    name: sequence(string) = msgspec.field(default_factory=list)
    position: sequence(float64) = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    effort: array(float64, 3) = msgspec.field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )


def main() -> None:
    print("Compiling or loading schema...")
    print(f"Cache dir: {CYTHON_CACHE_DIR}")
    print("Override at process start with XCDRJIT_CACHE_DIR=/path/to/cache")
    _ = JointStateLite._get_codec()
    print("Done")

    print("Creating data...")
    message = JointStateLite(
        header=Header(
            stamp=Stamp(sec=np.int32(17000), nanosec=np.uint32(1234)),
            frame_id=b"base_link",
        ),
        name=[b"joint_a", b"joint_b", b"joint_c"],
        position=np.array([0.5, 1.5, 2.5], dtype=np.float64),
        effort=np.array([3.5, 4.5, 5.5], dtype=np.float64),
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
    pprint(decoded_message)


if __name__ == "__main__":
    main()
