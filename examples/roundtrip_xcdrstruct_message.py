"""End-to-end xcdrjit example using XcdrStruct."""

import time
from pprint import pprint

import numpy as np

from xcdrjit.idl import (
    XcdrStruct,
    array,
    float64,
    int32,
    sequence,
    string,
    uint32,
)


class Stamp(XcdrStruct):
    sec: int32
    nanosec: uint32


class Header(XcdrStruct):
    stamp: Stamp
    frame_id: string


class JointStateLite(XcdrStruct):
    header: Header
    name: sequence(string)
    position: sequence(float64)
    effort: array(float64, 3)

class Something(XcdrStruct):
    stamp: Stamp
    frame_id: uint32




def main() -> None:
    message = JointStateLite(
        header=Header(
            stamp=Stamp(sec=np.int32(17000), nanosec=np.uint32(1234)),
            frame_id=b"base_link",
        ),
        name=[b"joint_a", b"joint_b", b"joint_c"],
        position=np.array([0.5, 1.5, 2.5], dtype=np.float64),
        effort=np.array([3.5, 4.5, 5.5], dtype=np.float64),
    )

    # First call compiles and caches the codec. Warm it once so the timings
    # below reflect steady-state serialization and deserialization.
    _ = message.serialize()

    started = time.perf_counter()
    payload = message.serialize()
    serialize_elapsed_us = (time.perf_counter() - started) * 1e6

    started = time.perf_counter()
    decoded_message = JointStateLite.deserialize(payload)
    deserialize_elapsed_us = (time.perf_counter() - started) * 1e6

    roundtrip_stable = (
        bytes(decoded_message.serialize()) == bytes(payload)
    )

    print(f"payload size: {len(payload)} bytes")
    print(f"serialize: {serialize_elapsed_us:.2f} us")
    print(f"deserialize: {deserialize_elapsed_us:.2f} us")
    print(f"roundtrip stable: {roundtrip_stable}")
    print()

    print("decoded message:")
    pprint(decoded_message)


if __name__ == "__main__":
    main()
