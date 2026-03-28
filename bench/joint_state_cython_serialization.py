"""Benchmark a generated cached JointState serializer against cyclonedds_idl."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cyclonedds_idl import IdlStruct, types

from _common import BenchmarkCase, benchmark_case, print_environment, print_results
from bench.schema import JOINT_STATE_SCHEMA
from xcdrjit import Time
from xcdrjit.idl import load_cython_serialize_function


SMALL_SEQUENCE_LENGTH = 8
LARGE_SEQUENCE_LENGTH = 10_000
BENCH_CACHE_DIR = Path(__file__).resolve().parent / "_generated_cython_cache" / "joint_state"


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class JointState(IdlStruct, typename="sensor_msgs/msg/JointState"):
    header: Header = field(default_factory=Header)
    name: types.sequence[str] = field(default_factory=list)
    position: types.sequence[types.float64] = field(default_factory=list)
    velocity: types.sequence[types.float64] = field(default_factory=list)
    effort: types.sequence[types.float64] = field(default_factory=list)


def make_float_sequence(count: int, start: float, stop: float) -> np.ndarray:
    return np.linspace(start, stop, num=count, dtype=np.float64)


def make_joint_names(count: int) -> list[bytes]:
    return [f"joint_{index:05d}".encode("ascii") for index in range(count)]


def build_joint_state_values(count: int) -> dict[str, object]:
    return {
        "header": {
            "stamp": {
                "sec": np.int32(1_700_000_000),
                "nanosec": np.uint32(123_456_789),
            },
            "frame_id": b"base_link",
        },
        "name": make_joint_names(count),
        "position": make_float_sequence(count, -1.0, 1.0),
        "velocity": make_float_sequence(count, 0.0, 10.0),
        "effort": make_float_sequence(count, 10.0, 0.0),
    }


def build_idl_message(values: dict[str, object]) -> JointState:
    return JointState(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("ascii"),
        ),
        name=[value.decode("ascii") for value in values["name"]],
        position=values["position"].tolist(),
        velocity=values["velocity"].tolist(),
        effort=values["effort"].tolist(),
    )


def build_case(label: str, count: int) -> BenchmarkCase:
    serialize = load_cython_serialize_function(JOINT_STATE_SCHEMA, cache_dir=BENCH_CACHE_DIR)
    values = build_joint_state_values(count)
    idl_message = build_idl_message(values)

    xcdrjit_bytes = bytes(serialize(values))
    cyclone_bytes = idl_message.serialize()
    assert xcdrjit_bytes == cyclone_bytes

    return BenchmarkCase(
        label=label,
        count=count,
        payload_size=len(cyclone_bytes),
        xcdrjit_fn=lambda: serialize(values),
        cyclonedds_fn=idl_message.serialize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7, help="Number of timed repeats per serializer.")
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.25,
        help="Minimum wall time in seconds to target for each timed measurement batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cases = [
        build_case("small", SMALL_SEQUENCE_LENGTH),
        build_case("large", LARGE_SEQUENCE_LENGTH),
    ]
    measurements = {
        case.label: benchmark_case(case, repeat=args.repeat, min_time=args.min_time)
        for case in cases
    }

    print_environment(
        "JointState Generated Serializer Benchmark",
        [
            "Schema: JOINT_STATE_SCHEMA from bench/schema.py",
            f"Cache dir: {BENCH_CACHE_DIR}",
            "xcdrjit call: serialize(values)",
            "Cyclone call: idl_message.serialize()",
            "Runtime input is one nested dict; xcdrjit flattens only the values in insertion order",
        ],
    )
    print_results(cases, measurements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
