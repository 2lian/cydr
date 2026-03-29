"""Benchmark a generated cached JointState serializer against cyclonedds_idl."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cyclonedds_idl import IdlStruct, types

from bench._common import BenchmarkCase, benchmark_case, print_environment, print_results
from bench.schema import JOINT_STATE_SCHEMA
from xcdrjit import Time, assert_messages_equal
from xcdrjit.idl import CYTHON_CACHE_DIR, XcdrStruct, float64, get_codec_for, int32, sequence, string, uint32


SMALL_SEQUENCE_LENGTH = 8
LARGE_SEQUENCE_LENGTH = 10_000
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


class StampStruct(XcdrStruct):
    sec: int32
    nanosec: uint32


class HeaderStruct(XcdrStruct):
    stamp: StampStruct
    frame_id: string


class JointStateStruct(XcdrStruct):
    header: HeaderStruct
    name: sequence(string)
    position: sequence(float64)
    velocity: sequence(float64)
    effort: sequence(float64)


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


def build_struct_message(values: dict[str, object]) -> JointStateStruct:
    return JointStateStruct(
        header=HeaderStruct(
            stamp=StampStruct(
                sec=values["header"]["stamp"]["sec"],
                nanosec=values["header"]["stamp"]["nanosec"],
            ),
            frame_id=values["header"]["frame_id"],
        ),
        name=values["name"],
        position=values["position"],
        velocity=values["velocity"],
        effort=values["effort"],
    )


def build_cases(label: str, count: int) -> tuple[BenchmarkCase, BenchmarkCase]:
    codec = get_codec_for(JOINT_STATE_SCHEMA)
    serialize = codec.serialize
    deserialize = codec.deserialize
    values = build_joint_state_values(count)
    idl_message = build_idl_message(values)
    struct_message = build_struct_message(values)
    payload = idl_message.serialize()

    xcdrjit_dict_bytes = bytes(serialize(values))
    xcdrjit_struct_bytes = bytes(struct_message.serialize())
    assert xcdrjit_dict_bytes == payload
    assert xcdrjit_struct_bytes == payload
    assert_messages_equal(deserialize(payload), values, JOINT_STATE_SCHEMA)
    assert_messages_equal(
        JointStateStruct.deserialize(payload)._to_message_dict(),
        values,
        JOINT_STATE_SCHEMA,
    )
    assert bytes(JointState.deserialize(payload).serialize()) == payload

    serialize_case = BenchmarkCase(
        label=label,
        count=count,
        payload_size=len(payload),
        functions={
            "xcdrjit_dict": lambda: serialize(values),
            "xcdrjit_struct": lambda: struct_message.serialize(),
            "cyclonedds_idl": idl_message.serialize,
        },
    )
    deserialize_case = BenchmarkCase(
        label=label,
        count=count,
        payload_size=len(payload),
        functions={
            "xcdrjit_dict": lambda: deserialize(payload),
            "xcdrjit_struct": lambda: JointStateStruct.deserialize(payload),
            "cyclonedds_idl": lambda: JointState.deserialize(payload),
        },
    )
    return serialize_case, deserialize_case


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

    serialize_cases: list[BenchmarkCase] = []
    deserialize_cases: list[BenchmarkCase] = []
    for label, count in (
        ("small", SMALL_SEQUENCE_LENGTH),
        ("large", LARGE_SEQUENCE_LENGTH),
    ):
        serialize_case, deserialize_case = build_cases(label, count)
        serialize_cases.append(serialize_case)
        deserialize_cases.append(deserialize_case)

    serialize_measurements = {
        case.label: benchmark_case(case, repeat=args.repeat, min_time=args.min_time)
        for case in serialize_cases
    }
    deserialize_measurements = {
        case.label: benchmark_case(case, repeat=args.repeat, min_time=args.min_time)
        for case in deserialize_cases
    }

    print_environment(
        "JointState Generated Codec Benchmark",
        [
            "Schema: JOINT_STATE_SCHEMA from bench/schema.py",
            f"Cache dir: {CYTHON_CACHE_DIR}",
            "Serialize: xcdrjit dict call is codec.serialize(values)",
            "Serialize: xcdrjit struct call is message.serialize()",
            "Serialize: Cyclone call is idl_message.serialize()",
            "Deserialize: xcdrjit dict call is codec.deserialize(payload)",
            "Deserialize: xcdrjit struct call is JointStateStruct.deserialize(payload)",
            "Deserialize: Cyclone call is JointState.deserialize(payload)",
            "Dict runtime input is one nested dict; struct runtime input is one XcdrStruct instance",
        ],
    )
    print("Serialize")
    print_results(serialize_cases, serialize_measurements)
    print()
    print("Deserialize")
    print_results(deserialize_cases, deserialize_measurements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
