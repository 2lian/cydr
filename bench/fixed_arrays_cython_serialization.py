"""Benchmark generated cached fixed-array serializers against cyclonedds_idl."""

import argparse
from dataclasses import dataclass, field

import numpy as np

from cyclonedds_idl import IdlStruct, types

from bench._common import BenchmarkCase, benchmark_case, print_environment, print_results
from bench.schema import Time
from xcdrjit import assert_messages_equal
from xcdrjit.idl import (
    CYTHON_CACHE_DIR,
    XcdrStruct,
    array,
    float32,
    float64,
    get_codec_for,
    int32,
    string,
    uint32,
)


SMALL_ARRAY_LENGTH = 16
LARGE_ARRAY_LENGTH = 10_000
@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class FixedArrayPayloadSmall(IdlStruct, typename="test/msg/FixedArrayPayloadSmall"):
    header: Header = field(default_factory=Header)
    gains: types.array[types.float64, 16] = field(default_factory=list)
    ids: types.array[types.uint32, 16] = field(default_factory=list)
    temperatures: types.array[types.float32, 16] = field(default_factory=list)
    labels: types.array[str, 4] = field(default_factory=list)


@dataclass
class FixedArrayPayloadLarge(IdlStruct, typename="test/msg/FixedArrayPayloadLarge"):
    header: Header = field(default_factory=Header)
    gains: types.array[types.float64, 10000] = field(default_factory=list)
    ids: types.array[types.uint32, 10000] = field(default_factory=list)
    temperatures: types.array[types.float32, 10000] = field(default_factory=list)
    labels: types.array[str, 4] = field(default_factory=list)


class StampStruct(XcdrStruct):
    sec: int32
    nanosec: uint32


class HeaderStruct(XcdrStruct):
    stamp: StampStruct
    frame_id: string


class FixedArrayPayloadSmallStruct(XcdrStruct):
    header: HeaderStruct
    gains: array(float64, SMALL_ARRAY_LENGTH)
    ids: array(uint32, SMALL_ARRAY_LENGTH)
    temperatures: array(float32, SMALL_ARRAY_LENGTH)
    labels: array(string, 4)


class FixedArrayPayloadLargeStruct(XcdrStruct):
    header: HeaderStruct
    gains: array(float64, LARGE_ARRAY_LENGTH)
    ids: array(uint32, LARGE_ARRAY_LENGTH)
    temperatures: array(float32, LARGE_ARRAY_LENGTH)
    labels: array(string, 4)


FIXED_ARRAY_SMALL_SCHEMA = {
    "header": {
        "stamp": {
            "sec": int32,
            "nanosec": uint32,
        },
        "frame_id": string,
    },
    "gains": array(float64, SMALL_ARRAY_LENGTH),
    "ids": array(uint32, SMALL_ARRAY_LENGTH),
    "temperatures": array(float32, SMALL_ARRAY_LENGTH),
    "labels": array(string, 4),
}


FIXED_ARRAY_LARGE_SCHEMA = {
    "header": {
        "stamp": {
            "sec": int32,
            "nanosec": uint32,
        },
        "frame_id": string,
    },
    "gains": array(float64, LARGE_ARRAY_LENGTH),
    "ids": array(uint32, LARGE_ARRAY_LENGTH),
    "temperatures": array(float32, LARGE_ARRAY_LENGTH),
    "labels": array(string, 4),
}


def build_values(count: int) -> dict[str, object]:
    return {
        "header": {
            "stamp": {
                "sec": np.int32(12345),
                "nanosec": np.uint32(67890),
            },
            "frame_id": b"fixed_arrays",
        },
        "gains": np.linspace(-1.0, 1.0, num=count, dtype=np.float64),
        "ids": np.arange(1000, 1000 + count, dtype=np.uint32),
        "temperatures": np.linspace(20.0, 80.0, num=count, dtype=np.float32),
        "labels": [b"front", b"rear", b"left", b"right"],
    }


def build_small_idl_message(values: dict[str, object]) -> FixedArrayPayloadSmall:
    return FixedArrayPayloadSmall(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("ascii"),
        ),
        gains=values["gains"].tolist(),
        ids=values["ids"].tolist(),
        temperatures=values["temperatures"].tolist(),
        labels=[value.decode("ascii") for value in values["labels"]],
    )


def build_large_idl_message(values: dict[str, object]) -> FixedArrayPayloadLarge:
    return FixedArrayPayloadLarge(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("ascii"),
        ),
        gains=values["gains"].tolist(),
        ids=values["ids"].tolist(),
        temperatures=values["temperatures"].tolist(),
        labels=[value.decode("ascii") for value in values["labels"]],
    )


def build_small_struct_message(values: dict[str, object]) -> FixedArrayPayloadSmallStruct:
    return FixedArrayPayloadSmallStruct(
        header=HeaderStruct(
            stamp=StampStruct(
                sec=values["header"]["stamp"]["sec"],
                nanosec=values["header"]["stamp"]["nanosec"],
            ),
            frame_id=values["header"]["frame_id"],
        ),
        gains=values["gains"],
        ids=values["ids"],
        temperatures=values["temperatures"],
        labels=values["labels"],
    )


def build_large_struct_message(values: dict[str, object]) -> FixedArrayPayloadLargeStruct:
    return FixedArrayPayloadLargeStruct(
        header=HeaderStruct(
            stamp=StampStruct(
                sec=values["header"]["stamp"]["sec"],
                nanosec=values["header"]["stamp"]["nanosec"],
            ),
            frame_id=values["header"]["frame_id"],
        ),
        gains=values["gains"],
        ids=values["ids"],
        temperatures=values["temperatures"],
        labels=values["labels"],
    )


def build_cases(label: str, count: int) -> tuple[BenchmarkCase, BenchmarkCase]:
    values = build_values(count)
    if count == SMALL_ARRAY_LENGTH:
        schema = FIXED_ARRAY_SMALL_SCHEMA
        idl_message = build_small_idl_message(values)
        struct_message = build_small_struct_message(values)
        struct_type = FixedArrayPayloadSmallStruct
        idl_type = FixedArrayPayloadSmall
    else:
        schema = FIXED_ARRAY_LARGE_SCHEMA
        idl_message = build_large_idl_message(values)
        struct_message = build_large_struct_message(values)
        struct_type = FixedArrayPayloadLargeStruct
        idl_type = FixedArrayPayloadLarge

    codec = get_codec_for(schema)
    serialize = codec.serialize
    deserialize = codec.deserialize
    payload = idl_message.serialize()

    xcdrjit_dict_bytes = bytes(serialize(values))
    xcdrjit_struct_bytes = bytes(struct_message.serialize())
    assert xcdrjit_dict_bytes == payload
    assert xcdrjit_struct_bytes == payload
    assert_messages_equal(deserialize(payload), values, schema)
    assert_messages_equal(struct_type.deserialize(payload)._to_message_dict(), values, schema)
    assert bytes(idl_type.deserialize(payload).serialize()) == payload

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
            "xcdrjit_struct": lambda: struct_type.deserialize(payload),
            "cyclonedds_idl": lambda: idl_type.deserialize(payload),
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
        ("small", SMALL_ARRAY_LENGTH),
        ("large", LARGE_ARRAY_LENGTH),
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
        "FixedArray Generated Codec Benchmark",
        [
            "Schema: local fixed-array schemas in bench/fixed_arrays_cython_serialization.py",
            f"Cache dir: {CYTHON_CACHE_DIR}",
            "Serialize: xcdrjit dict call is codec.serialize(values)",
            "Serialize: xcdrjit struct call is message.serialize()",
            "Serialize: Cyclone call is idl_message.serialize()",
            "Deserialize: xcdrjit dict call is codec.deserialize(payload)",
            "Deserialize: xcdrjit struct call is Struct.deserialize(payload)",
            "Deserialize: Cyclone call is IdlStruct.deserialize(payload)",
            "Fixed arrays exercised: float64, uint32, float32, and string arrays",
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
