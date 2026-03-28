"""Benchmark a generated cached EverySupportedSchema serializer against cyclonedds_idl."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cyclonedds_idl import IdlStruct, types

from _common import BenchmarkCase, benchmark_case, print_environment, print_results
from bench.schema import EVERY_SUPPORTED_SCHEMA
from xcdrjit import Time
from xcdrjit.idl import load_cython_serialize_function


SMALL_SEQUENCE_LENGTH = 8
LARGE_SEQUENCE_LENGTH = 10_000
BENCH_CACHE_DIR = Path(__file__).resolve().parent / "_generated_cython_cache" / "every_supported_schema"


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class EverySupportedSchema(IdlStruct, typename="test/msg/EverySupportedSchema"):
    boolean_value: bool = False
    byte_value: types.byte = 0
    signed_int8: types.int8 = 0
    unsigned_int8: types.uint8 = 0
    signed_int16: types.int16 = 0
    unsigned_int16: types.uint16 = 0
    signed_int32: types.int32 = 0
    unsigned_int32: types.uint32 = 0
    signed_int64: types.int64 = 0
    unsigned_int64: types.uint64 = 0
    float32_value: types.float32 = 0.0
    float64_value: types.float64 = 0.0
    text: str = ""
    header: Header = field(default_factory=Header)
    bool_sequence: types.sequence[bool] = field(default_factory=list)
    byte_array: types.array[types.byte, 3] = field(default_factory=list)
    int8_sequence: types.sequence[types.int8] = field(default_factory=list)
    uint8_array: types.array[types.uint8, 3] = field(default_factory=list)
    int16_sequence: types.sequence[types.int16] = field(default_factory=list)
    uint16_array: types.array[types.uint16, 2] = field(default_factory=list)
    int32_sequence: types.sequence[types.int32] = field(default_factory=list)
    uint32_array: types.array[types.uint32, 2] = field(default_factory=list)
    int64_sequence: types.sequence[types.int64] = field(default_factory=list)
    uint64_array: types.array[types.uint64, 2] = field(default_factory=list)
    float32_sequence: types.sequence[types.float32] = field(default_factory=list)
    float64_array: types.array[types.float64, 2] = field(default_factory=list)
    text_array: types.array[str, 2] = field(default_factory=list)
    text_sequence: types.sequence[str] = field(default_factory=list)


def make_bool_sequence(count: int) -> np.ndarray:
    return (np.arange(count, dtype=np.uint8) % 2 == 0)


def make_int_sequence(dtype: np.dtype, count: int, start: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=dtype)


def make_float_sequence(dtype: np.dtype, count: int, start: float, stop: float) -> np.ndarray:
    return np.linspace(start, stop, num=count, dtype=dtype)


def make_text_sequence(count: int) -> list[bytes]:
    return [f"text_{index:05d}".encode("utf-8") for index in range(count)]


def build_values(count: int) -> dict[str, object]:
    return {
        "boolean_value": True,
        "byte_value": 127,
        "signed_int8": -1,
        "unsigned_int8": 1,
        "signed_int16": -2,
        "unsigned_int16": 2,
        "signed_int32": -3,
        "unsigned_int32": 3,
        "signed_int64": -4,
        "unsigned_int64": 4,
        "float32_value": np.float32(1.5),
        "float64_value": np.float64(2.5),
        "text": "café😀".encode("utf-8"),
        "header": {
            "stamp": {
                "sec": 10,
                "nanosec": 123,
            },
            "frame_id": b"map",
        },
        "bool_sequence": make_bool_sequence(count),
        "byte_array": np.array([16, 32, 48], dtype=np.uint8),
        "int8_sequence": make_int_sequence(np.int8, count, -64),
        "uint8_array": np.array([1, 2, 3], dtype=np.uint8),
        "int16_sequence": make_int_sequence(np.int16, count, -1024),
        "uint16_array": np.array([4, 5], dtype=np.uint16),
        "int32_sequence": make_int_sequence(np.int32, count, -2048),
        "uint32_array": np.array([8, 9], dtype=np.uint32),
        "int64_sequence": make_int_sequence(np.int64, count, -4096),
        "uint64_array": np.array([12, 13], dtype=np.uint64),
        "float32_sequence": make_float_sequence(np.float32, count, -1.0, 1.0),
        "float64_array": np.array([3.5, -4.75], dtype=np.float64),
        "text_array": [b"a", "café".encode("utf-8")],
        "text_sequence": make_text_sequence(count),
    }


def build_idl_message(values: dict[str, object]) -> EverySupportedSchema:
    return EverySupportedSchema(
        boolean_value=bool(values["boolean_value"]),
        byte_value=int(values["byte_value"]),
        signed_int8=int(values["signed_int8"]),
        unsigned_int8=int(values["unsigned_int8"]),
        signed_int16=int(values["signed_int16"]),
        unsigned_int16=int(values["unsigned_int16"]),
        signed_int32=int(values["signed_int32"]),
        unsigned_int32=int(values["unsigned_int32"]),
        signed_int64=int(values["signed_int64"]),
        unsigned_int64=int(values["unsigned_int64"]),
        float32_value=float(values["float32_value"]),
        float64_value=float(values["float64_value"]),
        text=values["text"].decode("utf-8"),
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        bool_sequence=values["bool_sequence"].tolist(),
        byte_array=values["byte_array"].tolist(),
        int8_sequence=values["int8_sequence"].tolist(),
        uint8_array=values["uint8_array"].tolist(),
        int16_sequence=values["int16_sequence"].tolist(),
        uint16_array=values["uint16_array"].tolist(),
        int32_sequence=values["int32_sequence"].tolist(),
        uint32_array=values["uint32_array"].tolist(),
        int64_sequence=values["int64_sequence"].tolist(),
        uint64_array=values["uint64_array"].tolist(),
        float32_sequence=values["float32_sequence"].tolist(),
        float64_array=values["float64_array"].tolist(),
        text_array=[value.decode("utf-8") for value in values["text_array"]],
        text_sequence=[value.decode("utf-8") for value in values["text_sequence"]],
    )


def build_case(label: str, count: int) -> BenchmarkCase:
    serialize = load_cython_serialize_function(EVERY_SUPPORTED_SCHEMA, cache_dir=BENCH_CACHE_DIR)
    values = build_values(count)
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
        "EverySupportedSchema Generated Serializer Benchmark",
        [
            "Schema: EVERY_SUPPORTED_SCHEMA from bench/schema.py",
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
