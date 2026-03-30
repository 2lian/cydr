"""Benchmark scalar and fixed-array primitive fields against cyclonedds_idl."""

import argparse
import types as python_types
from dataclasses import dataclass

import numpy as np
from nptyping import NDArray, Shape

from cyclonedds_idl import IdlStruct, types

from bench._common import measure_runtime, print_environment
from cydr import assert_messages_equal
from cydr.idl import (
    CYDR_CACHE_DIR,
    XcdrStruct,
    boolean,
    byte,
    float32,
    float64,
    get_codec_for,
    int8,
    int16,
    int32,
    int64,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)


SMALL_ARRAY_LENGTH = 16
LARGE_ARRAY_LENGTH = 10_000
SMALL_STRING_LENGTH = 16
LARGE_STRING_LENGTH = 100_000
SMALL_STRING_ARRAY_ITEM_LENGTH = 8
LARGE_STRING_ARRAY_ITEM_LENGTH = 10


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    name: str
    schema_token: object
    idl_scalar_type: object
    numpy_dtype: np.dtype | None


@dataclass(frozen=True, slots=True)
class CaseSpec:
    primitive_name: str
    kind: str
    size_label: str
    count_or_length: int
    schema: dict[str, object]
    runtime_value: object
    idl_value: object
    idl_class: type[IdlStruct]
    struct_class: type[XcdrStruct]

    @property
    def label(self) -> str:
        return f"{self.primitive_name} {self.kind} {self.size_label}"


@dataclass(frozen=True, slots=True)
class CaseMeasurement:
    payload_size: int
    serialize_dict_us: float
    serialize_struct_us: float
    serialize_cyclone_us: float
    deserialize_dict_us: float
    deserialize_struct_us: float
    deserialize_cyclone_us: float


PRIMITIVES = [
    PrimitiveSpec("boolean", boolean, bool, np.dtype(np.bool_)),
    PrimitiveSpec("byte", byte, types.byte, np.dtype(np.uint8)),
    PrimitiveSpec("int8", int8, types.int8, np.dtype(np.int8)),
    PrimitiveSpec("uint8", uint8, types.uint8, np.dtype(np.uint8)),
    PrimitiveSpec("int16", int16, types.int16, np.dtype(np.int16)),
    PrimitiveSpec("uint16", uint16, types.uint16, np.dtype(np.uint16)),
    PrimitiveSpec("int32", int32, types.int32, np.dtype(np.int32)),
    PrimitiveSpec("uint32", uint32, types.uint32, np.dtype(np.uint32)),
    PrimitiveSpec("int64", int64, types.int64, np.dtype(np.int64)),
    PrimitiveSpec("uint64", uint64, types.uint64, np.dtype(np.uint64)),
    PrimitiveSpec("float32", float32, types.float32, np.dtype(np.float32)),
    PrimitiveSpec("float64", float64, types.float64, np.dtype(np.float64)),
    PrimitiveSpec("string", string, str, None),
]


def make_idl_class(
    class_name: str,
    typename: str,
    annotation: object,
) -> type[IdlStruct]:
    def body(namespace: dict[str, object]) -> None:
        namespace["__module__"] = __name__
        namespace["__annotations__"] = {"value": annotation}

    return dataclass(
        python_types.new_class(
            class_name,
            (IdlStruct,),
            {"kwds": {"typename": typename}},
            body,
        )
    )


def make_struct_class(class_name: str, annotation: object) -> type[XcdrStruct]:
    return type(
        class_name,
        (XcdrStruct,),
        {
            "__module__": __name__,
            "__annotations__": {"value": annotation},
        },
    )


def build_scalar_value(spec: PrimitiveSpec, size_label: str) -> object:
    if spec.name == "boolean":
        return True
    if spec.name == "byte":
        return np.uint8(127)
    if spec.name == "int8":
        return np.int8(-12)
    if spec.name == "uint8":
        return np.uint8(200)
    if spec.name == "int16":
        return np.int16(-1234)
    if spec.name == "uint16":
        return np.uint16(54321)
    if spec.name == "int32":
        return np.int32(-1234567)
    if spec.name == "uint32":
        return np.uint32(3456789012)
    if spec.name == "int64":
        return np.int64(-1234567890123)
    if spec.name == "uint64":
        return np.uint64(1234567890123456789)
    if spec.name == "float32":
        return np.float32(1.25)
    if spec.name == "float64":
        return np.float64(-2.5)
    if spec.name == "string":
        length = SMALL_STRING_LENGTH if size_label == "small" else LARGE_STRING_LENGTH
        return b"a" * length
    raise AssertionError(f"Unsupported scalar primitive {spec.name!r}")


def build_array_value(spec: PrimitiveSpec, count: int, size_label: str) -> object:
    if spec.name == "string":
        item_length = (
            SMALL_STRING_ARRAY_ITEM_LENGTH
            if size_label == "small"
            else LARGE_STRING_ARRAY_ITEM_LENGTH
        )
        item = b"x" * item_length
        return np.array([item] * count, dtype=np.bytes_)

    dtype = spec.numpy_dtype
    if dtype is None:
        raise AssertionError(f"Missing dtype for {spec.name!r}")

    if spec.name == "boolean":
        return (np.arange(count, dtype=np.uint8) % 2 == 0)
    if spec.name in {"byte", "uint8"}:
        return np.arange(count, dtype=np.uint8)
    if spec.name == "int8":
        return np.arange(count, dtype=np.int8)
    if spec.name == "int16":
        return np.arange(count, dtype=np.int16)
    if spec.name == "uint16":
        return np.arange(count, dtype=np.uint16)
    if spec.name == "int32":
        return np.arange(count, dtype=np.int32)
    if spec.name == "uint32":
        return np.arange(count, dtype=np.uint32)
    if spec.name == "int64":
        return np.arange(count, dtype=np.int64)
    if spec.name == "uint64":
        return np.arange(count, dtype=np.uint64)
    if spec.name == "float32":
        return np.linspace(-1.0, 1.0, num=count, dtype=np.float32)
    if spec.name == "float64":
        return np.linspace(-1.0, 1.0, num=count, dtype=np.float64)
    raise AssertionError(f"Unsupported array primitive {spec.name!r}")


def build_idl_value(spec: PrimitiveSpec, runtime_value: object) -> object:
    if spec.name == "string":
        if isinstance(runtime_value, np.ndarray):
            return [value.decode("ascii") for value in runtime_value]
        return runtime_value.decode("ascii")

    if isinstance(runtime_value, np.ndarray):
        return runtime_value.tolist()

    if isinstance(runtime_value, np.generic):
        return runtime_value.item()

    return runtime_value


def build_case_specs() -> list[CaseSpec]:
    cases: list[CaseSpec] = []

    for spec in PRIMITIVES:
        scalar_sizes = ("small", "big") if spec.name == "string" else ("scalar",)
        for size_label in scalar_sizes:
            runtime_value = build_scalar_value(spec, size_label)
            schema = {"value": spec.schema_token}
            suffix = size_label if spec.name == "string" else "scalar"
            idl_class = make_idl_class(
                f"{spec.name.title().replace('_', '')}{suffix.title()}Idl",
                f"bench/msg/{spec.name}_{suffix}",
                spec.idl_scalar_type,
            )
            struct_class = make_struct_class(
                f"{spec.name.title().replace('_', '')}{suffix.title()}Struct",
                spec.schema_token,
            )
            cases.append(
                CaseSpec(
                    primitive_name=spec.name,
                    kind="scalar",
                    size_label=size_label,
                    count_or_length=(
                        len(runtime_value)
                        if spec.name == "string"
                        else 1
                    ),
                    schema=schema,
                    runtime_value=runtime_value,
                    idl_value=build_idl_value(spec, runtime_value),
                    idl_class=idl_class,
                    struct_class=struct_class,
                )
            )

        for size_label, count in (("small", SMALL_ARRAY_LENGTH), ("big", LARGE_ARRAY_LENGTH)):
            runtime_value = build_array_value(spec, count, size_label)
            element_type = np.bytes_ if spec.name == "string" else spec.schema_token
            annotation = NDArray[Shape[str(count)], element_type]
            schema = {"value": annotation}
            idl_class = make_idl_class(
                f"{spec.name.title().replace('_', '')}Array{size_label.title()}Idl",
                f"bench/msg/{spec.name}_array_{size_label}",
                types.array[spec.idl_scalar_type, count],
            )
            struct_class = make_struct_class(
                f"{spec.name.title().replace('_', '')}Array{size_label.title()}Struct",
                annotation,
            )
            cases.append(
                CaseSpec(
                    primitive_name=spec.name,
                    kind="array",
                    size_label=size_label,
                    count_or_length=count,
                    schema=schema,
                    runtime_value=runtime_value,
                    idl_value=build_idl_value(spec, runtime_value),
                    idl_class=idl_class,
                    struct_class=struct_class,
                )
            )

    return cases


def benchmark_case_spec(case: CaseSpec, repeat: int, min_time: float) -> CaseMeasurement:
    codec = get_codec_for(case.schema)
    values = {"value": case.runtime_value}
    idl_message = case.idl_class(value=case.idl_value)
    struct_message = case.struct_class(case.runtime_value)
    payload = idl_message.serialize()

    assert bytes(codec.serialize(values)) == payload
    assert bytes(struct_message.serialize()) == payload
    assert_messages_equal(codec.deserialize(payload), values, case.schema)
    assert_messages_equal(case.struct_class.deserialize(payload)._to_nested_dict(), values, case.schema)
    assert bytes(case.idl_class.deserialize(payload).serialize()) == payload

    serialize_functions = {
        "cydr_dict": lambda: codec.serialize(values),
        "cydr_struct": lambda: struct_message.serialize(),
        "cyclonedds_idl": idl_message.serialize,
    }
    deserialize_functions = {
        "cydr_dict": lambda: codec.deserialize(payload),
        "cydr_struct": lambda: case.struct_class.deserialize(payload),
        "cyclonedds_idl": lambda: case.idl_class.deserialize(payload),
    }

    for fn in list(serialize_functions.values()) + list(deserialize_functions.values()):
        fn()
        fn()
        fn()

    serialize_measurements = {
        name: measure_runtime(fn, repeat=repeat, min_time=min_time)
        for name, fn in serialize_functions.items()
    }
    deserialize_measurements = {
        name: measure_runtime(fn, repeat=repeat, min_time=min_time)
        for name, fn in deserialize_functions.items()
    }

    return CaseMeasurement(
        payload_size=len(payload),
        serialize_dict_us=serialize_measurements["cydr_dict"].best_seconds * 1_000_000,
        serialize_struct_us=serialize_measurements["cydr_struct"].best_seconds * 1_000_000,
        serialize_cyclone_us=serialize_measurements["cyclonedds_idl"].best_seconds * 1_000_000,
        deserialize_dict_us=deserialize_measurements["cydr_dict"].best_seconds * 1_000_000,
        deserialize_struct_us=deserialize_measurements["cydr_struct"].best_seconds * 1_000_000,
        deserialize_cyclone_us=deserialize_measurements["cyclonedds_idl"].best_seconds * 1_000_000,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="Number of timed repeats per function.")
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.05,
        help="Minimum wall time in seconds to target for each timed measurement batch.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        help="Only run cases whose label contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N matched cases. Zero means no limit.",
    )
    return parser.parse_args()


def print_markdown_table(results: list[tuple[CaseSpec, CaseMeasurement]]) -> None:
    print()
    print("| Primitive | Kind | Size | Count/Len | Payload Bytes | Dict Ser us | Struct Ser us | Cyclone Ser us | Dict Deser us | Struct Deser us | Cyclone Deser us |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case, measurement in results:
        print(
            f"| `{case.primitive_name}` | `{case.kind}` | `{case.size_label}` | "
            f"{case.count_or_length} | {measurement.payload_size} | "
            f"{measurement.serialize_dict_us:.2f} | {measurement.serialize_struct_us:.2f} | {measurement.serialize_cyclone_us:.2f} | "
            f"{measurement.deserialize_dict_us:.2f} | {measurement.deserialize_struct_us:.2f} | {measurement.deserialize_cyclone_us:.2f} |"
        )


def main() -> int:
    args = parse_args()
    cases = build_case_specs()

    if args.match:
        cases = [case for case in cases if args.match in case.label]
    if args.limit > 0:
        cases = cases[: args.limit]

    print_environment(
        "Primitive Field Matrix Benchmark",
        [
            f"Cache dir: {CYDR_CACHE_DIR}",
            "Cases: scalar and fixed-array primitive fields",
            f"Small array length: {SMALL_ARRAY_LENGTH}",
            f"Large array length: {LARGE_ARRAY_LENGTH}",
            f"Small string length: {SMALL_STRING_LENGTH}",
            f"Large string length: {LARGE_STRING_LENGTH}",
            "Implementations: cydr dict, cydr struct, cyclonedds_idl",
        ],
    )

    results: list[tuple[CaseSpec, CaseMeasurement]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{total}] {case.label} ...")
        measurement = benchmark_case_spec(case, repeat=args.repeat, min_time=args.min_time)
        results.append((case, measurement))
        print(
            "  serialize us:"
            f" dict={measurement.serialize_dict_us:.2f}"
            f" struct={measurement.serialize_struct_us:.2f}"
            f" cyclone={measurement.serialize_cyclone_us:.2f}"
        )
        print(
            "  deserialize us:"
            f" dict={measurement.deserialize_dict_us:.2f}"
            f" struct={measurement.deserialize_struct_us:.2f}"
            f" cyclone={measurement.deserialize_cyclone_us:.2f}"
        )

    print_markdown_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
