from dataclasses import dataclass, field
import sys

import numpy as np
import pytest

from cyclonedds_idl import IdlStruct, types
from cydr import boolean, get_codec_for

from ..schema import Time

# Warm the helper backend through cydr's normal runtime path so this test uses
# the same pyximport/cache behavior as the library itself.
get_codec_for({"_warmup": boolean})

from cydr._every_supported_cython import (
    compute_serialized_size_every_supported_schema,
    deserialize_every_supported_schema,
    serialize_every_supported_schema,
)

if sys.version_info >= (3, 14):
    pytest.skip(
        "cyclonedds_idl test fixtures are not compatible with Python 3.14",
        allow_module_level=True,
    )


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


def _normalized_string_sequence(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        assert value.ndim == 1
        sequence = value.tolist()
    else:
        assert isinstance(value, list)
        sequence = value
    return [item.decode("utf-8") if isinstance(item, bytes) else item for item in sequence]


def _assert_string_collection_equal(actual: object, expected: object) -> None:
    assert _normalized_string_sequence(actual) == _normalized_string_sequence(expected)


FIELD_ORDER = [
    "boolean_value",
    "byte_value",
    "signed_int8",
    "unsigned_int8",
    "signed_int16",
    "unsigned_int16",
    "signed_int32",
    "unsigned_int32",
    "signed_int64",
    "unsigned_int64",
    "float32_value",
    "float64_value",
    "text",
    "header_stamp_sec",
    "header_stamp_nanosec",
    "header_frame_id",
    "bool_sequence",
    "byte_array",
    "int8_sequence",
    "uint8_array",
    "int16_sequence",
    "uint16_array",
    "int32_sequence",
    "uint32_array",
    "int64_sequence",
    "uint64_array",
    "float32_sequence",
    "float64_array",
    "text_array",
    "text_sequence",
]


def default_inputs() -> dict[str, object]:
    return {
        "boolean_value": False,
        "byte_value": 0,
        "signed_int8": 0,
        "unsigned_int8": 0,
        "signed_int16": 0,
        "unsigned_int16": 0,
        "signed_int32": 0,
        "unsigned_int32": 0,
        "signed_int64": 0,
        "unsigned_int64": 0,
        "float32_value": np.float32(0.0),
        "float64_value": np.float64(0.0),
        "text": b"",
        "header_stamp_sec": 0,
        "header_stamp_nanosec": 0,
        "header_frame_id": b"",
        "bool_sequence": np.array([], dtype=np.bool_),
        "byte_array": np.zeros(3, dtype=np.uint8),
        "int8_sequence": np.array([], dtype=np.int8),
        "uint8_array": np.zeros(3, dtype=np.uint8),
        "int16_sequence": np.array([], dtype=np.int16),
        "uint16_array": np.zeros(2, dtype=np.uint16),
        "int32_sequence": np.array([], dtype=np.int32),
        "uint32_array": np.zeros(2, dtype=np.uint32),
        "int64_sequence": np.array([], dtype=np.int64),
        "uint64_array": np.zeros(2, dtype=np.uint64),
        "float32_sequence": np.array([], dtype=np.float32),
        "float64_array": np.zeros(2, dtype=np.float64),
        "text_array": np.array([b"", b""], dtype=np.bytes_),
        "text_sequence": np.array([], dtype=np.bytes_),
    }


def non_default_inputs() -> dict[str, object]:
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
        "header_stamp_sec": 10,
        "header_stamp_nanosec": 123,
        "header_frame_id": b"map",
        "bool_sequence": np.array([True, False, True], dtype=np.bool_),
        "byte_array": np.array([16, 32, 48], dtype=np.uint8),
        "int8_sequence": np.array([-1, 0, 1], dtype=np.int8),
        "uint8_array": np.array([1, 2, 3], dtype=np.uint8),
        "int16_sequence": np.array([-2, 3], dtype=np.int16),
        "uint16_array": np.array([4, 5], dtype=np.uint16),
        "int32_sequence": np.array([-6, 7], dtype=np.int32),
        "uint32_array": np.array([8, 9], dtype=np.uint32),
        "int64_sequence": np.array([-10, 11], dtype=np.int64),
        "uint64_array": np.array([12, 13], dtype=np.uint64),
        "float32_sequence": np.array([1.5, -2.25], dtype=np.float32),
        "float64_array": np.array([3.5, -4.75], dtype=np.float64),
        "text_array": np.array([b"a", "café".encode("utf-8")], dtype=np.bytes_),
        "text_sequence": np.array([b"bbb", "😀".encode("utf-8")], dtype=np.bytes_),
    }


def build_case(prefix_length: int) -> dict[str, object]:
    values = default_inputs()
    populated = non_default_inputs()
    for field_name in FIELD_ORDER[:prefix_length]:
        values[field_name] = populated[field_name]
    return values


def serialize_cython(values: dict[str, object]) -> bytes:
    return bytes(
        serialize_every_supported_schema(
            values["boolean_value"],
            values["byte_value"],
            values["signed_int8"],
            values["unsigned_int8"],
            values["signed_int16"],
            values["unsigned_int16"],
            values["signed_int32"],
            values["unsigned_int32"],
            values["signed_int64"],
            values["unsigned_int64"],
            values["float32_value"],
            values["float64_value"],
            values["text"],
            values["header_stamp_sec"],
            values["header_stamp_nanosec"],
            values["header_frame_id"],
            values["bool_sequence"],
            values["byte_array"],
            values["int8_sequence"],
            values["uint8_array"],
            values["int16_sequence"],
            values["uint16_array"],
            values["int32_sequence"],
            values["uint32_array"],
            values["int64_sequence"],
            values["uint64_array"],
            values["float32_sequence"],
            values["float64_array"],
            values["text_array"],
            values["text_sequence"],
        )
    )


def serialize_cyclone(values: dict[str, object]) -> bytes:
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
                sec=int(values["header_stamp_sec"]),
                nanosec=int(values["header_stamp_nanosec"]),
            ),
            frame_id=values["header_frame_id"].decode("utf-8"),
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
    ).serialize()


def flatten_decoded(decoded: dict[str, object]) -> dict[str, object]:
    text_array = decoded["text_array"]
    text_sequence = decoded["text_sequence"]
    if hasattr(text_array, "to_numpy"):
        text_array = text_array.to_numpy()
    if hasattr(text_sequence, "to_numpy"):
        text_sequence = text_sequence.to_numpy()

    return {
        "boolean_value": decoded["boolean_value"],
        "byte_value": decoded["byte_value"],
        "signed_int8": decoded["signed_int8"],
        "unsigned_int8": decoded["unsigned_int8"],
        "signed_int16": decoded["signed_int16"],
        "unsigned_int16": decoded["unsigned_int16"],
        "signed_int32": decoded["signed_int32"],
        "unsigned_int32": decoded["unsigned_int32"],
        "signed_int64": decoded["signed_int64"],
        "unsigned_int64": decoded["unsigned_int64"],
        "float32_value": decoded["float32_value"],
        "float64_value": decoded["float64_value"],
        "text": decoded["text"],
        "header_stamp_sec": decoded["header"]["stamp"]["sec"],
        "header_stamp_nanosec": decoded["header"]["stamp"]["nanosec"],
        "header_frame_id": decoded["header"]["frame_id"],
        "bool_sequence": decoded["bool_sequence"],
        "byte_array": decoded["byte_array"],
        "int8_sequence": decoded["int8_sequence"],
        "uint8_array": decoded["uint8_array"],
        "int16_sequence": decoded["int16_sequence"],
        "uint16_array": decoded["uint16_array"],
        "int32_sequence": decoded["int32_sequence"],
        "uint32_array": decoded["uint32_array"],
        "int64_sequence": decoded["int64_sequence"],
        "uint64_array": decoded["uint64_array"],
        "float32_sequence": decoded["float32_sequence"],
        "float64_array": decoded["float64_array"],
        "text_array": text_array,
        "text_sequence": text_sequence,
    }


@pytest.mark.parametrize("prefix_length", range(1, len(FIELD_ORDER) + 1), ids=FIELD_ORDER)
def test_cython_every_supported_schema_matches_cyclone_incrementally(prefix_length: int) -> None:
    values = build_case(prefix_length)

    computed_size = compute_serialized_size_every_supported_schema(
        values["boolean_value"],
        values["byte_value"],
        values["signed_int8"],
        values["unsigned_int8"],
        values["signed_int16"],
        values["unsigned_int16"],
        values["signed_int32"],
        values["unsigned_int32"],
        values["signed_int64"],
        values["unsigned_int64"],
        values["float32_value"],
        values["float64_value"],
        values["text"],
        values["header_stamp_sec"],
        values["header_stamp_nanosec"],
        values["header_frame_id"],
        values["bool_sequence"],
        values["byte_array"],
        values["int8_sequence"],
        values["uint8_array"],
        values["int16_sequence"],
        values["uint16_array"],
        values["int32_sequence"],
        values["uint32_array"],
        values["int64_sequence"],
        values["uint64_array"],
        values["float32_sequence"],
        values["float64_array"],
        values["text_array"],
        values["text_sequence"],
    )
    cython_bytes = serialize_cython(values)
    cyclone_bytes = serialize_cyclone(values)

    assert computed_size == len(cython_bytes)
    assert cython_bytes == cyclone_bytes


@pytest.mark.parametrize("build_values", [default_inputs, non_default_inputs], ids=["default", "non_default"])
def test_deserialize_every_supported_schema_roundtrips_against_cyclone(build_values) -> None:
    values = build_values()

    cyclone_bytes = serialize_cyclone(values)
    decoded = deserialize_every_supported_schema(cyclone_bytes)
    roundtrip_values = flatten_decoded(decoded)
    roundtrip_bytes = serialize_cython(roundtrip_values)

    assert roundtrip_bytes == cyclone_bytes
    assert roundtrip_values["text"] == values["text"]
    assert roundtrip_values["header_frame_id"] == values["header_frame_id"]
    assert np.array_equal(roundtrip_values["bool_sequence"], values["bool_sequence"])
    assert np.array_equal(roundtrip_values["float64_array"], values["float64_array"])
    _assert_string_collection_equal(roundtrip_values["text_array"], values["text_array"])
    _assert_string_collection_equal(roundtrip_values["text_sequence"], values["text_sequence"])


def test_manual_serializer_accepts_list_bytes_for_string_collections() -> None:
    values = non_default_inputs()
    values["text_array"] = [b"a", "café".encode("utf-8")]
    values["text_sequence"] = [b"bbb", "😀".encode("utf-8")]

    computed_size = compute_serialized_size_every_supported_schema(
        values["boolean_value"],
        values["byte_value"],
        values["signed_int8"],
        values["unsigned_int8"],
        values["signed_int16"],
        values["unsigned_int16"],
        values["signed_int32"],
        values["unsigned_int32"],
        values["signed_int64"],
        values["unsigned_int64"],
        values["float32_value"],
        values["float64_value"],
        values["text"],
        values["header_stamp_sec"],
        values["header_stamp_nanosec"],
        values["header_frame_id"],
        values["bool_sequence"],
        values["byte_array"],
        values["int8_sequence"],
        values["uint8_array"],
        values["int16_sequence"],
        values["uint16_array"],
        values["int32_sequence"],
        values["uint32_array"],
        values["int64_sequence"],
        values["uint64_array"],
        values["float32_sequence"],
        values["float64_array"],
        values["text_array"],
        values["text_sequence"],
    )
    cython_bytes = serialize_cython(values)
    cyclone_bytes = serialize_cyclone(values)

    assert computed_size == len(cython_bytes)
    assert cython_bytes == cyclone_bytes


STRING_COLLECTION_VALUES = [b"hello", "café".encode("utf-8"), "😀".encode("utf-8")]


def _make_string_inputs_with(text_array, text_sequence) -> dict[str, object]:
    values = non_default_inputs()
    values["text_array"] = text_array
    values["text_sequence"] = text_sequence
    return values


def _cyclone_reference_bytes() -> bytes:
    return serialize_cyclone(
        _make_string_inputs_with(
            text_array=np.array(STRING_COLLECTION_VALUES[:2], dtype=np.bytes_),
            text_sequence=np.array(STRING_COLLECTION_VALUES, dtype=np.bytes_),
        )
    )


def test_string_collections_as_list_of_bytes_matches_cyclone() -> None:
    values = _make_string_inputs_with(
        text_array=list(STRING_COLLECTION_VALUES[:2]),
        text_sequence=list(STRING_COLLECTION_VALUES),
    )

    assert serialize_cython(values) == _cyclone_reference_bytes()


def test_string_collections_as_numpy_bytes_array_matches_cyclone() -> None:
    values = _make_string_inputs_with(
        text_array=np.array(STRING_COLLECTION_VALUES[:2], dtype=np.bytes_),
        text_sequence=np.array(STRING_COLLECTION_VALUES, dtype=np.bytes_),
    )

    assert serialize_cython(values) == _cyclone_reference_bytes()


def test_deserialize_every_supported_schema_returns_copies_for_numeric_collections() -> None:
    values = non_default_inputs()

    cyclone_bytes = serialize_cyclone(values)
    decoded = deserialize_every_supported_schema(cyclone_bytes)

    assert decoded["bool_sequence"].flags["OWNDATA"]
    assert decoded["float64_array"].flags["OWNDATA"]
