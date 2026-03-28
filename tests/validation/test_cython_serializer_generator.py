from dataclasses import dataclass, field

import numpy as np

from cyclonedds_idl import IdlStruct, types

from xcdrjit import Time
from xcdrjit.idl import (
    CythonFieldType,
    flatten_cython_fields,
    flatten_cython_value_list,
    generate_cython_serializer_code,
    load_cython_serialize_function,
)
from ..cache import SCHEMA_CACHE_DIR
from ..schema import EVERY_SUPPORTED_SCHEMA, EVERY_SUPPORTED_SCHEMA_FLAT


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


def build_values() -> dict[str, object]:
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
        "text_array": [b"a", "café".encode("utf-8")],
        "text_sequence": [b"bbb", "😀".encode("utf-8")],
    }


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
    ).serialize()


def test_flatten_cython_fields_flattens_nested_header_schema() -> None:
    flattened = flatten_cython_fields(EVERY_SUPPORTED_SCHEMA)

    assert flattened == EVERY_SUPPORTED_SCHEMA_FLAT
    assert list(flattened)[13:16] == [
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
    ]
    assert flattened["header_stamp_sec"] is CythonFieldType.INT32
    assert flattened["header_frame_id"] is CythonFieldType.STRING


def test_flatten_cython_value_list_ignores_keys_and_preserves_value_order() -> None:
    values = build_values()

    assert flatten_cython_value_list(values) == [
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
        values["header"]["stamp"]["sec"],
        values["header"]["stamp"]["nanosec"],
        values["header"]["frame_id"],
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
    ]


def test_generate_cython_serializer_code_renders_expected_lines() -> None:
    fields = flatten_cython_fields(EVERY_SUPPORTED_SCHEMA)
    assert fields["header_stamp_sec"] is CythonFieldType.INT32
    assert fields["header_frame_id"] is CythonFieldType.STRING

    serializer_name = "generated_every_supported_schema"
    source = generate_cython_serializer_code(serializer_name, EVERY_SUPPORTED_SCHEMA)

    assert f"cpdef Py_ssize_t compute_serialized_size_{serializer_name}(" in source
    assert "pos = advance_string_field(pos, header_frame_id, align_offset)" in source
    assert "pos = write_string_field(buffer, pos, header_frame_id, align_offset)" in source


def test_load_cython_serializer_matches_cyclone() -> None:
    serialize = load_cython_serialize_function(
        EVERY_SUPPORTED_SCHEMA,
        cache_dir=SCHEMA_CACHE_DIR,
    )
    values = build_values()
    generated_bytes = bytes(serialize(values))
    cyclone_bytes = serialize_cyclone(values)

    assert generated_bytes == cyclone_bytes


def test_load_cython_serializer_ignores_runtime_keys_and_uses_value_order() -> None:
    serialize = load_cython_serialize_function(
        EVERY_SUPPORTED_SCHEMA,
        cache_dir=SCHEMA_CACHE_DIR,
    )
    values = build_values()
    reordered_keys_same_values = {
        "a": values["boolean_value"],
        "b": values["byte_value"],
        "c": values["signed_int8"],
        "d": values["unsigned_int8"],
        "e": values["signed_int16"],
        "f": values["unsigned_int16"],
        "g": values["signed_int32"],
        "h": values["unsigned_int32"],
        "i": values["signed_int64"],
        "j": values["unsigned_int64"],
        "k": values["float32_value"],
        "l": values["float64_value"],
        "m": values["text"],
        "n": {
            "o": {
                "p": values["header"]["stamp"]["sec"],
                "q": values["header"]["stamp"]["nanosec"],
            },
            "r": values["header"]["frame_id"],
        },
        "s": values["bool_sequence"],
        "t": values["byte_array"],
        "u": values["int8_sequence"],
        "v": values["uint8_array"],
        "w": values["int16_sequence"],
        "x": values["uint16_array"],
        "y": values["int32_sequence"],
        "z": values["uint32_array"],
        "aa": values["int64_sequence"],
        "ab": values["uint64_array"],
        "ac": values["float32_sequence"],
        "ad": values["float64_array"],
        "ae": values["text_array"],
        "af": values["text_sequence"],
    }

    generated_bytes = bytes(serialize(reordered_keys_same_values))
    cyclone_bytes = serialize_cyclone(values)

    assert generated_bytes == cyclone_bytes
