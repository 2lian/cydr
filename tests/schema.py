"""Schema fixtures shared by validation tests."""

from __future__ import annotations

from xcdrjit.idl import (
    array,
    boolean,
    byte,
    flatten_cython_fields,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    sequence,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)


HEADER_SCHEMA = {
    "stamp": {
        "sec": int32,
        "nanosec": uint32,
    },
    "frame_id": string,
}


JOINT_STATE_SCHEMA = {
    "header": HEADER_SCHEMA,
    "name": sequence(string),
    "position": sequence(float64),
    "velocity": sequence(float64),
    "effort": sequence(float64),
}


EVERY_SUPPORTED_SCHEMA = {
    "boolean_value": boolean,
    "byte_value": byte,
    "signed_int8": int8,
    "unsigned_int8": uint8,
    "signed_int16": int16,
    "unsigned_int16": uint16,
    "signed_int32": int32,
    "unsigned_int32": uint32,
    "signed_int64": int64,
    "unsigned_int64": uint64,
    "float32_value": float32,
    "float64_value": float64,
    "text": string,
    "header": HEADER_SCHEMA,
    "bool_sequence": sequence(boolean),
    "byte_array": array(byte, 3),
    "int8_sequence": sequence(int8),
    "uint8_array": array(uint8, 3),
    "int16_sequence": sequence(int16),
    "uint16_array": array(uint16, 2),
    "int32_sequence": sequence(int32),
    "uint32_array": array(uint32, 2),
    "int64_sequence": sequence(int64),
    "uint64_array": array(uint64, 2),
    "float32_sequence": sequence(float32),
    "float64_array": array(float64, 2),
    "text_array": array(string, 2),
    "text_sequence": sequence(string),
}


EVERY_SUPPORTED_SCHEMA_FLAT = flatten_cython_fields(EVERY_SUPPORTED_SCHEMA)


__all__ = [
    "HEADER_SCHEMA",
    "JOINT_STATE_SCHEMA",
    "EVERY_SUPPORTED_SCHEMA",
    "EVERY_SUPPORTED_SCHEMA_FLAT",
]
