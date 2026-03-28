"""Schema fixtures shared by benchmarks."""

from __future__ import annotations

from xcdrjit.idl import CythonFieldType


HEADER_SCHEMA = {
    "stamp": {
        "sec": CythonFieldType.INT32,
        "nanosec": CythonFieldType.UINT32,
    },
    "frame_id": CythonFieldType.STRING,
}


JOINT_STATE_SCHEMA = {
    "header": HEADER_SCHEMA,
    "name": CythonFieldType.TEXT_SEQUENCE,
    "position": CythonFieldType.FLOAT64_SEQUENCE,
    "velocity": CythonFieldType.FLOAT64_SEQUENCE,
    "effort": CythonFieldType.FLOAT64_SEQUENCE,
}


EVERY_SUPPORTED_SCHEMA = {
    "boolean_value": CythonFieldType.BOOLEAN,
    "byte_value": CythonFieldType.BYTE,
    "signed_int8": CythonFieldType.INT8,
    "unsigned_int8": CythonFieldType.UINT8,
    "signed_int16": CythonFieldType.INT16,
    "unsigned_int16": CythonFieldType.UINT16,
    "signed_int32": CythonFieldType.INT32,
    "unsigned_int32": CythonFieldType.UINT32,
    "signed_int64": CythonFieldType.INT64,
    "unsigned_int64": CythonFieldType.UINT64,
    "float32_value": CythonFieldType.FLOAT32,
    "float64_value": CythonFieldType.FLOAT64,
    "text": CythonFieldType.STRING,
    "header": HEADER_SCHEMA,
    "bool_sequence": CythonFieldType.BOOL_SEQUENCE,
    "byte_array": CythonFieldType.BYTE_ARRAY_3,
    "int8_sequence": CythonFieldType.INT8_SEQUENCE,
    "uint8_array": CythonFieldType.UINT8_ARRAY_3,
    "int16_sequence": CythonFieldType.INT16_SEQUENCE,
    "uint16_array": CythonFieldType.UINT16_ARRAY_2,
    "int32_sequence": CythonFieldType.INT32_SEQUENCE,
    "uint32_array": CythonFieldType.UINT32_ARRAY_2,
    "int64_sequence": CythonFieldType.INT64_SEQUENCE,
    "uint64_array": CythonFieldType.UINT64_ARRAY_2,
    "float32_sequence": CythonFieldType.FLOAT32_SEQUENCE,
    "float64_array": CythonFieldType.FLOAT64_ARRAY_2,
    "text_array": CythonFieldType.TEXT_ARRAY_2,
    "text_sequence": CythonFieldType.TEXT_SEQUENCE,
}


__all__ = [
    "HEADER_SCHEMA",
    "JOINT_STATE_SCHEMA",
    "EVERY_SUPPORTED_SCHEMA",
]
