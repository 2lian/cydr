from __future__ import annotations

"""Helpers for generating Cython serializer modules from simple Python schemas.

The main user-facing entrypoint is `generate_cython_serializer_code(...)`.
It accepts a possibly nested `dict[str, CythonFieldType | dict[str, ...]]`
and returns the `.pyx` source as a string.
"""

from collections.abc import Mapping
from enum import StrEnum


class CythonFieldType(StrEnum):
    BOOLEAN = "boolean"
    BYTE = "byte"
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL_SEQUENCE = "bool_sequence"
    BYTE_ARRAY_3 = "byte_array_3"
    INT8_SEQUENCE = "int8_sequence"
    UINT8_ARRAY_3 = "uint8_array_3"
    INT16_SEQUENCE = "int16_sequence"
    UINT16_ARRAY_2 = "uint16_array_2"
    INT32_SEQUENCE = "int32_sequence"
    UINT32_ARRAY_2 = "uint32_array_2"
    INT64_SEQUENCE = "int64_sequence"
    UINT64_ARRAY_2 = "uint64_array_2"
    FLOAT32_SEQUENCE = "float32_sequence"
    FLOAT64_SEQUENCE = "float64_sequence"
    FLOAT64_ARRAY_2 = "float64_array_2"
    TEXT_ARRAY_2 = "text_array_2"
    TEXT_SEQUENCE = "text_sequence"


NestedCythonFieldValue = CythonFieldType | Mapping[str, object]
NestedCythonFields = Mapping[str, NestedCythonFieldValue]


def flatten_cython_fields(
    fields: NestedCythonFields,
    *,
    prefix: str = "",
) -> dict[str, CythonFieldType]:
    """Flatten a nested schema into `snake_case` field names.

    Example:
        {"header": {"stamp": {"sec": INT32}}}
        becomes
        {"header_stamp_sec": INT32}
    """
    flattened: dict[str, CythonFieldType] = {}
    for field_name, field_value in fields.items():
        full_name = f"{prefix}_{field_name}" if prefix else field_name
        if isinstance(field_value, CythonFieldType):
            flattened[full_name] = field_value
        elif isinstance(field_value, Mapping):
            flattened.update(flatten_cython_fields(field_value, prefix=full_name))
        else:
            raise TypeError(
                f"Unsupported field schema for {full_name!r}: {type(field_value).__name__}",
            )
    return flattened

FIELD_DECLARATIONS: dict[CythonFieldType, str] = {
    CythonFieldType.BOOLEAN: "bint {name}",
    CythonFieldType.BYTE: "uint8_t {name}",
    CythonFieldType.INT8: "int8_t {name}",
    CythonFieldType.UINT8: "uint8_t {name}",
    CythonFieldType.INT16: "int16_t {name}",
    CythonFieldType.UINT16: "uint16_t {name}",
    CythonFieldType.INT32: "int32_t {name}",
    CythonFieldType.UINT32: "uint32_t {name}",
    CythonFieldType.INT64: "int64_t {name}",
    CythonFieldType.UINT64: "uint64_t {name}",
    CythonFieldType.FLOAT32: "cnp.float32_t {name}",
    CythonFieldType.FLOAT64: "cnp.float64_t {name}",
    CythonFieldType.STRING: "bytes {name}",
    CythonFieldType.BOOL_SEQUENCE: "const cnp.npy_bool[::1] {name}",
    CythonFieldType.BYTE_ARRAY_3: "const uint8_t[::1] {name}",
    CythonFieldType.INT8_SEQUENCE: "const int8_t[::1] {name}",
    CythonFieldType.UINT8_ARRAY_3: "const uint8_t[::1] {name}",
    CythonFieldType.INT16_SEQUENCE: "const int16_t[::1] {name}",
    CythonFieldType.UINT16_ARRAY_2: "const uint16_t[::1] {name}",
    CythonFieldType.INT32_SEQUENCE: "const int32_t[::1] {name}",
    CythonFieldType.UINT32_ARRAY_2: "const uint32_t[::1] {name}",
    CythonFieldType.INT64_SEQUENCE: "const int64_t[::1] {name}",
    CythonFieldType.UINT64_ARRAY_2: "const uint64_t[::1] {name}",
    CythonFieldType.FLOAT32_SEQUENCE: "const cnp.float32_t[::1] {name}",
    CythonFieldType.FLOAT64_SEQUENCE: "const cnp.float64_t[::1] {name}",
    CythonFieldType.FLOAT64_ARRAY_2: "const cnp.float64_t[::1] {name}",
    CythonFieldType.TEXT_ARRAY_2: "list {name}",
    CythonFieldType.TEXT_SEQUENCE: "list {name}",
}


ADVANCE_CALLS: dict[CythonFieldType, str] = {
    CythonFieldType.BOOLEAN: "advance_boolean_field(pos)",
    CythonFieldType.BYTE: "advance_byte_field(pos, align_offset)",
    CythonFieldType.INT8: "advance_int8_field(pos, align_offset)",
    CythonFieldType.UINT8: "advance_uint8_field(pos, align_offset)",
    CythonFieldType.INT16: "advance_int16_field(pos, align_offset)",
    CythonFieldType.UINT16: "advance_uint16_field(pos, align_offset)",
    CythonFieldType.INT32: "advance_int32_field(pos, align_offset)",
    CythonFieldType.UINT32: "advance_uint32_field(pos, align_offset)",
    CythonFieldType.INT64: "advance_int64_field(pos, align_offset)",
    CythonFieldType.UINT64: "advance_uint64_field(pos, align_offset)",
    CythonFieldType.FLOAT32: "advance_float32_field(pos, align_offset)",
    CythonFieldType.FLOAT64: "advance_float64_field(pos, align_offset)",
    CythonFieldType.STRING: "advance_string_field(pos, {name}, align_offset)",
    CythonFieldType.BOOL_SEQUENCE: "advance_bool_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.BYTE_ARRAY_3: "advance_byte_array_field(pos, {name}, align_offset)",
    CythonFieldType.INT8_SEQUENCE: "advance_int8_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.UINT8_ARRAY_3: "advance_uint8_array_field(pos, {name}, align_offset)",
    CythonFieldType.INT16_SEQUENCE: "advance_int16_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.UINT16_ARRAY_2: "advance_uint16_array_field(pos, {name}, align_offset)",
    CythonFieldType.INT32_SEQUENCE: "advance_int32_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.UINT32_ARRAY_2: "advance_uint32_array_field(pos, {name}, align_offset)",
    CythonFieldType.INT64_SEQUENCE: "advance_int64_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.UINT64_ARRAY_2: "advance_uint64_array_field(pos, {name}, align_offset)",
    CythonFieldType.FLOAT32_SEQUENCE: "advance_float32_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.FLOAT64_SEQUENCE: "advance_float64_sequence_field(pos, {name}, align_offset)",
    CythonFieldType.FLOAT64_ARRAY_2: "advance_float64_array_field(pos, {name}, align_offset)",
    CythonFieldType.TEXT_ARRAY_2: "advance_text_array_field(pos, {name}, align_offset)",
    CythonFieldType.TEXT_SEQUENCE: "advance_text_sequence_field(pos, {name}, align_offset)",
}


WRITE_CALLS: dict[CythonFieldType, str] = {
    CythonFieldType.BOOLEAN: "write_boolean_field(buffer, pos, {name})",
    CythonFieldType.BYTE: "write_byte_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT8: "write_int8_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT8: "write_uint8_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT16: "write_int16_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT16: "write_uint16_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT32: "write_int32_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT32: "write_uint32_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT64: "write_int64_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT64: "write_uint64_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.FLOAT32: "write_float32_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.FLOAT64: "write_float64_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.STRING: "write_string_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.BOOL_SEQUENCE: "write_bool_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.BYTE_ARRAY_3: "write_byte_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT8_SEQUENCE: "write_int8_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT8_ARRAY_3: "write_uint8_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT16_SEQUENCE: "write_int16_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT16_ARRAY_2: "write_uint16_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT32_SEQUENCE: "write_int32_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT32_ARRAY_2: "write_uint32_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.INT64_SEQUENCE: "write_int64_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.UINT64_ARRAY_2: "write_uint64_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.FLOAT32_SEQUENCE: "write_float32_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.FLOAT64_SEQUENCE: "write_float64_sequence_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.FLOAT64_ARRAY_2: "write_float64_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.TEXT_ARRAY_2: "write_text_array_field(buffer, pos, {name}, align_offset)",
    CythonFieldType.TEXT_SEQUENCE: "write_text_sequence_field(buffer, pos, {name}, align_offset)",
}


def _signature_lines(fields: Mapping[str, CythonFieldType]) -> str:
    return ",\n    ".join(
        FIELD_DECLARATIONS[field_type].format(name=field_name)
        for field_name, field_type in fields.items()
    )


def _compute_body_lines(fields: Mapping[str, CythonFieldType]) -> str:
    return "\n".join(
        f"    pos = {ADVANCE_CALLS[field_type].format(name=field_name)}"
        for field_name, field_type in fields.items()
    )


def _serialize_body_lines(fields: Mapping[str, CythonFieldType]) -> str:
    return "\n".join(
        f"    pos = {WRITE_CALLS[field_type].format(name=field_name)}"
        for field_name, field_type in fields.items()
    )


def _argument_names(fields: Mapping[str, CythonFieldType], *, indent: int) -> str:
    padding = " " * indent
    return ",\n".join(f"{padding}{field_name}" for field_name in fields)


def generate_cython_serializer_code(
    serializer_name: str,
    schema: NestedCythonFields,
) -> str:
    """Return generated `.pyx` source for a serializer.

    `schema` may be flat or nested. Nested field names are flattened with `_`.

    Example:
        schema = {
            "text": CythonFieldType.STRING,
            "header": {
                "stamp": {
                    "sec": CythonFieldType.INT32,
                    "nanosec": CythonFieldType.UINT32,
                },
                "frame_id": CythonFieldType.STRING,
            },
        }

        pyx_source = generate_cython_serializer_code("my_message", schema)

        with open("xcdrjit/_my_message.pyx", "w", encoding="utf-8") as f:
            f.write(pyx_source)
    """
    return generate_cython_serializer_module(
        serializer_name,
        flatten_cython_fields(schema),
    )


def generate_cython_serializer_module(serializer_name: str, fields: Mapping[str, CythonFieldType]) -> str:
    """Low-level generator for already-flattened fields."""
    signature = _signature_lines(fields)
    compute_body = _compute_body_lines(fields)
    serialize_body = _serialize_body_lines(fields)
    call_args = _argument_names(fields, indent=8)

    return f'''# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

from cpython.bytearray cimport PyByteArray_AS_STRING
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cimport numpy as cnp

from xcdrjit._every_supported_cython cimport (
    ENCAPSULATION_HEADER_SIZE,
    advance_boolean_field,
    advance_byte_array_field,
    advance_byte_field,
    advance_bool_sequence_field,
    advance_float32_field,
    advance_float32_sequence_field,
    advance_float64_array_field,
    advance_float64_field,
    advance_float64_sequence_field,
    advance_int16_field,
    advance_int16_sequence_field,
    advance_int32_field,
    advance_int32_sequence_field,
    advance_int64_field,
    advance_int64_sequence_field,
    advance_int8_field,
    advance_int8_sequence_field,
    advance_string_field,
    advance_text_array_field,
    advance_text_sequence_field,
    advance_uint16_array_field,
    advance_uint16_field,
    advance_uint32_array_field,
    advance_uint32_field,
    advance_uint64_array_field,
    advance_uint64_field,
    advance_uint8_array_field,
    advance_uint8_field,
    write_boolean_field,
    write_byte_array_field,
    write_byte_field,
    write_bool_sequence_field,
    write_encapsulation_header,
    write_float32_field,
    write_float32_sequence_field,
    write_float64_array_field,
    write_float64_field,
    write_float64_sequence_field,
    write_int16_field,
    write_int16_sequence_field,
    write_int32_field,
    write_int32_sequence_field,
    write_int64_field,
    write_int64_sequence_field,
    write_int8_field,
    write_int8_sequence_field,
    write_string_field,
    write_text_array_field,
    write_text_sequence_field,
    write_uint16_array_field,
    write_uint16_field,
    write_uint32_array_field,
    write_uint32_field,
    write_uint64_array_field,
    write_uint64_field,
    write_uint8_array_field,
    write_uint8_field,
)


cpdef Py_ssize_t compute_serialized_size_{serializer_name}(
    {signature},
) except -1:
    cdef Py_ssize_t pos = ENCAPSULATION_HEADER_SIZE
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE

{compute_body}
    return pos


cpdef bytearray serialize_{serializer_name}(
    {signature},
):
    cdef Py_ssize_t total_size = compute_serialized_size_{serializer_name}(
{call_args},
    )
    cdef bytearray output = bytearray(total_size)
    cdef unsigned char* buffer = <unsigned char*> PyByteArray_AS_STRING(output)
    cdef Py_ssize_t pos = ENCAPSULATION_HEADER_SIZE
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE

    write_encapsulation_header(buffer)

{serialize_body}
    return output
'''
