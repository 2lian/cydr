from __future__ import annotations

"""Helpers for generating Cython codec modules from nested Python schemas."""

from collections.abc import Mapping
from dataclasses import dataclass

from .schema_types import (
    ArrayField,
    FlatField,
    NestedSchemaFields,
    PrimitiveKind,
    ScalarField,
    SequenceField,
    flatten_schema_fields,
)

NestedCythonFields = NestedSchemaFields


@dataclass(frozen=True, slots=True)
class PrimitiveCodegenInfo:
    scalar_decl: str
    view_decl: str
    scalar_advance: str
    scalar_write: str
    scalar_read: str
    itemsize_expr: str
    alignment: int
    dtype_expr: str
    pointer_expr: str


PRIMITIVE_CODEGEN: dict[PrimitiveKind, PrimitiveCodegenInfo] = {
    PrimitiveKind.BOOLEAN: PrimitiveCodegenInfo(
        scalar_decl="bint {name}",
        view_decl="const cnp.npy_bool[::1] {name}",
        scalar_advance="advance_boolean_field(pos)",
        scalar_write="write_boolean_field(buffer, pos, {name})",
        scalar_read="read_boolean_field(data, pos)",
        itemsize_expr="cython.sizeof(cnp.npy_bool)",
        alignment=1,
        dtype_expr="np.bool_",
        pointer_expr="bool_sequence_ptr({name})",
    ),
    PrimitiveKind.UINT8: PrimitiveCodegenInfo(
        scalar_decl="uint8_t {name}",
        view_decl="const uint8_t[::1] {name}",
        scalar_advance="advance_uint8_field(pos, align_offset)",
        scalar_write="write_uint8_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint8_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(uint8_t)",
        alignment=1,
        dtype_expr="np.uint8",
        pointer_expr="uint8_view_ptr({name})",
    ),
    PrimitiveKind.INT8: PrimitiveCodegenInfo(
        scalar_decl="int8_t {name}",
        view_decl="const int8_t[::1] {name}",
        scalar_advance="advance_int8_field(pos, align_offset)",
        scalar_write="write_int8_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int8_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(int8_t)",
        alignment=1,
        dtype_expr="np.int8",
        pointer_expr="int8_sequence_ptr({name})",
    ),
    PrimitiveKind.INT16: PrimitiveCodegenInfo(
        scalar_decl="int16_t {name}",
        view_decl="const int16_t[::1] {name}",
        scalar_advance="advance_int16_field(pos, align_offset)",
        scalar_write="write_int16_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int16_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(int16_t)",
        alignment=2,
        dtype_expr="np.int16",
        pointer_expr="int16_sequence_ptr({name})",
    ),
    PrimitiveKind.UINT16: PrimitiveCodegenInfo(
        scalar_decl="uint16_t {name}",
        view_decl="const uint16_t[::1] {name}",
        scalar_advance="advance_uint16_field(pos, align_offset)",
        scalar_write="write_uint16_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint16_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(uint16_t)",
        alignment=2,
        dtype_expr="np.uint16",
        pointer_expr="uint16_view_ptr({name})",
    ),
    PrimitiveKind.INT32: PrimitiveCodegenInfo(
        scalar_decl="int32_t {name}",
        view_decl="const int32_t[::1] {name}",
        scalar_advance="advance_int32_field(pos, align_offset)",
        scalar_write="write_int32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int32_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(int32_t)",
        alignment=4,
        dtype_expr="np.int32",
        pointer_expr="int32_sequence_ptr({name})",
    ),
    PrimitiveKind.UINT32: PrimitiveCodegenInfo(
        scalar_decl="uint32_t {name}",
        view_decl="const uint32_t[::1] {name}",
        scalar_advance="advance_uint32_field(pos, align_offset)",
        scalar_write="write_uint32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint32_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(uint32_t)",
        alignment=4,
        dtype_expr="np.uint32",
        pointer_expr="uint32_view_ptr({name})",
    ),
    PrimitiveKind.INT64: PrimitiveCodegenInfo(
        scalar_decl="int64_t {name}",
        view_decl="const int64_t[::1] {name}",
        scalar_advance="advance_int64_field(pos, align_offset)",
        scalar_write="write_int64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int64_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(int64_t)",
        alignment=8,
        dtype_expr="np.int64",
        pointer_expr="int64_sequence_ptr({name})",
    ),
    PrimitiveKind.UINT64: PrimitiveCodegenInfo(
        scalar_decl="uint64_t {name}",
        view_decl="const uint64_t[::1] {name}",
        scalar_advance="advance_uint64_field(pos, align_offset)",
        scalar_write="write_uint64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint64_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(uint64_t)",
        alignment=8,
        dtype_expr="np.uint64",
        pointer_expr="uint64_view_ptr({name})",
    ),
    PrimitiveKind.FLOAT32: PrimitiveCodegenInfo(
        scalar_decl="cnp.float32_t {name}",
        view_decl="const cnp.float32_t[::1] {name}",
        scalar_advance="advance_float32_field(pos, align_offset)",
        scalar_write="write_float32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_float32_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(cnp.float32_t)",
        alignment=4,
        dtype_expr="np.float32",
        pointer_expr="float32_sequence_ptr({name})",
    ),
    PrimitiveKind.FLOAT64: PrimitiveCodegenInfo(
        scalar_decl="cnp.float64_t {name}",
        view_decl="const cnp.float64_t[::1] {name}",
        scalar_advance="advance_float64_field(pos, align_offset)",
        scalar_write="write_float64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_float64_field(data, pos, align_offset)",
        itemsize_expr="cython.sizeof(cnp.float64_t)",
        alignment=8,
        dtype_expr="np.float64",
        pointer_expr="float64_sequence_ptr({name})",
    ),
    PrimitiveKind.STRING: PrimitiveCodegenInfo(
        scalar_decl="bytes {name}",
        view_decl="list {name}",
        scalar_advance="advance_string_field(pos, {name}, align_offset)",
        scalar_write="write_string_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_string_field(data, pos, align_offset)",
        itemsize_expr="",
        alignment=4,
        dtype_expr="",
        pointer_expr="",
    ),
}


def flatten_cython_fields(
    fields: NestedSchemaFields,
    *,
    prefix: str = "",
) -> dict[str, FlatField]:
    """Flatten a nested schema into `snake_case` field names."""
    return flatten_schema_fields(fields, prefix=prefix)


def _field_declaration(field_name: str, field_spec: FlatField) -> str:
    info = PRIMITIVE_CODEGEN[field_spec.primitive_kind]
    if isinstance(field_spec, ScalarField):
        return info.scalar_decl.format(name=field_name)
    return info.view_decl.format(name=field_name)


def _signature_lines(fields: Mapping[str, FlatField]) -> str:
    return ",\n    ".join(
        _field_declaration(field_name, field_spec)
        for field_name, field_spec in fields.items()
    )


def _array_length_expr(field_name: str, primitive_kind: PrimitiveKind) -> str:
    if primitive_kind is PrimitiveKind.STRING:
        return f"PyList_GET_SIZE({field_name})"
    return f"{field_name}.shape[0]"


def _compute_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    info = PRIMITIVE_CODEGEN[field_spec.primitive_kind]

    if isinstance(field_spec, ScalarField):
        return [f"    pos = {info.scalar_advance.format(name=field_name)}"]

    if isinstance(field_spec, ArrayField):
        length_expr = _array_length_expr(field_name, field_spec.primitive_kind)
        lines = [
            f'    require_fixed_length({length_expr}, {field_spec.length}, "{field_name}")'
        ]
        if field_spec.primitive_kind is PrimitiveKind.STRING:
            lines.append(
                f"    pos = advance_string_array_position(pos, {field_name}, align_offset)"
            )
        else:
            lines.append(
                "    pos = advance_primitive_array_position("
                f"pos, {length_expr}, {info.itemsize_expr}, {info.alignment}, align_offset)"
            )
        return lines

    if field_spec.primitive_kind is PrimitiveKind.STRING:
        return [f"    pos = advance_string_sequence_position(pos, {field_name}, align_offset)"]

    return [
        "    pos = advance_primitive_sequence_position("
        f"pos, {field_name}.shape[0], {info.itemsize_expr}, {info.alignment}, align_offset)"
    ]


def _compute_body_lines(fields: Mapping[str, FlatField]) -> str:
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_compute_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _serialize_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    info = PRIMITIVE_CODEGEN[field_spec.primitive_kind]

    if isinstance(field_spec, ScalarField):
        return [f"    pos = {info.scalar_write.format(name=field_name)}"]

    if isinstance(field_spec, ArrayField):
        length_expr = _array_length_expr(field_name, field_spec.primitive_kind)
        lines = [
            f'    require_fixed_length({length_expr}, {field_spec.length}, "{field_name}")'
        ]
        if field_spec.primitive_kind is PrimitiveKind.STRING:
            lines.append(
                f"    pos = write_string_array(buffer, pos, {field_name}, align_offset)"
            )
        else:
            lines.append(
                "    pos = write_primitive_array("
                f"buffer, pos, {info.pointer_expr.format(name=field_name)}, "
                f"{length_expr}, {info.itemsize_expr}, {info.alignment}, align_offset)"
            )
        return lines

    if field_spec.primitive_kind is PrimitiveKind.STRING:
        return [f"    pos = write_string_sequence(buffer, pos, {field_name}, align_offset)"]

    return [
        "    pos = write_primitive_sequence("
        f"buffer, pos, {info.pointer_expr.format(name=field_name)}, "
        f"{field_name}.shape[0], {info.itemsize_expr}, {info.alignment}, align_offset)"
    ]


def _serialize_body_lines(fields: Mapping[str, FlatField]) -> str:
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_serialize_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _argument_names(fields: Mapping[str, FlatField], *, indent: int) -> str:
    padding = " " * indent
    return ",\n".join(f"{padding}{field_name}" for field_name in fields)


def _deserialize_declarations(fields: Mapping[str, FlatField]) -> str:
    return "\n".join(f"    cdef object {field_name}" for field_name in fields)


def _deserialize_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    info = PRIMITIVE_CODEGEN[field_spec.primitive_kind]

    if isinstance(field_spec, ScalarField):
        return [f"    {field_name}, pos = {info.scalar_read.format(name=field_name)}"]

    if isinstance(field_spec, ArrayField):
        if field_spec.primitive_kind is PrimitiveKind.STRING:
            return [
                f"    {field_name}, pos = read_string_array_object(data, pos, {field_spec.length}, align_offset)"
            ]
        return [
            "    "
            f"{field_name}, pos = read_primitive_array_object("
            f"data, pos, {field_spec.length}, {info.itemsize_expr}, {info.alignment}, "
            f"align_offset, {info.dtype_expr})"
        ]

    if field_spec.primitive_kind is PrimitiveKind.STRING:
        return [
            f"    {field_name}, pos = read_string_sequence_object(data, pos, align_offset)"
        ]

    return [
        "    "
        f"{field_name}, pos = read_primitive_sequence_object("
        f"data, pos, {info.itemsize_expr}, {info.alignment}, align_offset, {info.dtype_expr})"
    ]


def _deserialize_body_lines(fields: Mapping[str, FlatField]) -> str:
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_deserialize_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _return_tuple_lines(fields: Mapping[str, FlatField]) -> str:
    if not fields:
        return "    return ()"

    values = ",\n".join(f"        {field_name}" for field_name in fields)
    return f"    return (\n{values},\n    )"


def generate_cython_serializer_code(
    serializer_name: str,
    schema: NestedSchemaFields,
) -> str:
    """Return generated `.pyx` source for a serializer/deserializer pair."""
    return generate_cython_serializer_module(
        serializer_name,
        flatten_cython_fields(schema),
    )


def generate_cython_serializer_module(
    serializer_name: str,
    fields: Mapping[str, FlatField],
) -> str:
    """Low-level generator for already-flattened fields."""
    signature = _signature_lines(fields)
    compute_body = _compute_body_lines(fields)
    serialize_body = _serialize_body_lines(fields)
    call_args = _argument_names(fields, indent=8)
    deserialize_declarations = _deserialize_declarations(fields)
    deserialize_body = _deserialize_body_lines(fields)
    return_tuple = _return_tuple_lines(fields)

    return f'''# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

from cpython.bytearray cimport PyByteArray_AS_STRING
from cpython.list cimport PyList_GET_SIZE
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

import numpy as np

cimport cython
cimport numpy as cnp

from xcdrjit._every_supported_cython cimport (
    ENCAPSULATION_HEADER_SIZE,
    advance_boolean_field,
    advance_float32_field,
    advance_float64_field,
    advance_int16_field,
    advance_int32_field,
    advance_int64_field,
    advance_int8_field,
    advance_primitive_array_position,
    advance_primitive_sequence_position,
    advance_string_array_position,
    advance_string_field,
    advance_string_sequence_position,
    advance_uint16_field,
    advance_uint32_field,
    advance_uint64_field,
    advance_uint8_field,
    bool_sequence_ptr,
    float32_sequence_ptr,
    float64_sequence_ptr,
    int16_sequence_ptr,
    int32_sequence_ptr,
    int64_sequence_ptr,
    int8_sequence_ptr,
    read_boolean_field,
    read_float32_field,
    read_float64_field,
    read_int16_field,
    read_int32_field,
    read_int64_field,
    read_int8_field,
    read_primitive_array_object,
    read_primitive_sequence_object,
    read_string_array_object,
    read_string_field,
    read_string_sequence_object,
    read_uint16_field,
    read_uint32_field,
    read_uint64_field,
    read_uint8_field,
    require_consumed,
    require_fixed_length,
    uint16_view_ptr,
    uint32_view_ptr,
    uint64_view_ptr,
    uint8_view_ptr,
    validate_encapsulation_header,
    write_boolean_field,
    write_encapsulation_header,
    write_float32_field,
    write_float64_field,
    write_int16_field,
    write_int32_field,
    write_int64_field,
    write_int8_field,
    write_primitive_array,
    write_primitive_sequence,
    write_string_array,
    write_string_field,
    write_string_sequence,
    write_uint16_field,
    write_uint32_field,
    write_uint64_field,
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


cpdef tuple deserialize_{serializer_name}(const unsigned char[::1] data):
    cdef Py_ssize_t pos = validate_encapsulation_header(data)
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE
{deserialize_declarations}

{deserialize_body}
    require_consumed(data, pos)
{return_tuple}
'''


__all__ = [
    "NestedCythonFields",
    "flatten_cython_fields",
    "generate_cython_serializer_code",
    "generate_cython_serializer_module",
]
