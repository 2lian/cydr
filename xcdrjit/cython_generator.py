"""Helpers for generating Cython codec modules from nested Python schemas."""

from collections.abc import Mapping
from dataclasses import dataclass

from .schema_types import (
    ArrayType,
    FlatField,
    NestedSchemaFields,
    SequenceType,
    boolean,
    float32,
    float64,
    flatten_schema_fields,
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


@dataclass(frozen=True, slots=True)
class PrimitiveCodegenInfo:
    """Cython code snippets for one primitive schema type.

    Each field is a format string or plain string fragment used by the code
    generators.  ``{name}`` is replaced with the Cython argument name;
    ``{pos}`` in ``scalar_read`` is replaced with ``&pos`` (pointer-advance
    API).  ``itemsize_expr``, ``dtype_expr``, and ``pointer_expr`` are empty
    strings for the ``string`` type where they do not apply.
    """

    scalar_decl: str
    view_decl: str
    scalar_advance: str
    scalar_write: str
    scalar_read: str
    itemsize_expr: str
    alignment: int
    dtype_expr: str
    pointer_expr: str


PRIMITIVE_CODEGEN: dict[object, PrimitiveCodegenInfo] = {
    boolean: PrimitiveCodegenInfo(
        scalar_decl="bint {name}",
        view_decl="const cnp.npy_bool[::1] {name}",
        scalar_advance="advance_boolean_field(pos)",
        scalar_write="write_boolean_field(buffer, pos, {name})",
        scalar_read="read_boolean_field(data, {pos})",
        itemsize_expr="cython.sizeof(cnp.npy_bool)",
        alignment=1,
        dtype_expr="np.bool_",
        pointer_expr="bool_sequence_ptr({name})",
    ),
    uint8: PrimitiveCodegenInfo(
        scalar_decl="uint8_t {name}",
        view_decl="const uint8_t[::1] {name}",
        scalar_advance="advance_uint8_field(pos, align_offset)",
        scalar_write="write_uint8_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint8_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(uint8_t)",
        alignment=1,
        dtype_expr="np.uint8",
        pointer_expr="uint8_view_ptr({name})",
    ),
    int8: PrimitiveCodegenInfo(
        scalar_decl="int8_t {name}",
        view_decl="const int8_t[::1] {name}",
        scalar_advance="advance_int8_field(pos, align_offset)",
        scalar_write="write_int8_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int8_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(int8_t)",
        alignment=1,
        dtype_expr="np.int8",
        pointer_expr="int8_sequence_ptr({name})",
    ),
    int16: PrimitiveCodegenInfo(
        scalar_decl="int16_t {name}",
        view_decl="const int16_t[::1] {name}",
        scalar_advance="advance_int16_field(pos, align_offset)",
        scalar_write="write_int16_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int16_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(int16_t)",
        alignment=2,
        dtype_expr="np.int16",
        pointer_expr="int16_sequence_ptr({name})",
    ),
    uint16: PrimitiveCodegenInfo(
        scalar_decl="uint16_t {name}",
        view_decl="const uint16_t[::1] {name}",
        scalar_advance="advance_uint16_field(pos, align_offset)",
        scalar_write="write_uint16_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint16_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(uint16_t)",
        alignment=2,
        dtype_expr="np.uint16",
        pointer_expr="uint16_view_ptr({name})",
    ),
    int32: PrimitiveCodegenInfo(
        scalar_decl="int32_t {name}",
        view_decl="const int32_t[::1] {name}",
        scalar_advance="advance_int32_field(pos, align_offset)",
        scalar_write="write_int32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int32_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(int32_t)",
        alignment=4,
        dtype_expr="np.int32",
        pointer_expr="int32_sequence_ptr({name})",
    ),
    uint32: PrimitiveCodegenInfo(
        scalar_decl="uint32_t {name}",
        view_decl="const uint32_t[::1] {name}",
        scalar_advance="advance_uint32_field(pos, align_offset)",
        scalar_write="write_uint32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint32_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(uint32_t)",
        alignment=4,
        dtype_expr="np.uint32",
        pointer_expr="uint32_view_ptr({name})",
    ),
    int64: PrimitiveCodegenInfo(
        scalar_decl="int64_t {name}",
        view_decl="const int64_t[::1] {name}",
        scalar_advance="advance_int64_field(pos, align_offset)",
        scalar_write="write_int64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_int64_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(int64_t)",
        alignment=8,
        dtype_expr="np.int64",
        pointer_expr="int64_sequence_ptr({name})",
    ),
    uint64: PrimitiveCodegenInfo(
        scalar_decl="uint64_t {name}",
        view_decl="const uint64_t[::1] {name}",
        scalar_advance="advance_uint64_field(pos, align_offset)",
        scalar_write="write_uint64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_uint64_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(uint64_t)",
        alignment=8,
        dtype_expr="np.uint64",
        pointer_expr="uint64_view_ptr({name})",
    ),
    float32: PrimitiveCodegenInfo(
        scalar_decl="cnp.float32_t {name}",
        view_decl="const cnp.float32_t[::1] {name}",
        scalar_advance="advance_float32_field(pos, align_offset)",
        scalar_write="write_float32_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_float32_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(cnp.float32_t)",
        alignment=4,
        dtype_expr="np.float32",
        pointer_expr="float32_sequence_ptr({name})",
    ),
    float64: PrimitiveCodegenInfo(
        scalar_decl="cnp.float64_t {name}",
        view_decl="const cnp.float64_t[::1] {name}",
        scalar_advance="advance_float64_field(pos, align_offset)",
        scalar_write="write_float64_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_float64_field(data, {pos}, align_offset)",
        itemsize_expr="cython.sizeof(cnp.float64_t)",
        alignment=8,
        dtype_expr="np.float64",
        pointer_expr="float64_sequence_ptr({name})",
    ),
    string: PrimitiveCodegenInfo(
        scalar_decl="bytes {name}",
        view_decl="list {name}",
        scalar_advance="advance_string_field(pos, {name}, align_offset)",
        scalar_write="write_string_field(buffer, pos, {name}, align_offset)",
        scalar_read="read_string_field(data, {pos}, align_offset)",
        itemsize_expr="",
        alignment=4,
        dtype_expr="",
        pointer_expr="",
    ),
}


def _field_info(field_spec: FlatField) -> tuple[object, PrimitiveCodegenInfo]:
    """Return the primitive type and its codegen info for one field spec.

    Args:
        field_spec: One flat field schema — a primitive token, ``ArrayType``,
            or ``SequenceType``.

    Returns:
        A ``(primitive_type, PrimitiveCodegenInfo)`` pair.  For collection
        specs the primitive type is the element type; for scalars it is the
        field spec itself.
    """
    primitive_type = (
        field_spec.element_type
        if isinstance(field_spec, (ArrayType, SequenceType))
        else field_spec
    )
    return primitive_type, PRIMITIVE_CODEGEN[primitive_type]


def _field_decl(field_name: str, field_spec: FlatField) -> str:
    """Return the Cython argument declaration for one field.

    Args:
        field_name: The Cython argument name for this field.
        field_spec: One flat field schema.

    Returns:
        A Cython type declaration string, e.g. ``"int32_t arg_0"`` for a
        scalar or ``"const int32_t[::1] arg_0"`` for an array/sequence.
    """
    _, info = _field_info(field_spec)
    if isinstance(field_spec, (ArrayType, SequenceType)):
        return info.view_decl.format(name=field_name)
    return info.scalar_decl.format(name=field_name)


def _signature_block(fields: Mapping[str, FlatField]) -> str:
    """Return the comma-separated argument declarations for a generated function.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A multi-line string of Cython argument declarations, ready to be
        placed between the parentheses of a ``cpdef`` function signature.
    """
    return ",\n    ".join(
        _field_decl(field_name, field_spec)
        for field_name, field_spec in fields.items()
    )


def _field_length_expr(field_name: str, primitive_type: object) -> str:
    """Return the Cython expression for the runtime length of one collection field.

    Args:
        field_name: The Cython argument name for this field.
        primitive_type: The element type of the collection.

    Returns:
        A Cython expression string that evaluates to the number of elements:
        ``"PyList_GET_SIZE(arg_0)"`` for string lists or ``"arg_0.shape[0]"``
        for NumPy views.
    """
    if primitive_type is string:
        return f"PyList_GET_SIZE({field_name})"
    return f"{field_name}.shape[0]"


def _size_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    """Return the size-advance Cython lines for one field.

    Args:
        field_name: The Cython argument name for this field.
        field_spec: One flat field schema.

    Returns:
        A list of Cython source lines (indented four spaces) that advance
        ``pos`` by the serialized byte count of this field.
    """
    primitive_type, info = _field_info(field_spec)

    if not isinstance(field_spec, (ArrayType, SequenceType)):
        return [f"    pos = {info.scalar_advance.format(name=field_name)}"]

    if isinstance(field_spec, ArrayType):
        length_expr = _field_length_expr(field_name, field_spec.element_type)
        lines = [
            f'    require_fixed_length({length_expr}, {field_spec.length}, "{field_name}")'
        ]
        if field_spec.element_type is string:
            lines.append(
                f"    pos = advance_string_array_position(pos, {field_name}, align_offset)"
            )
        else:
            lines.append(
                "    pos = advance_primitive_array_position("
                f"pos, {length_expr}, {info.itemsize_expr}, {info.alignment}, align_offset)"
            )
        return lines

    if primitive_type is string:
        return [f"    pos = advance_string_sequence_position(pos, {field_name}, align_offset)"]

    return [
        "    pos = advance_primitive_sequence_position("
        f"pos, {field_name}.shape[0], {info.itemsize_expr}, {info.alignment}, align_offset)"
    ]


def _size_body(fields: Mapping[str, FlatField]) -> str:
    """Return the full size-advance body for all fields.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A multi-line Cython source string with ``pos = ...`` advance
        statements for every field, ready to be placed inside
        ``compute_serialized_size_<hash>``.
    """
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_size_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _serialize_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    """Return the write Cython lines for one field.

    Args:
        field_name: The Cython argument name for this field.
        field_spec: One flat field schema.

    Returns:
        A list of Cython source lines (indented four spaces) that write
        this field into ``buffer`` and advance ``pos``.
    """
    primitive_type, info = _field_info(field_spec)

    if not isinstance(field_spec, (ArrayType, SequenceType)):
        return [f"    pos = {info.scalar_write.format(name=field_name)}"]

    if isinstance(field_spec, ArrayType):
        length_expr = _field_length_expr(field_name, field_spec.element_type)
        lines = [
            f'    require_fixed_length({length_expr}, {field_spec.length}, "{field_name}")'
        ]
        if field_spec.element_type is string:
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

    if primitive_type is string:
        return [f"    pos = write_string_sequence(buffer, pos, {field_name}, align_offset)"]

    return [
        "    pos = write_primitive_sequence("
        f"buffer, pos, {info.pointer_expr.format(name=field_name)}, "
        f"{field_name}.shape[0], {info.itemsize_expr}, {info.alignment}, align_offset)"
    ]


def _serialize_body(fields: Mapping[str, FlatField]) -> str:
    """Return the full write body for all fields.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A multi-line Cython source string with write statements for every
        field, ready to be placed inside ``serialize_<hash>``.
    """
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_serialize_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _argument_block(fields: Mapping[str, FlatField], indent: int) -> str:
    """Return the argument forwarding block used inside the serialize function.

    Args:
        fields: Ordered mapping from argument name to field schema.
        indent: Number of spaces to prepend to each argument name.

    Returns:
        A multi-line string of argument names (one per line, indented) for
        forwarding all arguments from ``serialize_<hash>`` to
        ``compute_serialized_size_<hash>``.
    """
    padding = " " * indent
    return ",\n".join(f"{padding}{field_name}" for field_name in fields)


def _deserialize_decl_block(fields: Mapping[str, FlatField]) -> str:
    """Return the ``cdef object`` variable declarations for the deserialize function.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A multi-line string of ``cdef object <name>`` declarations, one per
        field, ready to be placed at the top of ``deserialize_<hash>``.
    """
    return "\n".join(f"    cdef object {field_name}" for field_name in fields)


def _deserialize_lines_for_field(field_name: str, field_spec: FlatField) -> list[str]:
    """Return the read Cython lines for one field in the deserialize function.

    Args:
        field_name: The Cython argument name for this field.
        field_spec: One flat field schema.

    Returns:
        A list of Cython source lines (indented four spaces) that read this
        field from ``data``, advance ``pos`` via pointer, and assign the
        decoded value to ``field_name``.
    """
    primitive_type, info = _field_info(field_spec)

    if not isinstance(field_spec, (ArrayType, SequenceType)):
        # scalar_read uses {pos} as a placeholder; pass &pos for the pointer-advance API.
        return [f"    {field_name} = {info.scalar_read.format(pos='&pos')}"]

    if isinstance(field_spec, ArrayType):
        if field_spec.element_type is string:
            return [
                f"    {field_name} = read_string_array_object(data, &pos, {field_spec.length}, align_offset)"
            ]
        return [
            "    "
            f"{field_name} = read_primitive_array_object("
            f"data, &pos, {field_spec.length}, {info.itemsize_expr}, {info.alignment}, "
            f"align_offset, {info.dtype_expr})"
        ]

    if primitive_type is string:
        return [
            f"    {field_name} = read_string_sequence_object(data, &pos, align_offset)"
        ]

    return [
        "    "
        f"{field_name} = read_primitive_sequence_object("
        f"data, &pos, {info.itemsize_expr}, {info.alignment}, align_offset, {info.dtype_expr})"
    ]


def _deserialize_body(fields: Mapping[str, FlatField]) -> str:
    """Return the full read body for all fields.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A multi-line Cython source string with read-and-assign statements
        for every field, ready to be placed inside ``deserialize_<hash>``.
    """
    lines: list[str] = []
    for field_name, field_spec in fields.items():
        lines.extend(_deserialize_lines_for_field(field_name, field_spec))
    return "\n".join(lines)


def _return_tuple_block(fields: Mapping[str, FlatField]) -> str:
    """Return the ``return (...)`` statement for the deserialize function.

    Args:
        fields: Ordered mapping from argument name to field schema.

    Returns:
        A Cython ``return (...)`` statement listing all decoded field
        variables in schema order.
    """
    if not fields:
        return "    return ()"

    values = ",\n".join(f"        {field_name}" for field_name in fields)
    return f"    return (\n{values},\n    )"


def generate_cython_codec_source(
    serializer_name: str,
    schema: NestedSchemaFields,
) -> str:
    """Return generated ``.pyx`` source for a serializer/deserializer pair.

    Args:
        serializer_name: The name suffix used for all generated functions.
        schema: A nested schema mapping as accepted by ``get_codec_for``.

    Returns:
        A complete ``.pyx`` source string containing
        ``compute_serialized_size_<name>``, ``serialize_<name>``, and
        ``deserialize_<name>``.
    """
    return generate_cython_codec_module_source(
        serializer_name,
        flatten_schema_fields(schema),
    )


def generate_cython_codec_module_source(
    serializer_name: str,
    fields: Mapping[str, FlatField],
) -> str:
    """Low-level generator for already-flattened fields.

    Args:
        serializer_name: The name suffix used for all generated functions.
        fields: An ordered mapping from canonical argument name (``arg_0``,
            ``arg_1``, …) to flat field schema.

    Returns:
        A complete ``.pyx`` source string containing
        ``compute_serialized_size_<name>``, ``serialize_<name>``, and
        ``deserialize_<name>``.
    """
    signature = _signature_block(fields)
    compute_body = _size_body(fields)
    serialize_body = _serialize_body(fields)
    call_args = _argument_block(fields, 8)
    deserialize_declarations = _deserialize_decl_block(fields)
    deserialize_body = _deserialize_body(fields)
    return_tuple = _return_tuple_block(fields)

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
