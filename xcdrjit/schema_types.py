"""Schema descriptor helpers used by xcdrjit code generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np


# Public primitive tokens. These are intentionally simple values so schema
# definitions stay close to NumPy/Python types.
#: XCDR boolean field. Runtime values are regular Python ``bool`` scalars.
boolean: TypeAlias = np.bool_
#: XCDR byte field. Runtime values are 8-bit unsigned integers.
byte: TypeAlias = np.uint8
#: XCDR signed 8-bit integer field.
int8: TypeAlias = np.int8
#: XCDR unsigned 8-bit integer field.
uint8: TypeAlias = np.uint8
#: XCDR signed 16-bit integer field.
int16: TypeAlias = np.int16
#: XCDR unsigned 16-bit integer field.
uint16: TypeAlias = np.uint16
#: XCDR signed 32-bit integer field.
int32: TypeAlias = np.int32
#: XCDR unsigned 32-bit integer field.
uint32: TypeAlias = np.uint32
#: XCDR signed 64-bit integer field.
int64: TypeAlias = np.int64
#: XCDR unsigned 64-bit integer field.
uint64: TypeAlias = np.uint64
#: XCDR 32-bit floating-point field.
float32: TypeAlias = np.float32
#: XCDR 64-bit floating-point field.
float64: TypeAlias = np.float64
#: XCDR string field. Runtime values are UTF-8 ``bytes``.
string: TypeAlias = bytes


@dataclass(frozen=True, slots=True)
class ArrayType:
    """Fixed-size array schema descriptor.

    The runtime value must be a 1D NumPy array for numeric/boolean element
    types, or a ``list[bytes]`` for ``string``.
    """

    element_type: object
    length: int


@dataclass(frozen=True, slots=True)
class SequenceType:
    """Variable-size sequence schema descriptor.

    The runtime value must be a 1D NumPy array for numeric/boolean element
    types, or a ``list[bytes]`` for ``string``.
    """

    element_type: object


def array(element_type: object, length: int) -> ArrayType:
    """Return a fixed-size array schema descriptor.

    ``element_type`` must be one of the exported primitive schema tokens.
    ``length`` is exact, not a minimum.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError("array length must be a non-negative integer.")
    _primitive_kind_from_public_type(element_type)
    return ArrayType(element_type=element_type, length=length)


def sequence(element_type: object) -> SequenceType:
    """Return a variable-size sequence schema descriptor.

    ``element_type`` must be one of the exported primitive schema tokens.
    """
    _primitive_kind_from_public_type(element_type)
    return SequenceType(element_type=element_type)


class PrimitiveKind(StrEnum):
    BOOLEAN = "boolean"
    UINT8 = "uint8"
    INT8 = "int8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"


@dataclass(frozen=True, slots=True)
class ScalarField:
    primitive_kind: PrimitiveKind

    @property
    def cache_token(self) -> str:
        return self.primitive_kind.value


@dataclass(frozen=True, slots=True)
class ArrayField:
    primitive_kind: PrimitiveKind
    length: int

    @property
    def cache_token(self) -> str:
        return f"array:{self.primitive_kind.value}:{self.length}"


@dataclass(frozen=True, slots=True)
class SequenceField:
    primitive_kind: PrimitiveKind

    @property
    def cache_token(self) -> str:
        return f"sequence:{self.primitive_kind.value}"


FlatField = ScalarField | ArrayField | SequenceField
NestedSchemaFieldValue = object | Mapping[str, object]
NestedSchemaFields = Mapping[str, NestedSchemaFieldValue]


_PRIMITIVE_KIND_BY_PUBLIC_TYPE: dict[object, PrimitiveKind] = {
    boolean: PrimitiveKind.BOOLEAN,
    byte: PrimitiveKind.UINT8,
    int8: PrimitiveKind.INT8,
    uint8: PrimitiveKind.UINT8,
    int16: PrimitiveKind.INT16,
    uint16: PrimitiveKind.UINT16,
    int32: PrimitiveKind.INT32,
    uint32: PrimitiveKind.UINT32,
    int64: PrimitiveKind.INT64,
    uint64: PrimitiveKind.UINT64,
    float32: PrimitiveKind.FLOAT32,
    float64: PrimitiveKind.FLOAT64,
    string: PrimitiveKind.STRING,
}

_LEGACY_FIELD_BY_VALUE: dict[str, FlatField] = {
    "boolean": ScalarField(PrimitiveKind.BOOLEAN),
    "byte": ScalarField(PrimitiveKind.UINT8),
    "int8": ScalarField(PrimitiveKind.INT8),
    "uint8": ScalarField(PrimitiveKind.UINT8),
    "int16": ScalarField(PrimitiveKind.INT16),
    "uint16": ScalarField(PrimitiveKind.UINT16),
    "int32": ScalarField(PrimitiveKind.INT32),
    "uint32": ScalarField(PrimitiveKind.UINT32),
    "int64": ScalarField(PrimitiveKind.INT64),
    "uint64": ScalarField(PrimitiveKind.UINT64),
    "float32": ScalarField(PrimitiveKind.FLOAT32),
    "float64": ScalarField(PrimitiveKind.FLOAT64),
    "string": ScalarField(PrimitiveKind.STRING),
    "bool_sequence": SequenceField(PrimitiveKind.BOOLEAN),
    "byte_array_3": ArrayField(PrimitiveKind.UINT8, 3),
    "int8_sequence": SequenceField(PrimitiveKind.INT8),
    "uint8_array_3": ArrayField(PrimitiveKind.UINT8, 3),
    "int16_sequence": SequenceField(PrimitiveKind.INT16),
    "uint16_array_2": ArrayField(PrimitiveKind.UINT16, 2),
    "int32_sequence": SequenceField(PrimitiveKind.INT32),
    "uint32_array_2": ArrayField(PrimitiveKind.UINT32, 2),
    "int64_sequence": SequenceField(PrimitiveKind.INT64),
    "uint64_array_2": ArrayField(PrimitiveKind.UINT64, 2),
    "float32_sequence": SequenceField(PrimitiveKind.FLOAT32),
    "float64_sequence": SequenceField(PrimitiveKind.FLOAT64),
    "float64_array_2": ArrayField(PrimitiveKind.FLOAT64, 2),
    "text_array_2": ArrayField(PrimitiveKind.STRING, 2),
    "text_sequence": SequenceField(PrimitiveKind.STRING),
}


def _primitive_kind_from_public_type(element_type: object) -> PrimitiveKind:
    try:
        return _PRIMITIVE_KIND_BY_PUBLIC_TYPE[element_type]
    except KeyError as exc:
        raise TypeError(
            "Unsupported primitive schema type. "
            "Use one of: boolean, byte, int8, uint8, int16, uint16, "
            "int32, uint32, int64, uint64, float32, float64, string."
        ) from exc


def normalize_schema_field(field_value: object) -> FlatField:
    """Normalize one leaf schema value into an internal flat field descriptor."""
    if isinstance(field_value, ArrayType):
        return ArrayField(
            primitive_kind=_primitive_kind_from_public_type(field_value.element_type),
            length=field_value.length,
        )
    if isinstance(field_value, SequenceType):
        return SequenceField(
            primitive_kind=_primitive_kind_from_public_type(field_value.element_type),
        )
    if field_value in _PRIMITIVE_KIND_BY_PUBLIC_TYPE:
        return ScalarField(_primitive_kind_from_public_type(field_value))
    if isinstance(field_value, str) and field_value in _LEGACY_FIELD_BY_VALUE:
        return _LEGACY_FIELD_BY_VALUE[field_value]
    raise TypeError(
        f"Unsupported field schema value: {field_value!r}. "
        "Use a primitive type token, array(...), sequence(...), or a nested dict."
    )


def flatten_schema_fields(
    fields: NestedSchemaFields,
    *,
    prefix: str = "",
) -> dict[str, FlatField]:
    """Flatten a nested schema into `snake_case` field names."""
    flattened: dict[str, FlatField] = {}
    for field_name, field_value in fields.items():
        full_name = f"{prefix}_{field_name}" if prefix else field_name
        if isinstance(field_value, Mapping):
            flattened.update(flatten_schema_fields(field_value, prefix=full_name))
        else:
            flattened[full_name] = normalize_schema_field(field_value)
    return flattened


__all__ = [
    "ArrayField",
    "ArrayType",
    "FlatField",
    "NestedSchemaFields",
    "PrimitiveKind",
    "ScalarField",
    "SequenceField",
    "SequenceType",
    "array",
    "boolean",
    "byte",
    "flatten_schema_fields",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "normalize_schema_field",
    "sequence",
    "string",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
