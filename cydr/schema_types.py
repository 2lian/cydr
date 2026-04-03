"""Internal schema traversal and normalization helpers."""

import typing
from collections.abc import Mapping
from typing import TypeAlias, Union

import numpy as np
from numpy._typing import NDArray
from nptyping.nptyping_type import NPTypingType
from nptyping.shape import ShapeMeta

from .types import (
    PrimitiveSchemaType,
    boolean,
    byte,
    float32,
    float64,
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

# ---------------------------------------------------------------------------
# Internal primitive token map and validator
# ---------------------------------------------------------------------------

_TOKEN_BY_TYPE: dict[object, str] = {
    boolean: "boolean",
    byte:    "uint8",
    int8:    "int8",
    uint8:   "uint8",
    int16:   "int16",
    uint16:  "uint16",
    int32:   "int32",
    uint32:  "uint32",
    int64:   "int64",
    uint64:  "uint64",
    float32: "float32",
    float64: "float64",
    string:  "string",
}


def _normalize_primitive_type(element_type: type) -> PrimitiveSchemaType:
    """Validate and return one public primitive schema token.

    Raises:
        TypeError: If ``element_type`` is not one of the supported primitive
            schema tokens.
    """
    if element_type in _TOKEN_BY_TYPE:
        return element_type
    raise TypeError(
        "Unsupported primitive schema type. "
        "Use one of: boolean, byte, int8, uint8, int16, uint16, "
        "int32, uint32, int64, uint64, float32, float64, string."
    )


# ---------------------------------------------------------------------------
# NDArray annotation helpers
# ---------------------------------------------------------------------------

def _is_ndarray_annotation(x: object) -> bool:
    """Return True if *x* is a parameterised nptyping NDArray annotation."""
    return isinstance(x, type) and NPTypingType in x.__mro__


def _ndarray_element_type(annotation: object) -> PrimitiveSchemaType:
    """Return the validated primitive element token for one NDArray annotation."""
    _, dtype_type = annotation.__args__
    if dtype_type is np.bytes_:
        dtype_type = string
    return _normalize_primitive_type(dtype_type)


def _ndarray_fixed_length(annotation: object) -> int | None:
    """Return the fixed length for a fixed-size NDArray, or None for a sequence."""
    shape_arg, _ = annotation.__args__
    if shape_arg is typing.Any:
        return None
    if not isinstance(shape_arg, ShapeMeta):
        return None
    shape_str = (getattr(shape_arg, "__args__", None) or (None,))[0]
    if shape_str is None or shape_str == "*":
        return None
    return int(shape_str)


# ---------------------------------------------------------------------------
# Flat field types
# ---------------------------------------------------------------------------

FlatField: TypeAlias = PrimitiveSchemaType | NDArray
NestedSchemaFields: TypeAlias = Mapping[
    str,
    Union[PrimitiveSchemaType, NDArray, "NestedSchemaFields"],
]


def _primitive_token(element_type: PrimitiveSchemaType) -> str:
    """Return the stable internal token string for one primitive schema type."""
    return _TOKEN_BY_TYPE[element_type]


def field_schema_token(field_schema: FlatField) -> str:
    """Return the stable cache token for one flattened field schema.

    Args:
        field_schema: One flattened leaf schema — a primitive schema token or
            an NDArray annotation (sequence or fixed array).

    Returns:
        A stable token string used in schema hashing and code generation cache
        keys.
    """
    if _is_ndarray_annotation(field_schema):
        elem = _ndarray_element_type(field_schema)
        length = _ndarray_fixed_length(field_schema)
        if length is not None:
            return f"array:{_primitive_token(elem)}:{length}"
        return f"sequence:{_primitive_token(elem)}"
    return _primitive_token(field_schema)


def normalize_field_schema(field_schema: object) -> FlatField:
    """Normalize one leaf schema value into one canonical field schema.

    Args:
        field_schema: One non-nested schema leaf — a primitive token or an
            NDArray annotation (``NDArray[Any, dtype]`` or
            ``NDArray[Shape['n'], dtype]``).

    Returns:
        The same value, validated.

    Raises:
        TypeError: If ``field_schema`` is not a supported leaf schema value.
    """
    if _is_ndarray_annotation(field_schema):
        _ndarray_element_type(field_schema)  # validates element type
        shape_arg = field_schema.__args__[0]
        if not (shape_arg is typing.Any or isinstance(shape_arg, ShapeMeta)):
            raise TypeError(
                f"NDArray shape must be Any or Shape['n'] / Shape['*'], got {shape_arg!r}."
            )
        if isinstance(shape_arg, ShapeMeta):
            shape_str = (getattr(shape_arg, "__args__", None) or (None,))[0]
            if shape_str and "," in shape_str:
                raise TypeError(
                    f"cydr only supports 1-D arrays. Got multi-dimensional Shape['{shape_str}']."
                )
        return field_schema
    if field_schema in _TOKEN_BY_TYPE:
        return field_schema
    raise TypeError(
        f"Unsupported field schema value: {field_schema!r}. "
        "Use a primitive type token, NDArray[Any, dtype] for sequences, "
        "NDArray[Shape['n'], dtype] for fixed arrays, or a nested dict."
    )


def flatten_schema_fields(
    fields: NestedSchemaFields,
    prefix: str = "",
) -> dict[str, FlatField]:
    """Flatten one nested schema mapping into flat ``snake_case`` field names.

    Args:
        fields: A nested schema mapping. Values may be primitive schema tokens,
            NDArray annotations, or nested schema mappings.
        prefix: Optional field-name prefix used during recursive flattening.

    Returns:
        A flat mapping from flattened field name to normalized internal field
        descriptor. Nested fields are joined with underscores, so
        ``{"header": {"stamp": {"sec": int32}}}`` becomes
        ``{"header_stamp_sec": int32}``.

    Raises:
        TypeError: If ``fields`` is not a mapping, if a field name is not a
            string, or if one leaf schema value is unsupported.
    """
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"Schema must be a mapping from field name to field schema, got {type(fields)!r}."
        )

    flattened: dict[str, FlatField] = {}
    for field_name, field_value in fields.items():
        if not isinstance(field_name, str):
            raise TypeError(f"Schema field names must be strings, got {field_name!r}.")
        full_name = f"{prefix}_{field_name}" if prefix else field_name
        if isinstance(field_value, Mapping):
            flattened.update(flatten_schema_fields(field_value, prefix=full_name))
        else:
            flattened[full_name] = normalize_field_schema(field_value)
    return flattened
