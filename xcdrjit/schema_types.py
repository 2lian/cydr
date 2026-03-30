"""Schema descriptor helpers used by xcdrjit code generation."""

from collections.abc import Mapping
from dataclasses import dataclass
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

type PrimitiveSchemaType = (
    type[boolean]
    | type[uint8]
    | type[int8]
    | type[int16]
    | type[uint16]
    | type[int32]
    | type[uint32]
    | type[int64]
    | type[uint64]
    | type[float32]
    | type[float64]
    | type[bytes]
)


@dataclass(frozen=True, slots=True)
class ArrayType:
    """Fixed-size array schema descriptor.

    The runtime value must be a 1D NumPy array for numeric/boolean element
    types, or a ``list[bytes]`` for ``string``.
    """

    element_type: PrimitiveSchemaType
    length: int


@dataclass(frozen=True, slots=True)
class SequenceType:
    """Variable-size sequence schema descriptor.

    The runtime value must be a 1D NumPy array for numeric/boolean element
    types, or a ``list[bytes]`` for ``string``.
    """

    element_type: PrimitiveSchemaType



type FlatField = PrimitiveSchemaType | ArrayType | SequenceType
type NestedSchemaFields = Mapping[
    str,
    PrimitiveSchemaType | ArrayType | SequenceType | NestedSchemaFields,
]


_TOKEN_BY_TYPE: dict[PrimitiveSchemaType, str] = {
    boolean: "boolean",
    byte: "uint8",
    int8: "int8",
    uint8: "uint8",
    int16: "int16",
    uint16: "uint16",
    int32: "int32",
    uint32: "uint32",
    int64: "int64",
    uint64: "uint64",
    float32: "float32",
    float64: "float64",
    string: "string",
}


def _normalize_primitive_type(element_type: type) -> PrimitiveSchemaType:
    """Validate and return one public primitive schema token.

    Args:
        element_type: One of the exported primitive schema tokens such as
            ``int32``, ``float64``, or ``string``.

    Returns:
        The same token, narrowed to ``PrimitiveSchemaType`` for internal use.

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


def _primitive_token(element_type: PrimitiveSchemaType) -> str:
    """Return the stable internal token string for one primitive schema type.

    Args:
        element_type: One validated primitive schema token.

    Returns:
        A short stable token string used for internal cache keys and code
        generation decisions, such as ``"int32"`` or ``"string"``.
    """
    return _TOKEN_BY_TYPE[element_type]


def field_schema_token(field_schema: FlatField) -> str:
    """Return the stable cache token for one flattened field schema.

    Args:
        field_schema: One flattened leaf schema. This may be one primitive
            schema token, one ``ArrayType``, or one ``SequenceType``.

    Returns:
        A stable token string used in schema hashing and code generation cache
        keys.
    """
    if isinstance(field_schema, ArrayType):
        return f"array:{_primitive_token(field_schema.element_type)}:{field_schema.length}"
    if isinstance(field_schema, SequenceType):
        return f"sequence:{_primitive_token(field_schema.element_type)}"
    return _primitive_token(field_schema)


def normalize_field_schema(field_schema: object) -> FlatField:
    """Normalize one leaf schema value into one canonical field schema.

    Args:
        field_schema: One non-nested schema leaf. This may be:
            - one primitive schema token such as ``int32`` or ``string``
            - ``array(element_type, length)``
            - ``sequence(element_type)``

    Returns:
        A canonical flattened field schema:
            - one primitive schema token for scalar leaves
            - one ``ArrayType`` for fixed-size arrays
            - one ``SequenceType`` for variable-size sequences

    Raises:
        TypeError: If ``field_schema`` is not a supported leaf schema value.
    """
    if isinstance(field_schema, ArrayType):
        _normalize_primitive_type(field_schema.element_type)
        return field_schema
    if isinstance(field_schema, SequenceType):
        _normalize_primitive_type(field_schema.element_type)
        return field_schema
    if field_schema in _TOKEN_BY_TYPE:
        return field_schema
    raise TypeError(
        f"Unsupported field schema value: {field_schema!r}. "
        "Use a primitive type token, array(...), sequence(...), or a nested dict."
    )


def flatten_schema_fields(
    fields: NestedSchemaFields,
    prefix: str = "",
) -> dict[str, FlatField]:
    """Flatten one nested schema mapping into flat ``snake_case`` field names.

    Args:
        fields: A nested schema mapping. Values may be primitive schema tokens,
            ``array(...)``, ``sequence(...)``, or nested schema mappings.
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
