"""Public xcdrjit API for the dict-style codec.

Typical usage::

    from xcdrjit.idl import int32, uint32, float64, string, sequence, get_codec_for

    schema = {
        "header": {
            "stamp": {"sec": int32, "nanosec": uint32},
            "frame_id": string,
        },
        "name": sequence(string),
        "position": sequence(float64),
    }
    codec = get_codec_for(schema)
    payload = codec.serialize(values)
    decoded = codec.deserialize(payload)

Runtime conventions:

- ``string`` fields are UTF-8 ``bytes``, not ``str``
- Numeric/boolean arrays and sequences are 1-D NumPy arrays with the matching dtype
- Mapping keys are ignored at (de)serialize time; only value order matters
"""

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

from ._message_ops import assert_messages_equal
from ._runtime import (
    CYTHON_CACHE_DIR,
    Codec,
    flatten_runtime_values,
    get_codec_for,
    rebuild_runtime_values,
    schema_hash,
)
from .cython_generator import generate_cython_codec_source
from .schema_types import (
    ArrayType,
    NestedSchemaFields,
    SequenceType,
    _normalize_primitive_type,
    flatten_schema_fields,
)
from .structs import XcdrStruct

# ---------------------------------------------------------------------------
# Schema tokens
#
# Numeric tokens are NumPy scalar dtypes.  The same objects are used as keys
# in cython_generator.PRIMITIVE_CODEGEN, so these values must stay in sync
# with schema_types._TOKEN_BY_TYPE.
# ---------------------------------------------------------------------------

boolean: TypeAlias = np.bool_   #: XCDR boolean. Runtime value: Python bool or NumPy scalar.
byte: TypeAlias = np.uint8      #: XCDR byte (alias of uint8).
int8: TypeAlias = np.int8       #: XCDR signed 8-bit integer.
uint8: TypeAlias = np.uint8     #: XCDR unsigned 8-bit integer.
int16: TypeAlias = np.int16     #: XCDR signed 16-bit integer.
uint16: TypeAlias = np.uint16   #: XCDR unsigned 16-bit integer.
int32: TypeAlias = np.int32     #: XCDR signed 32-bit integer.
uint32: TypeAlias = np.uint32   #: XCDR unsigned 32-bit integer.
int64: TypeAlias = np.int64     #: XCDR signed 64-bit integer.
uint64: TypeAlias = np.uint64   #: XCDR unsigned 64-bit integer.
float32: TypeAlias = np.float32 #: XCDR 32-bit float.
float64: TypeAlias = np.float64 #: XCDR 64-bit float.
string: TypeAlias = bytes       #: XCDR string. Runtime value: UTF-8 bytes.


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------


def array(element_type: object, length: int) -> ArrayType:
    """Return a fixed-size array schema descriptor.

    Args:
        element_type: One primitive schema token such as ``int32`` or ``float64``.
        length: Exact element count. Must be a non-negative integer.

    Returns:
        An ``ArrayType`` descriptor for use as a field value in a schema dict.

    Raises:
        ValueError: If ``length`` is not a non-negative integer.
        TypeError: If ``element_type`` is not a supported primitive token.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError("array length must be a non-negative integer.")
    return ArrayType(
        element_type=_normalize_primitive_type(element_type),
        length=length,
    )


def sequence(element_type: object) -> SequenceType:
    """Return a variable-length sequence schema descriptor.

    Args:
        element_type: One primitive schema token such as ``float64`` or ``string``.

    Returns:
        A ``SequenceType`` descriptor for use as a field value in a schema dict.

    Raises:
        TypeError: If ``element_type`` is not a supported primitive token.
    """
    return SequenceType(element_type=_normalize_primitive_type(element_type))


# ---------------------------------------------------------------------------
# Codec warmup
# ---------------------------------------------------------------------------


def warmup_codec(
    value: Mapping[str, object] | XcdrStruct,
    schema: NestedSchemaFields | None = None,
) -> Codec:
    """Compile the codec and verify a full encode/decode roundtrip.

    Serializes ``value``, deserializes the result, asserts the decoded value
    matches the original, then asserts that re-serializing it produces the
    same bytes. Returns the compiled codec ready for steady-state use.

    For the dict API pass both ``value`` and ``schema``.
    For the struct API pass only an ``XcdrStruct`` instance; ``schema`` must
    be omitted.

    Args:
        value: One message value — a nested dict-like mapping or an
            ``XcdrStruct`` instance.
        schema: The nested schema dict when using the dict API. Omit for
            ``XcdrStruct`` values.

    Returns:
        The warmed ``Codec`` for the schema.

    Raises:
        TypeError: If ``value`` type and ``schema`` presence do not match.
        AssertionError: If the roundtrip check fails.
    """
    if isinstance(value, XcdrStruct):
        if schema is not None:
            raise TypeError(
                "warmup_codec does not accept a schema argument when value is an XcdrStruct."
            )
        codec = type(value)._get_codec()
        payload = value.serialize()
        decoded = type(value).deserialize(payload)
        assert_messages_equal(
            decoded._to_nested_dict(),
            value._to_nested_dict(),
            type(value)._schema_info().schema,
        )
        assert bytes(decoded.serialize()) == bytes(payload)
        return codec

    if isinstance(value, Mapping):
        if schema is None:
            raise TypeError(
                "warmup_codec requires a schema argument when value is a mapping."
            )
        codec = get_codec_for(schema)
        payload = codec.serialize(value)
        decoded = codec.deserialize(payload)
        assert_messages_equal(decoded, value, schema)
        assert bytes(codec.serialize(decoded)) == bytes(payload)
        return codec

    raise TypeError(
        "warmup_codec expects a nested dict-like mapping or an XcdrStruct instance."
    )
