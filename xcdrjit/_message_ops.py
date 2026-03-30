"""Private message helpers built around xcdrjit schemas."""

from collections.abc import Mapping

import numpy as np

from .schema_types import (
    ArrayType,
    FlatField,
    NestedSchemaFields,
    SequenceType,
    float32,
    float64,
    normalize_field_schema,
    string,
)

_MISSING = object()
_FLOAT_TYPES = {float32, float64}


def _format_path(path: str) -> str:
    """Return a display-friendly field path, falling back to ``"message"`` for the root.

    Args:
        path: Dot-separated field path, or empty string for the root.

    Returns:
        The path string, or ``"message"`` when ``path`` is empty.
    """
    return path or "message"


def _fail_mismatch(path: str, message: str) -> None:
    """Raise an ``AssertionError`` for a field mismatch at ``path``.

    Args:
        path: Dot-separated field path where the mismatch was detected.
        message: Human-readable description of the mismatch.
    """
    raise AssertionError(f"{_format_path(path)}: {message}")


def _assert_scalar_match(
    value_a: object,
    value_b: object,
    primitive_type: object,
    path: str,
) -> None:
    """Assert that two scalar values are equal under one primitive type.

    Float NaN values compare as equal (both NaN is a match).

    Args:
        value_a: First scalar value.
        value_b: Second scalar value.
        primitive_type: The schema primitive type token (e.g. ``int32``,
            ``float64``, ``string``).
        path: Dot-separated field path used in the assertion error message.
    """
    if primitive_type is string:
        if value_a != value_b:
            _fail_mismatch(path, f"{value_a!r} != {value_b!r}")
        return

    if primitive_type in _FLOAT_TYPES:
        if np.isnan(value_a) and np.isnan(value_b):
            return

    if value_a != value_b:
        _fail_mismatch(path, f"{value_a!r} != {value_b!r}")


def _assert_numpy_match(
    value_a: object,
    value_b: object,
    primitive_type: object,
    path: str,
) -> None:
    """Assert that two 1D NumPy arrays are equal under one primitive element type.

    Float arrays are compared with ``equal_nan=True``.

    Args:
        value_a: First array value.
        value_b: Second array value.
        primitive_type: The element primitive type token.
        path: Dot-separated field path used in the assertion error message.
    """
    if not isinstance(value_a, np.ndarray) or not isinstance(value_b, np.ndarray):
        _fail_mismatch(path, "expected NumPy arrays on both sides")
    if value_a.ndim != 1 or value_b.ndim != 1:
        _fail_mismatch(path, "expected 1D NumPy arrays")
    if value_a.dtype != value_b.dtype:
        _fail_mismatch(path, f"dtype {value_a.dtype!s} != {value_b.dtype!s}")

    equal_nan = primitive_type in _FLOAT_TYPES
    if not np.array_equal(value_a, value_b, equal_nan=equal_nan):
        _fail_mismatch(path, "array values differ")


def _assert_string_list_match(
    value_a: object,
    value_b: object,
    path: str,
) -> None:
    """Assert that two string sequences are equal element-by-element.

    Args:
        value_a: First ``list[bytes]`` value.
        value_b: Second ``list[bytes]`` value.
        path: Dot-separated field path used in the assertion error message.
    """
    try:
        length_a = len(value_a)
        length_b = len(value_b)
    except TypeError as exc:
        raise AssertionError(
            f"{_format_path(path)}: expected string sequences on both sides"
        ) from exc

    if length_a != length_b:
        _fail_mismatch(path, f"length {length_a} != {length_b}")

    for index, (item_a, item_b) in enumerate(zip(value_a, value_b, strict=False)):
        if item_a != item_b:
            _fail_mismatch(f"{path}[{index}]", f"{item_a!r} != {item_b!r}")


def _assert_field_match(
    value_a: object,
    value_b: object,
    field_schema: object,
    path: str,
) -> None:
    """Assert that two leaf field values are equal under one flat field schema.

    Dispatches to the appropriate scalar, NumPy, or string-list comparison.

    Args:
        value_a: First field value.
        value_b: Second field value.
        field_schema: The leaf schema for this field (primitive token,
            ``ArrayType``, or ``SequenceType``).
        path: Dot-separated field path used in the assertion error message.
    """
    normalized: FlatField = normalize_field_schema(field_schema)
    if not isinstance(normalized, (ArrayType, SequenceType)):
        _assert_scalar_match(value_a, value_b, normalized, path)
        return

    if normalized.element_type is string:
        _assert_string_list_match(value_a, value_b, path)
        return

    _assert_numpy_match(value_a, value_b, normalized.element_type, path)


def _assert_message_match(
    msg_a: object,
    msg_b: object,
    schema: NestedSchemaFields,
    path: str,
) -> None:
    """Recursively assert that two nested message dicts are equal under a schema.

    Values are compared in schema insertion order; mapping keys are ignored
    (only value order matters, matching the serializer convention).

    Args:
        msg_a: First nested ``dict``-like message.
        msg_b: Second nested ``dict``-like message.
        schema: The schema defining field names, types, and nesting shape.
        path: Dot-separated field path prefix used in assertion error messages.
    """
    if not isinstance(msg_a, Mapping) or not isinstance(msg_b, Mapping):
        _fail_mismatch(path, "expected nested dict-like mappings")

    values_a = iter(msg_a.values())
    values_b = iter(msg_b.values())

    for field_name, field_schema in schema.items():
        field_path = f"{path}.{field_name}" if path else field_name
        value_a = next(values_a, _MISSING)
        value_b = next(values_b, _MISSING)

        if value_a is _MISSING or value_b is _MISSING:
            _fail_mismatch(field_path, "missing value")

        if isinstance(field_schema, Mapping):
            _assert_message_match(value_a, value_b, field_schema, field_path)
        else:
            _assert_field_match(value_a, value_b, field_schema, field_path)

    extra_a = next(values_a, _MISSING)
    extra_b = next(values_b, _MISSING)
    if extra_a is not _MISSING or extra_b is not _MISSING:
        _fail_mismatch(path, "extra values do not match the schema")


def assert_messages_equal(
    msg_a: Mapping[str, object],
    msg_b: Mapping[str, object],
    schema: NestedSchemaFields,
) -> None:
    """Assert that two runtime messages are equal under one schema.

    The comparison follows the schema order and ignores the runtime mapping keys.
    Numeric and boolean arrays/sequences are compared with NumPy. String arrays
    and sequences are compared element-by-element as ``bytes``.
    """
    _assert_message_match(msg_a, msg_b, schema, "")
