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
    normalize_schema_field,
    string,
)

_MISSING = object()
_FLOAT_TYPES = {float32, float64}


def _describe_path(path: str) -> str:
    return path or "message"


def _raise_mismatch(path: str, message: str) -> None:
    raise AssertionError(f"{_describe_path(path)}: {message}")


def _assert_scalar_equal(
    value_a: object,
    value_b: object,
    primitive_type: object,
    path: str,
) -> None:
    if primitive_type is string:
        if value_a != value_b:
            _raise_mismatch(path, f"{value_a!r} != {value_b!r}")
        return

    if primitive_type in _FLOAT_TYPES:
        if np.isnan(value_a) and np.isnan(value_b):
            return

    if value_a != value_b:
        _raise_mismatch(path, f"{value_a!r} != {value_b!r}")


def _assert_numpy_sequence_equal(
    value_a: object,
    value_b: object,
    primitive_type: object,
    path: str,
) -> None:
    if not isinstance(value_a, np.ndarray) or not isinstance(value_b, np.ndarray):
        _raise_mismatch(path, "expected NumPy arrays on both sides")
    if value_a.ndim != 1 or value_b.ndim != 1:
        _raise_mismatch(path, "expected 1D NumPy arrays")
    if value_a.dtype != value_b.dtype:
        _raise_mismatch(path, f"dtype {value_a.dtype!s} != {value_b.dtype!s}")

    equal_nan = primitive_type in _FLOAT_TYPES
    if not np.array_equal(value_a, value_b, equal_nan=equal_nan):
        _raise_mismatch(path, "array values differ")


def _assert_string_sequence_equal(
    value_a: object,
    value_b: object,
    path: str,
) -> None:
    try:
        length_a = len(value_a)
        length_b = len(value_b)
    except TypeError as exc:
        raise AssertionError(
            f"{_describe_path(path)}: expected string sequences on both sides"
        ) from exc

    if length_a != length_b:
        _raise_mismatch(path, f"length {length_a} != {length_b}")

    for index, (item_a, item_b) in enumerate(zip(value_a, value_b, strict=False)):
        if item_a != item_b:
            _raise_mismatch(f"{path}[{index}]", f"{item_a!r} != {item_b!r}")


def _assert_leaf_equal(
    value_a: object,
    value_b: object,
    field_schema: object,
    path: str,
) -> None:
    normalized: FlatField = normalize_schema_field(field_schema)
    if not isinstance(normalized, (ArrayType, SequenceType)):
        _assert_scalar_equal(value_a, value_b, normalized, path)
        return

    if normalized.element_type is string:
        _assert_string_sequence_equal(value_a, value_b, path)
        return

    _assert_numpy_sequence_equal(value_a, value_b, normalized.element_type, path)


def _assert_messages_equal_recursive(
    msg_a: object,
    msg_b: object,
    schema: NestedSchemaFields,
    path: str,
) -> None:
    if not isinstance(msg_a, Mapping) or not isinstance(msg_b, Mapping):
        _raise_mismatch(path, "expected nested dict-like mappings")

    values_a = iter(msg_a.values())
    values_b = iter(msg_b.values())

    for field_name, field_schema in schema.items():
        field_path = f"{path}.{field_name}" if path else field_name
        value_a = next(values_a, _MISSING)
        value_b = next(values_b, _MISSING)

        if value_a is _MISSING or value_b is _MISSING:
            _raise_mismatch(field_path, "missing value")

        if isinstance(field_schema, Mapping):
            _assert_messages_equal_recursive(value_a, value_b, field_schema, field_path)
        else:
            _assert_leaf_equal(value_a, value_b, field_schema, field_path)

    extra_a = next(values_a, _MISSING)
    extra_b = next(values_b, _MISSING)
    if extra_a is not _MISSING or extra_b is not _MISSING:
        _raise_mismatch(path, "extra values do not match the schema")


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
    _assert_messages_equal_recursive(msg_a, msg_b, schema, "")
