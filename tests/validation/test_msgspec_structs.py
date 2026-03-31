from typing import Any, ClassVar

import numpy as np
from nptyping import Bool, Bytes, Float32, Float64, Int8, Int16, Int32, Int64, NDArray, Shape, UInt8, UInt16, UInt32, UInt64

from cydr.idl import (
    StringCollectionMode,
    XcdrStruct,
    assert_messages_equal,
    boolean,
    byte,
    flatten_schema_fields,
    flatten_runtime_values,
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
    warmup_codec,
)

from ..schema import EVERY_SUPPORTED_SCHEMA, HEADER_SCHEMA


class Stamp(XcdrStruct):
    sec: int32
    nanosec: uint32


class Header(XcdrStruct):
    stamp: Stamp
    frame_id: string


class EverySupportedMessage(XcdrStruct):
    boolean_value: boolean
    byte_value: byte
    signed_int8: int8
    unsigned_int8: uint8
    signed_int16: int16
    unsigned_int16: uint16
    signed_int32: int32
    unsigned_int32: uint32
    signed_int64: int64
    unsigned_int64: uint64
    float32_value: float32
    float64_value: float64
    text: string
    header: Header
    bool_sequence: NDArray[Any, Bool]
    byte_array: NDArray[Shape["3"], UInt8]
    int8_sequence: NDArray[Any, Int8]
    uint8_array: NDArray[Shape["3"], UInt8]
    int16_sequence: NDArray[Any, Int16]
    uint16_array: NDArray[Shape["2"], UInt16]
    int32_sequence: NDArray[Any, Int32]
    uint32_array: NDArray[Shape["2"], UInt32]
    int64_sequence: NDArray[Any, Int64]
    uint64_array: NDArray[Shape["2"], UInt64]
    float32_sequence: NDArray[Any, Float32]
    float64_array: NDArray[Shape["2"], Float64]
    text_array: NDArray[Shape["2"], Bytes]
    text_sequence: NDArray[Any, Bytes]


def build_values() -> dict[str, object]:
    return {
        "boolean_value": True,
        "byte_value": 127,
        "signed_int8": -1,
        "unsigned_int8": 1,
        "signed_int16": -2,
        "unsigned_int16": 2,
        "signed_int32": -3,
        "unsigned_int32": 3,
        "signed_int64": -4,
        "unsigned_int64": 4,
        "float32_value": np.float32(1.5),
        "float64_value": np.float64(2.5),
        "text": "café😀".encode("utf-8"),
        "header": {
            "stamp": {
                "sec": 10,
                "nanosec": 123,
            },
            "frame_id": b"map",
        },
        "bool_sequence": np.array([True, False, True], dtype=np.bool_),
        "byte_array": np.array([16, 32, 48], dtype=np.uint8),
        "int8_sequence": np.array([-1, 0, 1], dtype=np.int8),
        "uint8_array": np.array([1, 2, 3], dtype=np.uint8),
        "int16_sequence": np.array([-2, 3], dtype=np.int16),
        "uint16_array": np.array([4, 5], dtype=np.uint16),
        "int32_sequence": np.array([-6, 7], dtype=np.int32),
        "uint32_array": np.array([8, 9], dtype=np.uint32),
        "int64_sequence": np.array([-10, 11], dtype=np.int64),
        "uint64_array": np.array([12, 13], dtype=np.uint64),
        "float32_sequence": np.array([1.5, -2.25], dtype=np.float32),
        "float64_array": np.array([3.5, -4.75], dtype=np.float64),
        "text_array": np.array([b"a", "café".encode("utf-8")], dtype=np.bytes_),
        "text_sequence": np.array([b"bbb", "😀".encode("utf-8")], dtype=np.bytes_),
    }


def assert_flat_values_equal(actual: list[object], expected: list[object]) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(actual_value, np.ndarray) or isinstance(expected_value, np.ndarray):
            assert isinstance(actual_value, np.ndarray)
            assert isinstance(expected_value, np.ndarray)
            assert actual_value.dtype == expected_value.dtype
            assert np.array_equal(
                actual_value,
                expected_value,
                equal_nan=np.issubdtype(actual_value.dtype, np.floating),
            )
        else:
            assert actual_value == expected_value


def test_cydr_struct_schema_matches_expected_schema() -> None:
    assert Header._schema_info().schema == HEADER_SCHEMA
    assert EverySupportedMessage._schema_info().schema == EVERY_SUPPORTED_SCHEMA


def test_cydr_struct_flat_schema_matches_expected_flat_schema() -> None:
    assert (
        EverySupportedMessage._schema_info().flat_schema
        == flatten_schema_fields(EVERY_SUPPORTED_SCHEMA)
    )


def test_cydr_struct_to_nested_dict_matches_expected_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    assert_messages_equal(message._to_nested_dict(), values, EVERY_SUPPORTED_SCHEMA)


def test_cydr_struct_to_flat_matches_expected_flat_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    assert_flat_values_equal(message._to_flat(), flatten_runtime_values(values))


def test_cydr_struct_from_flat_matches_expected_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_flat(flatten_runtime_values(values))

    assert_messages_equal(message._to_nested_dict(), values, EVERY_SUPPORTED_SCHEMA)


def test_cydr_struct_integrates_with_generated_serializer_helpers() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)
    codec = EverySupportedMessage._get_codec()

    payload = bytes(codec.serialize(message._to_nested_dict()))

    assert codec.compute_size(values) == len(payload)
    assert payload == bytes(codec.serialize(message._to_flat()))
    assert payload == bytes(codec.serialize(values))
    assert_messages_equal(codec.deserialize(payload), values, EVERY_SUPPORTED_SCHEMA)


def test_cydr_struct_caches_codec_on_child_class_not_parent() -> None:
    class TinyMessage(XcdrStruct):
        value: int32

    assert XcdrStruct.__dict__.get("_cydr_codec") is None
    assert TinyMessage.__dict__.get("_cydr_codec") is None

    TinyMessage._get_codec()

    assert XcdrStruct.__dict__.get("_cydr_codec") is None
    codec = TinyMessage.__dict__.get("_cydr_codec")
    assert codec is not None
    assert codec.serialize is not None
    assert codec.compute_size is not None


def test_cydr_struct_ignores_classvar_fields_in_schema() -> None:
    class MessageWithClassVar(XcdrStruct):
        value: int32
        label: ClassVar[string] = b"ignored"

    assert MessageWithClassVar.__struct_fields__ == ("value",)
    assert MessageWithClassVar._schema_info().schema == {"value": int32}
    assert MessageWithClassVar._schema_info().flat_schema == {"value": int32}

    message = MessageWithClassVar(value=np.int32(7))
    payload = message.serialize()
    decoded = MessageWithClassVar.deserialize(payload)

    assert decoded.value == np.int32(7)


def test_cydr_struct_public_serialize_deserialize_roundtrip() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    payload = message.serialize()
    decoded = EverySupportedMessage.deserialize(payload)

    assert_messages_equal(decoded._to_nested_dict(), values, EVERY_SUPPORTED_SCHEMA)
    assert bytes(decoded.serialize()) == bytes(payload)


def test_cydr_struct_deserialize_can_return_list_string_collections() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    decoded = EverySupportedMessage.deserialize(
        message.serialize(),
        string_collections=StringCollectionMode.LIST,
    )

    assert decoded.text == values["text"]
    assert decoded.header.frame_id == values["header"]["frame_id"]
    assert isinstance(decoded.text_array, list)
    assert isinstance(decoded.text_sequence, list)
    assert decoded.text_array == values["text_array"].tolist()
    assert decoded.text_sequence == values["text_sequence"].tolist()
    assert bytes(decoded.serialize()) == bytes(message.serialize())


def test_warmup_codec_supports_cydr_struct_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    codec = warmup_codec(message)

    assert codec is EverySupportedMessage._get_codec()


def test_warmup_codec_rejects_schema_for_cydr_struct_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_nested_dict(values)

    with np.testing.assert_raises(TypeError):
        warmup_codec(message, EVERY_SUPPORTED_SCHEMA)
