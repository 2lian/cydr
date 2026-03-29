import numpy as np

from xcdrjit.idl import (
    XcdrStruct,
    array,
    assert_messages_equal,
    boolean,
    byte,
    flatten_cython_fields,
    flatten_cython_value_list,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    sequence,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
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
    bool_sequence: sequence(boolean)
    byte_array: array(byte, 3)
    int8_sequence: sequence(int8)
    uint8_array: array(uint8, 3)
    int16_sequence: sequence(int16)
    uint16_array: array(uint16, 2)
    int32_sequence: sequence(int32)
    uint32_array: array(uint32, 2)
    int64_sequence: sequence(int64)
    uint64_array: array(uint64, 2)
    float32_sequence: sequence(float32)
    float64_array: array(float64, 2)
    text_array: array(string, 2)
    text_sequence: sequence(string)


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
        "text_array": [b"a", "café".encode("utf-8")],
        "text_sequence": [b"bbb", "😀".encode("utf-8")],
    }


def assert_flat_values_equal(actual: list[object], expected: list[object]) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(actual_value, np.ndarray) or isinstance(expected_value, np.ndarray):
            assert isinstance(actual_value, np.ndarray)
            assert isinstance(expected_value, np.ndarray)
            assert actual_value.dtype == expected_value.dtype
            assert np.array_equal(actual_value, expected_value, equal_nan=True)
        else:
            assert actual_value == expected_value


def test_xcdr_struct_schema_dict_matches_expected_schema() -> None:
    assert Header._schema_state().schema_dict == HEADER_SCHEMA
    assert EverySupportedMessage._schema_state().schema_dict == EVERY_SUPPORTED_SCHEMA


def test_xcdr_struct_flat_schema_matches_expected_flat_schema() -> None:
    assert (
        EverySupportedMessage._schema_state().flat_schema
        == flatten_cython_fields(EVERY_SUPPORTED_SCHEMA)
    )


def test_xcdr_struct_to_message_dict_matches_expected_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_message_dict(values)

    assert_messages_equal(message._to_message_dict(), values, EVERY_SUPPORTED_SCHEMA)


def test_xcdr_struct_to_flat_matches_expected_flat_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_message_dict(values)

    assert_flat_values_equal(message._to_flat(), flatten_cython_value_list(values))


def test_xcdr_struct_from_flat_values_matches_expected_values() -> None:
    values = build_values()
    message = EverySupportedMessage._from_flat_values(flatten_cython_value_list(values))

    assert_messages_equal(message._to_message_dict(), values, EVERY_SUPPORTED_SCHEMA)


def test_xcdr_struct_integrates_with_generated_serializer_helpers() -> None:
    values = build_values()
    message = EverySupportedMessage._from_message_dict(values)
    codec = EverySupportedMessage._get_codec()

    payload = bytes(codec.serialize(message._to_message_dict()))

    assert codec.compute_size(values) == len(payload)
    assert payload == bytes(codec.serialize(message._to_flat()))
    assert payload == bytes(codec.serialize(values))
    assert_messages_equal(codec.deserialize(payload), values, EVERY_SUPPORTED_SCHEMA)


def test_xcdr_struct_caches_codec_on_child_class_not_parent() -> None:
    class TinyMessage(XcdrStruct):
        value: int32

    assert XcdrStruct.__dict__.get("_xcdr_codec") is None
    assert TinyMessage.__dict__.get("_xcdr_codec") is None

    TinyMessage._get_codec()

    assert XcdrStruct.__dict__.get("_xcdr_codec") is None
    codec = TinyMessage.__dict__.get("_xcdr_codec")
    assert codec is not None
    assert codec.serialize is not None
    assert codec.compute_size is not None


def test_xcdr_struct_public_serialize_deserialize_roundtrip() -> None:
    values = build_values()
    message = EverySupportedMessage._from_message_dict(values)

    payload = message.serialize()
    decoded = EverySupportedMessage.deserialize(payload)

    assert_messages_equal(decoded._to_message_dict(), values, EVERY_SUPPORTED_SCHEMA)
    assert bytes(decoded.serialize()) == bytes(payload)
