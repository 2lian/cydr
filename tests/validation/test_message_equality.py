import numpy as np
import pytest

from xcdrjit.idl import assert_messages_equal

from ..schema import EVERY_SUPPORTED_SCHEMA, JOINT_STATE_SCHEMA


def build_every_supported_values() -> dict[str, object]:
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


def build_joint_state_values() -> dict[str, object]:
    return {
        "header": {
            "stamp": {
                "sec": 17000,
                "nanosec": 1234,
            },
            "frame_id": b"base_link",
        },
        "name": [b"joint_a", b"joint_b", b"joint_c"],
        "position": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "velocity": np.array([3.5, 4.5, 5.5], dtype=np.float64),
        "effort": np.array([6.5, 7.5, 8.5], dtype=np.float64),
    }


def test_assert_messages_equal_accepts_equal_every_supported_messages() -> None:
    msg_a = build_every_supported_values()
    msg_b = build_every_supported_values()

    assert_messages_equal(msg_a, msg_b, EVERY_SUPPORTED_SCHEMA)


def test_assert_messages_equal_ignores_runtime_keys_and_uses_value_order() -> None:
    msg_a = build_joint_state_values()
    msg_b = {
        "a": {
            "b": {
                "c": msg_a["header"]["stamp"]["sec"],
                "d": msg_a["header"]["stamp"]["nanosec"],
            },
            "e": msg_a["header"]["frame_id"],
        },
        "f": msg_a["name"],
        "g": msg_a["position"],
        "h": msg_a["velocity"],
        "i": msg_a["effort"],
    }

    assert_messages_equal(msg_a, msg_b, JOINT_STATE_SCHEMA)


def test_assert_messages_equal_detects_scalar_mismatch() -> None:
    msg_a = build_every_supported_values()
    msg_b = build_every_supported_values()
    msg_b["header"]["stamp"]["sec"] = 11

    with pytest.raises(AssertionError, match=r"header\.stamp\.sec"):
        assert_messages_equal(msg_a, msg_b, EVERY_SUPPORTED_SCHEMA)


def test_assert_messages_equal_detects_numpy_array_mismatch() -> None:
    msg_a = build_joint_state_values()
    msg_b = build_joint_state_values()
    msg_b["position"] = np.array([0.5, 1.5, 99.0], dtype=np.float64)

    with pytest.raises(AssertionError, match=r"position"):
        assert_messages_equal(msg_a, msg_b, JOINT_STATE_SCHEMA)


def test_assert_messages_equal_detects_numpy_dtype_mismatch() -> None:
    msg_a = build_joint_state_values()
    msg_b = build_joint_state_values()
    msg_b["effort"] = msg_b["effort"].astype(np.float32)

    with pytest.raises(AssertionError, match=r"effort.*dtype"):
        assert_messages_equal(msg_a, msg_b, JOINT_STATE_SCHEMA)


def test_assert_messages_equal_detects_string_sequence_mismatch() -> None:
    msg_a = build_every_supported_values()
    msg_b = build_every_supported_values()
    msg_b["text_sequence"] = [b"bbb", b"other"]

    with pytest.raises(AssertionError, match=r"text_sequence\[1\]"):
        assert_messages_equal(msg_a, msg_b, EVERY_SUPPORTED_SCHEMA)
