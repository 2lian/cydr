from dataclasses import dataclass, field
import json

import numpy as np
import pytest

from cyclonedds_idl import IdlStruct, types
from cydr.idl import (
    flatten_schema_fields,
    get_codec_for,
    schema_hash,
)
from cydr.schema_types import field_schema_token
from ..cache import SCHEMA_CACHE_DIR
from ..schema import EVERY_SUPPORTED_SCHEMA, JOINT_STATE_SCHEMA, Time


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class EverySupportedSchema(IdlStruct, typename="test/msg/EverySupportedSchema"):
    boolean_value: bool = False
    byte_value: types.byte = 0
    signed_int8: types.int8 = 0
    unsigned_int8: types.uint8 = 0
    signed_int16: types.int16 = 0
    unsigned_int16: types.uint16 = 0
    signed_int32: types.int32 = 0
    unsigned_int32: types.uint32 = 0
    signed_int64: types.int64 = 0
    unsigned_int64: types.uint64 = 0
    float32_value: types.float32 = 0.0
    float64_value: types.float64 = 0.0
    text: str = ""
    header: Header = field(default_factory=Header)
    bool_sequence: types.sequence[bool] = field(default_factory=list)
    byte_array: types.array[types.byte, 3] = field(default_factory=list)
    int8_sequence: types.sequence[types.int8] = field(default_factory=list)
    uint8_array: types.array[types.uint8, 3] = field(default_factory=list)
    int16_sequence: types.sequence[types.int16] = field(default_factory=list)
    uint16_array: types.array[types.uint16, 2] = field(default_factory=list)
    int32_sequence: types.sequence[types.int32] = field(default_factory=list)
    uint32_array: types.array[types.uint32, 2] = field(default_factory=list)
    int64_sequence: types.sequence[types.int64] = field(default_factory=list)
    uint64_array: types.array[types.uint64, 2] = field(default_factory=list)
    float32_sequence: types.sequence[types.float32] = field(default_factory=list)
    float64_array: types.array[types.float64, 2] = field(default_factory=list)
    text_array: types.array[str, 2] = field(default_factory=list)
    text_sequence: types.sequence[str] = field(default_factory=list)


@dataclass
class JointState(IdlStruct, typename="sensor_msgs/msg/JointState"):
    header: Header = field(default_factory=Header)
    name: types.sequence[str] = field(default_factory=list)
    position: types.sequence[types.float64] = field(default_factory=list)
    velocity: types.sequence[types.float64] = field(default_factory=list)
    effort: types.sequence[types.float64] = field(default_factory=list)


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
        "text_array": np.array([b"a", "café".encode("utf-8")], dtype=np.bytes_),
        "text_sequence": np.array([b"bbb", "😀".encode("utf-8")], dtype=np.bytes_),
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
        "name": np.array([b"joint_a", b"joint_b", b"joint_c"], dtype=np.bytes_),
        "position": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "velocity": np.array([3.5, 4.5, 5.5], dtype=np.float64),
        "effort": np.array([6.5, 7.5, 8.5], dtype=np.float64),
    }


def serialize_every_supported_cyclone(values: dict[str, object]) -> bytes:
    return EverySupportedSchema(
        boolean_value=bool(values["boolean_value"]),
        byte_value=int(values["byte_value"]),
        signed_int8=int(values["signed_int8"]),
        unsigned_int8=int(values["unsigned_int8"]),
        signed_int16=int(values["signed_int16"]),
        unsigned_int16=int(values["unsigned_int16"]),
        signed_int32=int(values["signed_int32"]),
        unsigned_int32=int(values["unsigned_int32"]),
        signed_int64=int(values["signed_int64"]),
        unsigned_int64=int(values["unsigned_int64"]),
        float32_value=float(values["float32_value"]),
        float64_value=float(values["float64_value"]),
        text=values["text"].decode("utf-8"),
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        bool_sequence=values["bool_sequence"].tolist(),
        byte_array=values["byte_array"].tolist(),
        int8_sequence=values["int8_sequence"].tolist(),
        uint8_array=values["uint8_array"].tolist(),
        int16_sequence=values["int16_sequence"].tolist(),
        uint16_array=values["uint16_array"].tolist(),
        int32_sequence=values["int32_sequence"].tolist(),
        uint32_array=values["uint32_array"].tolist(),
        int64_sequence=values["int64_sequence"].tolist(),
        uint64_array=values["uint64_array"].tolist(),
        float32_sequence=values["float32_sequence"].tolist(),
        float64_array=values["float64_array"].tolist(),
        text_array=[value.decode("utf-8") for value in values["text_array"]],
        text_sequence=[value.decode("utf-8") for value in values["text_sequence"]],
    ).serialize()


def serialize_joint_state_cyclone(values: dict[str, object]) -> bytes:
    return JointState(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        name=[value.decode("utf-8") for value in values["name"]],
        position=values["position"].tolist(),
        velocity=values["velocity"].tolist(),
        effort=values["effort"].tolist(),
    ).serialize()


@pytest.mark.parametrize(
    ("schema", "values_factory", "cyclone_serializer"),
    [
        (
            EVERY_SUPPORTED_SCHEMA,
            build_every_supported_values,
            serialize_every_supported_cyclone,
        ),
        (
            JOINT_STATE_SCHEMA,
            build_joint_state_values,
            serialize_joint_state_cyclone,
        ),
    ],
)
def test_generated_cython_module_compiles_from_tmp_dir_and_runs_once(
    schema,
    values_factory,
    cyclone_serializer,
) -> None:
    values = values_factory()
    codec = get_codec_for(schema)
    serialize = codec.serialize
    deserialize = codec.deserialize
    generated_bytes = bytes(serialize(values))
    assert generated_bytes == cyclone_serializer(values)
    assert bytes(serialize(deserialize(generated_bytes))) == generated_bytes

    cached_codec = get_codec_for(schema)
    assert bytes(cached_codec.serialize(values)) == generated_bytes
    expected_hash = schema_hash(schema)
    assert codec.cache_dir.is_relative_to(SCHEMA_CACHE_DIR)
    assert (codec.cache_dir / f"{codec.module_name}.pyx").exists()

    manifest_path = codec.cache_dir / f"{codec.module_name}.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_hash"] == expected_hash
    assert manifest["module_name"] == codec.module_name
    assert manifest["serializer_name"] == f"schema_{expected_hash}"
    assert manifest["flat_schema"] == [
        field_schema_token(field_type)
        for field_type in flatten_schema_fields(schema).values()
    ]
