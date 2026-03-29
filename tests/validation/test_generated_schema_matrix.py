from dataclasses import dataclass, field

import numpy as np
import pytest

from cyclonedds_idl import IdlStruct, types

from xcdrjit import Time
from xcdrjit.idl import (
    array,
    boolean,
    float32,
    float64,
    get_codec_for,
    int16,
    int32,
    sequence,
    string,
    uint32,
    uint64,
)
from ..schema import HEADER_SCHEMA


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class ScalarEnvelope(IdlStruct, typename="test/msg/ScalarEnvelope"):
    active: bool = False
    priority: types.uint32 = 0
    title: str = ""
    header: Header = field(default_factory=Header)
    samples: types.sequence[types.float64] = field(default_factory=list)


@dataclass
class ActuatorSide(IdlStruct, typename="test/msg/ActuatorSide"):
    enabled: bool = False
    temperature: types.float32 = 0.0


@dataclass
class ActuatorBank(IdlStruct, typename="test/msg/ActuatorBank"):
    left: ActuatorSide = field(default_factory=ActuatorSide)
    right: ActuatorSide = field(default_factory=ActuatorSide)
    ids: types.array[str, 2] = field(default_factory=list)
    currents: types.sequence[types.int16] = field(default_factory=list)


@dataclass
class PacketWindow(IdlStruct, typename="test/msg/PacketWindow"):
    header: Header = field(default_factory=Header)
    flags: types.sequence[bool] = field(default_factory=list)
    window: types.array[types.uint64, 2] = field(default_factory=list)
    codes: types.sequence[types.int32] = field(default_factory=list)
    labels: types.sequence[str] = field(default_factory=list)


@dataclass
class Pose2D(IdlStruct, typename="test/msg/Pose2D"):
    x: types.float64 = 0.0
    y: types.float64 = 0.0


@dataclass
class StampedPoseLite(IdlStruct, typename="test/msg/StampedPoseLite"):
    header: Header = field(default_factory=Header)
    pose: Pose2D = field(default_factory=Pose2D)
    variances: types.array[types.float64, 2] = field(default_factory=list)
    tags: types.array[str, 2] = field(default_factory=list)
    residuals: types.sequence[types.float32] = field(default_factory=list)


@dataclass
class MissionInner(IdlStruct, typename="test/msg/MissionInner"):
    count: types.int32 = 0
    valid: bool = False


@dataclass
class MissionOuter(IdlStruct, typename="test/msg/MissionOuter"):
    inner: MissionInner = field(default_factory=MissionInner)
    note: str = ""


@dataclass
class MissionStatus(IdlStruct, typename="test/msg/MissionStatus"):
    meta: MissionOuter = field(default_factory=MissionOuter)
    moments: types.array[types.float64, 2] = field(default_factory=list)
    names: types.sequence[str] = field(default_factory=list)


SCALAR_ENVELOPE_SCHEMA = {
    "active": boolean,
    "priority": uint32,
    "title": string,
    "header": HEADER_SCHEMA,
    "samples": sequence(float64),
}


ACTUATOR_BANK_SCHEMA = {
    "left": {
        "enabled": boolean,
        "temperature": float32,
    },
    "right": {
        "enabled": boolean,
        "temperature": float32,
    },
    "ids": array(string, 2),
    "currents": sequence(int16),
}


PACKET_WINDOW_SCHEMA = {
    "header": HEADER_SCHEMA,
    "flags": sequence(boolean),
    "window": array(uint64, 2),
    "codes": sequence(int32),
    "labels": sequence(string),
}


STAMPED_POSE_LITE_SCHEMA = {
    "header": HEADER_SCHEMA,
    "pose": {
        "x": float64,
        "y": float64,
    },
    "variances": array(float64, 2),
    "tags": array(string, 2),
    "residuals": sequence(float32),
}


MISSION_STATUS_SCHEMA = {
    "meta": {
        "inner": {
            "count": int32,
            "valid": boolean,
        },
        "note": string,
    },
    "moments": array(float64, 2),
    "names": sequence(string),
}


def build_scalar_envelope_values() -> dict[str, object]:
    return {
        "active": True,
        "priority": np.uint32(7),
        "title": b"state",
        "header": {
            "stamp": {
                "sec": np.int32(42),
                "nanosec": np.uint32(999),
            },
            "frame_id": b"map",
        },
        "samples": np.array([1.0, -0.5, 3.25], dtype=np.float64),
    }


def serialize_scalar_envelope_cyclone(values: dict[str, object]) -> bytes:
    return ScalarEnvelope(
        active=bool(values["active"]),
        priority=int(values["priority"]),
        title=values["title"].decode("utf-8"),
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        samples=values["samples"].tolist(),
    ).serialize()


def build_actuator_bank_values() -> dict[str, object]:
    return {
        "left": {
            "enabled": True,
            "temperature": np.float32(20.5),
        },
        "right": {
            "enabled": False,
            "temperature": np.float32(21.75),
        },
        "ids": [b"left", b"right"],
        "currents": np.array([-10, 0, 10], dtype=np.int16),
    }


def serialize_actuator_bank_cyclone(values: dict[str, object]) -> bytes:
    return ActuatorBank(
        left=ActuatorSide(
            enabled=bool(values["left"]["enabled"]),
            temperature=float(values["left"]["temperature"]),
        ),
        right=ActuatorSide(
            enabled=bool(values["right"]["enabled"]),
            temperature=float(values["right"]["temperature"]),
        ),
        ids=[value.decode("utf-8") for value in values["ids"]],
        currents=values["currents"].tolist(),
    ).serialize()


def build_packet_window_values() -> dict[str, object]:
    return {
        "header": {
            "stamp": {
                "sec": np.int32(11),
                "nanosec": np.uint32(22),
            },
            "frame_id": b"odom",
        },
        "flags": np.array([True, False, True, True], dtype=np.bool_),
        "window": np.array([123, 456], dtype=np.uint64),
        "codes": np.array([-3, 5, 8], dtype=np.int32),
        "labels": [b"alpha", b"beta", b"gamma"],
    }


def serialize_packet_window_cyclone(values: dict[str, object]) -> bytes:
    return PacketWindow(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        flags=values["flags"].tolist(),
        window=values["window"].tolist(),
        codes=values["codes"].tolist(),
        labels=[value.decode("utf-8") for value in values["labels"]],
    ).serialize()


def build_stamped_pose_lite_values() -> dict[str, object]:
    return {
        "header": {
            "stamp": {
                "sec": np.int32(99),
                "nanosec": np.uint32(123456),
            },
            "frame_id": b"base_link",
        },
        "pose": {
            "x": np.float64(1.25),
            "y": np.float64(-2.5),
        },
        "variances": np.array([0.5, 0.25], dtype=np.float64),
        "tags": [b"front", b"vision"],
        "residuals": np.array([0.125, -0.25, 0.5], dtype=np.float32),
    }


def serialize_stamped_pose_lite_cyclone(values: dict[str, object]) -> bytes:
    return StampedPoseLite(
        header=Header(
            stamp=Time(
                sec=int(values["header"]["stamp"]["sec"]),
                nanosec=int(values["header"]["stamp"]["nanosec"]),
            ),
            frame_id=values["header"]["frame_id"].decode("utf-8"),
        ),
        pose=Pose2D(
            x=float(values["pose"]["x"]),
            y=float(values["pose"]["y"]),
        ),
        variances=values["variances"].tolist(),
        tags=[value.decode("utf-8") for value in values["tags"]],
        residuals=values["residuals"].tolist(),
    ).serialize()


def build_mission_status_values() -> dict[str, object]:
    return {
        "meta": {
            "inner": {
                "count": np.int32(314),
                "valid": True,
            },
            "note": b"ready",
        },
        "moments": np.array([1.5, -3.0], dtype=np.float64),
        "names": [b"alpha", b"omega"],
    }


def serialize_mission_status_cyclone(values: dict[str, object]) -> bytes:
    return MissionStatus(
        meta=MissionOuter(
            inner=MissionInner(
                count=int(values["meta"]["inner"]["count"]),
                valid=bool(values["meta"]["inner"]["valid"]),
            ),
            note=values["meta"]["note"].decode("utf-8"),
        ),
        moments=values["moments"].tolist(),
        names=[value.decode("utf-8") for value in values["names"]],
    ).serialize()


@pytest.mark.parametrize(
    ("case_name", "schema", "values_factory", "cyclone_serializer"),
    [
        (
            "scalar_envelope",
            SCALAR_ENVELOPE_SCHEMA,
            build_scalar_envelope_values,
            serialize_scalar_envelope_cyclone,
        ),
        (
            "actuator_bank",
            ACTUATOR_BANK_SCHEMA,
            build_actuator_bank_values,
            serialize_actuator_bank_cyclone,
        ),
        (
            "packet_window",
            PACKET_WINDOW_SCHEMA,
            build_packet_window_values,
            serialize_packet_window_cyclone,
        ),
        (
            "stamped_pose_lite",
            STAMPED_POSE_LITE_SCHEMA,
            build_stamped_pose_lite_values,
            serialize_stamped_pose_lite_cyclone,
        ),
        (
            "mission_status",
            MISSION_STATUS_SCHEMA,
            build_mission_status_values,
            serialize_mission_status_cyclone,
        ),
    ],
)
def test_generated_schemas_match_cyclone(
    case_name: str,
    schema,
    values_factory,
    cyclone_serializer,
) -> None:
    assert case_name
    codec = get_codec_for(schema)
    values = values_factory()

    generated = bytes(codec.serialize(values))
    expected = cyclone_serializer(values)

    assert codec.compute_size(values) == len(generated)
    assert generated == expected
    assert bytes(codec.serialize(codec.deserialize(expected))) == expected

    cached_codec = get_codec_for(schema)
    assert cached_codec.compute_size(values) == len(expected)
    assert bytes(cached_codec.serialize(values)) == expected
