"""Schema fixtures shared by validation tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from nptyping import Bool, Bytes, Float32, Float64, Int8, Int16, Int32, Int64, NDArray, Shape, UInt8, UInt16, UInt32, UInt64

from cyclonedds_idl import IdlStruct, types

from cydr.idl import (
    boolean,
    byte,
    flatten_schema_fields,
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


@dataclass
class Time(IdlStruct, typename="builtin_interfaces/msg/Time"):
    sec: types.int32 = 0
    nanosec: types.uint32 = 0


HEADER_SCHEMA = {
    "stamp": {
        "sec": int32,
        "nanosec": uint32,
    },
    "frame_id": string,
}


JOINT_STATE_SCHEMA = {
    "header": HEADER_SCHEMA,
    "name": NDArray[Any, Bytes],
    "position": NDArray[Any, Float64],
    "velocity": NDArray[Any, Float64],
    "effort": NDArray[Any, Float64],
}


EVERY_SUPPORTED_SCHEMA = {
    "boolean_value": boolean,
    "byte_value": byte,
    "signed_int8": int8,
    "unsigned_int8": uint8,
    "signed_int16": int16,
    "unsigned_int16": uint16,
    "signed_int32": int32,
    "unsigned_int32": uint32,
    "signed_int64": int64,
    "unsigned_int64": uint64,
    "float32_value": float32,
    "float64_value": float64,
    "text": string,
    "header": HEADER_SCHEMA,
    "bool_sequence": NDArray[Any, Bool],
    "byte_array": NDArray[Shape["3"], UInt8],
    "int8_sequence": NDArray[Any, Int8],
    "uint8_array": NDArray[Shape["3"], UInt8],
    "int16_sequence": NDArray[Any, Int16],
    "uint16_array": NDArray[Shape["2"], UInt16],
    "int32_sequence": NDArray[Any, Int32],
    "uint32_array": NDArray[Shape["2"], UInt32],
    "int64_sequence": NDArray[Any, Int64],
    "uint64_array": NDArray[Shape["2"], UInt64],
    "float32_sequence": NDArray[Any, Float32],
    "float64_array": NDArray[Shape["2"], Float64],
    "text_array": NDArray[Shape["2"], Bytes],
    "text_sequence": NDArray[Any, Bytes],
}


EVERY_SUPPORTED_SCHEMA_FLAT = flatten_schema_fields(EVERY_SUPPORTED_SCHEMA)


__all__ = [
    "Time",
    "HEADER_SCHEMA",
    "JOINT_STATE_SCHEMA",
    "EVERY_SUPPORTED_SCHEMA",
    "EVERY_SUPPORTED_SCHEMA_FLAT",
]
