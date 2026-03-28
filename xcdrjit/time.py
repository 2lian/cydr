from dataclasses import dataclass

from cyclonedds_idl import IdlStruct, types


@dataclass
class Time(IdlStruct, typename="builtin_interfaces/msg/Time"):
    sec: types.int32 = 0
    nanosec: types.uint32 = 0
