"""cydr schema type tokens.

All primitive tokens and nptyping re-exports live here.
Import this module to define dict-style schemas, and import ``XcdrStruct``
separately from ``cydr`` or ``cydr.idl`` for struct annotations::

    from cydr import types
    from cydr import XcdrStruct
    import typing

    # dict-style schema using nptyping annotations
    schema = {
        "stamp": {"sec": types.int32, "nanosec": types.uint32},
        "name":     types.NDArray[typing.Any, types.Bytes],
        "position": types.NDArray[typing.Any, types.Float64],
        "effort":   types.NDArray[types.Shape["3"], types.Float64],
    }

    # XcdrStruct using the same annotations
    class JointState(XcdrStruct):
        name:     types.NDArray[typing.Any, types.Bytes]
        position: types.NDArray[typing.Any, types.Float64]
        effort:   types.NDArray[types.Shape["3"], types.Float64]
"""

from typing import Any, TypeAlias  # re-export Any for use in NDArray[Any, dtype]

import numpy as np

# nptyping NDArray, Shape, and dtype aliases (re-exported for convenience)
from nptyping import (
    Bool,
    Bytes,
    Float,
    Float16,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    NDArray,
    Shape,
    UByte,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

# ---------------------------------------------------------------------------
# Primitive schema tokens
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

type PrimitiveSchemaType = (
    type[boolean]
    | type[uint8]
    | type[int8]
    | type[int16]
    | type[uint16]
    | type[int32]
    | type[uint32]
    | type[int64]
    | type[uint64]
    | type[float32]
    | type[float64]
    | type[bytes]
)
