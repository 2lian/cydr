"""Public cydr API for the dict-style codec.

Typical usage::

    from typing import Any
    from nptyping import Float64, NDArray
    from cydr.idl import int32, uint32, string, get_codec_for

    schema = {
        "header": {
            "stamp": {"sec": int32, "nanosec": uint32},
            "frame_id": string,
        },
        "position": NDArray[Any, Float64],
    }
    codec = get_codec_for(schema)
    payload = codec.serialize(values)
    decoded = codec.deserialize(payload)
    decoded_lists = codec.deserialize(payload, string_collections="list")

Runtime conventions:

- ``string`` fields are UTF-8 ``bytes``, not ``str``
- arrays and sequences are declared with ``nptyping.NDArray[...]``
- numeric/boolean collections are 1-D NumPy arrays with the matching dtype
- string collections are NumPy arrays with ``np.bytes_`` dtype
- ``codec.deserialize(..., string_collections="list")`` returns string collections as ``list[bytes]`` instead
- mapping keys are ignored at (de)serialize time; only value order matters
"""

from ._message_ops import assert_messages_equal
from ._runtime import (
    CYDR_CACHE_DIR,
    Codec,
    flatten_runtime_values,
    get_codec_for,
    rebuild_runtime_values,
    schema_hash,
)
from ._warmup import warmup_codec
from .cython_generator import generate_cython_codec_source
from .schema_types import (
    NestedSchemaFields,
    flatten_schema_fields,
)
from .structs import XcdrStruct
from .types import (
    boolean,
    byte,
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
