# xcdrjit

`xcdrjit` generates and caches Cython-backed XCDR1 serializers from simple nested Python schema dicts.

The public API is intentionally small and lives in [`xcdrjit.idl`](/home/elian/debug/cdr/xcdrjit/idl.py):

- `CythonFieldType`
- `generate_cython_serializer_code(...)`
- `load_cython_serializer(...)`
- `load_cython_serialize_function(...)`

The low-level helper backend used by generated modules is [`_every_supported_cython.pyx`](/home/elian/debug/cdr/xcdrjit/_every_supported_cython.pyx). It is kept as the worked example of the generated style.

A runnable example lives in [`examples/serialize_custom_message.py`](/home/elian/debug/cdr/examples/serialize_custom_message.py).

## Quick Start

```python
import numpy as np

from xcdrjit.idl import CythonFieldType, load_cython_serialize_function

schema = {
    "header": {
        "stamp": {
            "sec": CythonFieldType.INT32,
            "nanosec": CythonFieldType.UINT32,
        },
        "frame_id": CythonFieldType.STRING,
    },
    "name": CythonFieldType.TEXT_SEQUENCE,
    "position": CythonFieldType.FLOAT64_SEQUENCE,
}

serialize = load_cython_serialize_function(schema)

values = {
    "header": {
        "stamp": {
            "sec": 10,
            "nanosec": 123,
        },
        "frame_id": b"map",
    },
    "name": [b"joint_a", b"joint_b"],
    "position": np.array([1.0, 2.0], dtype=np.float64),
}

payload = serialize(values)
```

## Runtime Conventions

- Input to cached serializers is always one nested mapping.
- Keys are ignored at call time.
- Values are flattened recursively in insertion order.
- Scalar strings are `bytes`.
- String arrays and sequences are `list[bytes]`.
- Numeric arrays and sequences are 1D NumPy arrays with the expected dtype.

This means schema order and runtime value order must match.

## Generate Source Only

If you only want the generated `.pyx` source:

```python
from xcdrjit.idl import CythonFieldType, generate_cython_serializer_code

schema = {
    "text": CythonFieldType.STRING,
    "header": {
        "stamp": {
            "sec": CythonFieldType.INT32,
            "nanosec": CythonFieldType.UINT32,
        },
        "frame_id": CythonFieldType.STRING,
    },
    "values": CythonFieldType.FLOAT32_SEQUENCE,
}

pyx_source = generate_cython_serializer_code("my_message", schema)
```

## Cache

Generated serializers are cached by the hash of the flattened field-type sequence.

- Default cache dir: `~/.cache/xcdrjit`
- Override per call with `cache_dir=...`

Test and benchmark schemas are defined next to those call sites instead of inside the library package.
