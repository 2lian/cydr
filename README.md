# xcdrjit

`xcdrjit` is a fast, opinionated `XCDR1` serializer and deserializer for Python. At runtime, it dynamically generates a small Cython codec for your schema, compiles it once, caches it, and reuses it on later calls. There is no compilation step for the user, we do it Just In Time (jit).

The current target is common ROS 2 message shapes, not full DDS/XCDR coverage.

Priorities:

1. SPEED
2. Pythonic-style

- primitive schema tokens:
  - `boolean`, `byte`
  - `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
  - `float32`, `float64`
  - `string`
- collection schema helpers:
  - `array(element_type, length)`
  - `sequence(element_type)`
- `get_codec_for(...)`

Runnable examples:

- [`examples/roundtrip_custom_message.py`](examples/roundtrip_custom_message.py) for the low-level nested-dict interface
- [`examples/roundtrip_xcdrstruct_message.py`](examples/roundtrip_xcdrstruct_message.py) for the higher-level `XcdrStruct` interface

> [!WARNING]
> `xcdrjit` prioritizes speed and therefore supports only a focused subset of XCDR1: little-endian XCDR1, plain nested structs, and fixed arrays / sequences of scalars and strings.
>
> To stay fast, `xcdrjit` is intentionally opinionated about runtime types. In particular, schema `string` means UTF-8 `bytes` at runtime, not Python `str`. The library does not do implicit string encoding or decoding for you.
>
> Current constraints:
> - `string` fields are `bytes`, and string arrays / sequences are `list[bytes]`
> - arrays and sequences must be 1D NumPy arrays with the matching dtype
> - arrays or sequences of nested schemas are not supported
> - enums, unions, optionals, bounded strings, `char` / `wchar`, XCDR2 mutable or appendable encodings, and big-endian targets are not supported
>
> If you need anything outside that subset, use `cyclonedds_idl` directly. Contributions are welcome.

## Benchmarks

`xcdrjit` exists primarily to be faster than the general-purpose Python path while staying easy to call from Python.

Latest local benchmark results against `cyclonedds_idl`:

| Benchmark | Case | `xcdrjit` | `cyclonedds_idl` | Speedup |
|---|---:|---:|---:|---:|
| `JointState` | small | `2.51 us` | `15.40 us` | `6.14x` |
| `JointState` | large | `66.54 us` | `9208.58 us` | `138.38x` |
| `EverySupportedSchema` | small | `7.75 us` | `33.16 us` | `4.28x` |
| `EverySupportedSchema` | large | `68.08 us` | `8969.99 us` | `131.77x` |
| `FixedArray` | small | `2.42 us` | `9.88 us` | `4.09x` |
| `FixedArray` | large | `6.94 us` | `445.33 us` | `64.14x` |

These numbers were produced with:

- `pixi run bench-joint-state-cython -- --repeat 3 --min-time 0.05`
- `pixi run bench-every-supported-cython -- --repeat 3 --min-time 0.05`
- `pixi run bench-fixed-arrays-cython -- --repeat 3 --min-time 0.05`

## Quick Start

```python
import numpy as np

from xcdrjit.idl import (
    float64,
    get_codec_for,
    int32,
    sequence,
    string,
    uint32,
)

schema = {
    "header": {
        "stamp": {
            "sec": int32,
            "nanosec": uint32,
        },
        "frame_id": string,
    },
    "name": sequence(string),
    "position": sequence(float64),
}

my_codec = get_codec_for(schema)
my_serializer = my_codec.serialize
my_deserializer = my_codec.deserialize

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

payload = my_serializer(values)
decoded = my_deserializer(payload)
```

## Schema Types

- Scalar schema tokens are based on NumPy scalar dtypes:
  - `boolean` -> `np.bool_`
  - `byte`, `uint8` -> `np.uint8`
  - `int8` -> `np.int8`
  - `int16` -> `np.int16`
  - `uint16` -> `np.uint16`
  - `int32` -> `np.int32`
  - `uint32` -> `np.uint32`
  - `int64` -> `np.int64`
  - `uint64` -> `np.uint64`
  - `float32` -> `np.float32`
  - `float64` -> `np.float64`
- `string` is the exception: it maps to UTF-8 `bytes`, not a NumPy string dtype.
- `array(element_type, length)` defines a fixed-size collection of one schema token.
- `sequence(element_type)` defines a variable-size collection of one schema token.

Runtime values:

- scalar numeric and boolean fields can be plain Python scalars or NumPy scalars
- numeric and boolean arrays/sequences should be 1D NumPy arrays with the matching dtype from the schema token
- string fields are `bytes`
- string arrays/sequences are `list[bytes]`

## Runtime Conventions

- Keys are ignored when calling the (de)serializers.
- Ordering of dictionary entries is critical and changes the schema.
- Cached deserializers rebuild nested dicts using the original schema shape.

This means schema order and runtime value order must match.

## Cache

Generated codecs are cached by the hash of the flattened field-type sequence.

- Default cache dir: `./.xcdrjit_cache`
- If that cannot be created, `xcdrjit` falls back to a temporary directory and emits a `RuntimeWarning`
- Override globally with `XCDRJIT_CACHE_DIR=/path/to/cache`
- Override for the whole process with `XCDRJIT_CACHE_DIR=/path/to/cache` before import

Test and benchmark schemas are defined next to those call sites instead of inside the library package.
