# `cydr` Cython CDR

`cydr` is a fast, opinionated `XCDR1` serializer and deserializer for Python. At runtime, it dynamically generates a small Cython codec for your schema, compiles it down to C and uses it to (de)serialize payloads. There is no compilation step for the user, we do it Just In Time using Cython.

Priorities:

1. SPEED
2. Pythonic-style
3. JIT compilation

Supported types:
- Primitive schema tokens:
  - `boolean`, `byte`
  - `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
  - `float32`, `float64`
  - `string` (UTF-8)
- collection schema annotations:
  - `NDArray[Any, dtype]`
  - `NDArray[Shape["n"], dtype]`
- `get_codec_for(...)`

Runnable examples:

- [`examples/roundtrip_custom_message.py`](examples/roundtrip_custom_message.py) for the low-level nested-dict interface
- [`examples/roundtrip_xcdrstruct_message.py`](examples/roundtrip_xcdrstruct_message.py) for the higher-level `XcdrStruct` interface

> [!WARNING]
> `cydr` prioritizes speed and therefore supports only a focused subset of XCDR1: little-endian XCDR1, plain nested structs, and fixed arrays / sequences of scalars and strings.
>
> To stay fast, `cydr` is intentionally opinionated about runtime types. In particular, schema `string` means UTF-8 `bytes` at runtime, not Python `str`. The library does not do implicit string encoding or decoding for you.
>
> Current constraints:
> - `string` fields are `bytes`, and string arrays / sequences are NumPy arrays with `np.bytes_` dtype
>   - `np.bytes_` dtype were introduced in `numpy>=2`. To not limit `numpy==1.*` users you can still serialize `list[bytes]` and deserialize using `deserialize(..., string_collections="list")`
> - arrays and sequences must be 1D NumPy arrays with the matching dtype
> - string collections are much slower than numeric arrays, and large arrays / sequences of strings should be avoided if speed matters. One long `string` entry is fine.
>   - One `100 KB` `string` deserialized in about `2.83 us`, while `10,000` short strings totaling a similar byte volume took about `302.07 us`.
> - arrays or sequences of nested schemas are not supported
> - enums, unions, optionals, bounded strings, `char` / `wchar`, XCDR2 mutable or appendable encodings, and big-endian targets are not supported
>
> If you need anything outside this subset, use `cyclonedds_idl` directly. Contributions to extend our subset are however welcome!

## Benchmarks

`cydr` exists primarily to be faster than the general-purpose Python path while staying easy to call from Python.

Latest local benchmark results against `cyclonedds_idl`:

| Benchmark | Case | `cydr` | `cyclonedds_idl` | Speedup |
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
from typing import Any

import numpy as np
from nptyping import Bytes, Float64, NDArray

from cydr.idl import (
    get_codec_for,
    int32,
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
    "name": NDArray[Any, Bytes],
    "position": NDArray[Any, Float64],
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
    "name": np.array([b"joint_a", b"joint_b"], dtype=np.bytes_),
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
- `NDArray[Any, dtype]` defines a variable-length 1D collection of one primitive dtype.
- `NDArray[Shape["n"], dtype]` defines a fixed-length 1D collection of one primitive dtype.

Runtime values:

- scalar numeric and boolean fields can be plain Python scalars or NumPy scalars
- numeric and boolean arrays/sequences should be 1D NumPy arrays with the matching dtype from the schema token
- string fields are `bytes`
- string arrays/sequences are NumPy arrays with `np.bytes_` dtype

## Runtime Conventions

- Keys are ignored when calling the (de)serializers.
- Ordering of dictionary entries is critical and changes the schema.
- Cached deserializers rebuild nested dicts using the original schema shape.

This means schema order and runtime value order must match.

## Cache

Generated codecs are cached by the hash of the flattened field-type sequence.

- Default cache dir: `./.cydr_cache`
- If that cannot be created, `cydr` falls back to a temporary directory and emits a `RuntimeWarning`
- Override globally with `CYDR_CACHE_DIR=/path/to/cache`
- Override for the whole process with `CYDR_CACHE_DIR=/path/to/cache` before import

Test and benchmark schemas are defined next to those call sites instead of inside the library package.

| Primitive | Kind | Size | Count/Len | Payload Bytes | Dict Ser us | Struct Ser us | Cyclone Ser us | Dict Deser us | Struct Deser us | Cyclone Deser us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `boolean` | `scalar` | `scalar` | 1 | 5 | 0.38 | 0.58 | 1.89 | 0.75 | 0.64 | 1.86 |
| `boolean` | `array` | `small` | 16 | 20 | 0.57 | 0.77 | 2.29 | 1.61 | 1.62 | 2.10 |
| `boolean` | `array` | `big` | 10000 | 10004 | 0.71 | 0.97 | 109.74 | 1.95 | 1.86 | 40.02 |
| `byte` | `scalar` | `scalar` | 1 | 5 | 0.40 | 0.60 | 1.94 | 1.42 | 1.39 | 1.87 |
| `byte` | `array` | `small` | 16 | 20 | 0.55 | 0.76 | 1.89 | 1.58 | 1.57 | 1.80 |
| `byte` | `array` | `big` | 10000 | 10004 | 0.71 | 0.96 | 9.44 | 1.87 | 1.87 | 2.34 |
| `int8` | `scalar` | `scalar` | 1 | 5 | 0.40 | 0.61 | 1.94 | 1.41 | 1.37 | 1.93 |
| `int8` | `array` | `small` | 16 | 20 | 0.55 | 0.76 | 2.34 | 1.58 | 1.58 | 2.05 |
| `int8` | `array` | `big` | 10000 | 10004 | 0.70 | 0.95 | 144.75 | 1.87 | 1.87 | 65.93 |
| `uint8` | `scalar` | `scalar` | 1 | 5 | 0.40 | 0.68 | 2.19 | 1.53 | 1.27 | 1.91 |
| `uint8` | `array` | `small` | 16 | 20 | 0.56 | 0.76 | 1.89 | 1.51 | 1.59 | 1.79 |
| `uint8` | `array` | `big` | 10000 | 10004 | 0.71 | 1.12 | 10.93 | 2.16 | 2.14 | 2.71 |
| `int16` | `scalar` | `scalar` | 1 | 6 | 0.46 | 0.68 | 2.21 | 1.61 | 1.57 | 1.82 |
| `int16` | `array` | `small` | 16 | 36 | 0.55 | 0.76 | 2.35 | 1.56 | 1.59 | 2.11 |
| `int16` | `array` | `big` | 10000 | 20004 | 0.87 | 1.19 | 156.92 | 2.06 | 2.05 | 84.50 |
| `uint16` | `scalar` | `scalar` | 1 | 6 | 0.42 | 0.62 | 1.96 | 1.41 | 1.36 | 1.92 |
| `uint16` | `array` | `small` | 16 | 36 | 0.56 | 0.76 | 2.36 | 1.59 | 1.58 | 2.04 |
| `uint16` | `array` | `big` | 10000 | 20004 | 0.90 | 1.21 | 146.54 | 2.06 | 2.03 | 75.03 |
| `int32` | `scalar` | `scalar` | 1 | 8 | 0.41 | 0.61 | 1.97 | 1.38 | 1.37 | 1.92 |
| `int32` | `array` | `small` | 16 | 68 | 0.57 | 0.78 | 2.38 | 1.61 | 1.61 | 2.04 |
| `int32` | `array` | `big` | 10000 | 40004 | 1.66 | 1.96 | 156.73 | 2.47 | 2.47 | 85.61 |
| `uint32` | `scalar` | `scalar` | 1 | 8 | 0.41 | 0.61 | 1.98 | 1.35 | 1.39 | 1.99 |
| `uint32` | `array` | `small` | 16 | 68 | 0.56 | 0.77 | 2.37 | 1.60 | 1.61 | 2.07 |
| `uint32` | `array` | `big` | 10000 | 40004 | 1.68 | 1.96 | 151.57 | 2.36 | 2.39 | 92.83 |
| `int64` | `scalar` | `scalar` | 1 | 12 | 0.41 | 0.64 | 2.24 | 1.54 | 1.58 | 2.18 |
| `int64` | `array` | `small` | 16 | 132 | 0.63 | 0.87 | 2.67 | 1.82 | 1.81 | 2.30 |
| `int64` | `array` | `big` | 10000 | 80004 | 3.22 | 3.08 | 153.14 | 3.03 | 2.97 | 87.25 |
| `uint64` | `scalar` | `scalar` | 1 | 12 | 0.41 | 0.62 | 2.21 | 1.55 | 1.59 | 2.15 |
| `uint64` | `array` | `small` | 16 | 132 | 0.63 | 0.87 | 2.69 | 1.81 | 1.82 | 2.35 |
| `uint64` | `array` | `big` | 10000 | 80004 | 3.29 | 3.57 | 177.09 | 3.39 | 3.42 | 94.78 |
| `float32` | `scalar` | `scalar` | 1 | 8 | 0.45 | 0.69 | 2.19 | 1.56 | 1.58 | 2.17 |
| `float32` | `array` | `small` | 16 | 68 | 0.62 | 0.86 | 2.78 | 1.52 | 1.57 | 2.14 |
| `float32` | `array` | `big` | 10000 | 40004 | 1.73 | 1.99 | 181.49 | 2.44 | 2.43 | 123.19 |
| `float64` | `scalar` | `scalar` | 1 | 12 | 0.45 | 0.68 | 2.21 | 1.53 | 1.54 | 2.16 |
| `float64` | `array` | `small` | 16 | 132 | 0.62 | 0.87 | 2.76 | 1.80 | 1.80 | 2.42 |
| `float64` | `array` | `big` | 10000 | 80004 | 3.38 | 3.67 | 215.12 | 3.48 | 3.47 | 126.33 |
| `string` | `scalar` | `small` | 16 | 25 | 0.45 | 0.68 | 2.68 | 0.77 | 0.73 | 2.51 |
| `string` | `scalar` | `big` | 100000 | 100009 | 3.75 | 4.07 | 14.81 | 2.81 | 2.89 | 12.30 |
| `string` | `array` | `small` | 16 | 257 | 0.62 | 0.88 | 13.25 | 1.39 | 1.33 | 11.78 |
| `string` | `array` | `big` | 10000 | 160003 | 115.64 | 116.22 | 6960.46 | 314.88 | 313.95 | 6644.53 |
