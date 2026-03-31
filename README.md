# `cydr` Cython CDR

`cydr` is a fast, opinionated `XCDR1` serializer and deserializer for Python. At runtime, it dynamically generates a small Cython codec for your schema, compiles it down to C and uses it to (de)serialize payloads. There is no compilation step for the user, we do it Just In Time using Cython.

Priorities:

1. SPEED
2. Pythonic-style
3. JIT compilation

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
> - arrays and sequences must be 1D NumPy arrays with the matching dtype
> - string collections are much slower than numeric arrays, and large arrays / sequences of strings should be avoided if speed matters. One long `string` entry is fine.
>   - One `100 KB` `string` deserialized in about `2.83 us`, while `10,000` short strings totaling a similar byte volume took about `302.07 us`.
> - arrays or sequences of nested schemas are not supported
> - enums, unions, optionals, bounded strings, `char` / `wchar`, XCDR2 mutable or appendable encodings, and big-endian targets are not supported
>
> If you need anything outside this subset, use `cyclonedds_idl` directly. Contributions to extend our subset are however welcome!

## Benchmarks

`cydr` exists primarily to be faster than the general-purpose Python path while staying easy to call from Python.

The latest local `JointState` benchmark results against `cyclonedds_idl`:

Serialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `2.47 us` | `5.85x` |
| small | 8 | 372 | `cydr_struct` | `2.35 us` | `6.17x` |
| small | 8 | 372 | `cyclonedds_idl` | `14.12 us` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `50.20 us` | `171.83x` |
| large | 10000 | 400052 | `cydr_struct` | `57.09 us` | `151.44x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `8237.49 us` | `1.00x` |

Deserialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `3.75 us` | `3.74x` |
| small | 8 | 372 | `cydr_struct` | `3.06 us` | `4.50x` |
| small | 8 | 372 | `cyclonedds_idl` | `13.42 us` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `52.82 us` | `146.97x` |
| large | 10000 | 400052 | `cydr_struct` | `50.95 us` | `150.39x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `7422.52 us` | `1.00x` |

These numbers were produced with:

- `pixi run bench-joint-state-cython`

## Quick Start

```python
from typing import Any

import msgspec
import numpy as np
from nptyping import Bytes, Float64, NDArray

from cydr.idl import XcdrStruct
from cydr.types import int32, string, uint32


class Stamp(XcdrStruct):
    sec: int32 = np.int32(0)
    nanosec: uint32 = np.uint32(0)


class Header(XcdrStruct):
    stamp: Stamp = msgspec.field(default_factory=Stamp)
    frame_id: string = b""


class JointStateLite(XcdrStruct):
    header: Header = msgspec.field(default_factory=Header)
    name: NDArray[Any, Bytes] = msgspec.field(
        default_factory=lambda: np.empty(0, dtype=np.bytes_)
    )
    position: NDArray[Any, Float64] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    velocity: NDArray[Any, Float64] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    effort: NDArray[Any, Float64] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

JointStateLite.brew() # optional: forces caching and compilation of the schema

message = JointStateLite(
    header=Header(
        stamp=Stamp(sec=np.int32(10), nanosec=np.uint32(123)),
        frame_id=b"map",
    ),
    name=np.array([b"joint_a", b"joint_b"], dtype=np.bytes_),
    position=np.array([1.0, 2.0], dtype=np.float64),
)

payload = message.serialize()
decoded = JointStateLite.deserialize(payload)
```

## Schema Types
- **WIRE** -> **Python**

#### Primitive types
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
  - `string` -> UTF-8 `bytes`

#### Collection types
- `Sequence[Primitive]` -> `NDArray[Any, dtype]` defines a variable-length 1D collection of one primitive dtype.
- `Array[n, Primitive]` -> `NDArray[Shape["n"], dtype]` defines a fixed-length 1D collection of one primitive dtype.
- `Sequence[string]` -> `NDArray[Any, Bytes]` Represent a collection of strings with a numpy array of `np.bytes_` by default.
  - When deserializing you can choose the representation using the Enum `StringCollectionMode`. `.LIST` mode (`list[bytes]`) is more performant from small arrays, or you can use the most performant `.RAW` mode, to manipulate a C wrapper directly. Numpy 2 `.STRING_DTYPE` mode (`StringDType`) is usually slower.
  - When serializing, you can either pass `list[bytes]` or `array[np.bytes_]` or `array[np.StringDType]`

## Runtime Conventions

- Keys are ignored when calling the (de)serializers.
- Ordering of schema entries is critical and changes the schema.
- Deserializers rebuild messages using the schema shape provided at their creation.

This means schema order and runtime value order must match.

## Cache

Generated codecs are cached by the hash of the flattened field-type sequence.

- Default cache dir: `./.cydr_cache`
- If that cannot be created, `cydr` falls back to a temporary directory and emits a `RuntimeWarning`
- Override globally with `CYDR_CACHE_DIR=/path/to/cache`
- Override for the whole process with `CYDR_CACHE_DIR=/path/to/cache` before import

Test and benchmark schemas are defined next to those call sites instead of inside the library package.

# Benchmarks per primitives

| Primitive | Kind | Size | Count/Len | Payload Bytes | Dict Ser us | Struct Ser us | Cyclone Ser us | Dict Deser us | Struct Deser us | Cyclone Deser us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `boolean` | `scalar` | `scalar` | 1 | 5 | 0.47 | 0.71 | 2.46 | 0.95 | 0.80 | 2.38 |
| `boolean` | `array` | `small` | 16 | 20 | 0.64 | 0.92 | 2.83 | 1.04 | 0.93 | 2.42 |
| `boolean` | `array` | `big` | 250000 | 250004 | 8.81 | 9.14 | 3150.64 | 4.95 | 4.90 | 1694.94 |
| `byte` | `scalar` | `scalar` | 1 | 5 | 0.42 | 0.65 | 2.09 | 0.79 | 0.72 | 2.01 |
| `byte` | `array` | `small` | 16 | 20 | 0.57 | 0.81 | 1.97 | 0.91 | 0.82 | 1.87 |
| `byte` | `array` | `big` | 250000 | 250004 | 7.71 | 8.01 | 192.27 | 5.10 | 5.07 | 16.16 |
| `int8` | `scalar` | `scalar` | 1 | 5 | 0.43 | 0.65 | 2.09 | 0.81 | 0.73 | 2.04 |
| `int8` | `array` | `small` | 16 | 20 | 0.57 | 0.81 | 2.52 | 0.87 | 0.82 | 2.11 |
| `int8` | `array` | `big` | 250000 | 250004 | 7.80 | 8.10 | 5733.17 | 5.04 | 5.04 | 1960.90 |
| `uint8` | `scalar` | `scalar` | 1 | 5 | 0.42 | 0.65 | 2.09 | 0.83 | 0.72 | 2.05 |
| `uint8` | `array` | `small` | 16 | 20 | 0.57 | 0.81 | 1.97 | 0.91 | 0.76 | 1.88 |
| `uint8` | `array` | `big` | 250000 | 250004 | 7.74 | 8.10 | 196.64 | 5.05 | 5.05 | 14.45 |
| `int16` | `scalar` | `scalar` | 1 | 6 | 0.43 | 0.65 | 2.10 | 0.81 | 0.73 | 2.04 |
| `int16` | `array` | `small` | 16 | 36 | 0.57 | 0.81 | 2.52 | 0.87 | 0.82 | 2.15 |
| `int16` | `array` | `big` | 250000 | 500004 | 16.32 | 19.50 | 7527.46 | 11.66 | 11.75 | 2724.21 |
| `uint16` | `scalar` | `scalar` | 1 | 6 | 0.49 | 0.77 | 2.43 | 0.94 | 0.84 | 2.33 |
| `uint16` | `array` | `small` | 16 | 36 | 0.65 | 0.92 | 2.90 | 1.00 | 0.94 | 2.04 |
| `uint16` | `array` | `big` | 250000 | 500004 | 15.39 | 15.80 | 6377.00 | 10.53 | 10.58 | 2712.14 |
| `int32` | `scalar` | `scalar` | 1 | 8 | 0.43 | 0.67 | 2.11 | 0.75 | 0.73 | 2.33 |
| `int32` | `array` | `small` | 16 | 68 | 0.57 | 0.81 | 2.89 | 0.97 | 0.80 | 2.13 |
| `int32` | `array` | `big` | 250000 | 1000004 | 52.42 | 53.31 | 7638.93 | 38.43 | 38.22 | 3226.59 |
| `uint32` | `scalar` | `scalar` | 1 | 8 | 0.49 | 0.76 | 2.43 | 0.86 | 0.83 | 2.06 |
| `uint32` | `array` | `small` | 16 | 68 | 0.57 | 0.80 | 2.55 | 0.97 | 0.93 | 2.39 |
| `uint32` | `array` | `big` | 250000 | 1000004 | 44.47 | 45.22 | 6612.07 | 31.17 | 31.51 | 3116.57 |
| `int64` | `scalar` | `scalar` | 1 | 12 | 0.50 | 0.77 | 2.48 | 0.87 | 0.84 | 2.31 |
| `int64` | `array` | `small` | 16 | 132 | 0.58 | 0.83 | 2.51 | 0.83 | 0.82 | 2.13 |
| `int64` | `array` | `big` | 250000 | 2000004 | 107.15 | 107.91 | 6826.91 | 68.62 | 68.70 | 2958.38 |
| `uint64` | `scalar` | `scalar` | 1 | 12 | 0.44 | 0.68 | 2.14 | 0.83 | 0.73 | 2.06 |
| `uint64` | `array` | `small` | 16 | 132 | 0.57 | 0.81 | 2.50 | 0.83 | 0.82 | 2.21 |
| `uint64` | `array` | `big` | 250000 | 2000004 | 107.62 | 108.43 | 6891.52 | 70.17 | 69.52 | 2920.62 |
| `float32` | `scalar` | `scalar` | 1 | 8 | 0.43 | 0.66 | 2.10 | 0.80 | 0.83 | 2.32 |
| `float32` | `array` | `small` | 16 | 68 | 0.66 | 0.93 | 2.98 | 0.97 | 0.95 | 2.63 |
| `float32` | `array` | `big` | 250000 | 1000004 | 51.76 | 52.52 | 8315.61 | 39.41 | 39.31 | 3284.81 |
| `float64` | `scalar` | `scalar` | 1 | 12 | 0.42 | 0.63 | 2.11 | 0.80 | 0.73 | 2.04 |
| `float64` | `array` | `small` | 16 | 132 | 0.58 | 0.82 | 2.51 | 0.99 | 0.96 | 2.52 |
| `float64` | `array` | `big` | 250000 | 2000004 | 108.85 | 110.48 | 7392.84 | 73.73 | 73.56 | 3404.89 |
| `string` | `scalar` | `small` | 16 | 25 | 0.42 | 0.65 | 2.55 | 0.79 | 0.74 | 2.37 |
| `string` | `scalar` | `big` | 10000 | 10009 | 0.56 | 0.84 | 3.94 | 0.89 | 0.87 | 3.59 |
| `string` | `array` | `small` | 16 | 385 | 0.53 | 0.78 | 13.65 | 1.37 | 1.35 | 12.36 |
| `string` | `array` | `big` | 250000 | 2502000001 | 981533.50 | 977211.97 | 4446380.86 | 338691.79 | 352686.02 | 1765052.79 |
