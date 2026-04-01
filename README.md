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

Those benchmarks were performed on `JointState` messages containing 3 sequence of floats, 1 sequence of strings, one nested `Header` (containing a string, and another nested `Time` with 2 integers). The `Count` row of table idicates the length of the 4 sequences.

#### Serialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `2.22 us` | `6.53x` |
| small | 8 | 372 | `cydr_struct` | `1.99 us` | `7.05x` |
| small | 8 | 372 | `cyclonedds_idl` | `14.01 us` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `55.07 us` | `164.55x` |
| large | 10000 | 400052 | `cydr_struct` | `54.70 us` | `164.57x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `8993.45 us` | `1.00x` |

#### Deserialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `3.67 us` | `4.21x` |
| small | 8 | 372 | `cydr_struct` | `2.84 us` | `5.25x` |
| small | 8 | 372 | `cyclonedds_idl` | `14.80 us` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `46.63 us` | `170.36x` |
| large | 10000 | 400052 | `cydr_struct` | `41.01 us` | `172.04x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `7055.84 us` | `1.00x` |

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


class JointState(XcdrStruct):
    header: Header = msgspec.field(default_factory=Header)
    name: NDArray[Any, Bytes] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.bytes_)
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

JointState.brew() # optional: forces caching and compilation of the schema

message = JointState(
    header=Header(
        stamp=Stamp(sec=np.int32(10), nanosec=np.uint32(123)),
        frame_id=b"map",
    ),
    name=np.array([b"joint_a", b"joint_b"], dtype=np.bytes_),
    position=np.array([1.0, 2.0], dtype=np.float64),
)

payload = message.serialize()
decoded = JointState.deserialize(payload)
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
- Ordering of schema entries is what dictates the codec.
- Deserializers rebuild messages using the schema shape provided at their creation.

This means schema order and runtime value order must match. We do not verify the data as it is an expensive operation when talking ~2us in python (creating the Struct object actually takes longer than the codec at those speed)

## Cache

Generated codecs are cached by the hash of the flattened field-type sequence.

- Default cache dir: `./.cydr_cache`
- Override globally with `CYDR_CACHE_DIR=/path/to/cache`

# Benchmarks per primitives

| Primitive | Kind | Size | Count/Len | Payload Bytes | Dict Ser us | Struct Ser us | Cyclone Ser us | Dict Deser us | Struct Deser us | Cyclone Deser us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `boolean` | `scalar` | `scalar` | 1 | 5 | 0.41 | 0.63 | 2.09 | 0.83 | 0.73 | 2.05 |
| `boolean` | `array` | `small` | 16 | 20 | 0.58 | 0.83 | 2.51 | 0.91 | 0.80 | 2.18 |
| `boolean` | `array` | `big` | 250000 | 250004 | 7.55 | 7.81 | 3117.59 | 4.75 | 4.82 | 961.79 |
| `byte` | `scalar` | `scalar` | 1 | 5 | 0.43 | 0.65 | 2.11 | 0.77 | 0.71 | 2.03 |
| `byte` | `array` | `small` | 16 | 20 | 0.65 | 0.95 | 2.22 | 1.00 | 0.91 | 2.10 |
| `byte` | `array` | `big` | 250000 | 250004 | 7.60 | 7.88 | 190.03 | 4.96 | 4.99 | 15.08 |
| `int8` | `scalar` | `scalar` | 1 | 5 | 0.48 | 0.77 | 2.37 | 0.90 | 0.81 | 2.31 |
| `int8` | `array` | `small` | 16 | 20 | 0.65 | 0.93 | 2.91 | 0.96 | 0.92 | 2.43 |
| `int8` | `array` | `big` | 250000 | 250004 | 9.06 | 9.40 | 5571.20 | 5.07 | 5.01 | 2017.53 |
| `uint8` | `scalar` | `scalar` | 1 | 5 | 0.42 | 0.64 | 2.08 | 0.78 | 0.71 | 2.06 |
| `uint8` | `array` | `small` | 16 | 20 | 0.58 | 0.93 | 2.23 | 0.93 | 0.78 | 1.90 |
| `uint8` | `array` | `big` | 250000 | 250004 | 7.67 | 7.91 | 190.90 | 5.00 | 5.03 | 16.97 |
| `int16` | `scalar` | `scalar` | 1 | 6 | 0.42 | 0.66 | 2.09 | 0.81 | 0.72 | 2.06 |
| `int16` | `array` | `small` | 16 | 36 | 0.58 | 0.83 | 2.51 | 0.81 | 0.81 | 2.17 |
| `int16` | `array` | `big` | 250000 | 500004 | 18.11 | 18.61 | 7405.62 | 12.76 | 12.73 | 3236.44 |
| `uint16` | `scalar` | `scalar` | 1 | 6 | 0.50 | 0.76 | 2.46 | 0.83 | 0.81 | 2.35 |
| `uint16` | `array` | `small` | 16 | 36 | 0.66 | 0.95 | 2.81 | 0.92 | 0.92 | 2.07 |
| `uint16` | `array` | `big` | 250000 | 500004 | 15.48 | 15.99 | 6504.11 | 10.82 | 10.90 | 2721.29 |
| `int32` | `scalar` | `scalar` | 1 | 8 | 0.43 | 0.66 | 2.10 | 0.78 | 0.71 | 2.06 |
| `int32` | `array` | `small` | 16 | 68 | 0.56 | 0.84 | 2.53 | 0.80 | 0.81 | 2.19 |
| `int32` | `array` | `big` | 250000 | 1000004 | 45.15 | 53.37 | 7345.25 | 39.06 | 39.71 | 2821.39 |
| `uint32` | `scalar` | `scalar` | 1 | 8 | 0.43 | 0.67 | 2.14 | 0.78 | 0.72 | 2.09 |
| `uint32` | `array` | `small` | 16 | 68 | 0.58 | 0.82 | 2.53 | 0.80 | 0.81 | 2.18 |
| `uint32` | `array` | `big` | 250000 | 1000004 | 44.25 | 45.20 | 6672.12 | 32.67 | 32.86 | 3102.01 |
| `int64` | `scalar` | `scalar` | 1 | 12 | 0.43 | 0.67 | 2.15 | 0.79 | 0.71 | 2.05 |
| `int64` | `array` | `small` | 16 | 132 | 0.68 | 0.96 | 2.91 | 0.93 | 0.94 | 2.47 |
| `int64` | `array` | `big` | 250000 | 2000004 | 124.78 | 126.39 | 7625.85 | 80.02 | 79.53 | 3187.78 |
| `uint64` | `scalar` | `scalar` | 1 | 12 | 0.52 | 0.80 | 2.46 | 0.77 | 0.72 | 2.10 |
| `uint64` | `array` | `small` | 16 | 132 | 0.67 | 0.95 | 2.90 | 0.93 | 0.92 | 2.46 |
| `uint64` | `array` | `big` | 250000 | 2000004 | 109.25 | 109.09 | 6826.88 | 68.66 | 68.83 | 2828.40 |
| `float32` | `scalar` | `scalar` | 1 | 8 | 0.42 | 0.66 | 2.11 | 0.79 | 0.72 | 2.08 |
| `float32` | `array` | `small` | 16 | 68 | 0.58 | 0.81 | 2.59 | 0.81 | 0.81 | 2.25 |
| `float32` | `array` | `big` | 250000 | 1000004 | 54.01 | 54.65 | 8183.15 | 38.92 | 38.87 | 3735.67 |
| `float64` | `scalar` | `scalar` | 1 | 12 | 0.41 | 0.65 | 2.61 | 0.78 | 0.72 | 2.10 |
| `float64` | `array` | `small` | 16 | 132 | 0.58 | 0.84 | 2.57 | 0.80 | 0.81 | 2.26 |
| `float64` | `array` | `big` | 250000 | 2000004 | 133.97 | 140.94 | 8115.61 | 83.78 | 83.42 | 3679.49 |
| `string` | `scalar` | `small` | 16 | 25 | 0.41 | 0.66 | 2.58 | 0.82 | 0.72 | 2.44 |
| `string` | `scalar` | `big` | 10000 | 10009 | 0.56 | 0.84 | 3.97 | 0.86 | 0.82 | 3.56 |
| `string` | `array` | `small` | 16 | 385 | 0.53 | 0.78 | 13.60 | 1.48 | 1.52 | 13.50 |
| `string` | `array` | `big` | 250000 | 2502000001 | 968083.78 | 969586.07 | 4387828.77 | 459644.93 | 457855.89 | 1765190.79 |
