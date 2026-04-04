# `cydr` Cython CDR

| Requirements | Compatibility | Tests Matrix |
|---|:---|:---:|
| [![python](https://img.shields.io/badge/Python-3.10--3.14-%20blue?logo=python&logoColor=white)](./pyproject.toml) <br> [![numpy](https://img.shields.io/badge/NumPy-1.26%20%7C%202.x-%20blue?logo=numpy&logoColor=white)](./pixi.toml) <br> [![mit](https://img.shields.io/badge/License-MIT-gold)](https://opensource.org/license/mit) | [![linux](https://img.shields.io/badge/OS-Linux-black?logo=linux&logoColor=white)](./pixi.toml) <br> [![windows](https://img.shields.io/badge/OS-Windows-black?logo=windows)](./pixi.toml) <br> [![macos](https://img.shields.io/badge/OS-macOS-black?logo=apple)](./pixi.toml) <br> [![xcdr1](https://img.shields.io/badge/Wire-XCDR1-blue)](./README.md#runtime-conventions) | Python: `3.10`, `3.11`, `3.12`, `3.13`, `3.14` <br> NumPy: `1.26`, `2.x`<br> [![Tests](https://github.com/2lian/cydr/actions/workflows/tests.yml/badge.svg)](https://github.com/2lian/cydr/actions/workflows/tests.yml)|

`cydr` is a fast, pythonic, focused `XCDR1` serializer/deserializer for Python with a schema API and just-in-time compilation. After defining your message schema with a Python class, `cydr` generates a Cython codec and compiles it down to C unlocking sub-microsecond operations.

> [!NOTE]
> A working C compiler toolchain is required at runtime.

Priorities:

1. SPEED
2. Pythonic-style
3. JIT compilation

Runnable examples:

- [`examples/roundtrip_custom_message.py`](examples/roundtrip_custom_message.py) for the minimal low-level nested-dict interface
- [`examples/roundtrip_xcdrstruct_message.py`](examples/roundtrip_xcdrstruct_message.py) for the higher-level `XcdrStruct` interface based on [`msgspec`](https://github.com/jcrist/msgspec)

> [!WARNING]
> `cydr` prioritizes speed and therefore supports only a focused subset of XCDR1: little-endian XCDR1, plain nested structs, and fixed arrays / sequences of scalars and strings.
>
> To stay fast, `cydr` is exclusively based on numpy and opinionated about runtime types. In particular, schema `string` means UTF-8 `bytes` at runtime, not Python `str`.
>
> Constraints:
> - `string` fields are `bytes`, and string arrays / sequences are NumPy arrays with `np.bytes_` dtype. However this datastructure is not optimal for all applications, hence:
>   - You can alternatively pass `list[bytes]` for encoding.
>   - You can alternatively decode into `list[bytes]` or raw C representation.
> - Arrays and sequences must be 1D NumPy arrays with the matching dtype
> - Collections of strings are much slower than numeric arrays, thus large arrays / sequences of strings should be avoided if speed matters. One long `string` entry is fine.
> - Collections of schemas are not supported, only collections of primitive types.
> - Enums, unions, optionals, bounded strings, `char` / `wchar`, XCDR2 mutable or appendable encodings, and big-endian targets are not supported
>
> If you need anything outside this subset, use `cyclonedds_idl` directly. Contributions to extend our subset are however welcome!

## Benchmarks

Those benchmarks were performed on `JointState` messages containing 3 sequence of floats, 1 sequence of strings, one nested `Header` (containing a string, and another nested `Time` with 2 integers). The `Count` row of table idicates the length of the 4 sequences.

#### Serialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `2.22 μs` | `6.53x` |
| small | 8 | 372 | `cydr_struct` | `1.99 μs` | `7.05x` |
| small | 8 | 372 | `cyclonedds_idl` | `14.01 μs` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `55.07 μs` | `164.55x` |
| large | 10000 | 400052 | `cydr_struct` | `54.70 μs` | `164.57x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `8993.45 μs` | `1.00x` |

#### Deserialize

| Case | Count | Bytes | Implementation | Median | Speedup |
|---|---:|---:|---|---:|---:|
| small | 8 | 372 | `cydr_dict` | `3.67 μs` | `4.21x` |
| small | 8 | 372 | `cydr_struct` | `2.84 μs` | `5.25x` |
| small | 8 | 372 | `cyclonedds_idl` | `14.80 μs` | `1.00x` |
| large | 10000 | 400052 | `cydr_dict` | `46.63 μs` | `170.36x` |
| large | 10000 | 400052 | `cydr_struct` | `41.01 μs` | `172.04x` |
| large | 10000 | 400052 | `cyclonedds_idl` | `7055.84 μs` | `1.00x` |

## Quick Start

```python
from typing import Any
import msgspec
import numpy as np
from nptyping import Bytes, Float64, NDArray

from cydr.idl import XcdrStruct
from cydr.types import int32, string, uint32

# Define adn use your message schema using a simple python class
# Type hints (`: int32` / `: uint32`) describe the cdr type of the field
class Time(XcdrStruct):
    sec: int32 = np.int32(0)
    nanosec: uint32 = np.uint32(0)

# Create more schemas using one you previously created
class Header(XcdrStruct):
    stamp: Time = msgspec.field(default_factory=Stamp)
    frame_id: string = b""

# And also define collections (arrays of set/varialbe length) using numpy and nptyping
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

# Optional: forces caching and compilation of the schema
JointState.brew()

# Instatiate a message with its nested components and arrays
message = JointState(
    header=Header(
        stamp=Stamp(sec=np.int32(10), nanosec=np.uint32(123)),
        frame_id=b"map",
    ),
    name=np.array([b"joint_a", b"joint_b"], dtype=np.bytes_),
    position=np.array([1.0, 2.0], dtype=np.float64),
)

# Encode / Decode
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
  - When deserializing you can choose the representation using the Enum `StringCollectionMode`. `.LIST` mode (`list[bytes]`) is more performant from small arrays, or you can use the most performant `.RAW` mode, to manipulate a `list` like C wrapper directly.
  - When serializing, you can either pass `list[bytes]` or `array[np.bytes_]`

## Cache

Generated codecs are cached by the hash of the flattened field-type sequence.

- Default cache dir: `./.cydr_cache`
- Override globally with `CYDR_CACHE_DIR=/path/to/cache`

# Benchmarks per primitives

| Primitive | Kind | Size | Count/Len | Payload Bytes | Dict Ser μs | Struct Ser μs | Cyclone Ser μs | Dict Deser μs | Struct Deser μs | Cyclone Deser μs |
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
