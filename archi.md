# xcdrjit Architecture

## Goal

`xcdrjit` generates and caches Cython-backed XCDR1 codecs from nested Python schema dictionaries.

The current supported subset is:

- little-endian XCDR1 only
- plain nested structs
- primitive scalars:
  - `bool`
  - signed and unsigned integers
  - `float32`
  - `float64`
  - `string` represented as `bytes` on the Python side
- primitive collections:
  - fixed arrays of selected primitive types
  - sequences of selected primitive types
  - string arrays and string sequences represented as `list[bytes]`

The library assumes the host machine is little-endian.

## Public API

The user-facing API lives in [`xcdrjit/idl.py`](/home/elian/debug/cdr/xcdrjit/idl.py).

Naming plan:

- public schema tokens are short nouns: `int32`, `float64`, `string`
- public collection helpers are `array(...)` and `sequence(...)`
- public compiled entrypoint is `get_codec_for(...)`
- private compile/cache/runtime helpers live in [`xcdrjit/_runtime.py`](/home/elian/debug/cdr/xcdrjit/_runtime.py)

Current user-facing functions:

- primitive schema tokens:
  - `boolean`, `byte`
  - `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
  - `float32`, `float64`
  - `string`
- collection schema helpers:
  - `array(element_type, length)`
  - `sequence(element_type)`
- `get_codec_for(schema)`
- `flatten_schema_fields(schema)`
- `flatten_cython_value_list(values)`
- `inflate_cython_value_tree(schema, flat_values)`

The runtime convention is:

- schema input is a nested `dict[str, primitive | array(...) | sequence(...) | dict[str, ...]]`
- serializer runtime input is one nested mapping
- `get_codec_for(...)` returns one codec object with `.compute_size(...)`, `.serialize(...)`, and `.deserialize(...)`
- runtime keys are ignored at call time
- values are flattened recursively in insertion order

This makes the generated module reusable for schemas that have the same flattened type sequence but different field names.

## Main Layers

### 1. Schema / Codegen Layer

[`xcdrjit/cython_generator.py`](/home/elian/debug/cdr/xcdrjit/cython_generator.py)

Responsibilities:

- define the schema descriptor language
- flatten nested schemas into a flat ordered field list
- render generated `.pyx` source for both serialize and deserialize entrypoints

Important property:

- generated modules use canonical argument names `arg_0`, `arg_1`, ... instead of schema field names
- cache identity depends only on flattened type order, not field names

### 2. Runtime / Cache Layer

Public facade:

- [`xcdrjit/idl.py`](/home/elian/debug/cdr/xcdrjit/idl.py)

Private runtime implementation:

- [`xcdrjit/_runtime.py`](/home/elian/debug/cdr/xcdrjit/_runtime.py)

Responsibilities:

- hash a flattened schema type sequence
- choose a cache directory
- materialize the generated `.pyx` if missing
- compile or import the generated extension module from cache
- expose user-friendly Python wrappers

Important property:

- generated extensions are cached persistently
- default cache directory is `./.xcdrjit_cache`
- if the default cache cannot be created, the runtime falls back to a temporary directory and emits a `RuntimeWarning`
- the backend helper extension `xcdrjit._every_supported_cython` must be importable before a generated module can load

### 3. Backend Helper Layer

[`xcdrjit/_every_supported_cython.pyx`](/home/elian/debug/cdr/xcdrjit/_every_supported_cython.pyx)
[`xcdrjit/_every_supported_cython.pxd`](/home/elian/debug/cdr/xcdrjit/_every_supported_cython.pxd)

Responsibilities:

- low-level XCDR1 primitives
- alignment
- encapsulation header write
- scalar write helpers
- primitive array / sequence write helpers
- string write helpers
- manual worked example:
  - `compute_serialized_size_every_supported_schema(...)`
  - `serialize_every_supported_schema(...)`

This file is intentionally kept as the explicit example of the generated style.

## Serialization Data Flow

### Generated path

1. User defines a nested schema dict.
2. `get_codec_for(...).serialize(...)` flattens the schema.
3. The flattened type sequence is hashed.
4. A generated `.pyx` module is created or reused from cache.
5. The generated module cimports helper functions from `_every_supported_cython`.
6. The user-facing wrapper accepts one nested mapping, flattens only its values, and calls the generated serializer.

### Generated module shape

Today a generated module contains:

- `compute_serialized_size_<hash>(...)`
- `serialize_<hash>(...)`
- `deserialize_<hash>(data)`

All three functions are short straight-line sequences of helper calls.

## Runtime Value Representation

Serializer input values are expected to be already normalized for the current prototype:

- strings: `bytes`
- text arrays/sequences: `list[bytes]`
- primitive arrays/sequences: 1D NumPy arrays with matching dtype
- scalars: Python scalars or NumPy scalars that can convert to the declared Cython type

Nested structs are represented as nested Python mappings, but are flattened on the wire.

Deserializer output values mirror that convention:

- strings: `bytes`
- text arrays/sequences: `list[bytes]`
- primitive arrays/sequences: 1D NumPy arrays
- nested structs: nested `dict[str, object]`

## Cache Model

The cache key is:

- a versioned hash of the flattened field type sequence only

This means these two schemas intentionally reuse the same generated module if their flattened type order is identical:

- same types, same order, different field names
- same types, same order, different nesting labels

The wrapper layer carries the schema names and nesting shape.

The version prefix in the hash lets `xcdrjit` evolve the generated module shape
without colliding with older cached serializer-only modules.

## Testing Layout

### Direct backend tests

These verify the manual `_every_supported_cython.pyx` example.

### Generated-module tests

These verify:

- code generation
- cache reuse
- generated serializer correctness against Cyclone
- generated roundtrip behavior

The generated test cache is shared across runs in:

- [`.xcdrjit_cache`](/home/elian/debug/cdr/.xcdrjit_cache)

## Deserializer Design

The deserializer should mirror the serializer architecture instead of introducing a second unrelated path.

### Backend responsibilities

Low-level decode helpers live in `_every_supported_cython.pyx` / `.pxd` for:

- encapsulation header validation
- scalar reads
- string reads
- primitive fixed-array reads
- primitive sequence reads
- text array / text sequence reads
- end-of-buffer validation

They should follow the same XCDR1 assumptions as the serializer:

- 4-byte encapsulation header
- align offset = 4
- align max = 8
- little-endian host and wire format

### Generated module responsibilities

The generated deserialize function:

- accept a bytes-like buffer
- validate the encapsulation header
- decode fields in flat order
- return the flat decoded values in canonical order

The generated module should stay schema-name agnostic, just like the serializer path.

### Runtime wrapper responsibilities

The runtime layer in `idl.py`:

- load the generated flat deserializer
- rebuild a nested dict using the original user schema

That means deserializer output should be:

- nested mappings matching the schema shape
- scalar strings as `bytes`
- text arrays/sequences as `list[bytes]`
- primitive arrays/sequences as NumPy arrays

### Testing strategy

The implemented test order is:

1. direct helper/backend tests on `_every_supported_cython`
2. generated deserialize wrapper tests for `EverySupportedSchema`
3. generated roundtrip tests:
   - `serialize(values) -> bytes`
   - `deserialize(bytes) -> values2`
   - `serialize(values2) == bytes`
4. feed Cyclone-produced bytes into our deserializer and reserialize them

This keeps each stage observable and avoids debugging codegen and wire logic at the same time.
