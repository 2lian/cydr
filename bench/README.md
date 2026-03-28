# Benchmarks

Active benchmarks:

- `pixi run bench-joint-state-cython`
- `pixi run bench-every-supported-cython`

Both benchmark scripts are written around the public runtime API:

1. Define a nested schema dictionary using `CythonFieldType`.
2. Call `load_cython_serialize_function(schema, cache_dir=...)`.
3. Build one nested runtime `dict` that mirrors the schema.
4. Compare `bytes(serialize(values))` against `cyclonedds_idl`.
5. Time `xcdrjit` against `idl_message.serialize()` with `timeit`.

Runtime conventions in the benchmark match the public API:

- scalar strings are `bytes`
- string arrays and sequences are `list[bytes]`
- numeric arrays and sequences are 1D NumPy arrays
- keys are ignored at call time, so insertion order must match the schema

Generated `.pyx` and compiled extension modules are cached locally under:

- `bench/_generated_cython_cache/joint_state`
- `bench/_generated_cython_cache/every_supported_schema`

So you can inspect the generated code after a benchmark run.

You can override timing settings, for example:

```bash
pixi run bench-joint-state-cython -- --repeat 5 --min-time 0.1
pixi run bench-every-supported-cython -- --repeat 5 --min-time 0.1
```
