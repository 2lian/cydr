"""Private runtime helpers for generated xcdrjit codecs."""

import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Protocol

import numpy as np
import pyximport

from .cython_generator import (
    generate_cython_serializer_module,
)
from .schema_types import (
    FlatField,
    NestedSchemaFields,
    field_cache_token,
    flatten_schema_fields,
)

class ComputeSizeFunction(Protocol):
    """Callable returned by ``get_codec_for(...).compute_size``."""

    field_names: tuple[str, ...]
    schema_hash: str
    cache_dir: Path
    module_name: str

    def __call__(
        self,
        values: Mapping[str, object] | list[object] | tuple[object, ...],
        /,
    ) -> int: ...


class SerializerFunction(Protocol):
    """Callable returned by ``get_codec_for(...).serialize``."""

    field_names: tuple[str, ...]
    schema_hash: str
    cache_dir: Path
    module_name: str

    def __call__(
        self,
        values: Mapping[str, object] | list[object] | tuple[object, ...],
        /,
    ) -> bytearray: ...


class DeserializerFunction(Protocol):
    """Callable returned by ``get_codec_for(...).deserialize``."""

    field_names: tuple[str, ...]
    schema_hash: str
    cache_dir: Path
    module_name: str

    def __call__(
        self,
        data: object,
        /,
        *,
        flat: bool = False,
    ) -> dict[str, object] | tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class Codec:
    """Compiled codec trio for one schema."""

    compute_size: ComputeSizeFunction
    serialize: SerializerFunction
    deserialize: DeserializerFunction
    field_names: tuple[str, ...]
    schema_hash: str
    cache_dir: Path
    module_name: str


@dataclass(frozen=True, slots=True)
class GeneratedModuleInfo:
    """Resolved generated codec module and its cache metadata."""

    module: object
    module_name: str
    flattened_fields: dict[str, FlatField]
    schema_hash: str
    cache_dir: Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_PYXIMPORT_READY = False
_SCHEMA_HASH_VERSION = "xcdrjit-codegen-v4"
_DEFAULT_CACHE_NAME = ".xcdrjit_cache"
_FALLBACK_CACHE_DIR: Path | None = None
_ENV_CACHE_DIR = os.environ.get("XCDRJIT_CACHE_DIR")
_SHADOW_INIT = '"""Minimal shadow package for generated xcdrjit codecs."""\n'
_HELPER_BACKEND_FILES = (
    "_every_supported_cython.pyx",
    "_every_supported_cython.pxd",
)
_SCHEMA_HASH_LENGTH = 32


def _resolve_cython_cache_dir() -> Path:
    """Resolve the cache directory used for generated codecs.

    Resolution order:
    1. The ``XCDRJIT_CACHE_DIR`` environment variable.
    2. ``./.xcdrjit_cache`` in the current working directory.

    If the default working-directory cache cannot be created, xcdrjit falls back
    to a temporary directory and emits a ``RuntimeWarning``.
    """
    if _ENV_CACHE_DIR is not None:
        return Path(_ENV_CACHE_DIR)

    preferred_cache_dir = Path.cwd() / _DEFAULT_CACHE_NAME
    try:
        preferred_cache_dir.mkdir(parents=True, exist_ok=True)
        return preferred_cache_dir
    except OSError as exc:
        global _FALLBACK_CACHE_DIR
        if _FALLBACK_CACHE_DIR is None:
            _FALLBACK_CACHE_DIR = Path(tempfile.mkdtemp(prefix="xcdrjit_cache_"))
        warnings.warn(
            (
                f"Could not create default xcdrjit cache directory "
                f"{preferred_cache_dir!s}: {exc}. "
                f"Falling back to temporary cache directory "
                f"{_FALLBACK_CACHE_DIR!s}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _FALLBACK_CACHE_DIR


CYTHON_CACHE_DIR = _resolve_cython_cache_dir()


def schema_type_hash(schema: NestedSchemaFields) -> str:
    """Return a stable hash for the flattened schema type sequence."""
    flattened = flatten_schema_fields(schema)
    payload = (
        _SCHEMA_HASH_VERSION
        + "\0"
        + "\0".join(field_cache_token(field_type) for field_type in flattened.values())
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_SCHEMA_HASH_LENGTH]


def flatten_cython_value_list(values: Mapping[str, object]) -> list[object]:
    """Flatten nested runtime values in insertion order and ignore keys.

    This mirrors the calling convention of the generated codecs: only the value
    order matters at runtime. The schema supplies the expected structure and
    type layout, while the mapping keys are ignored when calling the compiled
    serializer.
    """
    flattened: list[object] = []

    for field_value in values.values():
        if isinstance(field_value, Mapping):
            flattened.extend(flatten_cython_value_list(field_value))
        else:
            flattened.append(field_value)

    return flattened


def inflate_cython_value_tree(
    schema: NestedSchemaFields,
    flat_values: list[object] | tuple[object, ...],
) -> dict[str, object]:
    """Rebuild a nested value mapping from flat decoded values."""
    remaining = iter(flat_values)

    def build(node: NestedSchemaFields) -> dict[str, object]:
        rebuilt: dict[str, object] = {}
        for field_name, field_value in node.items():
            if isinstance(field_value, Mapping):
                rebuilt[field_name] = build(field_value)
            else:
                try:
                    rebuilt[field_name] = next(remaining)
                except StopIteration as exc:
                    raise ValueError(
                        "Decoded value list is shorter than the schema."
                    ) from exc
        return rebuilt

    rebuilt = build(schema)
    sentinel = object()
    if next(remaining, sentinel) is not sentinel:
        raise ValueError("Decoded value list is longer than the schema.")
    return rebuilt


def _canonicalize_flattened_fields(
    flattened_fields: dict[str, FlatField],
) -> dict[str, FlatField]:
    return {
        f"arg_{index}": field_type
        for index, field_type in enumerate(flattened_fields.values())
    }


def _ensure_shadow_package(cache_dir: Path) -> None:
    shadow_package_dir = cache_dir / "xcdrjit"

    cache_dir.mkdir(parents=True, exist_ok=True)
    if shadow_package_dir.is_symlink():
        shadow_package_dir.unlink()
    shadow_package_dir.mkdir(parents=True, exist_ok=True)

    init_path = shadow_package_dir / "__init__.py"
    if not init_path.exists() or init_path.read_text(encoding="utf-8") != _SHADOW_INIT:
        init_path.write_text(_SHADOW_INIT, encoding="utf-8")

    for filename in _HELPER_BACKEND_FILES:
        source_path = _PACKAGE_DIR / filename
        target_path = shadow_package_dir / filename

        if target_path.is_symlink():
            try:
                if target_path.resolve() == source_path.resolve():
                    continue
            except OSError:
                pass
            target_path.unlink()

        if target_path.exists():
            shutil.copy2(source_path, target_path)
            continue

        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def _ensure_pyximport_ready() -> None:
    global _PYXIMPORT_READY
    if _PYXIMPORT_READY:
        return

    pyximport.install(
        language_level=3,
        inplace=True,
        setup_args={"include_dirs": np.get_include()},
    )
    _PYXIMPORT_READY = True


def _find_compiled_module_path(cache_dir: Path, module_name: str) -> Path | None:
    for suffix in EXTENSION_SUFFIXES:
        matches = sorted(cache_dir.glob(f"{module_name}*{suffix}"))
        if matches:
            return matches[0]
    return None


def _load_compiled_module(module_name: str, compiled_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, compiled_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load compiled serializer module from {compiled_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_helper_backend_loaded(cache_dir: Path) -> None:
    if "xcdrjit._every_supported_cython" in sys.modules:
        return

    _ensure_shadow_package(cache_dir)
    _ensure_pyximport_ready()

    sys.path.insert(0, str(cache_dir))
    importlib.invalidate_caches()
    try:
        importlib.import_module("xcdrjit._every_supported_cython")
    finally:
        sys.path.remove(str(cache_dir))


def _import_or_compile_generated_module(cache_dir: Path, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]

    _ensure_helper_backend_loaded(cache_dir)

    compiled_path = _find_compiled_module_path(cache_dir, module_name)
    if compiled_path is not None:
        return _load_compiled_module(module_name, compiled_path)

    sys.path.insert(0, str(cache_dir))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(cache_dir))


def _bind_schema_callable(
    function,
    field_names: tuple[str, ...],
    kind: str,
    schema_hash_value: str,
    cache_dir: Path,
):
    expected_arg_count = len(field_names)

    def wrapper(values: Mapping[str, object] | list[object] | tuple[object, ...]):
        if isinstance(values, Mapping):
            normalized_args = flatten_cython_value_list(values)
        elif isinstance(values, (list, tuple)):
            normalized_args = values
        else:
            raise TypeError(
                f"{kind} expects either one nested dict-like mapping or one flat list/tuple of values."
            )

        if len(normalized_args) != expected_arg_count:
            raise TypeError(
                f"{kind} expected {expected_arg_count} flattened values, "
                f"got {len(normalized_args)}."
            )
        return function(*normalized_args)

    wrapper.__name__ = kind
    wrapper.__doc__ = (
        f"{kind} for flattened fields {field_names}. "
        "Call with one nested dict-like mapping or one flat list/tuple. "
        "For mappings, keys are ignored and values are flattened in insertion order."
    )
    wrapper.field_names = field_names
    wrapper.schema_hash = schema_hash_value
    wrapper.cache_dir = cache_dir
    wrapper.module_name = function.__module__
    wrapper.__wrapped__ = function
    return wrapper


def _bind_deserialize_callable(
    function,
    schema: NestedSchemaFields,
    field_names: tuple[str, ...],
    schema_hash_value: str,
    cache_dir: Path,
):
    expected_arg_count = len(field_names)

    def wrapper(data, flat: bool = False):
        flat_values = function(data)
        if len(flat_values) != expected_arg_count:
            raise ValueError(
                f"deserialize expected {expected_arg_count} flattened values, "
                f"got {len(flat_values)}."
            )
        if flat:
            return flat_values
        return inflate_cython_value_tree(schema, flat_values)

    wrapper.__name__ = "deserialize"
    wrapper.__doc__ = (
        f"deserialize for flattened fields {field_names}. "
        "Returns a nested dict by default, or the flat decoded value tuple when called with flat=True."
    )
    wrapper.field_names = field_names
    wrapper.schema_hash = schema_hash_value
    wrapper.cache_dir = cache_dir
    wrapper.module_name = function.__module__
    wrapper.__wrapped__ = function
    return wrapper


def _load_generated_schema_module(
    schema: NestedSchemaFields,
) -> GeneratedModuleInfo:
    resolved_cache_dir = CYTHON_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    flattened_fields = flatten_schema_fields(schema)
    schema_hash_value = schema_type_hash(schema)
    module_name = f"schema_{schema_hash_value}"
    source_path = resolved_cache_dir / f"{module_name}.pyx"

    if not source_path.exists():
        canonical_fields = _canonicalize_flattened_fields(flattened_fields)
        source = generate_cython_serializer_module(module_name, canonical_fields)
        source_path.write_text(source, encoding="utf-8")

    return GeneratedModuleInfo(
        module=_import_or_compile_generated_module(resolved_cache_dir, module_name),
        module_name=module_name,
        flattened_fields=flattened_fields,
        schema_hash=schema_hash_value,
        cache_dir=resolved_cache_dir,
    )


def get_codec_for(
    schema: NestedSchemaFields,
) -> Codec:
    """Return the cached codec for a schema.

    The cache key is the hash of the flattened schema type sequence. The
    returned codec accepts one nested dict-like mapping whose values are
    flattened recursively in insertion order for ``compute_size`` and
    ``serialize``.

    - ``codec.compute_size(values)`` returns the serialized byte length
    - ``codec.serialize(values)`` returns a ``bytearray`` containing the XCDR1 payload
    - ``codec.deserialize(data)`` returns a nested ``dict[str, object]``
    """
    module_info = _load_generated_schema_module(schema)
    compute_impl = getattr(
        module_info.module,
        f"compute_serialized_size_{module_info.module_name}",
    )
    serialize_impl = getattr(module_info.module, f"serialize_{module_info.module_name}")
    deserialize_impl = getattr(
        module_info.module,
        f"deserialize_{module_info.module_name}",
    )
    field_names = tuple(module_info.flattened_fields)

    compute = _bind_schema_callable(
        compute_impl,
        field_names,
        "compute_serialized_size",
        module_info.schema_hash,
        module_info.cache_dir,
    )
    serialize = _bind_schema_callable(
        serialize_impl,
        field_names,
        "serialize",
        module_info.schema_hash,
        module_info.cache_dir,
    )
    deserialize = _bind_deserialize_callable(
        deserialize_impl,
        schema,
        field_names,
        module_info.schema_hash,
        module_info.cache_dir,
    )
    return Codec(
        compute_size=compute,
        serialize=serialize,
        deserialize=deserialize,
        field_names=field_names,
        schema_hash=module_info.schema_hash,
        cache_dir=module_info.cache_dir,
        module_name=module_info.module_name,
    )
