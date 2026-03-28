"""User-facing schema helpers for generating and loading Cython serializers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
from collections.abc import Mapping
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pyximport

from .cython_generator import (
    CythonFieldType,
    NestedCythonFields,
    flatten_cython_fields,
    generate_cython_serializer_code,
    generate_cython_serializer_module,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("XCDRJIT_CACHE_DIR", Path.home() / ".cache" / "xcdrjit")
)
_PYXIMPORT_READY = False


def get_cython_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the persistent cache directory used for generated serializers."""
    if cache_dir is None:
        return _DEFAULT_CACHE_DIR
    return Path(cache_dir)


def schema_type_hash(schema: NestedCythonFields) -> str:
    """Return a stable hash for the flattened schema type sequence."""
    flattened = flatten_cython_fields(schema)
    payload = "\0".join(field_type.value for field_type in flattened.values()).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def flatten_cython_value_list(values: Mapping[str, object]) -> list[object]:
    """Flatten nested runtime values in insertion order and ignore keys."""
    flattened: list[object] = []

    for field_value in values.values():
        if isinstance(field_value, Mapping):
            flattened.extend(flatten_cython_value_list(field_value))
        else:
            flattened.append(field_value)

    return flattened


def _canonicalize_flattened_fields(
    flattened_fields: dict[str, CythonFieldType],
) -> dict[str, CythonFieldType]:
    return {
        f"arg_{index}": field_type
        for index, field_type in enumerate(flattened_fields.values())
    }


def _ensure_shadow_package(cache_dir: Path) -> None:
    shadow_package_dir = cache_dir / "xcdrjit"
    if shadow_package_dir.exists():
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        shadow_package_dir.symlink_to(_PACKAGE_DIR, target_is_directory=True)
    except OSError:
        shutil.copytree(_PACKAGE_DIR, shadow_package_dir)


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

    def wrapper(values: Mapping[str, object]):
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{kind} expects a single nested dict-like mapping of values."
            )

        normalized_args = flatten_cython_value_list(values)
        if len(normalized_args) != expected_arg_count:
            raise TypeError(
                f"{kind} expected {expected_arg_count} flattened values, "
                f"got {len(normalized_args)}."
            )
        return function(*normalized_args)

    wrapper.__name__ = kind
    wrapper.__doc__ = (
        f"{kind} for flattened fields {field_names}. "
        "Call with one nested dict-like mapping; keys are ignored and values are flattened in insertion order."
    )
    wrapper.field_names = field_names
    wrapper.schema_hash = schema_hash_value
    wrapper.cache_dir = cache_dir
    wrapper.module_name = function.__module__
    wrapper.__wrapped__ = function
    return wrapper


def load_cython_serializer(
    schema: NestedCythonFields,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
):
    """Load or build a cached serializer for a possibly nested schema.

    The cache key is the hash of the flattened schema type sequence. The
    returned callables accept one nested dict-like mapping whose values are
    flattened recursively in insertion order.
    """
    resolved_cache_dir = get_cython_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    flattened_fields = flatten_cython_fields(schema)
    schema_hash_value = schema_type_hash(schema)
    module_name = f"xcdrjit_schema_{schema_hash_value}"
    source_path = resolved_cache_dir / f"{module_name}.pyx"

    if not source_path.exists():
        canonical_fields = _canonicalize_flattened_fields(flattened_fields)
        source = generate_cython_serializer_module(module_name, canonical_fields)
        source_path.write_text(source, encoding="utf-8")

    module = _import_or_compile_generated_module(resolved_cache_dir, module_name)
    compute_impl = getattr(module, f"compute_serialized_size_{module_name}")
    serialize_impl = getattr(module, f"serialize_{module_name}")
    field_names = tuple(flattened_fields)

    compute = _bind_schema_callable(
        compute_impl,
        field_names,
        "compute_serialized_size",
        schema_hash_value,
        resolved_cache_dir,
    )
    serialize = _bind_schema_callable(
        serialize_impl,
        field_names,
        "serialize",
        schema_hash_value,
        resolved_cache_dir,
    )
    return compute, serialize


def load_cython_serialize_function(
    schema: NestedCythonFields,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
):
    """Load or build a cached serializer and return only the serialize callable."""
    _, serialize = load_cython_serializer(schema, cache_dir=cache_dir)
    return serialize


__all__ = [
    "CythonFieldType",
    "flatten_cython_fields",
    "flatten_cython_value_list",
    "generate_cython_serializer_code",
    "get_cython_cache_dir",
    "load_cython_serialize_function",
    "load_cython_serializer",
    "schema_type_hash",
]
