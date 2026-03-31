"""Private runtime helpers for generated cydr codecs."""

import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
import pyximport
import pyximport.pyximport as pyximport_runtime

from .cython_generator import generate_cython_codec_module_source
from .schema_types import (
    FlatField,
    NestedSchemaFields,
    field_schema_token,
    flatten_schema_fields,
)


@dataclass(frozen=True, slots=True)
class Codec:
    """Compiled codec trio for one schema."""

    compute_size: Callable
    serialize: Callable
    deserialize: Callable
    field_names: tuple[str, ...]
    schema_hash: str
    cache_dir: Path
    module_name: str


@dataclass(frozen=True, slots=True)
class GeneratedModuleInfo:
    """Resolved generated codec module and its cache metadata.

    Returned by ``_load_generated_codec`` and consumed immediately by
    ``get_codec_for`` to build the public ``Codec``.
    """

    module: object
    module_name: str
    flattened_fields: dict[str, FlatField]
    schema_hash: str
    cache_dir: Path


class StringCollectionMode(IntEnum):
    """Container mode for decoded string collections."""

    NUMPY = 0
    LIST = 1
    STRING_DTYPE = 2
    RAW = 3


DEFAULT_STRING_COLLECTION_MODE = StringCollectionMode.NUMPY

_PACKAGE_DIR = Path(__file__).resolve().parent
_PYXIMPORT_READY = False
_SCHEMA_HASH_VERSION = "cydr-codegen-v7"
_DEFAULT_CACHE_NAME = ".cydr_cache"
_FALLBACK_CACHE_DIR: Path | None = None
_ENV_CACHE_DIR = os.environ.get("CYDR_CACHE_DIR")
_SHADOW_INIT = '"""Minimal shadow package for generated cydr codecs."""\n'
_HELPER_BACKEND_FILES = (
    "_every_supported_cython.pyx",
    "_every_supported_cython.pxd",
)
_SCHEMA_HASH_LENGTH = 32


def _resolve_cache_dir() -> Path:
    """Resolve the cache directory used for generated codecs.

    Resolution order:
    1. The ``CYDR_CACHE_DIR`` environment variable.
    2. ``./.cydr_cache`` in the current working directory.

    If the default working-directory cache cannot be created, cydr falls back
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
            _FALLBACK_CACHE_DIR = Path(tempfile.mkdtemp(prefix="cydr_cache_"))
        warnings.warn(
            (
                f"Could not create default cydr cache directory "
                f"{preferred_cache_dir!s}: {exc}. "
                f"Falling back to temporary cache directory "
                f"{_FALLBACK_CACHE_DIR!s}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _FALLBACK_CACHE_DIR


CYDR_CACHE_DIR = _resolve_cache_dir()


def schema_hash(schema: NestedSchemaFields) -> str:
    """Return a stable hash for the flattened schema type sequence."""
    flattened = flatten_schema_fields(schema)
    payload = (
        _SCHEMA_HASH_VERSION
        + "\0"
        + "\0".join(field_schema_token(field_type) for field_type in flattened.values())
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_SCHEMA_HASH_LENGTH]


def flatten_runtime_values(values: Mapping[str, object]) -> list[object]:
    """Flatten nested runtime values in insertion order and ignore keys.

    This mirrors the calling convention of the generated codecs: only the value
    order matters at runtime. The schema supplies the expected structure and
    type layout, while the mapping keys are ignored when calling the compiled
    serializer.
    """
    flattened: list[object] = []

    for field_value in values.values():
        if isinstance(field_value, Mapping):
            flattened.extend(flatten_runtime_values(field_value))
        else:
            flattened.append(field_value)

    return flattened


def rebuild_runtime_values(
    schema: NestedSchemaFields,
    flat_values: list[object] | tuple[object, ...],
) -> dict[str, object]:
    """Rebuild a nested value mapping from flat decoded values.

    Args:
        schema: The original nested schema used to define the nesting shape.
        flat_values: Flat sequence of decoded values in schema order, as
            returned by the generated ``deserialize_<hash>`` function.

    Returns:
        A nested ``dict[str, object]`` matching the shape of ``schema``.

    Raises:
        ValueError: If ``flat_values`` has fewer or more elements than the
            schema expects.
    """
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


def _ensure_shadow_backend_package(cache_dir: Path) -> Path:
    """Prepare the ``cydr`` shadow package inside the cache directory.

    Generated modules ``cimport`` from ``cydr._every_supported_cython``.
    To allow ``pyximport`` to resolve that import from ``cache_dir``, this
    function creates a minimal ``cydr/`` package there with the helper
    ``.pyx`` and ``.pxd`` files linked or copied from the real package.

    Args:
        cache_dir: The root cache directory where generated modules live.

    Returns:
        The shadow ``cydr`` package directory inside ``cache_dir``.
    """
    shadow_package_dir = cache_dir / "cydr"

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

    return shadow_package_dir


def _load_or_compile_generated_module(cache_dir: Path, module_name: str):
    """Load a generated codec module from cache, compiling it first if needed.

    If a compiled ``.so`` already exists in ``cache_dir`` it is loaded
    directly.  Otherwise the ``.pyx`` source is compiled via ``pyximport``.

    Before loading, the function checks whether ``cydr._every_supported_cython``
    is already imported from a path outside ``cache_dir`` (e.g. a stale
    in-tree ``.so``).  If so, it evicts that entry so that pyximport will
    compile and load the shadow-package version instead.

    Args:
        cache_dir: The directory that contains the generated ``.pyx`` source
            and will receive the compiled ``.so``.
        module_name: The importable module name, e.g. ``"schema_<hash>"``.

    Returns:
        The loaded module object.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    helper_module = sys.modules.get("cydr._every_supported_cython")
    if helper_module is not None:
        helper_path = getattr(helper_module, "__file__", None)
        cache_helper_dir = (cache_dir / "cydr").resolve()
        if helper_path is None or not Path(helper_path).resolve().is_relative_to(
            cache_helper_dir
        ):
            sys.modules.pop("cydr._every_supported_cython", None)

    # First make the shared Cython backend importable from the cache.
    if "cydr._every_supported_cython" not in sys.modules:
        shadow_package_dir = _ensure_shadow_backend_package(cache_dir)
        helper_module_name = "cydr._every_supported_cython"
        helper_source_path = shadow_package_dir / "_every_supported_cython.pyx"
        helper_compiled_path = None

        global _PYXIMPORT_READY
        if not _PYXIMPORT_READY:
            pyximport.install(
                language_level=3,
                inplace=True,
                setup_args={"include_dirs": np.get_include()},
            )
            _PYXIMPORT_READY = True

        for suffix in EXTENSION_SUFFIXES:
            matches = sorted(
                shadow_package_dir.glob(f"_every_supported_cython*{suffix}")
            )
            if matches:
                helper_compiled_path = matches[0]
                break

        if helper_compiled_path is None:
            helper_compiled_path = Path(
                pyximport_runtime.build_module(
                    helper_module_name,
                    str(helper_source_path),
                    pyxbuild_dir=str(cache_dir / "_pyxbld"),
                    inplace=True,
                    language_level=3,
                )
            )

        helper_spec = importlib.util.spec_from_file_location(
            helper_module_name,
            helper_compiled_path,
        )
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError(
                f"Could not load compiled helper module from {helper_compiled_path}"
            )
        helper_module = importlib.util.module_from_spec(helper_spec)
        sys.modules[helper_module_name] = helper_module
        try:
            helper_spec.loader.exec_module(helper_module)
        except Exception:
            sys.modules.pop(helper_module_name, None)
            raise

    # Reuse an already-compiled extension module when it exists.
    for suffix in EXTENSION_SUFFIXES:
        matches = sorted(cache_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            continue

        compiled_path = matches[0]
        spec = importlib.util.spec_from_file_location(module_name, compiled_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load compiled serializer module from {compiled_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    # Otherwise import the generated .pyx and let pyximport compile it.
    sys.path.insert(0, str(cache_dir))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(module_name)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(cache_dir))


def _wrap_schema_call(
    function,
    field_names: tuple[str, ...],
    kind: str,
    schema_hash_value: str,
    cache_dir: Path,
):
    """Wrap a generated compute-size or serialize function for the public API.

    The wrapper accepts either a nested dict-like mapping (values are
    flattened in insertion order, keys ignored) or a flat list/tuple, then
    forwards positional arguments to the underlying Cython function.

    Args:
        function: The generated Cython ``compute_serialized_size_<hash>`` or
            ``serialize_<hash>`` function.
        field_names: Tuple of flattened schema field names, used only for
            arity checking and introspection.
        kind: Human-readable label for error messages (``"compute_serialized_size"``
            or ``"serialize"``).
        schema_hash_value: The schema hash string, attached as an attribute.
        cache_dir: The cache directory, attached as an attribute.

    Returns:
        A Python callable wrapping the generated compute-size or serialize function.
    """
    expected_arg_count = len(field_names)

    def wrapper(values: Mapping[str, object] | list[object] | tuple[object, ...]):
        if isinstance(values, Mapping):
            normalized_args = flatten_runtime_values(values)
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


def _wrap_deserialize_call(
    function,
    schema: NestedSchemaFields,
    field_names: tuple[str, ...],
    schema_hash_value: str,
    cache_dir: Path,
):
    """Wrap a generated deserialize function for the public API.

    The wrapper calls the underlying Cython ``deserialize_<hash>`` function,
    checks the returned flat tuple length and either returns the flat values
    directly (``flat=True``) or rebuilds the nested dict via
    ``rebuild_runtime_values``.

    Args:
        function: The generated Cython ``deserialize_<hash>`` function.
        schema: The original nested schema, used to rebuild the nested dict.
        field_names: Tuple of flattened schema field names, used for arity
            checking and introspection.
        schema_hash_value: The schema hash string, attached as an attribute.
        cache_dir: The cache directory, attached as an attribute.

    Returns:
        A Python callable wrapping the generated deserialize function.
    """
    expected_arg_count = len(field_names)

    def wrapper(
        data,
        flat: bool = False,
        string_collections: StringCollectionMode = DEFAULT_STRING_COLLECTION_MODE,
    ):
        flat_values = function(data, int(string_collections))
        if len(flat_values) != expected_arg_count:
            raise ValueError(
                f"deserialize expected {expected_arg_count} flattened values, "
                f"got {len(flat_values)}."
            )
        if flat:
            return flat_values
        else:
            return rebuild_runtime_values(schema, flat_values)

    wrapper.__name__ = "deserialize"
    wrapper.__doc__ = (
        f"deserialize for flattened fields {field_names}. "
        "Returns a nested dict by default, or the flat decoded value tuple when called with flat=True. "
        "Use a StringCollectionMode value to choose the runtime container "
        "for decoded string collections."
    )
    wrapper.field_names = field_names
    wrapper.schema_hash = schema_hash_value
    wrapper.cache_dir = cache_dir
    wrapper.module_name = function.__module__
    wrapper.__wrapped__ = function
    return wrapper


def _load_generated_codec(
    schema: NestedSchemaFields,
) -> GeneratedModuleInfo:
    """Resolve, materialise, and load the generated codec module for a schema.

    Flattens the schema, hashes the type sequence, writes the ``.pyx`` source
    to the cache directory if it does not exist, then loads or compiles the
    extension module.

    Args:
        schema: A nested schema mapping as accepted by ``get_codec_for``.

    Returns:
        A ``GeneratedModuleInfo`` with the loaded module and its metadata.
    """
    resolved_cache_dir = CYDR_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    flattened_fields = flatten_schema_fields(schema)
    schema_hash_value = schema_hash(schema)
    module_name = f"schema_{schema_hash_value}"
    source_path = resolved_cache_dir / f"{module_name}.pyx"

    if not source_path.exists():
        canonical_fields = {
            f"arg_{index}": field_type
            for index, field_type in enumerate(flattened_fields.values())
        }
        source = generate_cython_codec_module_source(module_name, canonical_fields)
        source_path.write_text(source, encoding="utf-8")

    return GeneratedModuleInfo(
        module=_load_or_compile_generated_module(resolved_cache_dir, module_name),
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
    - ``codec.deserialize(data, string_collections=StringCollectionMode.LIST)`` returns string collections as ``list[bytes]``
    """
    module_info = _load_generated_codec(schema)
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

    compute = _wrap_schema_call(
        compute_impl,
        field_names,
        "compute_serialized_size",
        module_info.schema_hash,
        module_info.cache_dir,
    )
    serialize = _wrap_schema_call(
        serialize_impl,
        field_names,
        "serialize",
        module_info.schema_hash,
        module_info.cache_dir,
    )
    deserialize = _wrap_deserialize_call(
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
