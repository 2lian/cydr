"""msgspec.Struct-based helpers for declaring xcdrjit messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Self, get_type_hints

import msgspec

from ._runtime import Codec, get_codec_for
from .cython_generator import NestedCythonFields, flatten_cython_fields
from .schema_types import FlatField, normalize_schema_field


@dataclass(frozen=True, slots=True)
class _SchemaState:
    nested_structs_by_index: dict[int, type[XcdrStruct]]
    schema_dict: NestedCythonFields
    flat_schema: dict[str, FlatField]


class XcdrStruct(msgspec.Struct, gc=False):
    """High-level message base class built on ``msgspec.Struct``.

    Field annotations reuse the same schema tokens as the low-level dict API:

    - primitive tokens such as ``int32`` or ``float64``
    - ``array(element_type, length)``
    - ``sequence(element_type)``
    - nested ``XcdrStruct`` subclasses

    Instances can expose the same nested-dict and flat-value views used by the
    lower-level xcdrjit runtime.
    """

    _xcdr_schema_state: ClassVar[_SchemaState | None] = None
    _xcdr_codec: ClassVar[Codec | None] = None

    @classmethod
    def _schema_state(cls) -> _SchemaState:
        cached: _SchemaState | None = cls.__dict__.get("_xcdr_schema_state")
        if cached is not None:
            return cached

        resolved_annotations = get_type_hints(cls, include_extras=True)
        nested_structs_by_index: dict[int, type[XcdrStruct]] = {}
        schema: dict[str, object] = {}

        for index, field_name in enumerate(cls.__struct_fields__):
            annotation = resolved_annotations[field_name]
            nested_struct = (
                annotation
                if isinstance(annotation, type) and issubclass(annotation, XcdrStruct)
                else None
            )
            if nested_struct is not None:
                nested_structs_by_index[index] = nested_struct
                schema[field_name] = nested_struct._schema_state().schema_dict
                continue

            try:
                normalize_schema_field(annotation)
            except TypeError as exc:
                raise TypeError(
                    f"{cls.__name__}.{field_name} uses unsupported annotation "
                    f"{annotation!r}. Use xcdrjit schema tokens, array(...), "
                    f"sequence(...), or a nested XcdrStruct subclass."
                ) from exc
            schema[field_name] = annotation

        flat_schema = flatten_cython_fields(schema)

        cached = _SchemaState(
            nested_structs_by_index=nested_structs_by_index,
            schema_dict=schema,
            flat_schema=flat_schema,
        )
        cls._xcdr_schema_state = cached
        return cached

    def _to_message_dict(self) -> dict[str, object]:
        """Return the nested dict value representation of this message."""
        values = msgspec.structs.asdict(self)
        nested_keys: list[str] = []

        for field_name, value in values.items():
            if isinstance(value, XcdrStruct):
                nested_keys.append(field_name)

        for field_name in nested_keys:
            values[field_name] = values[field_name]._to_message_dict()
        return values

    def _to_flat(self) -> list[object]:
        """Return the flattened runtime values in schema order."""
        flattened = list(msgspec.structs.astuple(self))
        nested_indices: list[int] = []

        for index, value in enumerate(flattened):
            if isinstance(value, XcdrStruct):
                nested_indices.append(index)

        for index in reversed(nested_indices):
            flattened[index : index + 1] = flattened[index]._to_flat()
        return flattened

    @classmethod
    def _from_flat_values(cls, values: list[object] | tuple[object, ...]) -> Self:
        """Construct this struct from the flattened runtime value representation."""
        state = cls._schema_state()
        field_values = list(values)

        for index, nested_struct in state.nested_structs_by_index.items():
            nested_width = len(nested_struct._schema_state().flat_schema)
            nested_values = field_values[index : index + nested_width]

            field_values[index : index + nested_width] = [
                nested_struct._from_flat_values(nested_values)
            ]

        if len(field_values) < len(cls.__struct_fields__):
            raise ValueError("Flat value list is shorter than the schema.")
        if len(field_values) > len(cls.__struct_fields__):
            raise ValueError("Flat value list is longer than the schema.")
        return cls(*field_values)

    @classmethod
    def _from_message_dict(cls, values: Mapping[str, object]) -> Self:
        """Construct this struct from the nested dict value representation."""
        state = cls._schema_state()
        kwargs: dict[str, object] = {}
        for index, field_name in enumerate(cls.__struct_fields__):
            nested_struct = state.nested_structs_by_index.get(index)
            value = values[field_name]
            if nested_struct is not None:
                kwargs[field_name] = nested_struct._from_message_dict(value)
            else:
                kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def _get_codec(cls) -> Codec:
        """Return the cached codec for this struct."""
        cached: Codec | None = cls.__dict__.get("_xcdr_codec")

        if cached is not None:
            return cached

        cached = get_codec_for(cls._schema_state().schema_dict)
        cls._xcdr_codec = cached
        return cached

    def serialize(self) -> bytearray:
        """Serialize this message instance."""
        return self.__class__._get_codec().serialize(self._to_flat())

    @classmethod
    def deserialize(cls, data: object) -> Self:
        """Deserialize one payload into this struct type."""
        return cls._from_flat_values(cls._get_codec().deserialize(data, flat=True))


__all__ = ["XcdrStruct"]
