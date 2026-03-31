"""msgspec.Struct-based helpers for declaring cydr messages."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Optional, Self, get_type_hints

import msgspec

from ._runtime import (
    DEFAULT_STRING_COLLECTION_MODE,
    Codec,
    StringCollectionMode,
    get_codec_for,
)
from .schema_types import (
    FlatField,
    NestedSchemaFields,
    flatten_schema_fields,
    normalize_field_schema,
)


@dataclass(frozen=True, slots=True)
class _SchemaInfo:
    """Cached schema metadata for one ``XcdrStruct`` subclass.

    Built once per class by ``_schema_info()`` and stored on the class.
    """

    nested_structs_by_index: dict[int, type["XcdrStruct"]]
    schema: NestedSchemaFields
    flat_schema: dict[str, FlatField]


class XcdrStruct(msgspec.Struct, gc=False):
    """High-level message base class built on ``msgspec.Struct``.

    Field annotations reuse the same schema tokens as the low-level dict API:

    - primitive tokens such as ``int32`` or ``float64``
    - ``NDArray[Any, dtype]`` for variable-length sequences
    - ``NDArray[Shape['n'], dtype]`` for fixed-length arrays
    - nested ``XcdrStruct`` subclasses

    Instances can expose the same nested-dict and flat-value views used by the
    lower-level cydr runtime.
    """

    _cydr_schema_info: ClassVar[_SchemaInfo | None] = None
    _cydr_codec: ClassVar[Codec | None] = None

    @classmethod
    def _schema_info(cls) -> _SchemaInfo:
        """Return the cached ``_SchemaInfo`` for this class, building it on first call.

        Uses ``cls.__dict__`` directly (not attribute lookup) so that the
        ``ClassVar`` default on the base class is not returned for subclasses
        that have not yet built their own info.

        Returns:
            The ``_SchemaInfo`` for this struct class.

        Raises:
            TypeError: If any field annotation is not a supported schema token
                or nested ``XcdrStruct`` subclass.
        """
        cached: _SchemaInfo | None = cls.__dict__.get("_cydr_schema_info")
        if cached is not None:
            return cached

        resolved_annotations = get_type_hints(cls, include_extras=True)
        nested_structs_by_index: dict[int, type["XcdrStruct"]] = {}
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
                schema[field_name] = nested_struct._schema_info().schema
                continue

            try:
                normalize_field_schema(annotation)
            except TypeError as exc:
                raise TypeError(
                    f"{cls.__name__}.{field_name} uses unsupported annotation "
                    f"{annotation!r}. Use cydr schema tokens, "
                    f"NDArray[Any, dtype], NDArray[Shape['n'], dtype], or a nested XcdrStruct subclass."
                ) from exc
            schema[field_name] = annotation

        flat_schema = flatten_schema_fields(schema)

        cached = _SchemaInfo(
            nested_structs_by_index=nested_structs_by_index,
            schema=schema,
            flat_schema=flat_schema,
        )
        cls._cydr_schema_info = cached
        return cached

    def _to_nested_dict(self) -> dict[str, object]:
        """Return the nested dict value representation of this message."""
        values = msgspec.structs.asdict(self)
        for field_name, value in values.items():
            if isinstance(value, XcdrStruct):
                values[field_name] = value._to_nested_dict()
        return values

    def _to_flat(self) -> list[object]:
        """Return the flattened runtime values in schema order."""
        flattened = list(msgspec.structs.astuple(self))
        nested_indices = self._schema_info().nested_structs_by_index.keys()

        # for index, value in enumerate(flattened):
            # if isinstance(value, XcdrStruct):
                # nested_indices.append(index)

        for index in reversed(nested_indices):
            flattened[index : index + 1] = flattened[index]._to_flat()
        return flattened

    @classmethod
    def _from_flat(cls, values: list[object] | tuple[object, ...]) -> Self:
        """Construct this struct from the flattened runtime value representation."""
        schema_info = cls._schema_info()

        field_values = list(values) if type(values) is not list else values

        for index, nested_struct in schema_info.nested_structs_by_index.items():
            nested_width = len(nested_struct._schema_info().flat_schema)
            nested_values = field_values[index : index + nested_width]

            field_values[index : index + nested_width] = [
                nested_struct._from_flat(nested_values)
            ]

        if len(field_values) != len(cls.__struct_fields__):
            if len(field_values) < len(cls.__struct_fields__):
                raise ValueError("Flat value list is shorter than the schema.")
            if len(field_values) > len(cls.__struct_fields__):
                raise ValueError("Flat value list is longer than the schema.")
        return cls(*field_values)

    @classmethod
    def _from_nested_dict(cls, values: Mapping[str, object]) -> Self:
        """Construct this struct from the nested dict value representation.

        Args:
            values: A nested ``dict``-like mapping matching the schema shape
                of this struct.

        Returns:
            A new instance of this struct class.
        """
        schema_info = cls._schema_info()
        kwargs: dict[str, object] = {}
        for index, field_name in enumerate(cls.__struct_fields__):
            nested_struct = schema_info.nested_structs_by_index.get(index)
            value = values[field_name]
            if nested_struct is not None:
                kwargs[field_name] = nested_struct._from_nested_dict(value)
            else:
                kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def _get_codec(cls) -> Codec:
        """Return the cached codec for this struct."""
        cached: Codec | None = cls.__dict__.get("_cydr_codec")

        if cached is not None:
            return cached

        cached = get_codec_for(cls._schema_info().schema)
        cls._cydr_codec = cached
        return cached

    def serialize(self) -> bytearray:
        """Serialize this message instance."""
        return self.__class__._get_codec().serialize(self._to_flat())

    @classmethod
    def deserialize(
        cls, data: object, string_collections: Optional[StringCollectionMode] = None
    ) -> Self:
        """Deserialize one payload into this struct type."""
        return cls._from_flat(
            cls._get_codec().deserialize(
                data,
                flat=True,
                string_collections=(
                    string_collections
                    if string_collections is not None
                    else DEFAULT_STRING_COLLECTION_MODE
                ),
            )
        )

    @classmethod
    def brew(cls):
        cls.deserialize(cls().serialize())
