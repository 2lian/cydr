"""Public warmup helper for steady-state cydr timings."""

from collections.abc import Mapping

from ._message_ops import assert_messages_equal
from ._runtime import Codec, get_codec_for
from .schema_types import NestedSchemaFields
from .structs import XcdrStruct


def warmup_codec(
    value: Mapping[str, object] | XcdrStruct,
    schema: NestedSchemaFields | None = None,
) -> Codec:
    """Warm one codec and verify one full encode/decode roundtrip.

    Args:
        value: One runtime message value. This may be either one nested dict
            mapping or one ``XcdrStruct`` instance.
        schema: The nested schema when using the dict API. Omit it for
            ``XcdrStruct`` values.

    Returns:
        The warmed ``Codec``.

    Raises:
        TypeError: If ``value`` and ``schema`` do not match the expected API.
        AssertionError: If the roundtrip result differs from the input value.
    """
    if isinstance(value, XcdrStruct):
        if schema is not None:
            raise TypeError(
                "warmup_codec(value, schema) does not accept schema when value is an XcdrStruct."
            )

        codec = type(value)._get_codec()
        payload = value.serialize()
        decoded = type(value).deserialize(payload)
        assert_messages_equal(
            decoded._to_nested_dict(),
            value._to_nested_dict(),
            type(value)._schema_info().schema,
        )
        assert bytes(decoded.serialize()) == bytes(payload)
        return codec

    if isinstance(value, Mapping):
        if schema is None:
            raise TypeError(
                "warmup_codec(value, schema) requires schema when value is a mapping."
            )

        codec = get_codec_for(schema)
        payload = codec.serialize(value)
        decoded = codec.deserialize(payload)
        assert_messages_equal(decoded, value, schema)
        assert bytes(codec.serialize(decoded)) == bytes(payload)
        return codec

    raise TypeError(
        "warmup_codec(value, schema) expects either one nested dict-like mapping "
        "or one XcdrStruct instance."
    )
