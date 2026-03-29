"""Public warmup helpers for steady-state xcdrjit timings."""

from collections.abc import Mapping

from ._message_ops import assert_messages_equal
from ._runtime import Codec, get_codec_for
from .schema_types import NestedSchemaFields
from .structs import XcdrStruct


def warmup_codec(
    value: Mapping[str, object] | XcdrStruct,
    schema: NestedSchemaFields | None = None,
) -> Codec:
    """Warm one codec and assert that one encode/decode roundtrip is correct.

    Args:
        value: One runtime message value. This may be either:
            - one nested dict-like mapping for the low-level schema API
            - one ``XcdrStruct`` instance for the high-level struct API
        schema: The nested schema for ``value`` when using the dict API.
            This must be omitted for ``XcdrStruct`` values.

    Returns:
        The warmed ``Codec`` instance.

    Raises:
        TypeError: If ``value`` is neither a mapping nor an ``XcdrStruct``, or
            if ``schema`` is missing for the dict API, or provided for the
            struct API.
        AssertionError: If the roundtrip decode does not match the original
            message value, or if reserializing the decoded value changes the
            payload bytes.
    """
    if isinstance(value, XcdrStruct):
        if schema is not None:
            raise TypeError(
                "warmup_codec(value, schema) does not accept schema when value is an XcdrStruct."
            )

        message = value
        codec = type(message)._get_codec()
        payload = message.serialize()
        decoded = type(message).deserialize(payload)
        assert_messages_equal(
            decoded._to_message_dict(),
            message._to_message_dict(),
            type(message)._schema_state().schema_dict,
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
