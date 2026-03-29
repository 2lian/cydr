"""Public schema and codec API.

Naming plan:

- public schema tokens are short nouns: ``int32``, ``float64``, ``string``
- public collection builders are verbs: ``array(...)`` and ``sequence(...)``
- public compiled entrypoint is ``get_codec_for(...)``
- cache / import / compilation internals live in the private
  [`_runtime.py`](/home/elian/debug/cdr/xcdrjit/_runtime.py) module
"""

from ._message_ops import assert_messages_equal
from ._runtime import (
    CYTHON_CACHE_DIR,
    Codec,
    ComputeSizeFunction,
    DeserializerFunction,
    SerializerFunction,
    flatten_cython_value_list,
    get_codec_for,
    inflate_cython_value_tree,
    schema_type_hash,
)
from ._warmup import warmup_codec
from .cython_generator import (
    generate_cython_serializer_code,
)
from .schema_types import (
    ArrayType,
    NestedSchemaFields,
    SequenceType,
    array,
    boolean,
    byte,
    flatten_schema_fields,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    sequence,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .structs import XcdrStruct
