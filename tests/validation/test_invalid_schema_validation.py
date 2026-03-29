import pytest

from xcdrjit import get_codec_for, int32
from xcdrjit import _runtime


def test_get_codec_for_rejects_non_mapping_root_schema() -> None:
    with pytest.raises(TypeError, match="Schema must be a mapping"):
        get_codec_for(["not", "a", "schema"])  # type: ignore[arg-type]


def test_get_codec_for_rejects_non_string_field_name() -> None:
    with pytest.raises(TypeError, match="Schema field names must be strings"):
        get_codec_for({1: int32})  # type: ignore[dict-item]


def test_get_codec_for_rejects_invalid_field_before_codegen(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_codegen(*args, **kwargs):
        raise AssertionError("codegen should not run for an invalid schema")

    monkeypatch.setattr(_runtime, "generate_cython_serializer_module", fail_codegen)

    with pytest.raises(TypeError, match="Unsupported field schema value"):
        get_codec_for({"value": object()})
