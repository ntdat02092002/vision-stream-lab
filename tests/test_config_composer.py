from pathlib import Path

import pytest

from vision_stream_lab.configuration import compose_config_document, load_config_document


def write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_nested_refs_deep_merge_and_interpolation(tmp_path):
    write_yaml(
        tmp_path / "presets" / "detector.yaml",
        """
backend: onnx
model_path: models/base.onnx
confidence: 0.2
classes: [0, 1, 2]
label: ${.backend}:${.model_path}
""",
    )
    write_yaml(
        tmp_path / "usecase.yaml",
        """
inference:
  $ref: presets/detector.yaml
  confidence: 0.3
tracker:
  enabled: false
""",
    )
    write_yaml(
        tmp_path / "deployment.yaml",
        """
config:
  $ref: usecase.yaml
  inference:
    confidence: 0.55
    classes: [7]
    model_path: models/override.onnx
""",
    )

    result = load_config_document(
        tmp_path / "deployment.yaml",
        config_root=tmp_path,
    )

    assert result["config"]["inference"] == {
        "backend": "onnx",
        "model_path": "models/override.onnx",
        "confidence": 0.55,
        "classes": [7],
        "label": "onnx:models/override.onnx",
    }
    assert result["config"]["tracker"] == {"enabled": False}


def test_ref_rejects_cycles_and_paths_outside_config_root(tmp_path):
    write_yaml(tmp_path / "a.yaml", "$ref: b.yaml\n")
    write_yaml(tmp_path / "b.yaml", "$ref: a.yaml\n")
    with pytest.raises(ValueError, match="Circular config reference"):
        load_config_document(tmp_path / "a.yaml", config_root=tmp_path)

    write_yaml(tmp_path / "escape.yaml", "$ref: ../outside.yaml\n")
    with pytest.raises(ValueError, match="escapes config root"):
        load_config_document(tmp_path / "escape.yaml", config_root=tmp_path)


def test_composition_tracks_the_winning_source_for_each_leaf(tmp_path):
    write_yaml(tmp_path / "preset.yaml", "model: base.onnx\nconfidence: 0.2\n")
    write_yaml(
        tmp_path / "profile.yaml",
        "$ref: preset.yaml\nconfidence: 0.7\n",
    )

    document = compose_config_document(tmp_path / "profile.yaml", config_root=tmp_path)

    assert document.data == {"model": "base.onnx", "confidence": 0.7}
    assert document.sources == {
        "model": "preset.yaml",
        "confidence": "profile.yaml",
    }
