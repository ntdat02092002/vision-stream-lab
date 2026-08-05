from pathlib import Path

import pytest

from vision_stream_lab.configuration import load_config_document


def write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_ref_deep_merge_interpolation_and_deployment_override(tmp_path):
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

    result = load_config_document(
        tmp_path / "usecase.yaml",
        config_root=tmp_path,
        overrides={
            "inference": {
                "confidence": 0.55,
                "classes": [7],
                "model_path": "models/override.onnx",
            }
        },
    )

    assert result["inference"] == {
        "backend": "onnx",
        "model_path": "models/override.onnx",
        "confidence": 0.55,
        "classes": [7],
        "label": "onnx:models/override.onnx",
    }
    assert result["tracker"] == {"enabled": False}


def test_ref_rejects_cycles_and_paths_outside_config_root(tmp_path):
    write_yaml(tmp_path / "a.yaml", "$ref: b.yaml\n")
    write_yaml(tmp_path / "b.yaml", "$ref: a.yaml\n")
    with pytest.raises(ValueError, match="Circular config reference"):
        load_config_document(tmp_path / "a.yaml", config_root=tmp_path)

    write_yaml(tmp_path / "escape.yaml", "$ref: ../outside.yaml\n")
    with pytest.raises(ValueError, match="escapes config root"):
        load_config_document(tmp_path / "escape.yaml", config_root=tmp_path)
