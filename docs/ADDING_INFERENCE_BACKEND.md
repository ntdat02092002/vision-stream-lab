# Adding an inference backend or objective

This document defines the model-layer extension contract. Inference backends are
technical capabilities reused by use-case plugins; they are not business use
cases themselves.

## 1. Terminology

Keep these four concepts separate:

| Concept | Example | Owns |
|---|---|---|
| Objective | detection, classification, segmentation, recognition | Typed input/output contract |
| Model family | YOLO, RT-DETR, ResNet, ArcFace | Preprocessing and output decoding |
| Execution backend | ONNX Runtime, Ultralytics, Triton | Runtime/session/client adapter |
| Weight instance | `yolo11n.onnx`, `ppe_yolo.onnx` | Configuration, not Python source |

Do not create `yolo11n.py`, `yolo11s.py`, or one source file per weight. They
share one adapter and differ by `model_path` and config.

## 2. Repository structure

```text
inference/
├── core/
│   └── base.py                     objective-agnostic batch contract
│
└── detection/
    ├── base.py                     DetectionBackend specialization
    ├── config.py                   family-agnostic structural config contract
    ├── schema.py                   DetectionPrediction
    ├── plugin.py                   family plugin descriptor
    ├── registry.py                 auto-discovery by model_family
    ├── factory.py                  delegate to selected family plugin
    ├── noop/
    │   ├── config.py
    │   ├── plugin.py
    │   └── backend.py
    └── yolo/
        ├── config.py               YOLO backend configs and parser
        ├── plugin.py               YOLO backend factory
        ├── preprocessing.py        BGR → letterboxed BCHW tensor
        ├── postprocessing.py       decode, filter, NMS, restore coordinates
        ├── onnx.py                 ONNX Runtime session adapter
        ├── ultralytics.py          Ultralytics adapter
        └── triton.py               Triton client adapter
```

Add `classification/`, `segmentation/`, or `recognition/` only when an actual
model needs that objective. Do not create empty future folders.

## 3. Generic core contract

`inference/core/base.py` contains:

```python
class BatchInferenceBackend(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    def predict_batch(
        self,
        inputs: Sequence[InputT],
    ) -> tuple[OutputT, ...]: ...

    def warmup(self, batch_size: int = 1) -> None: ...
    def close(self) -> None: ...
```

Every backend must preserve these invariants:

1. Accept a batch, including an empty batch.
2. Return exactly one output per input.
3. Preserve input/output order.
4. Run synchronously; the caller owns scheduling and process lifecycle.
5. Do not contain camera routing, alerting, tracking, zones, or business rules.
6. Do not mutate input images.
7. Raise clear startup errors for missing packages, weights, providers, or
   incompatible model shapes.

The runtime already batches camera frames in the use-case worker. An inference
backend must not create another multiprocessing worker or an internal frame
queue. Triton may batch/scale remotely, but its client call remains synchronous
from the plugin pipeline's perspective.

## 4. Objective-specific contract

Detection specializes the generic contract:

```python
class DetectionBackend(
    BatchInferenceBackend[np.ndarray, DetectionPrediction]
):
    pass
```

Normalized output:

```python
@dataclass(frozen=True)
class DetectionPrediction:
    boxes: np.ndarray  # float32 [N, 6]
```

Detection box columns are:

```text
x1, y1, x2, y2, class_id, confidence
```

Coordinates must be restored to the original input image size before leaving
the backend. A use case should not need to know whether the model used
letterbox, center crop, direct resize, ONNX, Triton, or Ultralytics.

Other objectives should define their own normalized results. For example:

```python
@dataclass(frozen=True)
class ClassificationPrediction:
    scores: np.ndarray
    class_ids: np.ndarray


@dataclass(frozen=True)
class SegmentationPrediction:
    masks: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class RecognitionPrediction:
    embeddings: np.ndarray
```

Do not force all objectives into a universal dictionary or an Nx6 detection
shape.

## 5. Adding a detection execution backend

Example: add a TensorRT YOLO adapter.

### Step 1 — Extend backend selection

In `inference/detection/yolo/config.py`:

```python
class YoloBackendType(str, Enum):
    ...
    TENSORRT = "tensorrt"
```

Keep YAML values stable because deployments refer to them.

### Step 2 — Implement the adapter

Create `inference/detection/yolo/tensorrt.py`:

```python
from collections.abc import Sequence

import numpy as np

from ..base import DetectionBackend
from .config import TensorRtYoloConfig
from ..schema import DetectionPrediction


class TensorRtYoloBackend(DetectionBackend):
    def __init__(self, config: TensorRtYoloConfig, project_root):
        # Resolve paths and create the TensorRT engine/context here.
        ...

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[DetectionPrediction, ...]:
        if not images:
            return ()
        # 1. preprocess the whole batch
        # 2. execute one batch call
        # 3. decode and restore boxes to each original image
        outputs = ...
        predictions = ...
        if len(predictions) != len(images):
            raise RuntimeError("TensorRT output count does not match input batch")
        return tuple(predictions)

    def warmup(self, batch_size: int = 1) -> None:
        ...

    def close(self) -> None:
        ...
```

Reuse `yolo/preprocessing.py` and `yolo/postprocessing.py` only if the exported
model has the same tensor and output semantics. Do not reuse them merely because
the model is called YOLO.

### Step 3 — Add lazy factory construction

In `inference/detection/yolo/plugin.py`:

```python
if isinstance(config, TensorRtYoloConfig):
    from .yolo.tensorrt import TensorRtYoloBackend

    return TensorRtYoloBackend(config, project_root)
```

Keep heavy or optional imports inside the selected branch or adapter
constructor. A CPU ONNX deployment must not require TensorRT to be installed.

### Step 4 — Add configuration

Add an immutable backend-specific dataclass such as `TensorRtYoloConfig`, include
it in the family-owned `YoloConfig` union, and dispatch to it from
`parse_yolo_config()`. Do not put TensorRT-only fields on the ONNX or Triton
configs and do not use an untyped catch-all dictionary.

```yaml
# configs/inference/detection/yolo/yolo11n_tensorrt.yaml
model_family: yolo
backend: tensorrt
model_path: models/yolo11n.engine
image_size: 640
confidence: 0.4
iou: 0.45
max_detections: 300
```

Reference the preset from a plugin config:

```yaml
inference:
  $ref: inference/detection/yolo/yolo11n_tensorrt.yaml
```

OmegaConf composition finishes before the plugin parser runs. The parser must
therefore convert the resolved mapping into the backend-specific typed config
and reject unknown fields at startup. See [Configuration architecture](CONFIGURATION.md).

## 6. Adding another detection model family

Example: RT-DETR via ONNX.

Create a separate family folder because preprocessing/decoding differs:

```text
inference/detection/rt_detr/
├── __init__.py
├── config.py
├── plugin.py
├── preprocessing.py
├── postprocessing.py
└── onnx.py
```

Export a family descriptor from `rt_detr/plugin.py`:

```python
PLUGIN = DetectionFamilyPlugin(
    model_family="rt_detr",
    parse_config=parse_rt_detr_config,
    create_backend=create_rt_detr_backend,
)
```

No root enum, config union, registry table, or generic factory branch changes are
required. `detection/registry.py` discovers
`detection/<model_family>/plugin.py` by folder convention.

The family owns its backend discriminator locally:

```python
class RtDetrBackendType(str, Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
```

The corresponding preset uses independent dimensions:

```yaml
model_family: rt_detr
backend: onnx
model_path: models/rt_detr.onnx
```

Do not invent a combined value such as `backend: rt_detr_onnx`, and do not put
RT-DETR conditionals throughout `yolo/onnx.py`. An adapter file should represent
one coherent model-family/backend combination.

## 7. Adding a new objective

Example: recognition.

```text
inference/recognition/
├── __init__.py
├── base.py
├── config.py
├── schema.py
├── factory.py
├── noop.py
└── arcface/
    ├── preprocessing.py
    └── onnx.py
```

The objective owns:

- its normalized output schema;
- backend selection config;
- model-family preprocessing/postprocessing;
- factory and optional dependency loading.

Example base:

```python
class RecognitionBackend(
    BatchInferenceBackend[np.ndarray, RecognitionPrediction]
):
    pass
```

Do not modify `BatchInferenceBackend` merely to add embeddings or masks. Extend
the generic core only when every objective needs a new lifecycle capability.

## 8. Connecting inference to a use case

Inference packages never register themselves with the camera runtime. A use-case
pipeline chooses and owns them:

```python
class FaceAccessPipeline(UseCasePipeline):
    def __init__(self, config, project_root):
        self.detector = create_detection_backend(config.detection, project_root)
        self.recognizer = create_recognition_backend(
            config.recognition,
            project_root,
        )

    def process_batch(self, images, contexts=None):
        detections = self.detector.predict_batch(images)
        # crop faces, batch recognition, apply access rules, build results
        ...
```

This separation allows:

- object detection to use one detector;
- PPE to reuse detection with different weights and business rules;
- face access to compose detection plus recognition;
- a classical CV use case to use no inference package at all.

## 9. Testing requirements

Every new backend should test:

1. Empty batch returns an empty tuple.
2. N inputs produce N ordered outputs.
3. Output dtype and shape follow the objective schema.
4. Preprocessing preserves expected color/layout/range.
5. Postprocessing restores coordinates or masks to original image dimensions.
6. Confidence/class filtering and NMS semantics where applicable.
7. Fixed-batch and dynamic-batch model behavior.
8. Missing dependency/model/provider errors.
9. A real-model smoke test outside the fast unit suite.

Prefer pure tests for preprocess/postprocess and a fake session/client for
adapter tests. Keep heavyweight model downloads out of unit tests.

Run:

```powershell
python -m ruff check src tests scripts
python -m pytest -q
python scripts/smoke_onnx.py --video <video> --model <model.onnx>
```

Then run `scripts/smoke_runtime.py` to verify Windows process spawning, real
multi-camera batching, plugin pipeline integration, and output rendering.

## 10. Review checklist

Before merging a backend, verify:

- [ ] It implements the correct objective contract.
- [ ] It performs one true batch call when the model supports batching.
- [ ] It returns one ordered output per input.
- [ ] Optional dependencies are lazily imported.
- [ ] Weights are config, not source files.
- [ ] Model-family preprocessing/postprocessing stays outside business use cases.
- [ ] Camera IDs, queues, alerts, trackers, and zones are absent from inference.
- [ ] Backend code does not create another process or frame queue.
- [ ] Unit tests and real-model smoke tests pass.
