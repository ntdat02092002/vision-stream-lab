# Adding a use-case plugin

This document is the implementation contract for adding a use case without
changing camera capture, multiprocessing runtime, shared-frame transport,
monitoring, or alert lifecycle code.

## 1. Discovery convention

A plugin type is a lowercase snake-case string such as `frame_score` or
`person_intrusion`. The source package must use the same name:

```text
src/vision_stream_lab/usecases/<type>/plugin.py
```

`plugin.py` must export exactly one descriptor named `PLUGIN` whose `type`
equals the folder/config type:

```python
PLUGIN = UseCasePlugin(type="frame_score", ...)
```

The registry discovers this module dynamically. Do not add an enum value or
edit a central registry when adding a plugin.

Valid type names match:

```text
^[a-z][a-z0-9_]*$
```

## 2. Required files

Recommended package:

```text
src/vision_stream_lab/usecases/frame_score/
├── __init__.py
├── config.py       typed YAML config and validation
├── pipeline.py     model/business execution
├── state.py        plugin-owned cross-process result state
├── rendering.py    latest-result overlay policy
└── plugin.py       exports PLUGIN and wires all hooks
```

Only `plugin.py` is required by discovery. The other files are an ownership
convention that keeps the plugin maintainable.

## 3. Core pipeline contract

Every pipeline implements:

```python
class UseCasePipeline(ABC):
    def process_batch(
        self,
        images: list[np.ndarray],
        contexts: list[FrameContext] | None = None,
    ) -> list[UseCaseResult]: ...
```

`FrameContext` contains:

```python
@dataclass(frozen=True)
class FrameContext:
    camera_id: str
    sequence: int
    timestamp: float
```

`UseCaseResult` contains:

```python
@dataclass
class UseCaseResult:
    output_frame: np.ndarray
    event_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

Pipeline invariants:

1. Return exactly one result for every input image.
2. Preserve input order: result `i` belongs to image/context `i`.
3. `output_frame` must be a `uint8` image with the configured frame shape.
4. `output_frame` must remain pixel-aligned with the exact input image. It is
   used by `inference_only`, `delayed_matched`, and alert snapshots.
5. `event_count` is generic. It may mean detections, violations, anomalies,
   recognized items, or another count defined by the plugin.
6. `metadata` is private to the plugin's `publish_result` hook. Runtime code
   never reads plugin-specific keys.
7. Keep temporal state separated by `context.camera_id`.
8. Construct models in `create_pipeline`, which runs inside the use-case child
   process. Do not load a model at module import time.
9. The pipeline currently runs model and business logic sequentially in one
   process. Do not create extra processes or threads inside a plugin unless the
   architecture is deliberately changed and measured.

## 4. `UseCasePlugin` hook contract

The descriptor has five required hooks and one optional static-overlay hook:

```python
@dataclass(frozen=True)
class UseCasePlugin:
    type: str
    parse_config: Callable[[Mapping[str, Any]], Any]
    create_pipeline: Callable[[Any, Path], UseCasePipeline]
    create_shared_state: Callable[[Any, Any], Any]
    publish_result: Callable[[Any, UseCaseResult, FrameContext, Any], None]
    render_latest: Callable[
        [np.ndarray, Any, float, float, float, Any], np.ndarray
    ]
    render_static_overlay: Callable[
        [np.ndarray, str, Any, Any], np.ndarray
    ] | None = None
```

### `parse_config(raw) -> plugin_config`

Runs once in the main process during startup.

It should:

- convert raw YAML into immutable typed dataclasses;
- reject unknown fields;
- validate numeric ranges and incompatible combinations;
- remain lightweight and side-effect free;
- never load model weights or open cameras.

The returned config must be pickle-compatible because it is passed to a child
process with the Windows `spawn` multiprocessing method.

### `create_pipeline(plugin_config, project_root) -> UseCasePipeline`

Runs once inside each use-case deployment process.

It owns:

- model/backend construction;
- analyzers and business rules;
- optional per-camera trackers or temporal state.

Resolve relative model/resource paths against `project_root`.

### `create_shared_state(mp_context, plugin_config) -> Any`

Runs in the main process once per assigned camera before workers start.

The returned value is opaque to runtime core and is attached as
`UseCaseCameraState.plugin_state`. Use multiprocessing-compatible primitives
created from the supplied context:

- `context.Value(...)`
- `context.RawArray(...)`
- `context.Lock()`
- dataclasses containing those primitives

Do not return a normal mutable NumPy array and expect another process to see
updates. Do not put full video frames here; full images already use
`SharedFrameStore`.

Object detection, for example, owns its boxes and velocity buffers entirely in
`usecases/object_detection/state.py`. OCR may instead store text and regions;
pose may store keypoints; a stateless plugin may return `None`.

### `publish_result(shared_state, result, frame_context, plugin_config)`

Runs in the use-case worker after each successful pipeline result.

It converts plugin-private `result.metadata` into the plugin's shared-state
layout. Store `frame_context.sequence` and `frame_context.timestamp` whenever
the renderer needs identity, age, or alignment information.

The worker already writes `result.output_frame` into the exact-inference shared
frame store. This hook should only publish compact structured data needed for
`latest_predictions` rendering or monitoring extensions.

### `render_latest(image, shared_state, target_timestamp, now, ttl_ms, config)`

Runs in the main-process renderer thread for `latest_predictions` mode.

Arguments:

| Argument | Meaning |
|---|---|
| `image` | Latest raw camera frame; do not mutate it in place |
| `shared_state` | State returned by this plugin's factory |
| `target_timestamp` | Capture timestamp of `image` |
| `now` | Current wall-clock timestamp |
| `ttl_ms` | Maximum configured age of a reusable result |
| `config` | Typed plugin config |

The hook must return a `uint8` frame with the same shape. It owns snapshot
locking, TTL handling, extrapolation, and drawing semantics. If no valid result
exists, return the raw image.

`inference_only` and `delayed_matched` do not call this hook; they use the exact
annotated `output_frame` produced by the pipeline.

### `render_static_overlay(image, camera_id, shared_state, config)` (optional)

Runs only when `inference_only` or `delayed_matched` must display a raw fallback
frame because no exact inferred frame is available. It is intended for static
plugin-owned geometry such as zones or lines. The runtime does not call it for
an exact inferred frame, so overlays already drawn by the pipeline are not
duplicated. Plugins without static geometry should leave this hook unset.

## 5. Minimal stateless hooks

A plugin that does not support latest-result overlays may use:

```python
def create_shared_state(_context, _config):
    return None


def publish_result(_state, _result, _frame_context, _config):
    return None


def render_latest(image, _state, _target_timestamp, _now, _ttl_ms, _config):
    return image
```

Such a plugin still works correctly with `inference_only` and
`delayed_matched`. In `latest_predictions` it intentionally shows raw frames.

## 6. Minimal complete example

This example scores mean frame brightness and publishes the latest score.

### `config.py`

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameScoreConfig:
    alert_threshold: float = 180.0


def parse_config(raw: dict[str, Any]) -> FrameScoreConfig:
    unknown = set(raw) - {"alert_threshold"}
    if unknown:
        raise ValueError(f"Unknown frame_score fields: {sorted(unknown)}")
    config = FrameScoreConfig(**raw)
    if not 0 <= config.alert_threshold <= 255:
        raise ValueError("alert_threshold must be between 0 and 255")
    return config
```

### `pipeline.py`

```python
import cv2
import numpy as np

from ...schema.use_case import FrameContext, UseCaseResult
from ..base import UseCasePipeline
from .config import FrameScoreConfig


class FrameScorePipeline(UseCasePipeline):
    def __init__(self, config: FrameScoreConfig):
        self.config = config

    def process_batch(self, images, contexts=None):
        results = []
        for image in images:
            score = float(image.mean())
            output = image.copy()
            cv2.putText(
                output,
                f"brightness={score:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            results.append(
                UseCaseResult(
                    output_frame=output,
                    event_count=int(score >= self.config.alert_threshold),
                    metadata={"score": score},
                )
            )
        return results
```

### `state.py`

```python
from dataclasses import dataclass
from typing import Any

from ...schema.use_case import FrameContext, UseCaseResult
from .config import FrameScoreConfig


@dataclass
class SharedFrameScoreState:
    score: Any
    timestamp: Any
    lock: Any


def create_shared_state(context, _config):
    return SharedFrameScoreState(
        score=context.Value("d", 0.0),
        timestamp=context.Value("d", 0.0),
        lock=context.Lock(),
    )


def publish_result(state, result, frame_context, _config):
    with state.lock:
        state.score.value = float(result.metadata["score"])
        state.timestamp.value = frame_context.timestamp
```

### `rendering.py`

```python
import cv2


def render_latest(image, state, _target_timestamp, now, ttl_ms, _config):
    with state.lock:
        score = float(state.score.value)
        timestamp = float(state.timestamp.value)
    if not timestamp or (now - timestamp) * 1000 > ttl_ms:
        return image
    output = image.copy()
    cv2.putText(
        output,
        f"latest brightness={score:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return output
```

### `plugin.py`

```python
from .config import FrameScoreConfig, parse_config
from .pipeline import FrameScorePipeline
from .rendering import render_latest
from .state import create_shared_state, publish_result
from ..plugin import UseCasePlugin


def create_pipeline(config, _project_root):
    if not isinstance(config, FrameScoreConfig):
        raise TypeError("frame_score requires FrameScoreConfig")
    return FrameScorePipeline(config)


PLUGIN = UseCasePlugin(
    type="frame_score",
    parse_config=parse_config,
    create_pipeline=create_pipeline,
    create_shared_state=create_shared_state,
    publish_result=publish_result,
    render_latest=render_latest,
)
```

Adjust relative imports if the implementation is copied verbatim. The existing
`object_detection/plugin.py` is the authoritative in-repository example.

## 7. YAML declarations

Create plugin config:

```yaml
# configs/usecases/frame_score/default.yaml
alert_threshold: 180
```

Create alert policy:

```yaml
# configs/alerts/frame_score.yaml
enabled: false
output_dir: outputs/alerts/frame-score
min_events: 1
cooldown_seconds: 30
```

Add a deployment:

```yaml
# configs/deployments.yaml
frame-score-main:
  type: frame_score
  enabled: true
  cameras: [camera-01, camera-03]
  config:
    $ref: usecases/frame_score/default.yaml
    alert_threshold: 200
  alert:
    $ref: alerts/frame_score.yaml
```

Meanings:

| Field | Meaning |
|---|---|
| Mapping key | Unique deployment ID; used in process names, API, and dashboard |
| `type` | Plugin folder name and `PLUGIN.type` |
| `enabled` | Whether this deployment starts |
| `cameras` | Explicit camera IDs or `["*"]` |
| `config` | Composed plugin-owned mapping parsed by `parse_config` |
| `alert` | Generic alert/snapshot policy, inline or composed with `$ref` |
| `runtime` | Optional worker override; omitted fields inherit `app.runtime.worker_defaults` |

Multiple deployments may use the same plugin type with different models,
thresholds, or camera assignments. Each deployment currently owns one process
and one pipeline/model instance.

Plugin YAML may reuse inference presets through `$ref`. A deployment may also
put local fields next to its `config.$ref`; those fields deep-merge over the
profile before OmegaConf interpolation and `parse_config`. Keep parsing inside
the plugin so generic runtime code never learns its fields. See
[Configuration architecture](CONFIGURATION.md) for the exact merge rules.

## 8. Optional detector subsystem

The top-level `inference` package is a reusable detector library, not part of
runtime core and not a mandatory plugin dependency. A use case may:

- use `inference.detection.DetectionBackendConfig` and
  `create_detection_backend`;
- implement a different model backend;
- perform no deep-learning inference at all.

The normalized detection contract lives in `inference/detection/schema.py`,
while family-specific configs live under folders such as
`inference/detection/yolo/config.py`; neither belongs in generic `schema/`. See
[Adding an inference backend or objective](ADDING_INFERENCE_BACKEND.md) before
adding a new runtime adapter or model objective.

## 9. Tests required before enabling a plugin

Add tests for:

1. valid config parsing and unknown/invalid fields;
2. `process_batch` length/order and output shape/dtype;
3. independent temporal state for different cameras;
4. shared-state create/publish/read behavior;
5. stale-state TTL fallback to raw;
6. registry auto-discovery and `PLUGIN.type` match;
7. camera assignment/routing;
8. a Windows `spawn` smoke test when shared state or model construction changes.

Run:

```powershell
python -m ruff check src tests
python -m pytest -q
```

Then run the normal multi-camera smoke test before committing.

## 10. Core files a normal plugin must not modify

A normal new use case requires no changes to:

```text
main.py
configuration/loader.py
runtime/orchestrator.py
runtime/use_case_worker.py
runtime/shared_frames.py
runtime/output_renderer.py
schema/config.py
schema/frame.py
monitoring/api.py
usecases/registry.py
```

If adding a plugin appears to require one of these changes, first check whether
the behavior belongs in one of the five plugin hooks. Change core only when the
generic contract itself must evolve for every plugin.
