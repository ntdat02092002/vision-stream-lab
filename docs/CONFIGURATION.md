# Configuration architecture

The application is one recursively composed OmegaConf tree. Hydra is not used:
the service keeps its ordinary CLI, working directory, and process lifecycle.

## File graph

```text
configs/app.yaml
|- cameras: $ref cameras.yaml
`- deployments: $ref deployments.yaml
   `- <deployment>.config: $ref usecases/<profile>.yaml
      `- inference: $ref inference/<objective>/<family>/<preset>.yaml
```

`app.yaml` is the only loader entry point. Every `$ref` is resolved relative to
the `configs/` root, then OmegaConf resolves interpolation once over the complete
tree. The generic loader never opens deployment, plugin, inference, or alert
files separately.

## Responsibilities

| Layer | Owns |
|---|---|
| `app.yaml` | Worker defaults, instance sharding, frame shape, monitoring, root references |
| `cameras.yaml` | Camera IDs, sources, capture policy, optional shard pinning |
| `deployments.yaml` | Plugin selection, camera assignment, worker override, plugin config tree, alert policy |
| `usecases/*.yaml` | One plugin profile: algorithms, thresholds, trackers, zones, inference composition |
| `inference/<objective>/<family>/*.yaml` | Reusable model/backend preset for detection, segmentation, classification, recognition, etc. |
| `alerts/*.yaml` | Reusable generic alert side-effect policy |

Plugin-specific fields stay below `deployments.<id>.config`. The selected plugin
parses that mapping into its own typed schema. Generic runtime code therefore
does not need edits when a use case or inference objective is added.

Generic dataclasses are loaded through `OmegaConf.structured`, so unknown keys,
missing required values, and incompatible primitive types fail at the config
boundary. Dynamic plugin and inference unions are validated by their owning
parser after composition.

## ID-keyed collections

Cameras and deployments are mappings, not lists with repeated `id` fields:

```yaml
# cameras.yaml
gate-west:
  name: West gate
  source: rtsp://example/stream
  enabled: true
```

```yaml
# deployments.yaml
road-detection:
  type: object_detection
  cameras: [gate-west]
  config:
    $ref: usecases/object_detection.yaml
```

This shape gives every item a stable OmegaConf path such as
`deployments.road-detection.config.inference.confidence`. It also avoids list
merge ambiguity and redundant IDs.

## `$ref`, local overrides, and interpolation

`$ref` may appear at any mapping node. Fields next to it deep-merge over the
referenced mapping:

```yaml
inference:
  $ref: inference/detection/yolo/yolo11n_onnx.yaml
  confidence: 0.20
  classes: [0, 1, 2, 3, 5, 7]
```

Composition is always:

1. recursively resolve the referenced mapping;
2. deep-merge local sibling fields over it;
3. resolve `${...}` interpolation after the entire app tree is assembled;
4. parse generic schemas and plugin-owned typed schemas.

Mappings merge recursively. Scalars replace inherited values. Lists replace the
inherited list; they never append implicitly. Circular references, missing
files, non-mapping references, and paths escaping `configs/` fail at startup.

There is no separate `overrides` field or second programmatic merge pass. A
value that differs from a preset is written next to `$ref`, where it is visible
in the profile that owns it.

## Worker defaults and deployment runtime

Application defaults and instance sharding have separate namespaces:

```yaml
# app.yaml
runtime:
  worker_defaults:
    batch_size: 4
    batch_wait_ms: 12
    queue_timeout_ms: 250
  sharding:
    index: 0
    count: 1
```

A deployment writes only fields that genuinely differ:

```yaml
# deployments.yaml
heavy-segmentation:
  type: semantic_segmentation
  cameras: [gate-west]
  runtime:
    batch_size: 2
    batch_wait_ms: 25
  config:
    $ref: usecases/semantic_segmentation.yaml
```

The loader resolves `app.runtime.worker_defaults` with the deployment's
`runtime` mapping into `UseCaseRuntimeConfig`. Effective runtime fields retain
their app/deployment source for startup logs and `GET /api/status`.

## Plugin and inference extensibility

A deployment selects a use-case plugin with `type`. That plugin exclusively owns
the schema under `config`, and may compose one or more inference presets.

Inference presets are organized first by objective:

```text
configs/inference/
|- detection/
|  |- yolo/
|  |  `- yolo11n_onnx.yaml
|  `- rt_detr/
|     `- rt_detr_onnx.yaml
|- segmentation/
|  `- deeplabv3_onnx.yaml
|- classification/
|  `- resnet50_onnx.yaml
`- recognition/
   `- crnn_onnx.yaml
```

Within detection, `model_family` discovers
`inference/detection/<model_family>/plugin.py`. That family plugin owns its typed
backend discriminator, config union, parser, and backend factory. Adding
`rt_detr` therefore does not edit a root enum or factory. A new objective gets a
sibling Python package and preset directory; it does not add branches to the
generic configuration loader.

Multiple deployments may reference the same plugin profile while locally
overriding its `config` subtree:

```yaml
road-detection:
  type: object_detection
  cameras: [road-01, road-02]
  config:
    $ref: usecases/object_detection.yaml

person-detection:
  type: object_detection
  cameras: [lobby-01]
  config:
    $ref: usecases/object_detection.yaml
    inference:
      confidence: 0.55
      classes: [0]
```

Both mappings are fully composed before the same object-detection plugin parser
validates them, and each deployment creates its own worker/pipeline instance.

## Inspecting the resolved config

Print the exact tree seen by the loader:

```powershell
python scripts\resolve_config.py --config configs\app.yaml
```

Also print the winning source file for every leaf:

```powershell
python scripts\resolve_config.py --config configs\app.yaml --sources
```

Example source entries:

```text
deployments.object-detection.config.inference.model_path: inference/detection/yolo/yolo11n_onnx.yaml
deployments.object-detection.config.inference.confidence: usecases/object_detection.yaml
runtime.worker_defaults.batch_size: app.yaml
```

Use this command in CI and during deployment review; it does not load models or
open cameras.

## Why OmegaConf without Hydra

OmegaConf supplies recursive mapping merge, interpolation, YAML parsing, and a
resolved in-memory tree. Hydra would additionally own launch behavior, config
groups, run directories, sweep semantics, and CLI override grammar. Those are
useful for experiments but unnecessary for this long-running streaming service.

If training or sweep workflows are added later, Hydra can consume the same
plugin/inference profile files without changing runtime schemas.
