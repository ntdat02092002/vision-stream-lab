from enum import Enum


class OutputRenderMode(str, Enum):
    DELAYED_MATCHED = "delayed_matched"
    LATEST_PREDICTIONS = "latest_predictions"
    INFERENCE_ONLY = "inference_only"
