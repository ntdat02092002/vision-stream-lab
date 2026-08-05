from enum import Enum


class CameraSourceType(str, Enum):
    DEVICE = "device"
    VIDEO_FILE = "video_file"
    NETWORK_STREAM = "network_stream"


def detect_source_type(source: str | int) -> CameraSourceType:
    if isinstance(source, int):
        return CameraSourceType.DEVICE
    if "://" in source:
        return CameraSourceType.NETWORK_STREAM
    return CameraSourceType.VIDEO_FILE

