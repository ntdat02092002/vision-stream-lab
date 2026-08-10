from enum import Enum
from urllib.parse import urlsplit


class CameraSourceType(str, Enum):
    DEVICE = "device"
    VIDEO_FILE = "video_file"
    SEGMENTED_STREAM = "segmented_stream"
    NETWORK_STREAM = "network_stream"


class SourceTimingMode(str, Enum):
    AUTO = "auto"
    REALTIME = "realtime"
    MEDIA_TIMELINE = "media_timeline"


def detect_source_type(source: str | int) -> CameraSourceType:
    if isinstance(source, int):
        return CameraSourceType.DEVICE
    if "://" in source:
        path = urlsplit(source).path.lower()
        if path.endswith((".m3u8", ".mpd")):
            return CameraSourceType.SEGMENTED_STREAM
        return CameraSourceType.NETWORK_STREAM
    return CameraSourceType.VIDEO_FILE
