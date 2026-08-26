from __future__ import annotations

from enum import IntEnum

import cv2
import numpy as np

from .config import RedLightViolationConfig, RenderingConfig
from .spatial import ResolvedGeometry, resolve_camera_geometry


class BoxRenderState(IntEnum):
    NORMAL = 0
    TRACKING = 1
    VIOLATION = 2


def _contour(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    result = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    result[:, 0] = result[:, 0].clip(0, width - 1)
    result[:, 1] = result[:, 1].clip(0, height - 1)
    return result.reshape(-1, 1, 2)


def _draw_violation_hud(image: np.ndarray, violation_count: int) -> None:
    height, width = image.shape[:2]
    panel_width = min(268, width - 12)
    x1, x2 = width - panel_width - 12, width - 12
    y1, y2 = 12, min(62, height - 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 15, 22), -1)
    image[:] = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    cv2.putText(
        image,
        f"VIOLATIONS {violation_count}",
        (x1 + 12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_light_hud(image: np.ndarray, light_state: str) -> None:
    normalized = str(light_state).strip().lower()
    color = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 200, 0),
    }.get(normalized, (140, 140, 140))
    label = normalized.upper() if normalized in {"red", "yellow", "green"} else "UNKNOWN"

    height, width = image.shape[:2]
    panel_width = min(230, width - 12)
    x1, x2 = 12, 12 + panel_width
    y2 = height - 12
    y1 = max(0, y2 - 50)

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 15, 22), -1)
    image[:] = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    center = (x1 + 25, y1 + 25)
    cv2.circle(image, center, 10, color, -1, cv2.LINE_AA)
    cv2.circle(image, center, 10, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"LIGHT {label}",
        (x1 + 45, y1 + 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _box_style(state: int, config: RenderingConfig) -> tuple[tuple[int, int, int], str]:
    if state == BoxRenderState.VIOLATION:
        return config.violation_box_color, "VIOLATION"
    if state == BoxRenderState.TRACKING:
        return config.tracking_box_color, "TRACKING"
    return config.box_color, "VEHICLE"


def annotate_frame(
    image: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    box_states: np.ndarray,
    geometry: ResolvedGeometry | None,
    violation_count: int,
    config: RenderingConfig,
    *,
    static_only: bool = False,
    current_light_state: str = "unknown",
) -> np.ndarray:
    output = image.copy()
    if geometry is not None:
        if config.show_approach_roi:
            overlay = output.copy()
            cv2.fillPoly(
                overlay,
                [_contour(geometry.approach_roi, output.shape)],
                config.approach_roi_color,
                cv2.LINE_AA,
            )
            output = cv2.addWeighted(overlay, 0.16, output, 0.84, 0)
        if config.show_roi:
            cv2.polylines(
                output,
                [_contour(geometry.roi, output.shape)],
                True,
                config.roi_color,
                config.thickness,
                cv2.LINE_AA,
            )
        if config.show_gate:
            points = np.rint(geometry.stop_line).astype(int)
            cv2.line(
                output,
                tuple(points[0]),
                tuple(points[1]),
                config.stop_line_color,
                config.thickness + 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "STOP LINE",
                (int(points[0, 0]), max(20, int(points[0, 1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.stop_line_color,
                2,
                cv2.LINE_AA,
            )
    if static_only:
        return output
    if config.show_boxes:
        values = np.asarray(boxes).reshape(-1, 6)
        statuses = np.asarray(box_states, dtype=np.int8).reshape(-1)
        if len(values) != len(track_ids) or len(values) != len(statuses):
            raise ValueError("Boxes, track IDs, and render states must have equal length")
        for box, track_id, status in zip(values, track_ids, statuses):
            x1, y1, x2, y2, _class_id, confidence = box
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            color, label = _box_style(int(status), config)
            cv2.rectangle(output, p1, p2, color, config.thickness)
            cv2.putText(
                output,
                f"{label} #{int(track_id)} {confidence:.2f}",
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    if config.show_counts:
        _draw_violation_hud(output, violation_count)
    if config.show_light_state:
        _draw_light_hud(output, current_light_state)
    return output


def project_boxes(
    boxes: np.ndarray,
    velocities: np.ndarray,
    prediction_timestamp: float,
    target_timestamp: float,
    max_extrapolation_ms: float,
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    projected = np.asarray(boxes, dtype=np.float32).reshape(-1, 6).copy()
    if not len(projected):
        return projected, np.empty(0, dtype=bool)
    elapsed = max(0.0, min(target_timestamp - prediction_timestamp, max_extrapolation_ms / 1000))
    projected[:, :4] += np.asarray(velocities, dtype=np.float32).reshape(-1, 4) * elapsed
    height, width = frame_shape[:2]
    projected[:, [0, 2]] = projected[:, [0, 2]].clip(0, width)
    projected[:, [1, 3]] = projected[:, [1, 3]].clip(0, height)
    valid = (projected[:, 2] > projected[:, 0]) & (projected[:, 3] > projected[:, 1])
    return projected[valid], valid


def render_latest(
    image: np.ndarray,
    shared_state: object,
    target_timestamp: float,
    now: float,
    ttl_ms: float,
    config: RedLightViolationConfig,
) -> np.ndarray:
    from .state import read_snapshot

    snapshot = read_snapshot(shared_state)
    age_ms = (now - snapshot.timestamp) * 1000
    if not snapshot.timestamp or age_ms > ttl_ms:
        return annotate_frame(
            image,
            np.empty((0, 6), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
            snapshot.geometry,
            0,
            config.rendering,
            static_only=True,
        )
    boxes, valid = project_boxes(
        snapshot.boxes,
        snapshot.velocities,
        snapshot.timestamp,
        target_timestamp,
        config.tracker.max_extrapolation_ms,
        image.shape,
    )
    return annotate_frame(
        image,
        boxes,
        snapshot.track_ids[valid],
        snapshot.box_states[valid],
        snapshot.geometry,
        snapshot.violation_count,
        config.rendering,
        current_light_state=snapshot.current_light_state,
    )


def render_static_overlay(
    image: np.ndarray,
    camera_id: str,
    config: RedLightViolationConfig,
) -> np.ndarray:
    geometry = resolve_camera_geometry(config.spatial, camera_id, image.shape)
    return annotate_frame(
        image,
        np.empty((0, 6), dtype=np.float32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int8),
        geometry,
        0,
        config.rendering,
        static_only=True,
    )
