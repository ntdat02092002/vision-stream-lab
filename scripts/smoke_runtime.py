from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastapi.testclient import TestClient

from vision_stream_lab.configuration import load_config
from vision_stream_lab.main import VisionRuntime
from vision_stream_lab.monitoring import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = VisionRuntime(config)
    runtime.start()
    try:
        time.sleep(args.seconds)
        app = create_app(
            config,
            runtime.raw_store,
            runtime.use_cases.output_stores,
            runtime.states,
            runtime.use_cases.use_case_states,
        )
        with TestClient(app) as client:
            response = client.get("/api/status")
            response.raise_for_status()
            status = response.json()
            print(status)
            primary = status["primary_use_case"]
            for camera in status["cameras"]:
                assert camera["captured"] > 0, camera
                assert camera["use_cases"][primary]["inferred"] > 0, camera
                frame = client.get(
                    f"/api/cameras/{camera['id']}/frame.jpg?use_case={primary}"
                )
                assert frame.status_code == 200
                assert frame.headers["content-type"] == "image/jpeg"
                assert len(frame.content) > 1000
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
