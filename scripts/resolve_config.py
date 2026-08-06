from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from vision_stream_lab.configuration import compose_config_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the fully composed config and optionally each leaf's source file."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Print config-path to source-file mappings after the resolved YAML.",
    )
    args = parser.parse_args()

    document = compose_config_document(args.config, config_root=args.config.parent)
    print(OmegaConf.to_yaml(OmegaConf.create(document.data), resolve=True).rstrip())
    if args.sources:
        print("\n# Sources")
        for config_path, source in sorted(document.sources.items()):
            print(f"{config_path}: {source}")


if __name__ == "__main__":
    main()
