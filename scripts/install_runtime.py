"""Install a reproducible CPU or NVIDIA CUDA runtime for local YOLO inference."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

Variant = Literal["cpu", "gpu"]

PYTORCH_INDEXES: dict[Variant, str] = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "gpu": "https://download.pytorch.org/whl/cu128",
}


def detect_nvidia_gpu() -> tuple[bool, str]:
    """Return whether the NVIDIA driver exposes at least one GPU."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, "nvidia-smi was not found"

    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"nvidia-smi could not be queried: {error}"

    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not gpu_names:
        detail = result.stderr.strip() or "no NVIDIA GPU was reported"
        return False, detail
    return True, ", ".join(gpu_names)


def resolve_variant(requested: str) -> tuple[Variant, str]:
    """Resolve auto from hardware while keeping deployment choices explicit."""
    if requested in ("cpu", "gpu"):
        return requested, "selected explicitly"

    available, detail = detect_nvidia_gpu()
    if available:
        return "gpu", f"NVIDIA GPU detected: {detail}"
    return "cpu", f"no usable NVIDIA GPU detected: {detail}"


def build_install_command(
    variant: Variant,
    *,
    with_dev: bool = False,
    with_export: bool = False,
) -> list[str]:
    extras = [variant]
    if with_dev:
        extras.append("dev")
    if with_export:
        extras.append("export")
    project_spec = f".[{','.join(extras)}]"
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        project_spec,
        "--extra-index-url",
        PYTORCH_INDEXES[variant],
    ]


def verify_runtime(variant: Variant, *, project_root: Path) -> None:
    expected_cuda = variant == "gpu"
    check = f"""
import torch
import torchvision

available = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if available else "CPU"
print(
    f"torch={{torch.__version__}} torchvision={{torchvision.__version__}} "
    f"cuda_available={{available}} device={{device}}"
)
raise SystemExit(0 if available is {expected_cuda!r} else 1)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", check],
            cwd=project_root,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        expectation = "CUDA-enabled GPU" if expected_cuda else "CPU-only Torch"
        raise SystemExit(f"Runtime verification failed; expected {expectation}.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Use auto for local setup; use cpu/gpu explicitly in deployments.",
    )
    parser.add_argument("--with-dev", action="store_true", help="Install the dev extra.")
    parser.add_argument(
        "--with-export",
        action="store_true",
        help="Install ONNX export dependencies too.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved install command without changing the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    variant, reason = resolve_variant(args.variant)
    command = build_install_command(
        variant,
        with_dev=args.with_dev,
        with_export=args.with_export,
    )

    print(f"Runtime variant: {variant} ({reason})")
    print("Install command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return

    subprocess.run(command, cwd=project_root, check=True)
    verify_runtime(variant, project_root=project_root)


if __name__ == "__main__":
    main()
