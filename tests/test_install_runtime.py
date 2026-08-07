from scripts import install_runtime


def test_auto_selects_gpu_when_nvidia_is_available(monkeypatch):
    monkeypatch.setattr(
        install_runtime,
        "detect_nvidia_gpu",
        lambda: (True, "Test GPU"),
    )

    variant, reason = install_runtime.resolve_variant("auto")

    assert variant == "gpu"
    assert "Test GPU" in reason


def test_auto_falls_back_to_cpu_without_nvidia(monkeypatch):
    monkeypatch.setattr(
        install_runtime,
        "detect_nvidia_gpu",
        lambda: (False, "nvidia-smi was not found"),
    )

    variant, reason = install_runtime.resolve_variant("auto")

    assert variant == "cpu"
    assert "nvidia-smi" in reason


def test_explicit_variant_does_not_probe_hardware(monkeypatch):
    def fail_if_called():
        raise AssertionError("hardware detection should not run")

    monkeypatch.setattr(install_runtime, "detect_nvidia_gpu", fail_if_called)

    assert install_runtime.resolve_variant("gpu") == ("gpu", "selected explicitly")
    assert install_runtime.resolve_variant("cpu") == ("cpu", "selected explicitly")


def test_gpu_command_uses_pinned_extra_and_cuda_index():
    command = install_runtime.build_install_command(
        "gpu",
        with_dev=True,
        with_export=True,
    )

    assert ".[gpu,dev,export]" in command
    assert install_runtime.PYTORCH_INDEXES["gpu"] in command
