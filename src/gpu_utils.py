"""GPU detection for faster-whisper (CUDA via CTranslate2)."""

from __future__ import annotations


def cuda_device_count() -> int:
    try:
        import ctranslate2 as ct2

        return int(ct2.get_cuda_device_count())
    except Exception:
        return 0


def cuda_available() -> bool:
    return cuda_device_count() > 0


def resolve_whisper_device(*, use_gpu: bool, device: str | None = None) -> tuple[str, str]:
    """
    Return (device, compute_type) for WhisperModel.
    compute_type: int8 on CPU; float16 on CUDA (override with env WHISPER_COMPUTE_TYPE).
    """
    import os

    if device:
        dev = device.strip().lower()
        if dev == "cuda" and not cuda_available():
            dev = "cpu"
        ctype = os.getenv("WHISPER_COMPUTE_TYPE", "")
        if not ctype:
            ctype = "float16" if dev == "cuda" else "int8"
        return dev, ctype

    if use_gpu and cuda_available():
        ctype = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        return "cuda", ctype

    return "cpu", os.getenv("WHISPER_COMPUTE_TYPE", "int8")
