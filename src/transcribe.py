"""Speech-to-text using pretrained faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.audio_utils import load_wav_mono_float32
from src.gpu_utils import resolve_whisper_device
from src.vtt_parser import Segment


def transcribe_wav(
    wav_path: Path,
    model_size: str = "small",
    device: str | None = None,
    compute_type: str | None = None,
    use_gpu: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[Segment]:
    dev, ctype = resolve_whisper_device(use_gpu=use_gpu, device=device)
    if compute_type:
        ctype = compute_type

    if on_progress:
        on_progress(f"Loading Whisper model ({model_size}) on {dev} ({ctype})…")
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device=dev, compute_type=ctype)
    except Exception as e:
        if dev == "cuda":
            if on_progress:
                on_progress(f"CUDA failed ({e}), falling back to CPU…")
            dev, ctype = "cpu", "int8"
            model = WhisperModel(model_size, device=dev, compute_type=ctype)
        else:
            raise

    if on_progress:
        on_progress("Loading audio…")
    audio = load_wav_mono_float32(wav_path)
    if on_progress:
        label = "GPU" if dev == "cuda" else "CPU"
        on_progress(f"Transcribing on {label}…")
    segments_iter, _info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )
    segments: list[Segment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            Segment(start=float(seg.start), end=float(seg.end), speaker="Speaker", text=text)
        )
    if on_progress:
        on_progress(f"Transcription complete ({len(segments)} segments, {dev}).")
    return segments
