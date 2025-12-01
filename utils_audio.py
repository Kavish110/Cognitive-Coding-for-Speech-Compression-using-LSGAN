# scripts/utils_audio.py
"""
Lightweight audio utilities for the cognitive_speech_compression project.

All tensors are shaped as:
- Waveforms: [B, 1, T] or [1, 1, T] for mono audio
Functions accept [1, T] or [T] in a few cases and will auto-reshape as needed.
"""

from __future__ import annotations
import os
import math
from typing import List, Tuple, Optional

import torch
import torchaudio


# ---------------------------
# I/O
# ---------------------------

def load_audio(path: str, target_sr: Optional[int] = None, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio with torchaudio, optionally resample and downmix to mono.
    Returns (wav, sr) where wav is [1, 1, T] float32 in [-1, 1].
    """
    wav, sr = torchaudio.load(path)  # [C, T], float32 or int
    wav = wav.float()
    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # [1, T]
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    wav = wav.clamp(-1.0, 1.0).unsqueeze(0)  # [1, 1, T]
    return wav, sr


def save_audio(path: str, wav: torch.Tensor, sr: int):
    """
    Save mono audio. Accepts [B,1,T], [1,T], or [T]. Writes first channel if batched.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if wav.dim() == 3:     # [B,1,T]
        wav = wav[0]
    if wav.dim() == 2:     # [1,T]
        wav = wav[0]
    wav = wav.detach().cpu().clamp(-1, 1)
    torchaudio.save(path, wav.unsqueeze(0), sr)


# ---------------------------
# Level / Loudness helpers
# ---------------------------

def peak_normalize(x: torch.Tensor, peak: float = 0.99, eps: float = 1e-8) -> torch.Tensor:
    """
    Peak-normalize to target 'peak' (<=1.0). Works with [B,1,T], [1,T], or [T].
    """
    s = _ensure_b1t(x)
    m = s.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    y = (s / m) * peak
    return y.view_as(s) if s is x else _restore_shape(x, y)


def rms(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """RMS over time for each item in batch. Returns shape [B,1,1]."""
    s = _ensure_b1t(x)
    r = torch.sqrt((s ** 2).mean(dim=-1, keepdim=True).clamp_min(eps))
    return r


def dbfs(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """dBFS (20*log10(rms)) per batch item, returns [B,1,1]."""
    return 20.0 * torch.log10(rms(x, eps))


def match_loudness(x: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    """
    Scale to target dBFS. Returns waveform with same shape as input.
    """
    s = _ensure_b1t(x)
    cur_db = dbfs(s)
    gain_db = target_dbfs - cur_db
    gain = (10.0 ** (gain_db / 20.0)).to(s.dtype)
    y = s * gain
    return _restore_shape(x, y)


# ---------------------------
# Resample / Pad / Trim / Framing
# ---------------------------

def resample(x: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Resample [B,1,T] (or [1,T]/[T]) from orig_sr to new_sr."""
    s = _ensure_b1t(x)
    y = torchaudio.functional.resample(s, orig_sr, new_sr)
    return _restore_shape(x, y)


def pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pad (right) or trim waveform to target_len. Shape-preserving.
    """
    s = _ensure_b1t(x)
    T = s.size(-1)
    if T == target_len:
        return x
    if T < target_len:
        pad = target_len - T
        y = torch.nn.functional.pad(s, (0, pad))
    else:
        y = s[..., :target_len]
    return _restore_shape(x, y)


def segment_audio(x: torch.Tensor, segment_seconds: float, sr: int, hop_seconds: Optional[float] = None) -> List[torch.Tensor]:
    """
    Slice audio into segments of length segment_seconds with optional hop.
    Returns list of [1,1,segment_T] tensors.
    """
    s = _ensure_b1t(x)
    seg_T = int(round(segment_seconds * sr))
    hop_T = seg_T if hop_seconds is None else int(round(hop_seconds * sr))
    T = s.size(-1)
    out = []
    for start in range(0, max(T - seg_T + 1, 1), hop_T):
        out.append(s[..., start:start + seg_T])
        if start + seg_T >= T:
            break
    return out


def frame_signal(x: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    """
    Frame [B,1,T] into [B,1,frame_len, num_frames] with hop_len.
    """
    s = _ensure_b1t(x)
    y = s.unfold(dimension=-1, size=frame_len, step=hop_len)  # [B,1,frame_len,F]
    return y


def overlap_add(frames: torch.Tensor, hop_len: int) -> torch.Tensor:
    """
    Overlap-add inverse of frame_signal. 'frames' is [B,1,frame_len,F].
    Returns [B,1,T].
    """
    B, C, L, F = frames.shape
    T = (F - 1) * hop_len + L
    y = torch.zeros(B, C, T, dtype=frames.dtype, device=frames.device)
    wsum = torch.zeros_like(y)
    for i in range(F):
        start = i * hop_len
        y[..., start:start + L] += frames[..., i]
        wsum[..., start:start + L] += 1.0
    wsum = wsum.clamp_min(1.0)
    return y / wsum


# ---------------------------
# Augmentation / Pre-emphasis
# ---------------------------

def snr_mix(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mix 'noise' into 'clean' at desired SNR (dB). Both [B,1,T] (or broadcastable).
    """
    c = _ensure_b1t(clean)
    n = _ensure_b1t(noise)
    # Match lengths
    if n.size(-1) < c.size(-1):
        reps = math.ceil(c.size(-1) / n.size(-1))
        n = n.repeat(1, 1, reps)[..., :c.size(-1)]
    elif n.size(-1) > c.size(-1):
        n = n[..., :c.size(-1)]
    # Compute scaling
    p_sig = (c ** 2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    p_noise = (n ** 2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    target_ratio = 10.0 ** (-snr_db / 10.0)  # noise_power / signal_power
    scale = torch.sqrt((p_sig * target_ratio) / p_noise)
    mix = c + scale * n
    return _restore_shape(clean, mix)


def pre_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Pre-emphasis filter y[t] = x[t] - coeff * x[t-1]."""
    s = _ensure_b1t(x)
    y = torch.cat([s[..., :1], s[..., 1:] - coeff * s[..., :-1]], dim=-1)
    return _restore_shape(x, y)


def de_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Inverse of pre-emphasis (IIR)."""
    s = _ensure_b1t(x)
    y = torch.zeros_like(s)
    y[..., 0] = s[..., 0]
    for t in range(1, s.size(-1)):
        y[..., t] = s[..., t] + coeff * y[..., t - 1]
    return _restore_shape(x, y)


# ---------------------------
# Feature helpers (thin wrappers)
# ---------------------------

def mel_spectrogram(x: torch.Tensor, sr: int, n_fft: int = 1024, hop_length: int = 256,
                    win_length: int = 1024, n_mels: int = 80, center: bool = True, power: float = 1.0):
    """
    Convenience mel wrapper returning a tensor shaped [B, n_mels, T_mel].
    """
    s = _ensure_b1t(x)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, center=center, power=power
    ).to(s.device)
    return mel(s)


# ---------------------------
# Codec / bookkeeping utils
# ---------------------------

def bitrate_from_bits(bits_tensor: torch.Tensor, hop_ms: float) -> float:
    """
    Estimate kbps given Δ-mod bits tensor [B, T_frames, C] and frame hop in ms.
    kbps = (C * frames_per_sec) / 1000, frames_per_sec = 1000 / hop_ms
    """
    if bits_tensor.dim() != 3:
        raise ValueError("bits_tensor must be [B, T_frames, C]")
    _, _, C = bits_tensor.shape
    frames_per_sec = 1000.0 / float(hop_ms)
    bps = C * frames_per_sec
    return float(bps) / 1000.0


# ---------------------------
# Internal shape helpers
# ---------------------------

def _ensure_b1t(x: torch.Tensor) -> torch.Tensor:
    """Coerce to [B,1,T] without copying data when possible."""
    if x.dim() == 1:         # [T]
        return x.view(1, 1, -1)
    if x.dim() == 2:         # [1,T] or [C,T]
        if x.size(0) == 1:
            return x.view(1, 1, -1)
        else:
            # downmix channels silently (rare path)
            return x.mean(dim=0, keepdim=True).view(1, 1, -1)
    if x.dim() == 3:         # [B,1,T] or [B,C,T]
        if x.size(1) == 1:
            return x
        else:
            return x.mean(dim=1, keepdim=True)
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")


def _restore_shape(ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Restore y to the original shape of ref when _ensure_b1t was used."""
    if ref.dim() == 1:   # [T]
        return y.view(-1)
    if ref.dim() == 2:   # [C,T] or [1,T]
        return y.view(1, -1)
    if ref.dim() == 3:   # [B,1,T] or [B,C,T]
        return y
    return y
