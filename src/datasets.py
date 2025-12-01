import os
import glob
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        root,
        target_sr=16000,
        frame_ms=20,
        normalize=True,
        min_audio_ms=200,
    ):
        self.root = root
        self.target_sr = target_sr
        self.frame_len = int((frame_ms / 1000) * target_sr)
        self.min_audio_len = int((min_audio_ms / 1000) * target_sr)
        self.normalize = normalize

        # Scan dataset
        self.files = sorted(glob.glob(f"{root}/**/*.flac", recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f"No FLAC files found in {root}")

        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=target_sr)

    def _load(self, path):
        # Use soundfile to load FLAC (avoids torchcodec dependency)
        wav, sr = sf.read(path, dtype='float32')
        
        # Convert to tensor and ensure shape [C, T]
        wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        # Resample if required
        if sr != self.target_sr:
            wav = self.resampler(wav)

        wav = wav.squeeze(0)

        # Normalize waveform
        if self.normalize:
            wav = wav / (wav.abs().max() + 1e-6)

        return wav

    def _chunk(self, wav):
        """Segments audio into fixed 20 ms frames for the encoder."""
        if len(wav) < self.min_audio_len:
            return None

        frames = wav.unfold(0, self.frame_len, self.frame_len).T
        return frames

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load(path)
        frames = self._chunk(wav)

        if frames is None:
            return None

        return frames

    def __len__(self):
        return len(self.files)


class LibriSpeechWavDataset(Dataset):
    """Loads full waveforms instead of chunked frames, for training codec models."""
    def __init__(
        self,
        root,
        target_sr=16000,
        segment_seconds=2.0,
        normalize=True,
        min_audio_ms=200,
    ):
        self.root = root
        self.target_sr = target_sr
        self.segment_len = int(segment_seconds * target_sr)
        self.min_audio_len = int((min_audio_ms / 1000) * target_sr)
        self.normalize = normalize

        # Scan dataset
        self.files = sorted(glob.glob(f"{root}/**/*.flac", recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f"No FLAC files found in {root}")

        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=target_sr)

    def _load(self, path):
        # Use soundfile to load FLAC (avoids torchcodec dependency)
        wav, sr = sf.read(path, dtype='float32')
        
        # Convert to tensor and ensure shape [C, T]
        wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        # Resample if required
        if sr != self.target_sr:
            wav = self.resampler(wav)

        wav = wav.squeeze(0)

        # Normalize waveform
        if self.normalize:
            wav = wav / (wav.abs().max() + 1e-6)

        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load(path)
        
        # Pad or trim to segment length
        if len(wav) < self.segment_len:
            wav = torch.nn.functional.pad(wav, (0, self.segment_len - len(wav)))
        else:
            wav = wav[:self.segment_len]
        
        # Return [1, T] for unsqueezing into [B, 1, T] in dataloader
        return wav.unsqueeze(0)

    def __len__(self):
        return len(self.files)


def librospeech_collate(batch):
    """Flattens all frames from each file into one training batch."""
    frames = [x for x in batch if x is not None]

    if len(frames) == 0:
        return None

    frames = torch.cat(frames, dim=0)
    return frames

def wav_collate(batch):
    """Stacks waveforms into a batch [B, 1, T]."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    # Each item is [1, T], stack them to get [B, 1, T]
    return torch.stack(batch, dim=0)



# Usage example ---------------------------------------------------------
if __name__ == "__main__":
    dataset = LibriSpeechDataset(
        root="/path/to/LibriSpeech/train-clean-100",
        frame_ms=20,
        normalize=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=librospeech_collate,
        num_workers=4,
    )

    for batch in dataloader:
        if batch is None:
            continue

        print("Batch:", batch.shape)   # [B, 320]  for 20 ms @ 16 kHz
        break
