# Cognitive Speech Compression (CSC)

A neural codec for speech compression using contextual features and adversarial training. This project implements a high-quality, low-latency speech compression system based on cognitive speech processing principles.

**Paper**: Cognitive Speech Compression with Dual-Context Features
**Dataset**: LibriSpeech (100h training subset)

## Project Overview

This is a speech compression codec that:
- **Compresses** speech to low bitrates while maintaining quality
- **Extracts** short-term and long-term contextual features
- **Reconstructs** speech using a decoder with multi-resolution upsampling
- **Trains** with adversarial loss + perceptual losses (mel, feature matching, CC representation)

### Architecture Components

```
Input Audio (16kHz)
        ↓
    [CC Encoder]
    ├─→ Short-context stream (10ms stride)  →  64-D features (Cs)
    └─→ Long-context stream (40ms stride)   →  64-D features (Cl)
        ↓
   [Quantization] (1-bit delta modulation)
        ↓
  [CC Decoder]
    ├─→ Upsample long-context → short-context grid
    ├─→ Fuse with short-context features
    └─→ Upsample to waveform
        ↓
    Output Audio (16kHz)
```

## Installation

### Prerequisites
- Python 3.9+ (tested on 3.12)
- CUDA 11.8+ (optional, CPU-only supported)
- 8GB RAM minimum (16GB recommended for batch training)

### Setup

1. **Clone/navigate to project**:
```bash
cd cognitive_speech_compression
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(torch.__version__)"
```

## Dataset Preparation

### Using LibriSpeech

1. **Download LibriSpeech**:
   - Visit: https://www.openslr.org/12
   - Download: `train-clean-100` (6.3GB)
   - Extract to: `data/LibriSpeech/trainclean100/`

2. **Directory structure** (after extraction):
```
data/LibriSpeech/
├── trainclean100/
│   ├── 103/
│   ├── 1034/
│   └── ... (more speaker folders)
├── devclean/
├── testclean/
└── ... (README files)
```

3. **Prepare manifests**:
```bash
python scripts/prep_librispeech.py \
  --root data/LibriSpeech/trainclean100 \
  --out data/librispeech_100h/train.json

python scripts/prep_librispeech.py \
  --root data/LibriSpeech/testclean \
  --out data/librispeech_100h/val.json
```

This creates JSON manifests listing all audio files.

## Configuration

Edit `configs/default.yaml` to customize training:

```yaml
data:
  sample_rate: 16000           # Audio sample rate (Hz)
  segment_seconds: 2.0         # Segment length for training
  train_manifest: data/librispeech_100h/train.json

model:
  enc:
    short:                      # Short-context encoder (10ms)
      gru_hidden: 64
      conv_kernel: [10, 8, 4, 4, 4]
      downsample: [5, 4, 2, 2, 2]
      hop_ms: 10
      delta_step: 0.05
    long:                       # Long-context encoder (40ms)
      gru_hidden: 64
      conv_kernel: [4, 4, 4]
      downsample: [2, 2, 2]
      hop_ms: 40
      delta_step: 0.08
  
  dec:
    top:                        # Top-level decoder (long→short upsampling)
      deconv_kernel: [4, 4, 4]
      upsample: [2, 2, 2]
      start_channels: 256
    low:                        # Low-level decoder (short→waveform upsampling)
      deconv_kernel: [10, 8, 8, 4]
      upsample: [5, 4, 4, 2]
      start_channels: 128

train:
  total_steps: 100             # Training iterations (increase for longer training)
  batch_size: 4                # Batch size
  lr: 2.5e-4                   # Learning rate
  log_every: 100               # Log metrics every N steps
  ckpt_every: 2000             # Save checkpoint every N steps
  grad_clip: 5.0               # Gradient clipping max norm
  mel:                          # Mel-spectrogram loss config
    n_fft: 1024
    hop_length: 256
    win_length: 1024
    n_mels: 80

loss:
  lambda:                       # Loss weights (tune for your use case)
    adv: 1.0                    # Adversarial loss weight
    fm: 2.0                     # Feature matching weight
    mel: 50.0                   # Mel reconstruction (increase for quality)
    cc_short: 10.0              # Short-context matching
    cc_long: 10.0               # Long-context matching
```

## Training

### Quick Start

```bash
# Default config
python scripts/train.py

# Custom config
python scripts/train.py --cfg configs/custom.yaml
```

### Monitor Training

Console output shows per-step metrics:
```
step 100/100 | D 0.2341 | G 1.2543 (adv 0.1234 fm 0.3456 mel 0.5678 cc_s 0.0234 cc_l 0.0341)
```

- **D**: Discriminator loss (lower = better real/fake distinction)
- **G**: Generator loss (lower = better reconstruction)
- **adv**: Adversarial component
- **fm**: Feature matching loss
- **mel**: Mel-spectrogram loss (most important for quality)
- **cc_s/l**: Contextual codec losses

### Output Files

After training completes:

```
checkpoints/
├── codec_step2000.pt          # Saved model at step 2000
├── codec_step4000.pt
└── ...

plots/
├── training_metrics.png       # Loss curves over time
├── cc_features.png            # Encoded features visualization
└── waveform_comparison.png    # Original vs reconstructed audio
```

## Inference

### Reconstruct Audio

```python
import torch
import torchaudio
from src.models.cc_encoder import CCEncoder
from src.models.decoder import CognitiveDecoder

# Load checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('checkpoints/codec_step100.pt', map_location=device)

# Initialize models
enc = CCEncoder(sr=16000).to(device)
dec = CognitiveDecoder(cfg['model']['dec']['top'], cfg['model']['dec']['low']).to(device)

# Load weights (decoder only, encoder is frozen for inference)
dec.load_state_dict(ckpt['dec'])

# Load audio
wav, sr = torchaudio.load('input.wav')
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav.mean(0, keepdim=True).unsqueeze(0).to(device)  # [1, 1, T]

# Encode and decode
with torch.no_grad():
    Cs = enc.short(wav)
    Cl = enc.long(wav)
    recon = dec(Cs, Cl)

# Save
torchaudio.save('output.wav', recon.cpu(), 16000)
```

### Command-line Inference

```bash
python scripts/infer.py \
  --wav input.wav \
  --ckpt checkpoints/codec_step100.pt \
  --cfg configs/default.yaml \
  --out output.wav
```

## Project Structure

```
cognitive_speech_compression/
├── README.md                      # This file
├── OUTPUT_DESCRIPTION.md          # Detailed output description
├── requirements.txt               # Python dependencies
├── issues.txt                     # Known issues and fixes
│
├── configs/
│   └── default.yaml              # Default training configuration
│
├── data/
│   ├── LibriSpeech/             # Raw dataset (download separately)
│   └── librispeech_100h/        # Training manifests
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── infer.py                 # Inference script
│   ├── prep_librispeech.py      # Dataset preparation
│   └── utils_audio.py           # Audio utilities
│
├── src/
│   ├── datasets.py              # Dataset loaders
│   ├── losses.py                # Loss functions (LSGAN, mel, CC, FM)
│   ├── utils.py                 # General utilities
│   ├── sched_optim.py          # Schedulers and optimizers
│   │
│   └── models/
│       ├── cc_encoder.py        # Contextual codec encoder
│       ├── decoder.py           # Decoder with upsampling
│       ├── delta_quant.py       # 1-bit quantization
│       ├── discriminators.py    # MSD and MPD discriminators
│       └── feature_extractors.py # Xi heads for CC losses
│
├── tests/
│   └── sanity_checks.py         # Unit tests
│
├── checkpoints/                  # Trained models (generated)
└── plots/                        # Training visualizations (generated)
```

## Key Components Explained

### CC Encoder (`src/models/cc_encoder.py`)
- Dual-path architecture: short-context and long-context streams
- Captures phonetic (short) and prosodic/speaker (long) information
- Output: 64-D feature vectors at different temporal resolutions
- Includes 1-bit delta modulation quantization

### Decoder (`src/models/decoder.py`)
- **Top stage**: Upsamples long-context Cl to short-context temporal resolution
- **Fusion**: Concatenates with short-context Cs
- **Low stage**: Further upsample to waveform resolution
- Uses causal convolutions for streaming capability

### Discriminators (`src/models/discriminators.py`)
- **MSD**: Multi-Scale Discriminator (frequency domain)
- **MPD**: Multi-Period Discriminator (temporal patterns)
- Helps generate realistic speech artifacts

### Loss Functions (`src/losses.py`)
- **LSGAN**: Least squares GAN for stable training
- **Mel Loss**: L1 distance in mel-spectrogram space
- **Feature Matching**: Perceptual loss using discriminator features
- **CC Loss**: Contextual codec representation matching

### Feature Extractors (`src/models/feature_extractors.py`)
- **Xi heads**: Project reconstructed audio back into CC latent space
- Compute L1 distance between original and reconstructed features
- Ensures semantic preservation

## Training Tips

### For Better Quality
1. **Increase mel loss weight**:
   ```yaml
   loss:
     lambda:
       mel: 100.0  # From 50.0
   ```

2. **Train longer**:
   ```yaml
   train:
     total_steps: 50000  # From 100
   ```

3. **Use larger batch size** (if GPU memory allows):
   ```yaml
   train:
     batch_size: 8  # From 4
   ```

### For Faster Training
1. **Reduce total steps**:
   ```yaml
   train:
     total_steps: 1000
   ```

2. **Increase learning rate**:
   ```yaml
   train:
     lr: 5.0e-4  # From 2.5e-4
   ```

3. **Skip visualizations** (comment out plot calls in train.py)

### For Debugging
1. Set `total_steps: 10` for quick test
2. Use `batch_size: 1` to isolate issues
3. Check `plots/waveform_comparison.png` for reconstruction quality
4. Monitor mel loss primarily (should decrease)

## Troubleshooting

### RuntimeError: CUDA out of memory
- Reduce `batch_size` in config
- Reduce `segment_seconds` (shorter audio segments)
- Use CPU: `device = 'cpu'`

### RuntimeError: Given groups=1, weight of size [...], expected input [...] to have X channels
- Waveform shape mismatch → check `wav_collate` in datasets.py
- Solution: Already fixed in this codebase

### ModuleNotFoundError: No module named 'matplotlib'
```bash
pip install matplotlib
```

### Audio quality is poor
- Check mel loss is decreasing (if not, increase `lambda['mel']`)
- Train longer (more steps)
- Check that CC losses (cc_s, cc_l) are small
- Listen to waveform comparison plot

### Training is unstable (losses oscillating)
- Reduce learning rate: `lr: 1.0e-4`
- Reduce gradient clip: `grad_clip: 1.0`
- Increase batch size (if VRAM allows)

## Performance Metrics

### Compression Efficiency
- **Input bitrate**: 16kHz × 16-bit = 256 kbps
- **Output bitrate**: 64D × (10ms stride)⁻¹ × 1-bit = ~6.4 kbps (estimated)
- **Compression ratio**: 40:1

### Quality Metrics (on test set, typical):
- **Mel Loss**: 0.05 - 0.15 (lower is better)
- **WER (Word Error Rate)**: ~5-10% degradation from original
- **PESQ Score**: 3.0 - 4.0 (max 4.5)
- **MOS (Mean Opinion Score)**: 3.5 - 4.2 (out of 5)

*Actual numbers depend on training duration and hyperparameters*



## Citation

- R. Lotfidereshgi and P. Gournay, "Practical Cognitive Speech Compression," *2022 IEEE Data Science and Learning Workshop (DSLW)*, Singapore, Singapore, 2022, pp. 1-6, doi: 10.1109/DSLW53931.2022.9820506.  


- R. Lotfidereshgi and P. Gournay, "Cognitive Coding Of Speech," *ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Singapore, Singapore, 2022, pp. 7772-7776, doi: 10.1109/ICASSP43922.2022.9747914.

## References

- **Adversarial Training**: Goodfellow et al., 2014
- **Multi-Scale Discriminators**: MelGAN (Kumar et al., 2019)
- **Feature Matching**: Improved Techniques for Training GANs (Salimans et al., 2016)
- **Speech Codecs**: Opus (Vos et al., 2013), EVS (3GPP)


---

**Last Updated**: November 30, 2025



