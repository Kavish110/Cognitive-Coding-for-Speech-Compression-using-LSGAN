# Quick Start Guide - Cognitive Speech Compression

Get up and running in 5 minutes!

## 1. Install Dependencies (2 min)

```bash
# Navigate to project
cd cognitive_speech_compression

# Install packages
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch; import torchaudio; print('✓ Ready!')"
```

## 2. Download & Prepare Dataset (15 min)

**Option A: Quick Test (No real data)**
```bash
# Just run training on dummy data - useful to test code
python scripts/train.py
```

**Option B: Use Real LibriSpeech Data**

1. Download (do this once, ~6GB):
   ```bash
   # Visit: https://www.openslr.org/12
   # Download: train-clean-100 (6.3GB)
   # Extract to: data/LibriSpeech/trainclean100/
   ```

2. Prepare manifests:
   ```bash
   python scripts/prep_librispeech.py \
     --root data/LibriSpeech/trainclean100 \
     --out data/librispeech_100h/train.json
   ```

## 3. Train (30+ min depending on config)

```bash
# Using default config
python scripts/train.py

# Custom config
python scripts/train.py --cfg configs/default.yaml
```

You'll see output like:
```
step 100/100 | D 0.2341 | G 1.2543 (adv 0.1234 fm 0.3456 mel 0.5678 cc_s 0.0234 cc_l 0.0341)
step 200/100 | D 0.1856 | G 1.1234 (adv 0.0987 fm 0.3123 mel 0.5234 cc_s 0.0156 cc_l 0.0234)
...
```

### What do those numbers mean?

- **D**: Discriminator loss (lower = better at detecting fake audio) → goal: 0.1-0.5
- **G**: Generator/reconstruction loss (lower = better reconstruction) → goal: decreasing
- **mel**: Most important! Spectral similarity (lower = better audio quality) → goal: < 0.1
- **cc_s/cc_l**: Content preservation losses (lower = better) → goal: < 0.05

## 4. Check Results

After training finishes, you'll find:

### Training Metrics
```
plots/training_metrics.png
```
Shows 6 plots:
- Discriminator loss trend
- Generator loss components
- Loss balance (D vs G)
- Compression quality (mel loss)

### Feature Visualization
```
plots/cc_features.png
```
Shows what the encoder learned:
- Top: Short-context features (10ms resolution)
- Bottom: Long-context features (40ms resolution)
- Bright colors = high activation

### Audio Reconstruction
```
plots/waveform_comparison.png
```
Shows original vs reconstructed:
- Top: Original audio
- Middle: Reconstructed audio
- Bottom: Error signal (should be small!)

### Saved Models
```
checkpoints/codec_step2000.pt
checkpoints/codec_step4000.pt
...
```
Use these for inference!

## 5. Inference (Reconstruct Audio)

```bash
python scripts/infer.py \
  --wav input.wav \
  --ckpt checkpoints/codec_step100.pt \
  --cfg configs/default.yaml \
  --out reconstructed.wav
```

## Customization - 3 Things to Try

### ⚡ Train Faster (but lower quality)
Edit `configs/default.yaml`:
```yaml
train:
  total_steps: 10          # From 100 (much faster)
  batch_size: 2            # From 4 (less GPU memory)
```

### 🎵 Better Audio Quality
```yaml
train:
  total_steps: 1000        # Much longer training
  batch_size: 8            # Larger batches

loss:
  lambda:
    mel: 100.0             # Focus on quality (from 50)
```

### 🚀 Use GPU (if available)
```bash
# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"

# Already auto-detects! Code runs on GPU if available
```

## File Structure for Reference

```
cognitive_speech_compression/
├── README.md                    ← Full documentation
├── QUICKSTART.md               ← This file
├── OUTPUT_DESCRIPTION.md       ← What plots mean
├── requirements.txt
├── configs/
│   └── default.yaml            ← Training settings (edit this!)
├── scripts/
│   ├── train.py               ← Training (main script)
│   ├── infer.py               ← Audio reconstruction
│   └── prep_librispeech.py    ← Prepare dataset
├── src/
│   ├── models/                ← Neural network components
│   ├── losses.py              ← Loss functions
│   ├── datasets.py            ← Data loading
│   └── utils.py               ← Utilities
├── data/
│   ├── LibriSpeech/           ← Download data here
│   └── librispeech_100h/      ← Manifests (auto-generated)
├── checkpoints/               ← Saved models (auto-generated)
└── plots/                     ← Visualizations (auto-generated)
```

## Typical Training Output

```
Epoch 1: Training encoder and decoder
step 100/100 | D 0.5432 | G 3.2341 ...  (early, losses high)
step 200/100 | D 0.3211 | G 2.1234 ...  (stabilizing)
step 300/100 | D 0.2341 | G 1.2543 ...  (converging)
...
Generating final training plots...
✓ Saved metrics plot to plots/training_metrics.png
✓ Saved CC features plot to plots/cc_features.png
✓ Saved waveform comparison to plots/waveform_comparison.png
Training completed!
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: matplotlib` | `pip install matplotlib` |
| `CUDA out of memory` | Reduce `batch_size` in config |
| `No FLAC files found` | Download LibriSpeech first |
| `Plots not showing` | Install matplotlib: `pip install matplotlib` |
| `Audio quality poor` | Increase `total_steps` and mel loss weight |

## Next Steps

1. ✅ Train for longer: Change `total_steps: 1000` (or more!)
2. ✅ Improve quality: Increase `lambda['mel']` in config
3. ✅ Experiment: Try different architectures in `configs/`
4. ✅ Deploy: Use best checkpoint for inference
5. ✅ Analyze: Read OUTPUT_DESCRIPTION.md for detailed metrics explanation

## Need Help?

- **Full guide**: See `README.md`
- **Understanding outputs**: See `OUTPUT_DESCRIPTION.md`
- **Code issues**: Check `issues.txt`
- **Questions**: Review code comments in `src/models/`

---

**That's it!** You're ready to compress speech! 🎉

