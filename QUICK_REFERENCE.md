# 🎯 Cognitive Speech Compression - Quick Reference Card

## 📋 What This Project Does

- **Compresses** speech audio to ~6.4 kbps (40:1 compression ratio)
- **Preserves** quality using dual-context feature extraction
- **Trains** with adversarial learning + perceptual losses
- **Reconstructs** high-quality speech from compressed features

---

## ⚡ Get Started in 3 Steps

### 1️⃣ Install (2 min)
```bash
cd cognitive_speech_compression
pip install -r requirements.txt
```

### 2️⃣ Train (30+ min)
```bash
python scripts/train.py
```

### 3️⃣ Check Results
```
plots/
├── training_metrics.png      (Loss curves)
├── cc_features.png           (Learned representations)
└── waveform_comparison.png   (Original vs Reconstructed)
```

---

## 📚 Documentation Guide

| File | Purpose | Read if... |
|------|---------|-----------|
| **QUICKSTART.md** | 5-minute setup guide | You want to start immediately |
| **README.md** | Complete documentation | You want full understanding |
| **OUTPUT_DESCRIPTION.md** | Understanding outputs | You're confused by the results |
| **DOCUMENTATION.md** | Doc index & overview | You need navigation help |
| **issues.txt** | Known bugs & fixes | Something breaks |

---

## 🔧 Configuration (Edit `configs/default.yaml`)

### For Quality 🎵
```yaml
loss:
  lambda:
    mel: 100.0  # Increase from 50.0
train:
  total_steps: 10000  # Increase from 100
```

### For Speed ⚡
```yaml
train:
  total_steps: 10  # Decrease to 10
  batch_size: 2    # Decrease from 4
```

### For Stability 🛡️
```yaml
train:
  lr: 1.0e-4      # Lower from 2.5e-4
  grad_clip: 1.0  # Lower from 5.0
```

---

## 📊 Monitoring Metrics

While training, watch for:

| Metric | Good Range | What It Means |
|--------|-----------|------------------|
| **D Loss** | 0.1-0.5 | Discriminator performance |
| **G Loss** | 0.5-3.0 | Reconstruction quality |
| **Mel Loss** | <0.15 | Audio quality (most important!) |
| **CC Loss** | <0.1 | Content preservation |
| **D/G Ratio** | 0.5-2.0 | Training balance |

---

## 🎯 Key Commands

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --cfg configs/default.yaml

# Reconstruct audio
python scripts/infer.py --wav input.wav --ckpt checkpoints/codec_step100.pt --out output.wav

# Prepare dataset
python scripts/prep_librispeech.py --root data/LibriSpeech/trainclean100 --out data/librispeech_100h/train.json
```

---

## 📁 Important Directories

```
checkpoints/          ← Saved models (best = highest step #)
plots/               ← Visualizations (after training)
configs/             ← Configuration files
src/models/          ← Neural network code
data/LibriSpeech/    ← Dataset (download separately)
```

---

## ⚠️ Quick Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: matplotlib` | `pip install matplotlib` |
| `No FLAC files found` | Download LibriSpeech to `data/LibriSpeech/` |
| `CUDA out of memory` | Reduce `batch_size` in config |
| `Audio quality poor` | Increase `mel` loss weight & train longer |
| `Training unstable` | Reduce learning rate: `lr: 1.0e-4` |

---

## 🏗️ Architecture Overview

```
Input Audio [16kHz]
    ↓
[CC Encoder]
├─ Short-context (10ms) → 64-D features
└─ Long-context (40ms) → 64-D features
    ↓
[Quantization] (1-bit delta-mod)
    ↓
[CC Decoder]
├─ Upsample long → short grid
├─ Fuse with short features
└─ Upsample to waveform
    ↓
Output Audio [16kHz]
```

**Compression**: 256 kbps → ~6.4 kbps (40x!)

---

## 📈 Expected Results After Training

### Console Output
```
step 100 | D 0.234 | G 1.254 (adv 0.123 fm 0.345 mel 0.567 cc_s 0.023 cc_l 0.034)
```

### Plots Generated
- **training_metrics.png**: 6-panel loss visualization
- **cc_features.png**: 2-panel feature heatmaps (800ms window)
- **waveform_comparison.png**: Original vs reconstructed comparison

### Model Saved
- `checkpoints/codec_stepN.pt`: N = 2000, 4000, ... (your last checkpoint)

---

## 🚀 Next Steps

After successful training:

1. ✅ **Evaluate**: Listen to audio in `waveform_comparison.png`
2. ✅ **Fine-tune**: Adjust loss weights if needed
3. ✅ **Deploy**: Use best checkpoint for inference
4. ✅ **Experiment**: Try different architectures/datasets

---

## 💡 Pro Tips

1. **Check Mel Loss First**: If it's not decreasing, training isn't learning
2. **Balance Matters**: D/G ratio should hover around 1.0
3. **Train Longer for Quality**: Start with `total_steps: 10000`
4. **Save Often**: Checkpoints save every 2000 steps
5. **GPU Helps**: Training ~10x faster on GPU vs CPU

---

## 📞 File Quick Links

- **Full docs**: `README.md`
- **Quick start**: `QUICKSTART.md`
- **Understanding plots**: `OUTPUT_DESCRIPTION.md`
- **Known issues**: `issues.txt`
- **Dependencies**: `requirements.txt`
- **This file**: `QUICK_REFERENCE.md`

---

## ✨ Success Checklist

- [ ] `pip install -r requirements.txt` runs without errors
- [ ] `python scripts/train.py` starts training
- [ ] Console shows decreasing loss values
- [ ] Plots are generated after training
- [ ] Waveform comparison shows realistic reconstruction
- [ ] CC feature heatmaps show diverse patterns

---

**You're all set! Happy compressing! 🎉**

Last Updated: November 30, 2025
