import os, math, yaml, torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import LibriSpeechWavDataset, wav_collate
from src.models.cc_encoder import CCEncoder
from src.models.decoder import CognitiveDecoder, match_length
from src.models.discriminators import MSD, MPD
from src.models.feature_extractors import XiHead
from src.losses import (lsgan_d, lsgan_g, feature_matching_loss, mel_loss, cc_repr_loss)

def plot_training_metrics(metrics_dict, save_dir="plots"):
    """Plot training metrics: loss curves, accuracy-like metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # D Loss
    ax = axes[0, 0]
    ax.plot(metrics_dict['d_loss'], label='D Loss', color='red', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Loss')
    ax.grid(True)
    ax.legend()
    
    # G Loss components
    ax = axes[0, 1]
    ax.plot(metrics_dict['adv_loss'], label='Adversarial', alpha=0.7)
    ax.plot(metrics_dict['fm_loss'], label='Feature Matching', alpha=0.7)
    ax.plot(metrics_dict['mel_loss'], label='Mel Loss', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Generator Loss Components')
    ax.grid(True)
    ax.legend()
    
    # CC Loss
    ax = axes[0, 2]
    ax.plot(metrics_dict['cc_short_loss'], label='CC Short', alpha=0.7)
    ax.plot(metrics_dict['cc_long_loss'], label='CC Long', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('CC Representation Losses')
    ax.grid(True)
    ax.legend()
    
    # Total G Loss
    ax = axes[1, 0]
    ax.plot(metrics_dict['g_loss'], label='Total G Loss', color='blue', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Generator Loss')
    ax.grid(True)
    ax.legend()
    
    # Loss ratio (should be balanced)
    ax = axes[1, 1]
    d_loss = np.array(metrics_dict['d_loss'])
    g_loss = np.array(metrics_dict['g_loss'])
    ratio = np.divide(d_loss, g_loss + 1e-8)
    ax.plot(ratio, label='D_Loss / G_Loss', color='purple', linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Balanced')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ratio')
    ax.set_title('D/G Loss Ratio (should be ~1)')
    ax.grid(True)
    ax.legend()
    
    # Compression efficiency (lower mel loss = better)
    ax = axes[1, 2]
    ax.plot(metrics_dict['mel_loss'], label='Mel Loss (Lower=Better)', color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Compression Quality (Mel Loss)')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    print(f"Saved metrics plot to {os.path.join(save_dir, 'training_metrics.png')}")
    plt.close()

def plot_cc_features(Cs, Cl, save_dir="plots"):
    """Plot raw and quantized CC features over 800ms of speech."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first sample from batch
    cs = Cs[0].detach().cpu().numpy()  # [T_short, 64]
    cl = Cl[0].detach().cpu().numpy()  # [T_long, 64]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Encoder CC Features (800ms Speech Segment)', fontsize=14, fontweight='bold')
    
    # Short-term features
    ax = axes[0]
    im0 = ax.imshow(cs.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title('Short-term Context (Cs) - 10ms Stride')
    ax.set_xlabel('Time Frames (10ms each)')
    ax.set_ylabel('Feature Dimension (0-63)')
    plt.colorbar(im0, ax=ax, label='Feature Magnitude')
    
    # Long-term features
    ax = axes[1]
    im1 = ax.imshow(cl.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title('Long-term Context (Cl) - 40ms Stride')
    ax.set_xlabel('Time Frames (40ms each)')
    ax.set_ylabel('Feature Dimension (0-63)')
    plt.colorbar(im1, ax=ax, label='Feature Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cc_features.png'), dpi=150, bbox_inches='tight')
    print(f"Saved CC features plot to {os.path.join(save_dir, 'cc_features.png')}")
    plt.close()

def plot_waveform_comparison(wav_orig, wav_recon, sr=16000, save_dir="plots"):
    """Plot original vs reconstructed waveform."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first sample from batch, convert to numpy
    orig = wav_orig[0, 0].detach().cpu().numpy()  # [T]
    recon = wav_recon[0, 0].detach().cpu().numpy()  # [T]
    
    # Trim to same length
    min_len = min(len(orig), len(recon))
    orig = orig[:min_len]
    recon = recon[:min_len]
    
    # Time axis in seconds
    time = np.arange(len(orig)) / sr
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Speech Waveform: Original vs Reconstructed', fontsize=14, fontweight='bold')
    
    # Original
    ax = axes[0]
    ax.plot(time, orig, linewidth=0.8, color='blue', alpha=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Original Waveform')
    ax.grid(True, alpha=0.3)
    
    # Reconstructed
    ax = axes[1]
    ax.plot(time, recon, linewidth=0.8, color='green', alpha=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Reconstructed Waveform')
    ax.grid(True, alpha=0.3)
    
    # Difference (error)
    ax = axes[2]
    error = orig - recon
    ax.plot(time, error, linewidth=0.8, color='red', alpha=0.7, label='Error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Error')
    ax.set_title(f'Reconstruction Error (MAE: {np.mean(np.abs(error)):.5f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'waveform_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved waveform comparison to {os.path.join(save_dir, 'waveform_comparison.png')}")
    plt.close()

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_tr = LibriSpeechWavDataset(cfg['data']['train_manifest'], target_sr=cfg['data']['sample_rate'], segment_seconds=cfg['data']['segment_seconds'])
    dl = DataLoader(ds_tr, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=0, drop_last=True, collate_fn=wav_collate)

    enc = CCEncoder(sr=cfg['data']['sample_rate'],
                    short_cfg=cfg['model']['enc']['short'],
                    long_cfg=cfg['model']['enc']['long']).to(device)
    dec = CognitiveDecoder(cfg['model']['dec']['top'], cfg['model']['dec']['low']).to(device)
    msd, mpd = MSD().to(device), MPD().to(device)

    # ξ heads for CC-representation distances
    xi_s = XiHead(sr=cfg['data']['sample_rate'], kind='short').to(device)
    xi_l = XiHead(sr=cfg['data']['sample_rate'], kind='long').to(device)

    g_params = list(dec.parameters()) + list(xi_s.parameters()) + list(xi_l.parameters())
    d_params = list(msd.parameters()) + list(mpd.parameters())
    opt_g = torch.optim.AdamW(g_params, lr=cfg['train']['lr'], betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(d_params, lr=cfg['train']['lr'], betas=(0.8, 0.99))

    # Metrics tracking
    metrics = {
        'd_loss': [],
        'g_loss': [],
        'adv_loss': [],
        'fm_loss': [],
        'mel_loss': [],
        'cc_short_loss': [],
        'cc_long_loss': [],
    }
    
    first_sample_wavs = None
    first_sample_Cs = None
    first_sample_Cl = None

    # Local LSGAN implementations that return torch.Tensors (preserving autograd)
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    step, total = 0, cfg['train']['total_steps']
    while step < total:
        for wav in dl:
            if wav is None:
                continue
            step += 1
            # move input to device
            # wav is already [B, 1, T] from wav_collate
            wav = wav.to(device)
            Cs = enc.short(wav)
            Cl = enc.long(wav)
            x_hat = dec(Cs, Cl)
            
            # Match reconstruction length to input
            x_hat = match_length(x_hat, wav)
            
            # Save first sample for visualization
            if first_sample_wavs is None:
                first_sample_wavs = wav.clone().detach()
                first_sample_Cs = Cs.clone().detach()
                first_sample_Cl = Cl.clone().detach()

            # ===== DISCRIMINATE =====
            # Real/fake
            real_logits_msd, real_feats_msd = msd(wav)
            fake_logits_msd, fake_feats_msd = msd(x_hat.detach())
            real_logits_mpd, real_feats_mpd = mpd(wav)
            fake_logits_mpd, fake_feats_mpd = mpd(x_hat.detach())

            # ----- Train D (LSGAN) -----
            opt_d.zero_grad(set_to_none=True)
            d_loss = lsgan_d(real_logits_msd + real_logits_mpd,
                             fake_logits_msd + fake_logits_mpd)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(d_params, cfg['train']['grad_clip'])
            opt_d.step()

            # ----- Train G -----
            opt_g.zero_grad(set_to_none=True)
            # Re-compute for generator - real features don't change, just get new fake ones with gradients
            real_logits_msd_g, real_feats_msd_g = msd(wav)
            g_fake_logits_msd, g_fake_feats_msd = msd(x_hat)
            real_logits_mpd_g, real_feats_mpd_g = mpd(wav)
            g_fake_logits_mpd, g_fake_feats_mpd = mpd(x_hat)
            adv = lsgan_g(g_fake_logits_msd + g_fake_logits_mpd)

            # Feature matching (sum over all scales/periods)
            fm  = feature_matching_loss(real_feats_msd_g, g_fake_feats_msd)
            fm += feature_matching_loss(real_feats_mpd_g, g_fake_feats_mpd)

            # Mel L1
            mel = mel_loss(wav, x_hat, cfg['data']['sample_rate'], cfg['train']['mel'])

            # CC representation distances (Eq. 3) on short/long heads
            cc_s = cc_repr_loss(xi_s, wav, x_hat)
            cc_l = cc_repr_loss(xi_l, wav, x_hat)

            lam = cfg['loss']['lambda']
            g_loss = (lam['adv']*adv + lam['fm']*fm + lam['mel']*mel +
                      lam['cc_short']*cc_s + lam['cc_long']*cc_l)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(g_params, cfg['train']['grad_clip'])
            opt_g.step()
            
            # Log metrics
            metrics['d_loss'].append(d_loss.item())
            metrics['g_loss'].append(g_loss.item())
            metrics['adv_loss'].append(adv.item())
            metrics['fm_loss'].append(fm.item())
            metrics['mel_loss'].append(mel.item())
            metrics['cc_short_loss'].append(cc_s.item())
            metrics['cc_long_loss'].append(cc_l.item())

            if step % cfg['train']['log_every'] == 0:
                print(f"step {step}/{total} | D {d_loss:.4f} | G {g_loss:.4f} (adv {adv:.4f} fm {fm:.4f} mel {mel:.4f} cc_s {cc_s:.4f} cc_l {cc_l:.4f})")

            if step % cfg['train']['ckpt_every'] == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({'dec': dec.state_dict(), 'xi_s': xi_s.state_dict(),
                            'xi_l': xi_l.state_dict()},
                           f"checkpoints/codec_step{step}.pt")
                
                # Generate visualizations at checkpoints
                if first_sample_wavs is not None:
                    with torch.no_grad():
                        recon = dec(first_sample_Cs, first_sample_Cl)
                        recon = match_length(recon, first_sample_wavs)
                    plot_waveform_comparison(first_sample_wavs, recon, sr=cfg['data']['sample_rate'])
                    plot_cc_features(first_sample_Cs, first_sample_Cl)

            if step >= total:
                break
    
    # Final visualizations
    print("\nGenerating final training plots...")
    plot_training_metrics(metrics)
    if first_sample_wavs is not None:
        with torch.no_grad():
            recon = dec(first_sample_Cs, first_sample_Cl)
            recon = match_length(recon, first_sample_wavs)
        plot_waveform_comparison(first_sample_wavs, recon, sr=cfg['data']['sample_rate'])
        plot_cc_features(first_sample_Cs, first_sample_Cl)
    
    print("Training completed!")
    print(f"Checkpoints saved in: checkpoints/")
    print(f"Plots saved in: plots/")


if __name__ == "__main__":
    main()
