import torch
import torch.nn.functional as F
import torchaudio

def lsgan_d(real_logits, fake_logits):
    """Compute LSGAN discriminator loss. Returns tensor for autograd."""
    real_list = real_logits if isinstance(real_logits, (list, tuple)) else [real_logits]
    fake_list = fake_logits if isinstance(fake_logits, (list, tuple)) else [fake_logits]
    loss = None
    for r in real_list:
        term = ((r - 1)**2).mean()
        loss = term if loss is None else loss + term
    for f in fake_list:
        term = (f**2).mean()
        loss = term if loss is None else loss + term
    return loss if loss is not None else torch.tensor(0., dtype=torch.float32)

def lsgan_g(fake_logits):
    """Compute LSGAN generator loss. Returns tensor for autograd."""
    fake_list = fake_logits if isinstance(fake_logits, (list, tuple)) else [fake_logits]
    loss = None
    for f in fake_list:
        term = ((f - 1)**2).mean()
        loss = term if loss is None else loss + term
    return loss if loss is not None else torch.tensor(0., dtype=torch.float32)

def feature_matching_loss(real_feats, fake_feats):
    """Match features with automatic length alignment."""
    loss = None
    for r_scale, f_scale in zip(real_feats, fake_feats):
        for r, f in zip(r_scale, f_scale):
            # Handle potential length mismatches
            if r.size(-1) != f.size(-1):
                # Interpolate f to match r's length
                f = F.interpolate(f.unsqueeze(0), size=r.size(-1), mode='linear', align_corners=False).squeeze(0)
            term = F.l1_loss(f, r)
            loss = term if loss is None else loss + term
    return loss if loss is not None else torch.tensor(0., dtype=torch.float32)

def mel_spectrogram(x, sr, n_fft, hop_length, win_length, n_mels):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, center=True, power=1.0)
    return mel(x)

def mel_loss(x, x_hat, sr, cfg):
    """Compute mel spectrogram loss with automatic length alignment."""
    mel_x = mel_spectrogram(x, sr, **cfg)
    mel_y = mel_spectrogram(x_hat, sr, **cfg)
    
    # Handle potential length mismatches in time dimension
    if mel_y.size(-1) != mel_x.size(-1):
        mel_y = F.interpolate(mel_y, size=mel_x.size(-1), mode='linear', align_corners=False)
    
    return F.l1_loss(mel_y, mel_x)

def cc_repr_loss(xi_func, x_real, x_fake):
    """Compute CC representation loss with automatic length alignment."""
    z_r = xi_func(x_real)
    z_f = xi_func(x_fake)
    
    # Handle potential length mismatches in time dimension
    if z_f.size(1) != z_r.size(1):
        z_f = F.interpolate(z_f.transpose(1, 2), size=z_r.size(1), mode='linear', align_corners=False).transpose(1, 2)
    
    return F.l1_loss(z_f, z_r)

