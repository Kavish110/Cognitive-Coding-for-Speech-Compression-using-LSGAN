import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MRFFBlock(nn.Module):
    def __init__(self, ch, kernels=(3,7,11)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(ch, ch, k, padding=0) for k in kernels])
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        outs = []
        for conv in self.branches:
            k = conv.kernel_size[0]
            # left pad to keep causality
            x_pad = nn.functional.pad(x, (k-1, 0))
            outs.append(conv(x_pad))
        y = sum(outs) / len(outs)
        return self.act(y)

class UpStage(nn.Module):
    def __init__(self, in_ch, k_list, up_factors, start_ch, mrff_k=(3,7,11), mrff_repeat=3, out_ch=1):
        """
        Generic upsampling stage that applies a pointwise conv then a sequence of
        ConvTranspose1d blocks followed by MRFF blocks. By default the stage
        outputs `out_ch` channels.
        """
        super().__init__()
        ch = start_ch
        self.first = nn.Conv1d(in_ch, ch, 3, padding=0)
        self.first_pad = 2  # (3-1)
        blocks = []
        for k, u in zip(k_list, up_factors):
            blocks += [nn.ConvTranspose1d(ch, ch//2, k, stride=u, padding=0), nn.LeakyReLU(0.2, True)]
            ch = ch//2
            for _ in range(mrff_repeat):
                blocks += [MRFFBlock(ch, mrff_k)]
        self.net = nn.Sequential(*blocks)
        self.final = nn.Conv1d(ch, out_ch, 7, padding=0)
        self.final_pad = 6  # (7-1)

    def forward(self, x):
        x = nn.functional.pad(x, (self.first_pad, 0))
        x = self.first(x)
        x = self.net(x)
        # final conv with left-only pad to keep causality
        x = nn.functional.pad(x, (self.final_pad, 0))
        x = self.final(x)
        return x

class CognitiveDecoder(nn.Module):
    """
    Stage 1: upsample long-term Cl to short-term grid
    Stage 2: fuse with short-term Cs (projected to single channel), then upsample to waveform
    """
    def __init__(self, top_cfg, low_cfg):
        super().__init__()
        # up_long will normally produce a short rate representation
        self.up_long = UpStage(
            in_ch=64,
            k_list=top_cfg['deconv_kernel'],
            up_factors=top_cfg['upsample'],
            start_ch=top_cfg['start_channels'],
            mrff_k=tuple(top_cfg['mrff_k']),
            mrff_repeat=top_cfg['mrff_repeat'],
            out_ch=1   # produce single channel to match cs_proj output
        )
        # After top upsampling, concat with Cs along channel dim
        # in_ch = 1 (up_long) + 1 (cs_proj)
        self.up_fuse = UpStage(
            in_ch=2,
            k_list=low_cfg['deconv_kernel'],
            up_factors=low_cfg['upsample'],
            start_ch=low_cfg['start_channels'],
            mrff_k=tuple(low_cfg['mrff_k']),
            mrff_repeat=low_cfg['mrff_repeat'],
            out_ch=1
        )
        # project Cs from 64 to 1 channel prior to fusion
        self.cs_proj = nn.Conv1d(64, 1, 1)

    def _time_align(self, src: torch.Tensor, tgt_length: int) -> torch.Tensor:
        """
        Ensure src has time dimension equal to tgt_length.
        If lengths differ, use linear interpolation along time axis.
        src shape: [B, C, T_src]
        Returns: [B, C, tgt_length]
        """
        if src.size(-1) == tgt_length:
            return src
        # F.interpolate expects shape [B, C, L]
        return F.interpolate(src, size=tgt_length, mode='linear', align_corners=False)

    def forward(self, Cs, Cl):
        # Cs, Cl: [B, T_frames, 64] -> [B, 64, T_frames]
        Cl = Cl.transpose(1, 2)
        Cs = Cs.transpose(1, 2)

        # upsample long to short grid. Expect up_long to output [B, 1, T_short]
        up_l = self.up_long(Cl)                   # [B, 1, T_up]
        cs_1 = self.cs_proj(Cs)                   # [B, 1, T_short]

        # time align up_l to cs_1 if needed
        target_T = cs_1.size(-1)
        if up_l.size(-1) != target_T:
            up_l = self._time_align(up_l, target_T)

        fused = torch.cat([up_l, cs_1], dim=1)    # [B, 2, T_short]
        wav = self.up_fuse(fused)                 # [B, 1, T_wav]
        return wav
def match_length(gen, real):
    """
    gen: [B,1,T_gen]
    real: [B,1,T_real]
    returns gen resized to T_real
    """
    T_gen = gen.size(-1)
    T_real = real.size(-1)

    if T_gen > T_real:
        return gen[..., :T_real]          # crop
    elif T_gen < T_real:
        pad = T_real - T_gen
        return nn.functional.pad(gen, (0, pad))  # pad right
    else:
        return gen
