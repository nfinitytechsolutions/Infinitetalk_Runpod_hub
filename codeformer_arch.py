"""Vendored CodeFormer architecture — eliminates the basicsr build dependency.

Only the inference-path classes are included. Original source:
  https://github.com/sczhou/CodeFormer (MIT License)

Dependencies: torch only (no basicsr, no compiled CUDA extensions).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VQGAN components (from basicsr.archs.vqgan_arch)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)
        d = (
            z_flattened.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, min_encoding_indices


def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (c ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        curr_res = resolution

        blocks = [nn.Conv2d(in_channels, nf, 3, 1, 1)]

        in_ch_mult = (1,) + tuple(ch_mult)
        for i in range(self.num_resolutions):
            block_in = nf * in_ch_mult[i]
            block_out = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in))
                curr_res //= 2

        blocks += [ResnetBlock(block_in, block_in), AttnBlock(block_in), ResnetBlock(block_in, block_in)]
        blocks += [normalize(block_in), nn.SiLU(), nn.Conv2d(block_in, emb_dim, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = nf * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))

        blocks = [nn.Conv2d(emb_dim, block_in, 3, 1, 1)]
        blocks += [ResnetBlock(block_in, block_in), AttnBlock(block_in), ResnetBlock(block_in, block_in)]

        for i in reversed(range(self.num_resolutions)):
            block_out = nf * ch_mult[i]
            for _ in range(self.num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if i != 0:
                blocks.append(Upsample(block_in))
                curr_res *= 2

        blocks += [normalize(block_in), nn.SiLU(), nn.Conv2d(block_in, out_channels, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# CodeFormer Transformer
# ---------------------------------------------------------------------------

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if query_pos is not None else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


def _get_pos_embed(embed_dim, pos_len):
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    pos = torch.arange(pos_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(pos), torch.cos(pos)], dim=1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class CodeFormer(nn.Module):
    """CodeFormer face restoration network.

    Args:
        dim_embd: Embedding dimension (default 512).
        codebook_size: Number of codebook entries (default 1024).
        n_head: Transformer attention heads (default 8).
        n_layers: Transformer layers (default 9).
        connect_list: Feature map resolutions to connect (default ['32','64','128','256']).
    """

    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=("32", "64", "128", "256"),
        fix_modules=("quantize", "generator"),
    ):
        super().__init__()

        in_channels = 3
        nf = 64
        emb_dim = 256
        ch_mult = [1, 2, 2, 4, 4, 8]
        num_res_blocks = 2
        resolution = 512
        attn_resolutions = [16]

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Embedding(latent_size, dim_embd)

        self.feat_emb = nn.Linear(emb_dim, dim_embd)

        # Transformer
        self.ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(n_layers)
        ])

        # logits predict heads
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            "16": 512,
            "32": 256,
            "64": 256,
            "128": 128,
            "256": 64,
            "512": 64,
        }

        # Fidelity adaptation layers
        self.fuse_encoder_block = nn.ModuleDict()
        self.fuse_generator_block = nn.ModuleDict()
        for f_size in connect_list:
            in_ch = self.channels[f_size]
            self.fuse_encoder_block[f_size] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(in_ch, in_ch, 3, 1, 1)
            )
            self.fuse_generator_block[f_size] = nn.Sequential(
                nn.Conv2d(in_ch * 2, in_ch, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(in_ch, in_ch, 3, 1, 1)
            )

        # VQGAN
        self.encoder = Encoder(
            in_channels=in_channels, nf=nf, emb_dim=emb_dim, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, resolution=resolution, attn_resolutions=attn_resolutions,
        )
        self.decoder = Decoder(
            out_channels=in_channels, nf=nf, emb_dim=emb_dim, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, resolution=resolution, attn_resolutions=attn_resolutions,
        )
        self.quantize = VectorQuantizer(codebook_size, emb_dim, beta=0.25)

        # Fix encoder + quantizer (only train transformer + fuse layers)
        if fix_modules is not None:
            for module_name in fix_modules:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0.0, detach_16=True, code_only=False, adain=False):
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 512, 512), normalized to [-1, 1].
            w: Fidelity weight (0 = full codebook, 1 = full encoder features).
            adain: Use adaptive instance normalization for blending.

        Returns:
            (output, codebook_loss, quant_indices) or just (output,) depending on code_only.
        """
        # Encoder forward — collect intermediate features
        enc_feat_dict = {}
        out = x
        for i, block in enumerate(self.encoder.blocks):
            out = block(out)
            if isinstance(block, ResnetBlock):
                # Check spatial resolution
                h = out.shape[2]
                f_size = str(h)
                if f_size in self.connect_list:
                    enc_feat_dict[f_size] = out.clone()

        # Quantize
        lq_feat = out
        quant_feat, codebook_loss, quant_indices = self.quantize(lq_feat)

        if code_only:
            return quant_indices, codebook_loss

        # Transformer prediction
        pos_emb = self.position_emb.weight.unsqueeze(1).repeat(1, x.shape[0], 1)  # (HW, B, dim)

        # Reshape encoder features for transformer
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))  # (HW, B, dim)

        # Transformer forward
        query_emb = feat_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # Predict codebook indices
        logits = self.idx_pred_layer(query_emb)  # (HW, B, codebook_size)
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)

        quant_feat = self.quantize.embedding(top_idx.squeeze(2))  # (HW, B, emb_dim)
        # Reshape back to spatial
        B = x.shape[0]
        h = w_ = int(math.sqrt(quant_feat.shape[0]))
        quant_feat = quant_feat.permute(1, 2, 0).view(B, -1, h, w_)

        if detach_16:
            quant_feat = quant_feat.detach()

        # Decoder forward with fidelity blending
        out = quant_feat
        fuse_list = [str(2**i) for i in range(int(math.log2(512)), 3, -1)]  # ['256','128','64','32','16']

        for i, block in enumerate(self.decoder.blocks):
            out = block(out)
            if isinstance(block, ResnetBlock):
                h = out.shape[2]
                f_size = str(h)
                if f_size in self.connect_list and f_size in enc_feat_dict:
                    enc_feat = self.fuse_encoder_block[f_size](enc_feat_dict[f_size])
                    if adain:
                        out = self._adaptive_instance_normalization(out, enc_feat, w)
                    else:
                        out = out * (1 - w) + enc_feat * w
                    out = self.fuse_generator_block[f_size](torch.cat([enc_feat, out], dim=1))

        return out, codebook_loss, quant_indices

    @staticmethod
    def _adaptive_instance_normalization(feat, enc_feat, w):
        """AdaIN blending between decoder features and encoder features."""
        size = feat.size()
        feat_mean, feat_std = _calc_mean_std(feat)
        enc_mean, enc_std = _calc_mean_std(enc_feat)
        # Blend statistics
        target_mean = (1 - w) * feat_mean + w * enc_mean
        target_std = (1 - w) * feat_std + w * enc_std
        # Normalize and apply blended statistics
        normalized = (feat - feat_mean.expand(size)) / (feat_std.expand(size) + 1e-6)
        return normalized * target_std.expand(size) + target_mean.expand(size)


def _calc_mean_std(feat, eps=1e-5):
    """Calculate channel-wise mean and std."""
    size = feat.size()
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std
