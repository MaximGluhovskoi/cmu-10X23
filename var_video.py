# var_video.py
import torch
import torch.nn as nn
from dataclasses import dataclass


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, L, D)
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,L,Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask
        attn = attn_logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B,H,L,Hd)
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class VARConfig:
    """
    Config for frame-conditioned VAR on 16x16 latent tokens.
    """
    codebook_size: int = 256   # <-- must match tokenizer num_embeddings
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_h: int = 16
    max_w: int = 16
    action_dim: int = 11
    dropout: float = 0.1


class FrameConditionedVAR(nn.Module):
    """
    Predicts next-frame tokens from previous-frame tokens + action:

        p(F_t | F_{t-1}, a_{t-1})

    prev_tokens:   (B, H, W) long
    target_tokens: (B, H, W) long
    actions:       (B, A) float
    """

    def __init__(self, cfg: VARConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.codebook_size, cfg.d_model)
        self.row_embed = nn.Embedding(cfg.max_h, cfg.d_model)
        self.col_embed = nn.Embedding(cfg.max_w, cfg.d_model)

        self.query_token = nn.Parameter(torch.randn(cfg.d_model))
        self.action_token = nn.Parameter(torch.randn(cfg.d_model))
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.codebook_size)

    def _positional_embeddings(self, b, h, w, device):
        row_ids = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
        col_ids = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
        row_ids = row_ids.reshape(b, -1)
        col_ids = col_ids.reshape(b, -1)
        return row_ids, col_ids

    def forward(self, prev_tokens, target_tokens, actions):
        """
        prev_tokens:   (B, H, W)
        target_tokens: (B, H, W)
        actions:       (B, A)

        returns logits: (B, H, W, V)
        """
        device = prev_tokens.device
        b, h, w = prev_tokens.shape
        cfg = self.cfg

        L = h * w

        prev_ids = prev_tokens.view(b, L)  # (B,L)
        row_ids, col_ids = self._positional_embeddings(b, h, w, device)

        prev_emb = self.token_embed(prev_ids)
        prev_emb = prev_emb + self.row_embed(row_ids.clamp(max=cfg.max_h - 1)) \
                             + self.col_embed(col_ids.clamp(max=cfg.max_w - 1))

        query_vec = self.query_token.view(1, 1, -1)
        tgt_emb = query_vec.expand(b, L, -1)
        tgt_emb = tgt_emb + self.row_embed(row_ids.clamp(max=cfg.max_h - 1)) \
                           + self.col_embed(col_ids.clamp(max=cfg.max_w - 1))

        act_emb = self.action_proj(actions) + self.action_token  # (B,D)
        act_emb = act_emb.unsqueeze(1)  # (B,1,D)

        x = torch.cat([prev_emb, tgt_emb, act_emb], dim=1)  # (B,2L+1,D)

        attn_mask = None

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        tgt_out = x[:, L:2 * L, :]  # (B,L,D)
        logits = self.head(tgt_out)  # (B,L,V)
        logits = logits.view(b, h, w, cfg.codebook_size)
        return logits
