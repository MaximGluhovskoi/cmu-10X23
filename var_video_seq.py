# var_video_seq.py
import torch
import torch.nn as nn
from dataclasses import dataclass


def _causal_mask(sz_history: int, sz_query: int, device):
    """
    Build a mask where history tokens can attend to history+queries,
    but queries cannot influence history (block query -> history backflow).
    Shape: (total, total)
    """
    total = sz_history + sz_query
    mask = torch.zeros(total, total, device=device)
    # prevent queries attending to future queries (causal within query block)
    # and prevent history attending to queries (one-way flow)
    # block history <- queries
    mask[:sz_history, sz_history:] = float("-inf")
    # block query -> future queries (upper triangle within query block)
    q_block = torch.triu(torch.ones(sz_query, sz_query, device=device), diagonal=1) * float("-inf")
    mask[sz_history:, sz_history:] = q_block
    return mask


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class SequenceVARConfig:
    codebook_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    max_h: int = 16
    max_w: int = 16
    max_t: int = 32
    action_dim: int = 11
    dropout: float = 0.1


class SequenceConditionedVAR(nn.Module):
    """
    Predicts next-frame tokens conditioned on a window of past frames:
        p(F_t | F_{0:t-1}, a_{0:t-1})

    Inputs:
      tokens_hist: (B, T, H, W) long
      actions_hist: (B, T, A) float   # actions aligned per frame (use last for next)
    Output:
      logits_next: (B, H, W, V)
    """

    def __init__(self, cfg: SequenceVARConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.codebook_size, cfg.d_model)
        self.row_embed = nn.Embedding(cfg.max_h, cfg.d_model)
        self.col_embed = nn.Embedding(cfg.max_w, cfg.d_model)
        self.time_embed = nn.Embedding(cfg.max_t, cfg.d_model)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.codebook_size)

    def forward(self, tokens_hist, actions_hist):
        """
        tokens_hist: (B, T, H, W) longs
        actions_hist: (B, T, A) floats (action per frame; last action used for next frame)
        """
        device = tokens_hist.device
        b, t, h, w = tokens_hist.shape
        cfg = self.cfg
        assert t <= cfg.max_t, "Sequence length exceeds max_t"

        # embeddings for history tokens
        L_hist = t * h * w
        row_ids = torch.arange(h, device=device).view(1, 1, h, 1).expand(b, t, h, w)
        col_ids = torch.arange(w, device=device).view(1, 1, 1, w).expand(b, t, h, w)
        time_ids = torch.arange(t, device=device).view(1, t, 1, 1).expand(b, t, h, w)

        tok_flat = tokens_hist.view(b, L_hist)
        row_flat = row_ids.view(b, L_hist)
        col_flat = col_ids.view(b, L_hist)
        time_flat = time_ids.view(b, L_hist)

        tok_emb = self.token_embed(tok_flat)
        tok_emb = tok_emb + self.row_embed(row_flat.clamp(max=cfg.max_h - 1)) \
                           + self.col_embed(col_flat.clamp(max=cfg.max_w - 1)) \
                           + self.time_embed(time_flat.clamp(max=cfg.max_t - 1))

        # action embedding per frame applied to tokens of that frame
        act_emb = self.action_proj(actions_hist).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,D)
        act_emb = act_emb.expand(-1, -1, h, w, -1).contiguous().view(b, L_hist, cfg.d_model)
        tok_emb = tok_emb + act_emb

        # query tokens for next frame (start as zeros)
        L_query = h * w
        row_q = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w).reshape(b, L_query)
        col_q = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w).reshape(b, L_query)
        time_q = torch.full((b, L_query), min(t, cfg.max_t - 1), device=device, dtype=torch.long)
        act_q = self.action_proj(actions_hist[:, -1]).unsqueeze(1).expand(b, L_query, cfg.d_model)

        query_emb = self.row_embed(row_q.clamp(max=cfg.max_h - 1)) \
                    + self.col_embed(col_q.clamp(max=cfg.max_w - 1)) \
                    + self.time_embed(time_q) + act_q

        x = torch.cat([tok_emb, query_emb], dim=1)  # (B, L_hist+L_query, D)

        attn_mask = _causal_mask(L_hist, L_query, device)
        attn_mask = attn_mask.unsqueeze(0).expand(b, -1, -1)  # batch_first

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        query_out = x[:, L_hist:, :]  # (B, L_query, D)
        logits = self.head(query_out).view(b, h, w, cfg.codebook_size)
        return logits
