# tokenizer_ms_vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VectorQuantizer2D(nn.Module):
    """
    VQ layer for 2D feature maps.

    z: (N, D, H, W)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.02):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        z: (N, D, H, W)
        returns:
          z_q_st: (N, D, H, W)
          indices: (N, H, W)
          vq_loss: scalar
        """
        n, d, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, d)  # (N*H*W, D)

        # Squared distances to codebook entries
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )  # (N*H*W, K)

        indices = distances.argmin(dim=1)  # (N*H*W,)
        z_q = self.embedding(indices).view(n, h, w, d).permute(0, 3, 1, 2).contiguous()  # (N,D,H,W)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Losses
        commitment_loss = self.commitment_cost * F.mse_loss(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        vq_loss = commitment_loss + codebook_loss

        indices_reshaped = indices.view(n, h, w)  # (N,H,W)

        # Simple anti-dead-code trick: if some codes are barely used, reinit them
        with torch.no_grad():
            usage = torch.bincount(indices, minlength=self.num_embeddings).float()
            dead_codes = (usage < 10).nonzero(as_tuple=False).view(-1)
            if dead_codes.numel() > 0:
                self.embedding.weight[dead_codes] = (
                    torch.randn_like(self.embedding.weight[dead_codes]) * 0.1
                )

        return z_q_st, indices_reshaped, vq_loss


class Encoder2D(nn.Module):
    """
    2D encoder for 64x64 images.

    Input:  x (N, C, 64, 64)
    Output: z (N, D, 16, 16)  after 2x stride-2 downsamples
    """

    def __init__(self, in_channels=3, base_channels=64, latent_dim=128, num_res_blocks=0):
        super().__init__()
        ch = base_channels
        self.net = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),

            # 32 -> 16
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU(),

            nn.Conv2d(ch * 2, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, latent_dim),
                    nn.SiLU(),
                )
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, x):
        x = self.net(x)
        for block in self.res_blocks:
            x = x + block(x)
        return x


class Decoder2D(nn.Module):
    """
    2D decoder for 64x64 images.

    Input:  z (N, D, 16, 16)
    Output: x (N, C, 64, 64)
    """

    def __init__(self, out_channels=3, base_channels=64, latent_dim=128, num_res_blocks=0):
        super().__init__()
        ch = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU(),

            # 16 -> 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch * 2, ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),

            # 32 -> 64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, latent_dim),
                    nn.SiLU(),
                )
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, z):
        for block in self.res_blocks:
            z = z + block(z)
        return self.net(z)


class MultiScaleVQTokenizerVideo2D(nn.Module):
    """
    Video wrapper around a single-scale 2D VQ-VAE.

    x: (B, C, T, 64, 64) in [-1,1]

    encode_tokens(x):
        -> tokens: list length 1, tokens[0] shape (B, T, 16, 16)
        -> vq_loss

    decode_tokens(tokens):
        -> x_recon (B, C, T, 64, 64)
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        latent_dim=128,
        num_embeddings=256,
        commitment_cost=0.02,
        num_res_blocks=0,
    ):
        super().__init__()
        self.n_scales = 1
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.encoder = Encoder2D(in_channels, base_channels, latent_dim, num_res_blocks=num_res_blocks)
        self.decoder = Decoder2D(in_channels, base_channels, latent_dim, num_res_blocks=num_res_blocks)
        self.vq = VectorQuantizer2D(num_embeddings, latent_dim, commitment_cost)

    def encode_tokens(self, x: torch.Tensor):
        """
        x: (B, C, T, 64, 64)
        returns:
          tokens[0]: (B, T, 16, 16)
          vq_loss: scalar
          z_q: (B, T, D, 16, 16) quantized latents (straight-through)
        """
        b, c, t, h, w = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)  # (B*T,C,H,W)

        z = self.encoder(x2d)  # (B*T,D,16,16)
        z_q_st, indices, vq_loss = self.vq(z)  # indices: (B*T,16,16)

        tokens = indices.view(b, t, 16, 16).contiguous()  # (B,T,16,16)
        z_q = z_q_st.view(b, t, self.latent_dim, 16, 16).contiguous()  # (B,T,D,16,16)
        return [tokens], vq_loss, z_q

    def decode_latents(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        z_q: (B, T, D, 16, 16) quantized latents
        returns: (B, C, T, 64, 64)
        """
        b, t, d, h_p, w_p = z_q.shape
        z_q2d = z_q.view(b * t, d, h_p, w_p)  # (B*T,D,16,16)
        x2d = self.decoder(z_q2d)  # (B*T,C,64,64)
        _, c, h, w = x2d.shape
        x = x2d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        return x

    def decode_tokens(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        tokens: list length 1, tokens[0]: (B, T, 16, 16)
        returns: (B, C, T, 64, 64)
        """
        assert len(tokens) == 1
        indices = tokens[0]  # (B,T,16,16)
        b, t, h_p, w_p = indices.shape
        d = self.latent_dim

        z_q = self.vq.embedding(indices.view(-1)).view(b, t, h_p, w_p, d)
        z_q = z_q.permute(0, 1, 4, 2, 3).contiguous()  # (B,T,D,16,16)
        return self.decode_latents(z_q)


MultiScaleVQTokenizer3D = MultiScaleVQTokenizerVideo2D
