from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import PromptSRConfig


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def window_partition(x: torch.Tensor, window_size: int):
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    hp, wp = x.shape[-2:]
    # x: [B, C, Hp, Wp]
    # -> [B, C, nH, Ws, nW, Ws]
    #    B: batch, C: channel, nH/nW: window 网格数, Ws: window_size
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    # -> [B, nH, nW, Ws, Ws, C]
    # -> [B*nH*nW, Ws*Ws, C]
    #    每个 window 拉平成 token 序列，token 数 = Ws*Ws
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, c)
    return x, hp, wp


def window_reverse(windows: torch.Tensor, window_size: int, b: int, c: int, hp: int, wp: int, h: int, w: int):
    # windows: [B*nH*nW, Ws*Ws, C]
    # -> [B, nH, nW, Ws, Ws, C]
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
    # -> [B, C, nH, Ws, nW, Ws]
    # -> [B, C, Hp, Wp]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, hp, wp)
    return x[:, :, :h, :w]


class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        windows, hp, wp = window_partition(x, self.window_size)
        n_windows = windows.shape[0]
        # windows: [B*nH*nW, T, C], T = Ws*Ws
        # q: [B*nH*nW, heads, T, head_dim]
        q = self.q(windows).reshape(n_windows, windows.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # kv linear 后: [B*nH*nW, T, 2*C]
        # -> [B*nH*nW, T, 2, heads, head_dim]
        # -> [2, B*nH*nW, heads, T, head_dim]
        kv = self.kv(windows).reshape(n_windows, windows.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # k/v: [B*nH*nW, heads, T, head_dim]
        # attn: [B*nH*nW, heads, T, T]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # (attn @ v): [B*nH*nW, heads, T, head_dim]
        # -> [B*nH*nW, T, heads, head_dim]
        # -> [B*nH*nW, T, C]
        out = (attn @ v).transpose(1, 2).reshape(n_windows, windows.shape[1], self.dim)
        out = self.proj(out)
        out = window_reverse(out, self.window_size, b, c, hp, wp, h, w)
        return out


def index_reverse(index: torch.Tensor) -> torch.Tensor:
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def feature_shuffle(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    dim = index.dim()
    assert x.shape[:dim] == index.shape, f"x ({x.shape}) and index ({index.shape}) shape incompatible"
    for _ in range(x.dim() - index.dim()):
        # 例如 index: [B, N]，x: [B, N, C]
        # 这里把 index 扩成 [B, N, 1]，后面再 expand 到 [B, N, C]
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


class CategorySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, category_size: int = 128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, similarity_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        # x: [B, C, H, W]
        # flatten(2): [B, C, N]
        # transpose(1, 2): [B, N, C]
        #   N = H*W，每个空间位置变成一个 token
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        # qkv: [B, N, 3C]
        # 这里只是复用 token，拼成 ATD 风格的 q/k/v 输入布局
        qkv = torch.cat([tokens, tokens, tokens], dim=-1)  # mimic ATD AC_MSA input

        if similarity_map.shape[1] <= similarity_map.shape[2]:
            tk_id = similarity_map.argmax(dim=1)  # [B, N]
        else:
            tk_id = similarity_map.argmax(dim=2)  # [B, N]

        gs = min(n, self.category_size)
        ng = (n + gs - 1) // gs

        _, sort_idx = torch.sort(tk_id, dim=-1, stable=False)
        inv_idx = index_reverse(sort_idx)
        # qkv: [B, N, 3C]
        # sort_idx: [B, N]
        # shuffled_qkv: [B, N, 3C]，但 token 顺序已按类别重排
        shuffled_qkv = feature_shuffle(qkv, sort_idx)
        pad_n = ng * gs - n
        if pad_n > 0:
            # pad_tail: [B, pad_n, 3C]
            # 补到 ng*gs，便于后面整组 attention
            pad_tail = torch.flip(shuffled_qkv[:, n - pad_n:n, :], dims=[1])
            shuffled_qkv = torch.cat((shuffled_qkv, pad_tail), dim=1)

        # shuffled_qkv: [B, ng*gs, 3C]
        # -> [B, ng, gs, 3, heads, head_dim]
        #    ng: group 数, gs: 每组 token 数
        # -> [3, B, ng, heads, gs, head_dim]
        y = shuffled_qkv.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = y[0], y[1], y[2]
        # q/k/v: [B, ng, heads, gs, head_dim]
        # attn: [B, ng, heads, gs, gs]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # (attn @ v): [B, ng, heads, gs, head_dim]
        # -> [B, ng, gs, heads, head_dim]
        # -> [B, ng*gs, C]
        # -> [:, :N, :] 去掉 padding token
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]
        # 用逆置换把 token 顺序恢复回原始空间顺序: [B, N, C]
        out = feature_shuffle(out, inv_idx)
        out = self.proj(out)
        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return 0.05 * out


class GlobalAnchorPromptingLayer(nn.Module):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        dim = cfg.embed_dim
        self.cfg = cfg
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.anchor_proj = nn.Conv2d(dim, dim, 1)
        self.kp_proj = nn.Conv1d(dim, dim, 1)
        self.vp_proj = nn.Conv1d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, prev_prompt: torch.Tensor | None = None):
        b, c, h, w = x.shape
        # x: [B, C, H, W] -> [B, H*W, C]
        q = self.q_proj(x).flatten(2).transpose(1, 2)
        k = self.k_proj(x).flatten(2).transpose(1, 2)
        v = self.v_proj(x).flatten(2).transpose(1, 2)

        pooled_h = max(1, h // self.cfg.anchor_downsample)
        pooled_w = max(1, w // self.cfg.anchor_downsample)
        pooled = F.interpolate(x, size=(pooled_h, pooled_w), mode='bilinear', align_corners=False)
        # pooled: [B, C, Hp, Wp] -> anchor token a: [B, Hp*Wp, C]
        a = self.anchor_proj(pooled).flatten(2).transpose(1, 2)
        # a: [B, Na, C], k^T: [B, C, N]
        # m_coarse: [B, Na, N]
        m_coarse = torch.matmul(a, k.transpose(1, 2)) / math.sqrt(c)
        # softmax(m_coarse): [B, Na, N]
        # v: [B, N, C]
        # p: [B, Na, C]
        p = torch.matmul(torch.softmax(m_coarse, dim=-1), v)

        if prev_prompt is not None and prev_prompt.shape == p.shape:
            alpha = self.cfg.anchor_update_alpha
            p = alpha * prev_prompt + (1.0 - alpha) * p

        # p: [B, Na, C] -> p_t: [B, C, Na]
        p_t = p.transpose(1, 2)
        # Conv1d 后再转回 token 形式: [B, Na, C]
        kp = self.kp_proj(p_t).transpose(1, 2)
        vp = self.vp_proj(p_t).transpose(1, 2)

        # q: [B, N, C]
        # (kp + a)^T: [B, C, Na]
        # m_fine: [B, N, Na]
        m_fine = torch.matmul(q, (kp + a).transpose(1, 2)) / math.sqrt(c)
        # softmax(m_fine): [B, N, Na]
        # vp: [B, Na, C]
        # x_prompted: [B, N, C]
        x_prompted = torch.matmul(torch.softmax(m_fine, dim=-1), vp)
        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        x_prompted = x_prompted.transpose(1, 2).reshape(b, c, h, w)
        x_prompted = self.out_proj(x_prompted)
        return x_prompted, p, m_coarse, m_fine


class LocalPromptingLayer(nn.Module):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        dim = cfg.embed_dim
        hidden_dim = max(dim, int(dim * cfg.mlp_ratio))
        self.wsa = WindowSelfAttention(dim, cfg.window_size, cfg.num_heads)
        self.csa = CategorySelfAttention(dim, cfg.num_heads, cfg.num_categories)
        self.norm = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor, similarity_map: torch.Tensor) -> torch.Tensor:
        x = 0.5 * self.wsa(x) + self.csa(x, similarity_map) + x
        x = 0.1 * self.ffn(self.norm(x)) + x
        return x


class CascadePromptingBlock(nn.Module):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        dim = cfg.embed_dim
        hidden_dim = max(dim, int(dim * cfg.mlp_ratio))
        self.gapl = GlobalAnchorPromptingLayer(cfg)
        self.gapl_norm = LayerNorm2d(dim)
        self.gapl_ffn = FeedForward(dim, hidden_dim)
        self.coarse_lpl = LocalPromptingLayer(cfg)
        self.fine_lpl = LocalPromptingLayer(cfg)

    def forward(self, x: torch.Tensor, prev_prompt: torch.Tensor | None = None):
        x_prompted, prompt, m_coarse, m_fine = self.gapl(x, prev_prompt=prev_prompt)
        x_prompted = x + 0.5 * x_prompted
        x_prompted = 0.1 * self.gapl_ffn(self.gapl_norm(x_prompted)) + x_prompted
        x_coarse = self.coarse_lpl(x_prompted, m_coarse)
        x_fine = self.fine_lpl(x_coarse, m_fine)
        return x_fine, prompt


class ResidualGroup(nn.Module):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        self.blocks = nn.ModuleList([CascadePromptingBlock(cfg) for _ in range(cfg.num_cpbs_per_group)])
        self.conv = nn.Conv2d(cfg.embed_dim, cfg.embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        prompt = None
        for block in self.blocks:
            x, prompt = block(x, prev_prompt=prompt)
        x = self.conv(x)
        return residual + 0.1 * x


class PromptSRModel(nn.Module):
    def __init__(self, cfg: PromptSRConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.head = nn.Conv2d(cfg.in_channels, dim, 3, padding=1)
        self.groups = nn.ModuleList([ResidualGroup(cfg) for _ in range(cfg.num_residual_groups)])
        self.body_tail = nn.Conv2d(dim, dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, cfg.in_channels * (cfg.scale ** 2), 3, padding=1),
            nn.PixelShuffle(cfg.scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.head(x)
        feat = x0
        for group in self.groups:
            feat = group(feat)
        x_df = self.body_tail(feat)
        out = self.upsample(x_df + x0)
        return out
