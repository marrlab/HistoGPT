""" 
PyTorch Flamingo Model
Author: Lucidrain / Deepmind
"""

import torch
import torch.nn.functional as F

from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)


class FeedForward(nn.Module):
    """
    Positionwise Feedforward Layer
    """
    def __init__(self, dim, mul=4):
        """
        :param dim: input / output dimension
        :param mul: hidden layer multiplier
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mul, bias=False),
            nn.GELU(),
            nn.Linear(dim * mul, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    """
    Multi-Headed Perceiver Latent-Attention
    """
    def __init__(self, dim_ltn, dim_head=64, num_heads=8):
        """
        :param: dim_ltn: dimension of latent space
        :param: dim_head: dimension of each head
        :param: num_heads: number of attention heads
        """
        super().__init__()

        dim_emb = dim_head * num_heads

        self.scale = dim_head**-0.5
        self.num_heads = num_heads

        self.norm_src = nn.LayerNorm(dim_ltn)
        self.norm_ltn = nn.LayerNorm(dim_ltn)

        self.to_q = nn.Linear(dim_ltn, dim_emb, bias=False)
        self.to_kv = nn.Linear(dim_ltn, dim_emb * 2, bias=False)
        self.to_out = nn.Linear(dim_emb, dim_ltn, bias=False)

    def forward(self, src, ltn):
        """
        :param src: input array of shape ['batch', 'time', 'seq', 'dim']
        :param ltn: learned latent vector of lower dimensional size
        """
        h = self.num_heads

        src = self.norm_src(src)
        ltn = self.norm_ltn(ltn)

        q = self.to_q(ltn)
        context = torch.cat((src, ltn), dim=-2)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)

        attn = einsum("... i d, ... j d  -> ... i j", q, k) * self.scale
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out)

        return out


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler Network
    """
    def __init__(
        self, dim_ltn, dim_head, num_heads, num_ltn, num_emb, num_layers,
        ffw_mul
    ):
        """
        :param: dim_ltn: dimension of latent space
        :param: dim_head: dimension of each head
        :param: num_heads: number of attention heads
        :param: num_ltn: number of learned latent variables
        :param: num_emb: time steps in the feature space
        :param: num_layers: depth of the perceiver resampler
        :param: ffw_mul: hidden layer multiplier
        """
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_ltn, dim_ltn))
        self.pos_emb = nn.Parameter(torch.randn(num_emb, 1, dim_ltn))

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim_ltn)

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim_ltn, dim_head, num_heads),
                        FeedForward(dim_ltn, ffw_mul),
                    ]
                )
            )

    def forward(self, src):
        """
        :param src: input array of shape ['batch', 'time', 'seq', 'dim']
        :return out: fixed number of output tokens
        """
        if src.ndim == 3:
            src = rearrange(src, "b n d -> b 1 n d")

        b, m = src.shape[0], src.shape[1]
        src = src + self.pos_emb[:m]

        ltn = repeat(self.latents, "n d -> b m n d", b=b, m=m)

        for mha, ffw in self.layers:
            ltn = mha(src, ltn) + ltn
            ltn = ffw(ltn) + ltn

        return self.norm(ltn)


class MaskedCrossAttention(nn.Module):
    """
    Multi-Headed Masked Cross-Attention
    """
    def __init__(self, dim_ltn, dim_head, num_heads, only_attend_to_immediate):
        super().__init__()

        dim_emb = dim_head * num_heads

        self.heads = num_heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim_ltn, dim_emb, bias=False)
        self.to_kv = nn.Linear(dim_ltn, dim_emb * 2, bias=False)
        self.to_out = nn.Linear(dim_emb, dim_ltn, bias=False)

        self.norm = nn.LayerNorm(dim_ltn)
        self.only_attend_to_immediate = only_attend_to_immediate

    def forward(self, text, image, locations=None):
        """
        :param text: input text features (y)
        :param image: input visual features (x)
        """
        h = self.heads
        _, t, m = image.shape[:3]

        text = self.norm(text)
        image = rearrange(image, "b t n d -> b (t n) d")

        q = self.to_q(text)
        k, v = self.to_kv(image).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        attn = einsum("... i d, ... j d -> ... i j", q, k) * self.scale

        if exists(locations):

            txt_time = locations.cumsum(dim=-1)
            viz_time = torch.arange(t, device=image.device) + 1

            mask_op = torch.eq if self.only_attend_to_immediate else torch.ge

            txt_to_viz_mask = mask_op(
                rearrange(txt_time, "b i -> b 1 i 1"),
                repeat(viz_time, "j -> 1 1 1 (j m)", m=m),
            )
            attn = attn.masked_fill(
                ~txt_to_viz_mask, -torch.finfo(attn.dtype).max
            )

        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)

        if exists(locations) and self.only_attend_to_immediate:

            txt_without_viz_mask = txt_time == 0
            txt_without_viz_mask = rearrange(
                txt_without_viz_mask, "b i -> b 1 i 1"
            )

            attn = attn.masked_fill(txt_without_viz_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class GatedCrossAttentionBlock(nn.Module):
    """
    Gated Cross-Attention Dense Block (GATED XATTN-DENSE)
    Flamingo: A Visual text Model for Few-Shot Learning
    """
    def __init__(
        self, dim_ltn, dim_head, num_heads, ffw_mul, only_attend_to_immediate
    ):
        super().__init__()

        self.mha = MaskedCrossAttention(
            dim_ltn, dim_head, num_heads, only_attend_to_immediate
        )
        self.mha_gate = nn.Parameter(torch.tensor([0.0]))

        self.ffw = FeedForward(dim_ltn, ffw_mul)
        self.ffw_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, text, image, locations=None):
        text = (self.mha(text, image, locations) * self.mha_gate.tanh() + text)
        text = self.ffw(text) * self.ffw_gate.tanh() + text
        return text
