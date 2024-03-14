""" 
PyTorch BioGPT Model
Author: HuggingFace / Microsoft

PyTorch Flamingo Model
Author: Lucidrain / DeepMind

PyTorch HistoGPT Model
Author: Manuel Tran / Helmholtz Munich
"""

import json
import math
import random
from typing import Optional, NamedTuple, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, einsum
from torch.nn import CrossEntropyLoss

from einops import rearrange, repeat
from einops_exts import rearrange_many

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.biogpt import BioGptConfig

from flamingo_pytorch import PerceiverResampler, GatedCrossAttentionBlock
from flamingo_pytorch.flamingo_pytorch import FeedForward, PerceiverAttention


def exists(val):
    return val is not None


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
):
    """
    Creates a triangular attention mask used for causal language modeling.
    """
    bsz, tgt_len = input_ids_shape

    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))

    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
):
    """
    Expands the attention mask from `[bsz, len]` to `[bsz, 1, tgt_len, src_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = (
        mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    )
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min
    )


class PRConfig(NamedTuple):
    '''
    Perceiver Resampler Configuration

    Description:
    ------------
    Contains all parameters required to define a perceiver resampler model.
    '''
    dim_feats: int = 768
    dim_model: int = 1536
    dim_head: int = 96
    num_heads: int = 16
    num_layers: int = 6
    num_latents: int = 640

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


class MaskedCrossAttention(nn.Module):
    def __init__(
        self, *, dim, dim_head=64, heads=8, only_attend_immediate_media=True
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.attn = None
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(t, device=x.device) + 1

            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, 'b i -> b 1 i 1'),
                repeat(media_time, 'j -> 1 1 1 (j m)', m=m)
            )
            sim = sim.masked_fill(
                ~text_to_media_mask, -torch.finfo(sim.dtype).max
            )

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, 'b i -> b 1 i 1'
            )
            attn = attn.masked_fill(text_without_media_mask, 0.)

        self.attn = attn

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, media, media_locations=None):
        x = self.attn(x, media, media_locations=media_locations
                     ) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler Module
    
    Description:
    ------------
    A sequence dimension reduction model that takes as input a variable number 
    of latent vectors and resamples them to a fixed number of latent vectors.
    """
    def __init__(
        self,
        dim_feats: int,
        dim_model: int,
        dim_head: int,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        num_media: int = 1,
        ffwn_mult: int = 4
    ):
        super().__init__()
        self.linear = nn.Linear(dim_feats, dim_model, bias=False)
        self.media_pos = nn.Parameter(torch.randn(num_media, 1, dim_model))
        self.latents = nn.Parameter(torch.randn(num_latents, dim_model))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim_model, dim_head=dim_head, heads=num_heads
                        ),
                        FeedForward(dim=dim_model, mult=ffwn_mult)
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor):
        x = self.linear(x.float())

        x = x.unsqueeze(1) if x.ndim == 3 else x
        b, m, _, _ = x.size()

        x = x + self.media_pos[:m]
        latents = self.latents.repeat(b, m, 1, 1)

        for attn, ffwn in self.layers:
            latents = latents + attn(x, latents)
            latents = latents + ffwn(latents)

        return self.norm(latents)


class HistoGPTLearnedPositionalEmbedding(nn.Embedding):
    """
    HistoGPT Learned Positional Embeddings

    Description:
    ------------
    A linear layer that learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0
    ):
        """
        `input_ids_shape` is expected to be [bsz x seq_len]
        """
        attention_mask = attention_mask.long()

        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) *
            attention_mask
        ).long() - 1
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class HistoGPTAttention(nn.Module):
    """
    HistoGPT Scaled Dot-Product Multi-Head Attention 

    Description:
    ------------
    A standard transformer self-attention layer with multiple heads.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads,
                        self.head_dim).transpose(1, 2).contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """
        input shape: batch x time x channel
        """

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            is_cross_attention and past_key_value is not None and
            past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]

        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len) +
                attention_mask
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(
                1, -1, 1, 1
            ) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class HistoGPTDecoderLayer(nn.Module):
    """
    HistoGPT Decoder Layer for Causal Language Modeling
    
    Description:
    ------------
    An autoregressive transformer decoder layer that uses an 
    upper triangular attention mask for language modeling.
    """
    def __init__(self, config: BioGptConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = HistoGPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        self.dropout = config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        input shape: batch x seq_len x embed_dim
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )
        if use_cache:
            outputs += (present_key_value, )
        return outputs


class HistoGPTPreTrainedModel(PreTrainedModel):
    """
    HistoGPT Pre-Trained Model Wrapper Class

    Description:
    ------------
    An astract class that handles weights initialization and an
    interface for downloading and loading the pre-trained models.
    """

    config_class = BioGptConfig
    base_model_prefix = "histogpt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        initialize the weights
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HistoGPTModel):
            module.gradient_checkpointing = value


class HistoGPTModel(HistoGPTPreTrainedModel):
    """
    HistoGPT Model For Generating Histopathology Reports

    Description:
    ------------
    A transformer decoder model combined with a perceiver resampler using 
    interleaved gated cross-attention layers following Deepmind's Flamingo.
    """
    def __init__(self, config: BioGptConfig, params: PRConfig):
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.embed_scale = (
            math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        )
        self.perceiver_resampler = PerceiverResampler(
            dim_feats=params.dim_feats,
            dim_model=params.dim_model,
            dim_head=params.dim_head,
            num_heads=params.num_heads,
            num_layers=params.num_layers,
            num_latents=params.num_latents,
        )
        self.perceiver_exitgate = nn.Linear(
            in_features=params.dim_model,
            out_features=config.hidden_size,
            bias=False,
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, self.embed_dim, self.padding_idx
        )
        self.embed_positions = HistoGPTLearnedPositionalEmbedding(
            config.max_position_embeddings, self.embed_dim
        )
        self.layers = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        GatedCrossAttentionBlock(
                            dim=config.hidden_size,
                            dim_head=(
                                config.hidden_size // config.num_attention_heads
                            ),
                            heads=config.num_attention_heads,
                            ff_mult=4,
                            only_attend_immediate_media=True
                        ),
                        HistoGPTDecoderLayer(config),
                    ]
                )
            )
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        input shape: batch x seq_len or batch x seq_len x embed_dim
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]

        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        positions = self.embed_positions(attention_mask, past_key_values_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = None
        next_decoder_cache = () if use_cache else None

        if exists(image_ids):
            image_embed = self.perceiver_resampler(image_ids)
            image_embed = self.perceiver_exitgate(image_embed)

        for idx, (xattn_layer, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )

            else:
                if exists(image_ids):
                    hidden_states = xattn_layer(
                        x=hidden_states,
                        media=image_embed,
                        media_locations=None,
                    )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(
                        head_mask[idx] if head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        hidden_states = self.layer_norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class HistoGPTForCausalLM(HistoGPTPreTrainedModel):
    """
    HistoGPT for Causal Language Modeling
    
    Description:
    ------------
    A wrapper class for autorgressive causal language modeling.
    """
    _keys_to_ignore_on_load_missing = ["output_projection.weight"]

    def __init__(self, config, params):
        super().__init__(config)
        self.histogpt = HistoGPTModel(config, params)
        self.output_projection = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.post_init()

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        labels: targets for language modeling, note that 
        the labels **are shifted** inside the model
        """
        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.histogpt(
            input_ids,
            image_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)
        lm_loss = None

        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous(
            )
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores, ) + outputs[1:]
            return ((lm_loss, ) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}

        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered_past
