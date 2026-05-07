"""
Transformer architecture for DA6401 Assignment 3.

The implementation follows "Attention Is All You Need" using only basic
PyTorch building blocks. Masks use True for positions that must be hidden.
"""

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_w = F.softmax(scores, dim=-1)

    if mask is not None:
        attn_w = attn_w.masked_fill(mask, 0.0)
        denom = attn_w.sum(dim=-1, keepdim=True)
        attn_w = torch.where(denom > 0, attn_w / denom.clamp_min(1e-12), attn_w)

    output = torch.matmul(attn_w, V)
    return output, attn_w


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.

    Args:
        Q: Query tensor of shape (..., seq_q, d_k).
        K: Key tensor of shape (..., seq_k, d_k).
        V: Value tensor of shape (..., seq_k, d_v).
        mask: Optional boolean mask broadcastable to (..., seq_q, seq_k).
              True entries are masked out.

    Returns:
        A tuple (output, attention_weights).
    """
    return _attention_impl(Q, K, V, mask=mask, use_scaling=True)


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Return encoder padding mask of shape [batch, 1, 1, src_len]."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Return combined target padding and causal mask [batch, 1, tgt_len, tgt_len]."""
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask.expand(batch_size, 1, tgt_len, tgt_len) | causal_mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented from linear projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _project(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        batch_size = x.size(0)
        return (
            linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._project(query, self.q_linear)
        k = self._project(key, self.k_linear)
        v = self._project(value, self.v_linear)

        attn_out, attn_w = _attention_impl(
            q,
            k,
            v,
            mask=mask,
            use_scaling=self.use_scaling,
        )
        self.attn_weights = attn_w.detach()
        attn_out = torch.matmul(self.dropout(attn_w), v)

        batch_size = query.size(0)
        concat = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.out_linear(concat)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding registered as a non-trainable buffer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding used only for the positional ablation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


class PositionwiseFeedForward(nn.Module):
    """Two-layer point-wise feed-forward network with ReLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.dropout(self.self_attn(y, y, y, src_mask))
        y = self.norm2(x)
        x = x + self.dropout(self.feed_forward(y))
        return x


class DecoderLayer(nn.Module):
    """Pre-LayerNorm Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.dropout(self.self_attn(y, y, y, tgt_mask))
        y = self.norm2(x)
        x = x + self.dropout(self.cross_attn(y, memory, memory, src_mask))
        y = self.norm3(x)
        x = x + self.dropout(self.feed_forward(y))
        return x


class Encoder(nn.Module):
    """Stack of identical encoder layers with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of identical decoder layers with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for German-to-English translation."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        positional_encoding: str = "sinusoidal",
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len
        self.positional_encoding = positional_encoding
        self.use_scaling = use_scaling

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        pos_cls = (
            LearnedPositionalEncoding
            if positional_encoding == "learned"
            else PositionalEncoding
        )
        self.src_pos = pos_cls(d_model, dropout, max_len)
        self.tgt_pos = pos_cls(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scaling)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scaling)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    @property
    def model_config(self) -> dict:
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "N": self.N,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_len": self.max_len,
            "positional_encoding": self.positional_encoding,
            "use_scaling": self.use_scaling,
        }

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        return self.encoder(self.src_pos(x), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        decoded = self.decoder(self.tgt_pos(x), memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
