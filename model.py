"""
Transformer architecture for DA6401 Assignment 3.

The implementation follows "Attention Is All You Need" using only basic
PyTorch building blocks. Masks use True for positions that must be hidden.
"""

import copy
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_DIR = Path(__file__).resolve().parent


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

    DEFAULT_CHECKPOINT_FILENAME = "baseline_noam_scaled_sinusoidal_ls01_best.pt"
    DEFAULT_CHECKPOINT_PATH = MODULE_DIR / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME
    DEFAULT_CHECKPOINT_ID = "1Y6UmF-_9Wc2-oVls1QFlZl-98diXiD5C"
    DEFAULT_CHECKPOINT_FOLDER_URL = (
        "https://drive.google.com/drive/folders/"
        "1J4uz-pIFyqxxzDphWbB_FQa58TUtrzrb?usp=sharing"
    )

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        positional_encoding: str = "sinusoidal",
        use_scaling: bool = True,
        vocab_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_folder_url: Optional[str] = None,
        load_pretrained: Optional[bool] = None,
    ) -> None:
        super().__init__()

        explicit_vocab_sizes = src_vocab_size is not None or tgt_vocab_size is not None
        explicit_checkpoint = (
            checkpoint_path is not None
            or checkpoint_id is not None
            or checkpoint_folder_url is not None
        )
        should_load_pretrained = (
            load_pretrained
            if load_pretrained is not None
            else explicit_checkpoint or not explicit_vocab_sizes
        )

        checkpoint = None
        if should_load_pretrained:
            checkpoint_file = self._ensure_checkpoint(
                checkpoint_path,
                checkpoint_id,
                checkpoint_folder_url,
            )
            checkpoint = self._load_torch_checkpoint(checkpoint_file)
            checkpoint_config = self._checkpoint_config(checkpoint)
            if checkpoint_config:
                src_vocab_size = checkpoint_config.get("src_vocab_size", src_vocab_size)
                tgt_vocab_size = checkpoint_config.get("tgt_vocab_size", tgt_vocab_size)
                d_model = checkpoint_config.get("d_model", d_model)
                N = checkpoint_config.get("N", N)
                num_heads = checkpoint_config.get("num_heads", num_heads)
                d_ff = checkpoint_config.get("d_ff", d_ff)
                dropout = checkpoint_config.get("dropout", dropout)
                max_len = checkpoint_config.get("max_len", max_len)
                positional_encoding = checkpoint_config.get(
                    "positional_encoding",
                    positional_encoding,
                )
                use_scaling = checkpoint_config.get("use_scaling", use_scaling)
            else:
                inferred = self._infer_config_from_state_dict(
                    self._checkpoint_state_dict(checkpoint)
                )
                src_vocab_size = src_vocab_size or inferred.get("src_vocab_size")
                tgt_vocab_size = tgt_vocab_size or inferred.get("tgt_vocab_size")
                d_model = inferred.get("d_model", d_model)
                d_ff = inferred.get("d_ff", d_ff)
                N = inferred.get("N", N)
                max_len = inferred.get("max_len", max_len)
                positional_encoding = inferred.get(
                    "positional_encoding",
                    positional_encoding,
                )

        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.src_vocab = None
        self.tgt_vocab = None

        if vocab_path is None and (not explicit_vocab_sizes or should_load_pretrained):
            vocab_path = self._find_vocab_path()

        if vocab_path and Path(vocab_path).exists():
            self._load_vocabs(Path(vocab_path))
        elif not explicit_vocab_sizes:
            self.src_vocab = self._minimal_vocab()
            self.tgt_vocab = self._minimal_vocab()

        try:
            self.src_tokenizer = spacy.blank("de").tokenizer
            self.tgt_tokenizer = spacy.blank("en").tokenizer
        except Exception:
            pass

        if self.src_vocab is not None and isinstance(self.src_vocab, dict) and not checkpoint:
            src_vocab_size = len(self.src_vocab.get("itos", []))
        if self.tgt_vocab is not None and isinstance(self.tgt_vocab, dict) and not checkpoint:
            tgt_vocab_size = len(self.tgt_vocab.get("itos", []))

        if src_vocab_size is None:
            src_vocab_size = 10000
        if tgt_vocab_size is None:
            tgt_vocab_size = 10000

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

        if should_load_pretrained and checkpoint is not None:
            self._load_checkpoint_state(checkpoint)

    @staticmethod
    def _minimal_vocab() -> dict:
        tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
        return {"stoi": {token: idx for idx, token in enumerate(tokens)}, "itos": tokens}

    def _find_vocab_path(self) -> Optional[Path]:
        """Search for vocabularies in common locations."""
        candidates = [
            MODULE_DIR / "vocabs.json",
            MODULE_DIR / "outputs" / "vocabs.json",
            Path.cwd() / "vocabs.json",
            Path.cwd() / "outputs" / "vocabs.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_vocabs(self, vocab_path: Path) -> None:
        """Load vocabularies from a JSON file."""
        try:
            with vocab_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.src_vocab = data.get("src_vocab")
            self.tgt_vocab = data.get("tgt_vocab")
        except Exception as e:
            print(f"Warning: Failed to load vocabs from {vocab_path}: {e}")

    def _ensure_checkpoint(
        self,
        checkpoint_path: Optional[str],
        checkpoint_id: Optional[str],
        checkpoint_folder_url: Optional[str],
    ) -> Path:
        checkpoint_file = self._resolve_checkpoint_path(
            checkpoint_path or str(self.DEFAULT_CHECKPOINT_PATH)
        )
        if checkpoint_file.exists():
            return checkpoint_file

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_id = checkpoint_id or self.DEFAULT_CHECKPOINT_ID
        checkpoint_folder_url = checkpoint_folder_url or self.DEFAULT_CHECKPOINT_FOLDER_URL
        self._download_checkpoint(checkpoint_file, checkpoint_id, checkpoint_folder_url)
        return checkpoint_file

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
        path = Path(checkpoint_path)
        if path.is_absolute():
            return path
        module_path = MODULE_DIR / path
        if module_path.exists() or not path.exists():
            return module_path
        return path.resolve()

    def _download_checkpoint(
        self,
        checkpoint_file: Path,
        checkpoint_id: Optional[str],
        checkpoint_folder_url: Optional[str],
    ) -> None:
        try:
            import gdown
        except ImportError:
            print("Warning: install gdown or place the checkpoint under checkpoints/.")
            return

        if checkpoint_id:
            file_url = f"https://drive.google.com/uc?id={checkpoint_id}"
            if self._try_gdown_downloads(
                [
                    lambda: gdown.download(
                        id=checkpoint_id,
                        output=str(checkpoint_file),
                        quiet=False,
                    ),
                    lambda: gdown.download(
                        url=file_url,
                        output=str(checkpoint_file),
                        quiet=False,
                    ),
                    lambda: gdown.download(file_url, str(checkpoint_file), quiet=False),
                ],
                checkpoint_file,
                "checkpoint file",
            ):
                return

        if checkpoint_folder_url:
            self._try_gdown_folder_downloads(
                gdown,
                checkpoint_folder_url,
                checkpoint_file,
            )

    @staticmethod
    def _try_gdown_downloads(downloaders: list, checkpoint_file: Path, label: str) -> bool:
        last_error = None
        for download in downloaders:
            try:
                result = download()
            except Exception as e:
                last_error = e
                continue
            if checkpoint_file.exists():
                return True
            if result:
                downloaded_path = Path(result)
                if downloaded_path.exists():
                    downloaded_path.replace(checkpoint_file)
                    return True
        if last_error is not None:
            print(f"Warning: Failed to download {label}: {last_error}")
        return False

    def _try_gdown_folder_downloads(
        self,
        gdown,
        checkpoint_folder_url: str,
        checkpoint_file: Path,
    ) -> bool:
        last_error = None
        for download in (
            lambda: gdown.download_folder(
                url=checkpoint_folder_url,
                output=str(checkpoint_file.parent),
                quiet=False,
            ),
            lambda: gdown.download_folder(
                checkpoint_folder_url,
                str(checkpoint_file.parent),
                quiet=False,
            ),
        ):
            try:
                files = download()
            except Exception as e:
                last_error = e
                continue
            if checkpoint_file.exists():
                return True
            if not files:
                continue
            for downloaded in files:
                downloaded_path = Path(downloaded)
                if downloaded_path.suffix in {".pt", ".pth"} and downloaded_path.exists():
                    downloaded_path.replace(checkpoint_file)
                    return True
        if last_error is not None:
            print(f"Warning: Failed to download checkpoint folder: {last_error}")
        return False

    @staticmethod
    def _load_torch_checkpoint(checkpoint_file: Path):
        if not checkpoint_file.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_file}")
            return None
        try:
            return torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(checkpoint_file, map_location="cpu")
        except Exception:
            try:
                return torch.load(checkpoint_file, map_location="cpu")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint from {checkpoint_file}: {e}")
                return None

    @staticmethod
    def _checkpoint_config(checkpoint) -> dict:
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
            return checkpoint["model_config"]
        return {}

    @staticmethod
    def _checkpoint_state_dict(checkpoint) -> Optional[dict]:
        if checkpoint is None:
            return None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        return None

    @staticmethod
    def _infer_config_from_state_dict(state_dict: Optional[dict]) -> dict:
        if not state_dict:
            return {}

        config = {}
        src_embed = state_dict.get("src_embed.weight")
        tgt_embed = state_dict.get("tgt_embed.weight")
        generator = state_dict.get("generator.weight")
        ff_weight = state_dict.get("encoder.layers.0.feed_forward.linear1.weight")

        if src_embed is not None:
            config["src_vocab_size"], config["d_model"] = src_embed.shape
        if tgt_embed is not None:
            config["tgt_vocab_size"] = tgt_embed.shape[0]
            config.setdefault("d_model", tgt_embed.shape[1])
        if generator is not None:
            config["tgt_vocab_size"] = generator.shape[0]
        if ff_weight is not None:
            config["d_ff"] = ff_weight.shape[0]

        layer_ids = []
        for key in state_dict:
            parts = key.split(".")
            if len(parts) > 3 and parts[:2] == ["encoder", "layers"]:
                try:
                    layer_ids.append(int(parts[2]))
                except ValueError:
                    pass
        if layer_ids:
            config["N"] = max(layer_ids) + 1

        if "src_pos.embedding.weight" in state_dict:
            config["positional_encoding"] = "learned"
            config["max_len"] = state_dict["src_pos.embedding.weight"].shape[0]
        elif "src_pos.pe" in state_dict:
            config["positional_encoding"] = "sinusoidal"
            config["max_len"] = state_dict["src_pos.pe"].shape[1]

        return config

    def _load_checkpoint_state(self, checkpoint) -> None:
        state_dict = self._checkpoint_state_dict(checkpoint)
        if state_dict is None:
            return
        try:
            self.load_state_dict(state_dict)
            print(f"Loaded model weights from {self.DEFAULT_CHECKPOINT_FILENAME}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint state: {e}")

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

    def infer(
        self,
        german_sentence: str,
        max_decode_len: int = 100,
    ) -> str:
        """Translate a German sentence with greedy decoding."""
        self.eval()
        device = next(self.parameters()).device

        if self.src_tokenizer is None:
            try:
                self.src_tokenizer = spacy.blank("de").tokenizer
            except Exception:
                return ""

        tokens = [token.text.lower() for token in self.src_tokenizer(german_sentence)]

        if self.src_vocab is None or not isinstance(self.src_vocab, dict):
            return ""

        src_stoi = self.src_vocab.get("stoi", {})
        unk_idx = src_stoi.get("<unk>", 0)
        sos_idx = src_stoi.get("<sos>", 2)
        eos_idx = src_stoi.get("<eos>", 3)
        pad_idx = src_stoi.get("<pad>", 1)

        src_ids = [sos_idx] + [src_stoi.get(token, unk_idx) for token in tokens] + [eos_idx]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        src_mask = make_src_mask(src_tensor, pad_idx=pad_idx)

        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)

        tgt_vocab = self.tgt_vocab
        if tgt_vocab is None or not isinstance(tgt_vocab, dict):
            return ""

        tgt_stoi = tgt_vocab.get("stoi", {})
        tgt_sos_idx = tgt_stoi.get("<sos>", 2)
        tgt_eos_idx = tgt_stoi.get("<eos>", 3)
        tgt_pad_idx = tgt_stoi.get("<pad>", 1)
        tgt_itos = tgt_vocab.get("itos", [])

        ys = torch.full((1, 1), tgt_sos_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_decode_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=tgt_pad_idx).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_word = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_word], dim=1)
                if next_word.item() == tgt_eos_idx:
                    break

        output_ids = ys[0].tolist()
        output_tokens = []
        specials = {"<unk>", "<pad>", "<sos>", "<eos>"}

        for idx in output_ids:
            token = tgt_itos[idx] if idx < len(tgt_itos) else "<unk>"
            if token == "<eos>":
                break
            if token not in specials:
                output_tokens.append(token)

        return " ".join(output_tokens)
