"""Multi30k loading, spaCy tokenization, vocabulary, and batching helpers."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


@dataclass
class Vocabulary:
    itos: list[str]
    stoi: dict[str, int]
    default_index: int = 0

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[list[str]],
        min_freq: int = 2,
        specials: Optional[list[str]] = None,
    ) -> "Vocabulary":
        specials = specials or SPECIALS
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        itos = list(specials)
        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq >= min_freq and token not in specials:
                itos.append(token)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(itos=itos, stoi=stoi, default_index=stoi["<unk>"])

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        return cls(
            itos=list(data["itos"]),
            stoi={token: int(idx) for token, idx in data["stoi"].items()},
            default_index=int(data.get("default_index", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "itos": self.itos,
            "stoi": self.stoi,
            "default_index": self.default_index,
        }

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def sos_idx(self) -> int:
        return self.stoi["<sos>"]

    @property
    def eos_idx(self) -> int:
        return self.stoi["<eos>"]

    def lookup_token(self, idx: int) -> str:
        return self.itos[int(idx)]

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self.lookup_token(idx) for idx in indices]

    def encode(self, tokens: list[str], add_specials: bool = True) -> list[int]:
        ids = [self[token] for token in tokens]
        if add_specials:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(
        self,
        indices: Iterable[int],
        skip_specials: bool = True,
        stop_at_eos: bool = True,
    ) -> list[str]:
        tokens = []
        specials = set(SPECIALS)
        for idx in indices:
            token = self.lookup_token(int(idx))
            if stop_at_eos and token == "<eos>":
                break
            if skip_specials and token in specials:
                continue
            tokens.append(token)
        return tokens


def _normalise_split(split: str) -> str:
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    return aliases.get(split, split)


class Multi30kDataset(Dataset):
    """German-to-English Multi30k dataset with spaCy tokenization."""

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        lower: bool = True,
        max_len: Optional[int] = None,
    ) -> None:
        self.split = _normalise_split(split)
        self.lower = lower
        self.max_len = max_len
        self.src_tokenizer = spacy.blank("de").tokenizer
        self.tgt_tokenizer = spacy.blank("en").tokenizer
        self.raw_dataset = load_dataset("bentrevett/multi30k", split=self.split)

        self.tokenized = [
            {
                "src": self.tokenize(example["de"], lang="src"),
                "tgt": self.tokenize(example["en"], lang="tgt"),
            }
            for example in self.raw_dataset
        ]

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab(min_freq=min_freq)

        self.data = self.process_data()

    def tokenize(self, text: str, lang: str) -> list[str]:
        if self.lower:
            text = text.lower()
        tokenizer = self.src_tokenizer if lang == "src" else self.tgt_tokenizer
        return [token.text for token in tokenizer(text)]

    def build_vocab(self, min_freq: int = 2) -> tuple[Vocabulary, Vocabulary]:
        self.src_vocab = Vocabulary.build(
            (example["src"] for example in self.tokenized),
            min_freq=min_freq,
        )
        self.tgt_vocab = Vocabulary.build(
            (example["tgt"] for example in self.tokenized),
            min_freq=min_freq,
        )
        return self.src_vocab, self.tgt_vocab

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        assert self.src_vocab is not None and self.tgt_vocab is not None
        data = []
        for example in self.tokenized:
            src_ids = self.src_vocab.encode(example["src"], add_specials=True)
            tgt_ids = self.tgt_vocab.encode(example["tgt"], add_specials=True)
            if self.max_len is not None:
                if len(src_ids) > self.max_len or len(tgt_ids) > self.max_len:
                    continue
            data.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab.pad_idx)
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab.pad_idx)
        return src, tgt


def save_vocabs(src_vocab: Vocabulary, tgt_vocab: Vocabulary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"src_vocab": src_vocab.to_dict(), "tgt_vocab": tgt_vocab.to_dict()}, f)


def load_vocabs(path: str | Path) -> tuple[Vocabulary, Vocabulary]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Vocabulary.from_dict(data["src_vocab"]), Vocabulary.from_dict(data["tgt_vocab"])


def build_dataloaders(
    batch_size: int = 64,
    min_freq: int = 2,
    lower: bool = True,
    max_len: Optional[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    train_data = Multi30kDataset("train", min_freq=min_freq, lower=lower, max_len=max_len)
    val_data = Multi30kDataset(
        "validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        lower=lower,
        max_len=max_len,
    )
    test_data = Multi30kDataset(
        "test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        lower=lower,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, train_data.src_vocab, train_data.tgt_vocab
