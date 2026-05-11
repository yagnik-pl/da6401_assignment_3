"""
Training, inference, evaluation, and experiment logging for Assignment 3.

The code keeps all model computation in PyTorch and writes CSV artifacts for
the W&B report analyses:
  - epoch/step curves for Noam vs fixed LR
  - query/key gradient norms for scaling ablation
  - encoder attention-head heat maps
  - validation BLEU for sinusoidal vs learned position encodings
  - prediction confidence for label smoothing ablation
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import random
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Vocabulary, build_dataloaders, save_vocabs
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing, ignoring the padding index."""

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        non_pad = target != self.pad_idx

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            if self.smoothing > 0:
                denom = max(1, self.vocab_size - 2)
                true_dist.fill_(self.smoothing / denom)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_(~non_pad.unsqueeze(1), 0.0)

        loss = -(true_dist * log_probs).sum(dim=1)
        return loss.sum() / non_pad.sum().clamp_min(1)


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """Run one train/eval epoch and return average token-level loss."""
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    pad_idx = getattr(loss_fn, "pad_idx", 1)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx=1)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            tokens = (tgt_out != pad_idx).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    return total_loss / max(1, total_tokens)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a translation token by token using greedy decoding."""
    was_training = model.training
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            if end_symbol is not None and torch.all(next_word.squeeze(1) == end_symbol):
                break

    model.train(was_training)
    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Evaluate corpus-level BLEU on a dataloader of (src, tgt) batches."""
    model.eval()
    hypotheses: list[list[str]] = []
    references: list[list[str]] = []
    sos_idx = _vocab_index(tgt_vocab, "<sos>", 2)
    eos_idx = _vocab_index(tgt_vocab, "<eos>", 3)

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            for i in range(src.size(0)):
                src_i = src[i : i + 1]
                src_mask = make_src_mask(src_i, pad_idx=1)
                pred = greedy_decode(
                    model,
                    src_i,
                    src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )
                hypotheses.append(_ids_to_tokens(pred[0].tolist(), tgt_vocab))
                references.append(_ids_to_tokens(tgt[i].tolist(), tgt_vocab))

    return _corpus_bleu(hypotheses, references)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model, optimizer, scheduler, and reconstruction config."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model.model_config,
        },
        path_obj,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Load a checkpoint into the given model and optional optimizer/scheduler."""
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])


def load_model_from_checkpoint(path: str, device: str = "cpu") -> Transformer:
    """Load a Transformer model from checkpoint with saved configuration.

    This function reconstructs the model using the model_config saved in the checkpoint,
    ensuring all required arguments (src_vocab_size, tgt_vocab_size) are provided.
    """
    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint.get("model_config")

    if model_config is None:
        raise ValueError(f"Checkpoint at {path} does not contain 'model_config'")

    model = Transformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_training_experiment() -> None:
    """CLI entry point for one training run or all report ablations."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    device = _device(args.device)

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        lower=True,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )
    save_vocabs(src_vocab, tgt_vocab, output_dir / "vocabs.json")

    base_config = {
        "epochs": args.epochs,
        "d_model": args.d_model,
        "N": args.layers,
        "num_heads": args.heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "warmup_steps": args.warmup_steps,
        "base_lr": args.base_lr,
        "fixed_lr": args.fixed_lr,
        "max_decode_len": args.decode_max_len,
        "device": device,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "checkpoint_dir": checkpoint_dir,
        "csv_dir": csv_dir,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
    }

    if args.experiment == "all_report":
        runs = _report_runs(base_config)
    else:
        runs = [
            {
                **base_config,
                "run_name": args.run_name,
                "scheduler_type": args.scheduler,
                "label_smoothing": args.label_smoothing,
                "positional_encoding": args.positional_encoding,
                "use_scaling": not args.no_attention_scaling,
                "log_grad_norms": args.log_grad_norms,
                "log_prediction_confidence": True,
            }
        ]

    completed = []
    for config in runs:
        result = _train_single_run(
            config,
            train_loader,
            val_loader,
            test_loader,
            src_vocab,
            tgt_vocab,
        )
        completed.append(result)
        _append_csv(csv_dir / "report_index.csv", result)

    _write_report_readme(csv_dir, completed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 Transformer experiments")
    parser.add_argument("--experiment", choices=["single", "all_report"], default="single")
    parser.add_argument("--run-name", default="baseline_noam_scaled_sinusoidal_ls01")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--base-lr", type=float, default=1.0)
    parser.add_argument("--fixed-lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["noam", "fixed"], default="noam")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--positional-encoding", choices=["sinusoidal", "learned"], default="sinusoidal")
    parser.add_argument("--no-attention-scaling", action="store_true")
    parser.add_argument("--log-grad-norms", action="store_true")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--decode-max-len", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="da6401-a3")
    return parser.parse_args()


def _report_runs(base: dict) -> list[dict]:
    baseline = {
        **base,
        "run_name": "baseline_noam_scaled_sinusoidal_ls01",
        "scheduler_type": "noam",
        "label_smoothing": 0.1,
        "positional_encoding": "sinusoidal",
        "use_scaling": True,
        "log_grad_norms": True,
        "log_prediction_confidence": True,
    }
    return [
        baseline,
        {
            **baseline,
            "run_name": "fixed_lr_scaled_sinusoidal_ls01",
            "scheduler_type": "fixed",
            "log_grad_norms": False,
        },
        {
            **baseline,
            "run_name": "no_scale_noam_sinusoidal_ls01",
            "use_scaling": False,
        },
        {
            **baseline,
            "run_name": "learned_pos_noam_scaled_ls01",
            "positional_encoding": "learned",
            "log_grad_norms": False,
        },
        {
            **baseline,
            "run_name": "noam_scaled_sinusoidal_ls00",
            "label_smoothing": 0.0,
            "log_grad_norms": False,
        },
    ]


def _train_single_run(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
) -> dict:
    device = config["device"]
    run_name = config["run_name"]
    csv_dir: Path = config["csv_dir"]
    checkpoint_dir: Path = config["checkpoint_dir"]

    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=5000,
        positional_encoding=config["positional_encoding"],
        use_scaling=config["use_scaling"],
    ).to(device)

    lr = config["base_lr"] if config["scheduler_type"] == "noam" else config["fixed_lr"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])
        if config["scheduler_type"] == "noam"
        else None
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=config["tgt_vocab_size"],
        pad_idx=tgt_vocab.pad_idx,
        smoothing=config["label_smoothing"],
    )

    wandb_run = _maybe_start_wandb(config)
    best_val_loss = math.inf
    best_epoch = -1
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    last_path = checkpoint_dir / f"{run_name}_last.pt"
    global_step = 0

    for epoch in range(1, config["epochs"] + 1):
        train_stats, global_step = _run_epoch_with_metrics(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device,
            run_name,
            csv_dir,
            global_step,
            config,
        )
        val_stats, _ = _run_epoch_with_metrics(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device,
            run_name,
            csv_dir,
            global_step,
            config,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_row = {
            "run_name": run_name,
            "epoch": epoch,
            "scheduler": config["scheduler_type"],
            "use_scaling": config["use_scaling"],
            "positional_encoding": config["positional_encoding"],
            "label_smoothing": config["label_smoothing"],
            "learning_rate": lr_now,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_prediction_confidence": train_stats["prediction_confidence"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_prediction_confidence": val_stats["prediction_confidence"],
        }
        _append_csv(csv_dir / "all_epoch_metrics.csv", epoch_row)
        _append_csv(csv_dir / f"{run_name}_epoch_metrics.csv", epoch_row)
        if wandb_run is not None:
            wandb_run.log(epoch_row, step=global_step)

        save_checkpoint(model, optimizer, scheduler, epoch, str(last_path))
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, str(best_path))

    load_checkpoint(str(best_path), model)
    val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device, config["max_decode_len"])
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device, config["max_decode_len"])
    bleu_row = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "val_bleu": val_bleu,
        "test_bleu": test_bleu,
        "best_checkpoint": str(best_path),
    }
    _append_csv(csv_dir / "all_bleu_scores.csv", bleu_row)
    _append_csv(csv_dir / f"{run_name}_bleu_scores.csv", bleu_row)
    if wandb_run is not None:
        wandb_run.log(bleu_row, step=global_step)
        wandb_run.finish()

    if run_name == "baseline_noam_scaled_sinusoidal_ls01":
        _export_encoder_attention(
            model,
            test_loader,
            src_vocab,
            csv_dir / "encoder_attention_heads.csv",
            run_name,
            device,
        )

    return {
        "run_name": run_name,
        "scheduler": config["scheduler_type"],
        "use_scaling": config["use_scaling"],
        "positional_encoding": config["positional_encoding"],
        "label_smoothing": config["label_smoothing"],
        "epoch_metrics_csv": str(csv_dir / f"{run_name}_epoch_metrics.csv"),
        "step_metrics_csv": str(csv_dir / f"{run_name}_step_metrics.csv"),
        "grad_norms_csv": str(csv_dir / f"{run_name}_grad_norms.csv"),
        "prediction_confidence_csv": str(csv_dir / f"{run_name}_prediction_confidence.csv"),
        "bleu_csv": str(csv_dir / f"{run_name}_bleu_scores.csv"),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "val_bleu": val_bleu,
        "test_bleu": test_bleu,
    }


def _run_epoch_with_metrics(
    data_iter: DataLoader,
    model: Transformer,
    loss_fn: LabelSmoothingLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[NoamScheduler],
    epoch: int,
    is_train: bool,
    device: str,
    run_name: str,
    csv_dir: Path,
    global_step: int,
    config: dict,
) -> tuple[dict, int]:
    model.train(is_train)
    pad_idx = loss_fn.pad_idx
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    confidence_sum = 0.0
    split = "train" if is_train else "val"
    iterator = tqdm(data_iter, desc=f"{run_name} {split} epoch {epoch}", leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(iterator, start=1):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx=1)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits, tgt_out)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                global_step += 1
                if config.get("log_grad_norms") and global_step <= 1000:
                    grad_row = {
                        "run_name": run_name,
                        "step": global_step,
                        "use_scaling": config["use_scaling"],
                        **_attention_grad_norms(model),
                    }
                    _append_csv(csv_dir / "scaling_ablation_grad_norms.csv", grad_row)
                    _append_csv(csv_dir / f"{run_name}_grad_norms.csv", grad_row)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_stats = _batch_accuracy_confidence(logits.detach(), tgt_out, pad_idx)
            tokens = batch_stats["tokens"]
            total_loss += loss.item() * tokens
            total_tokens += tokens
            correct += batch_stats["correct"]
            confidence_sum += batch_stats["confidence_sum"]

            if is_train:
                step_row = {
                    "run_name": run_name,
                    "split": split,
                    "epoch": epoch,
                    "step": global_step,
                    "batch": batch_idx,
                    "scheduler": config["scheduler_type"],
                    "use_scaling": config["use_scaling"],
                    "positional_encoding": config["positional_encoding"],
                    "label_smoothing": config["label_smoothing"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": loss.item(),
                    "accuracy": batch_stats["correct"] / max(1, tokens),
                    "prediction_confidence": batch_stats["confidence_sum"] / max(1, tokens),
                }
                _append_csv(csv_dir / "all_step_metrics.csv", step_row)
                _append_csv(csv_dir / f"{run_name}_step_metrics.csv", step_row)
                if config.get("log_prediction_confidence"):
                    _append_csv(csv_dir / "label_smoothing_prediction_confidence.csv", step_row)
                    _append_csv(csv_dir / f"{run_name}_prediction_confidence.csv", step_row)

            avg_loss = total_loss / max(1, total_tokens)
            avg_acc = correct / max(1, total_tokens)
            iterator.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc:.3f}")

    return (
        {
            "loss": total_loss / max(1, total_tokens),
            "accuracy": correct / max(1, total_tokens),
            "prediction_confidence": confidence_sum / max(1, total_tokens),
        },
        global_step,
    )


def _batch_accuracy_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> dict:
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    mask = target != pad_idx
    correct = ((pred == target) & mask).sum().item()
    target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    confidence_sum = target_probs.masked_select(mask).sum().item()
    return {
        "tokens": mask.sum().item(),
        "correct": correct,
        "confidence_sum": confidence_sum,
    }


def _attention_grad_norms(model: Transformer) -> dict:
    q_sq = 0.0
    k_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm_sq = float(param.grad.detach().norm(2).item() ** 2)
        if name.endswith("q_linear.weight"):
            q_sq += grad_norm_sq
        elif name.endswith("k_linear.weight"):
            k_sq += grad_norm_sq
    return {
        "query_weight_grad_norm": math.sqrt(q_sq),
        "key_weight_grad_norm": math.sqrt(k_sq),
    }


def _export_encoder_attention(
    model: Transformer,
    test_loader: DataLoader,
    src_vocab: Vocabulary,
    csv_path: Path,
    run_name: str,
    device: str,
) -> None:
    model.eval()
    src, _ = next(iter(test_loader))
    src = src[:1].to(device)
    src_mask = make_src_mask(src, pad_idx=src_vocab.pad_idx)
    with torch.no_grad():
        model.encode(src, src_mask)
    attn = model.encoder.layers[-1].self_attn.attn_weights
    if attn is None:
        return
    attn = attn[0].detach().cpu()
    tokens = [src_vocab.lookup_token(idx) for idx in src[0].detach().cpu().tolist()]

    for head in range(attn.size(0)):
        for q_pos, q_tok in enumerate(tokens):
            for k_pos, k_tok in enumerate(tokens):
                _append_csv(
                    csv_path,
                    {
                        "run_name": run_name,
                        "head": head,
                        "query_position": q_pos,
                        "key_position": k_pos,
                        "query_token": q_tok,
                        "key_token": k_tok,
                        "attention_weight": float(attn[head, q_pos, k_pos].item()),
                    },
                )


def _corpus_bleu(hypotheses: list[list[str]], references: list[list[str]], max_n: int = 4) -> float:
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            ref_counts = _ngram_counts(ref, n)
            clipped_counts[n - 1] += sum(
                min(count, ref_counts.get(ngram, 0))
                for ngram, count in hyp_counts.items()
            )
            total_counts[n - 1] += max(0, len(hyp) - n + 1)

    if hyp_len == 0:
        return 0.0

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0 or clipped == 0:
            return 0.0
        precisions.append(clipped / total)

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * score


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _ids_to_tokens(indices: Iterable[int], vocab) -> list[str]:
    tokens = []
    for idx in indices:
        token = _lookup_token(vocab, int(idx))
        if token == "<eos>":
            break
        if token not in {"<unk>", "<pad>", "<sos>", "<eos>"}:
            tokens.append(token)
    return tokens


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
    if hasattr(vocab, "get_itos"):
        return vocab.get_itos()[idx]
    raise TypeError("tgt_vocab must provide lookup_token, itos, or get_itos")


def _vocab_index(vocab, token: str, default: int) -> int:
    if hasattr(vocab, "stoi") and token in vocab.stoi:
        return vocab.stoi[token]
    if hasattr(vocab, "get_stoi") and token in vocab.get_stoi():
        return vocab.get_stoi()[token]
    try:
        return int(vocab[token])
    except Exception:
        return default


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _maybe_start_wandb(config: dict):
    if not config.get("use_wandb"):
        return None
    import wandb

    return wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config={k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))},
        reinit=True,
    )


def _write_report_readme(csv_dir: Path, rows: list[dict]) -> None:
    text = [
        "# CSV files for W&B report",
        "",
        "Load `all_epoch_metrics.csv` for train/validation loss and validation accuracy overlays.",
        "Load `scaling_ablation_grad_norms.csv` for first-1000-step Q/K gradient norms.",
        "Load `encoder_attention_heads.csv` for last encoder layer head heat maps.",
        "Load `all_bleu_scores.csv` for validation/test BLEU comparisons.",
        "Load `label_smoothing_prediction_confidence.csv` for prediction-confidence plots.",
        "",
        "Runs:",
    ]
    for row in rows:
        text.append(
            f"- {row['run_name']}: scheduler={row['scheduler']}, "
            f"scale={row['use_scaling']}, pos={row['positional_encoding']}, "
            f"eps={row['label_smoothing']}, val_bleu={row['val_bleu']:.2f}"
        )
    (csv_dir / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


if __name__ == "__main__":
    run_training_experiment()
