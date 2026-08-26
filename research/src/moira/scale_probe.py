"""Bounded CUDA capacity probes for measured MOIRA milestone surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import nn

from .config import ModelConfig
from .masks import CognitiveMode
from .model import MoiraModel


@dataclass(frozen=True, slots=True)
class ScaleProbeResult:
    milestone: str
    optimizer: str
    optimizer_state_precision: str
    parameters: int
    sequence_length: int
    loss: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    seconds: float
    checkpoint_written: bool = False


def scale_probe_config(milestone: str) -> ModelConfig:
    configs = {
        "m1": ModelConfig.m1_cpu(),
        "m2": ModelConfig.m2_gpu(),
        "m3": ModelConfig(8_192, 1_280, 32, 20, 3_840, 8, 32, 16, 64, 128),
    }
    try:
        return configs[milestone]
    except KeyError as error:
        raise ValueError("milestone must be m1, m2, or m3") from error


def run_scale_probe(
    milestone: str,
    *,
    optimizer_name: str = "adamw",
    sequence_length: int = 64,
    seed: int = 20260826,
) -> ScaleProbeResult:
    """Execute exactly one synthetic CUDA optimizer step without a checkpoint."""
    config = scale_probe_config(milestone)
    if type(sequence_length) is not int or not 4 <= sequence_length <= config.max_seq_len:
        raise ValueError("sequence_length must fit the selected milestone")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be non-negative")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    torch.manual_seed(seed)
    started = time.perf_counter()
    model = MoiraModel(config).to(device="cuda", dtype=torch.float32)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False
        )
        state_precision = "fp32"
    elif optimizer_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ValueError("adamw8bit requires the optional bitsandbytes package") from error
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=3e-4, weight_decay=0.01
        )
        state_precision = "int8-quantized"
    else:
        raise ValueError("optimizer must be adamw or adamw8bit")
    tokens = torch.randint(
        0, config.vocab_size, (1, sequence_length), device="cuda"
    )
    targets = torch.randint(
        0, config.vocab_size, (1, sequence_length), device="cuda"
    )
    torch.cuda.reset_peak_memory_stats()
    logits = model(tokens, mode=CognitiveMode.ATROPOS).logits
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, config.vocab_size), targets.reshape(-1)
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("scale probe loss became non-finite")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    return ScaleProbeResult(
        milestone=milestone,
        optimizer=optimizer_name,
        optimizer_state_precision=state_precision,
        parameters=model.parameter_ledger().total_trainable,
        sequence_length=sequence_length,
        loss=float(loss.detach().cpu().item()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.perf_counter() - started,
    )
