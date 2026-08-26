"""Validated configuration for CPU MOIRA research runs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    thread_radius: int
    spool_rank: int
    knot_slots: int
    capsule_width: int
    max_seq_len: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    expected_trainable_parameters: int | None = None

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.ffn_dim,
            self.thread_radius,
            self.spool_rank,
            self.knot_slots,
            self.capsule_width,
            self.max_seq_len,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_dim < self.d_model:
            raise ValueError("ffn_dim must be at least d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def m0_cpu(cls) -> "ModelConfig":
        return cls(64, 32, 2, 4, 64, 3, 8, 4, 16, 32)

    @classmethod
    def m1_cpu(cls) -> "ModelConfig":
        """The fixed 33M M1 reference surface for local shape and ablation checks."""
        return cls(
            vocab_size=8_192,
            d_model=512,
            n_layers=12,
            n_heads=8,
            ffn_dim=1_024,
            thread_radius=8,
            spool_rank=32,
            knot_slots=16,
            capsule_width=64,
            max_seq_len=512,
            expected_trainable_parameters=33_032_562,
        )

    @classmethod
    def agent_smoke(cls) -> "ModelConfig":
        """Sub-million local surface with room for canonical action trajectories."""
        return cls(
            vocab_size=512,
            d_model=64,
            n_layers=2,
            n_heads=4,
            ffn_dim=128,
            thread_radius=8,
            spool_rank=8,
            knot_slots=8,
            capsule_width=16,
            max_seq_len=768,
        )

    @classmethod
    def m2_gpu(cls) -> "ModelConfig":
        """The fixed 153M M2 reference surface for bounded GPU training."""
        return cls(
            vocab_size=8_192,
            d_model=896,
            n_layers=20,
            n_heads=14,
            ffn_dim=2_304,
            thread_radius=8,
            spool_rank=32,
            knot_slots=16,
            capsule_width=64,
            max_seq_len=256,
            expected_trainable_parameters=153_349_122,
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    kind: str = "synthetic-indexed-v1"
    seed: int = 13031997
    train_split: int = 0
    eval_split: int = 1
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.kind != "synthetic-indexed-v1":
            raise ValueError("unsupported data kind")
        if self.seed < 0 or self.train_split < 0 or self.eval_split < 0:
            raise ValueError("data seeds and splits must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    schedule: str = "cosine-with-warmup"
    warmup_steps: int = 2
    schedule_total_steps: int = 20
    precision: str = "fp32"
    clip_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("unsupported optimizer")
        if self.schedule != "cosine-with-warmup":
            raise ValueError("unsupported schedule")
        if self.precision != "fp32":
            raise ValueError("unsupported precision")
        if self.learning_rate <= 0 or self.clip_grad_norm <= 0:
            raise ValueError("optimization rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_accumulation_steps <= 0 or self.schedule_total_steps <= 0:
            raise ValueError("optimization counts must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.schedule_total_steps:
            raise ValueError("warmup_steps must be less than schedule_total_steps")


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    seed: int = 20260824
    deterministic: str = "strict"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.deterministic != "strict":
            raise ValueError("unsupported deterministic mode")


@dataclass(frozen=True, slots=True)
class RunConfig:
    log_every_steps: int = 1
    checkpoint_every_steps: int = 2
    keep_last_checkpoints: int = 2

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.log_every_steps,
                self.checkpoint_every_steps,
                self.keep_last_checkpoints,
            )
        ):
            raise ValueError("run counts must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    schema_version: int
    model: ModelConfig
    data: DataConfig
    optimization: OptimizationConfig
    reproducibility: ReproducibilityConfig
    run: RunConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.data.train_split == self.data.eval_split:
            raise ValueError("train_split and eval_split must be different")

    @classmethod
    def m0_cpu(cls) -> "TrainingConfig":
        return cls(
            schema_version=1,
            model=ModelConfig.m0_cpu(),
            data=DataConfig(),
            optimization=OptimizationConfig(),
            reproducibility=ReproducibilityConfig(),
            run=RunConfig(),
        )
