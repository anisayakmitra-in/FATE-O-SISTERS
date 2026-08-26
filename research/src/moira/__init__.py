"""CPU-testable research components for MOIRA."""

from .config import ModelConfig, TrainingConfig
from .ablation import (
    AblationResult,
    assert_equal_parameter_surface,
    equal_parameter_ablation_report,
)
from .dataset import (
    DatasetExample,
    DatasetManifest,
    SourceCatalog,
    SourceSpec,
    assign_split,
    canonical_sha256,
    file_sha256,
    iter_jsonl_examples,
    load_image_tensor,
    normalize_row,
)
from .ledger import ParameterLedger, ParameterSubsystem, build_parameter_ledger
from .loom import CapsuleProjector, FeatureCapsuleEncoder, VisionPatchEncoder
from .m1_train import (
    ByteTokenizer,
    CapabilityBatch,
    CapabilityLoom,
    CausalBatch,
    M1TrainingResult,
    build_capability_batch,
    build_causal_batch,
    iter_cataloged_capability_examples,
    train_m1_capabilities,
    train_m1_jsonl,
)
from .m1_checkpoint import (
    M1LoadedCheckpoint,
    checkpoint_manifest_hash,
    load_m1_checkpoint,
    save_m1_checkpoint,
)

__all__ = [
    "ModelConfig",
    "AblationResult",
    "equal_parameter_ablation_report",
    "assert_equal_parameter_surface",
    "SourceCatalog",
    "SourceSpec",
    "DatasetManifest",
    "DatasetExample",
    "assign_split",
    "canonical_sha256",
    "file_sha256",
    "iter_jsonl_examples",
    "load_image_tensor",
    "normalize_row",
    "VisionPatchEncoder",
    "FeatureCapsuleEncoder",
    "CapsuleProjector",
    "ByteTokenizer",
    "CapabilityBatch",
    "CapabilityLoom",
    "CausalBatch",
    "M1TrainingResult",
    "build_capability_batch",
    "build_causal_batch",
    "iter_cataloged_capability_examples",
    "train_m1_capabilities",
    "train_m1_jsonl",
    "M1LoadedCheckpoint",
    "checkpoint_manifest_hash",
    "load_m1_checkpoint",
    "save_m1_checkpoint",
    "ParameterLedger",
    "ParameterSubsystem",
    "TrainingConfig",
    "build_parameter_ledger",
]
