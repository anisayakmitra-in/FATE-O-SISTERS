# MOIRA M1 data and capability workflow

This milestone adds a manifest-first data layer and a 33,032,562-parameter M1
reference surface. It does not download external weights, upload to Hugging
Face, or claim that M1 training is complete.

## Check the catalog

The catalog is [research/configs/m1-sources.toml](../../research/configs/m1-sources.toml).
It records the source URI, revision, license, intended modality, split, and the
last Dataset Viewer size check. Dataset files and model weights stay in a
caller-owned cache or workspace.

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path "research/src").Path
python -m moira dataset validate --catalog research/configs/m1-sources.toml
python -m moira m1 inspect
```

`dataset validate` is local and read-only. `m1 inspect` builds the model on the
meta device, checks the exact parameter ledger, and confirms that the component
ablation variants retain the same parameter surface.

## Sources checked on 2026-08-25

The catalog separates training data from teacher or evaluation references.

| Source | Modality | License recorded | M1 use |
| --- | --- | --- | --- |
| `zai-org/LongCite-45k` | Text | Apache-2.0 | Citation SFT |
| `open-r1/OpenR1-Math-220k` | Text | Apache-2.0 | Math reasoning SFT |
| `HuggingFaceTB/smol-smoltalk` | Text | Apache-2.0 | Compact instruction SFT |
| `nvidia/OpenCodeReasoning` | Code | CC-BY-4.0 | Code generation and verification |
| `HuggingFaceM4/ChartQA` | Vision-text | GPL-3.0 | Chart QA evaluation and adaptation |
| `Spawning/PD12M` | Vision-text | CDLA-Permissive-2.0 | Public-domain image-text slices |
| `PolyAI/minds14` | Audio | CC-BY-4.0 | Speech transcription and intent |
| `lerobot/pusht` | Robotics | MIT | Action and time-series capsules |

The catalog also lists optional teacher references from Z.ai GLM, DeepSeek R1
Distill, Xiaomi MiMo-VL and MiMo-Audio, Qwen3, Mistral, Ai2 OLMo 2, SmolLM2,
GLM-4.5-Air, DeepSeek-V3, IBM Granite, Mixtral, Qwen2-Audio, Whisper,
DINOv2, SmolVLM, TimesFM, Graphormer, ESM2, and FLUX.1 schnell. A teacher
reference is not part of the MOIRA parameter count and is never downloaded by
the repository commands.

## Additional open sources checked on 2026-08-26

The second catalog tranche broadens the capability loom without making any of
these sources runtime dependencies:

| Source | Capability | License recorded | M1 status |
| --- | --- | --- | --- |
| `open-web-math` | Mathematical web text | ODC-By-1.0 + Common Crawl terms | Catalog reference; deterministic slicing required |
| `nvidia/OpenMathInstruct-2` | Math reasoning | CC-BY-4.0 | Text normalizer available |
| `openai/gsm8k` | Math verification | MIT | Test split held out; text normalizer available |
| `facebook/voxpopuli` | Multilingual speech | CC0 plus European Parliament notice | Catalog reference; preserve the notice |
| `google/fleurs` | Multilingual speech | CC-BY-4.0 | Catalog one language subset at a time |
| `snap-stanford/ogb` | Graph learning | MIT code; per-dataset terms vary | License review required per graph dataset |
| `facebook/esm2_t6_8M_UR50D` | Protein sequences | MIT | Optional teacher reference |
| `google/timesfm-1.0-200m` | Time-series forecasting | Apache-2.0 | Optional teacher reference |
| `black-forest-labs/FLUX.1-schnell` | Diffusion image generation | Apache-2.0 | Gated 12B reference; never required by M1 |

The catalog deliberately records mixed or gated sources as references instead
of silently treating them as enterprise-cleared training data. Per-row source
licenses, consent, attribution, deletion, and redistribution checks still
belong in the materialized-slice manifest.

There are three important source boundaries:

- `OpenR1-Math-220k` contains DeepSeek-R1 teacher generations but is maintained
  by open-r1. The catalog does not call it a DeepSeek-owned dataset.
- Xiaomi's official MiMo repositories provide model or evaluation references.
  The public Hugging Face Dataset Viewer did not expose a dataset under
  `XiaomiMiMo/MiMo-Embodied`, so the catalog records that repository as an
  evaluation reference instead of inventing a training corpus.
- ChartQA, Minds14, OpenCodeReasoning, and PD12M carry different licenses. A
  license entry is a review starting point, not a blanket enterprise approval.
  Keep per-row license fields when a source provides them.

## Local JSONL normalization

The stream adapter reads one record at a time, enforces an 8 MiB record bound,
computes a SHA-256 record digest, and assigns deterministic splits. It knows the
schemas above and preserves unconsumed fields as metadata.

To inspect caller-owned JSONL without network access:

```powershell
python -m moira dataset sample `
  --catalog research/configs/m1-sources.toml `
  --source zai-longcite-45k `
  --path C:\data\longcite-slice.jsonl `
  --max-records 5
```

The sample command emits record IDs and modalities. It does not fetch a Hub
file. A JSONL row must match the selected source schema; generic text rows use
`text` or `prompt` as input and `target`, `response`, or `output` as target.

## Capability loom

`VisionPatchEncoder` turns a bounded RGB tensor into spatial capsules.
`FeatureCapsuleEncoder` covers audio, time series, graphs, structured values,
and robotics state vectors through the same typed boundary. `CapsuleProjector`
maps a validated capsule to the MOIRA hidden width. Each capsule carries a task,
scope, modality, source hash, slot types, uncertainty, and evidence status.

Image decoding is optional. `load_image_tensor` uses Pillow only when a caller
asks for local image decoding; Pillow is not a required M1 dependency. External
ViT, ASR, diffusion, VAE, GAN, or other specialist weights remain teacher or
organ references until a separate alignment contract is measured.

## Bounded M1 training path

The M1 JSONL command uses a deterministic UTF-8 byte tokenizer and target-only
causal loss. It accepts only cataloged text or code sources, runs in memory, and
does not write a checkpoint:

```powershell
python -m moira m1 train-jsonl `
  --catalog research/configs/m1-sources.toml `
  --source zai-longcite-45k `
  --path C:\data\longcite-slice.jsonl `
  --steps 1 `
  --batch-size 1 `
  --device cpu
```

The step and batch limits are deliberately small. Checkpoint size limits,
resume semantics, tokenizer versioning, GPU precision, held-out evaluation,
and equal-parameter learning curves remain M1 exit work. The existing
checkpointed trainer remains the M0 deterministic reference.

Do not place private conversations in a foundation corpus by accident. Keep
consent, tenant scope, deletion policy, source revision, preprocessing version,
split seed, and content hash beside every materialized dataset slice.
