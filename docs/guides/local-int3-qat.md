# Local INT3-to-INT6 QAT

This guide runs MOIRA's proof-scale, master-retained weight-QAT path. It does
not full-train a 3B model and does not provide a native INT3 GPU kernel.

## What the run proves

The optimizer owns FP32 master weights. Each `Linear` and `Embedding` weight
row gets its own quantization scale and a signed symmetric INT3 forward view,
while the straight-through backward pass updates the masters. Promotion changes
that view to INT6 and continues training. Because the masters survive, the INT6
stage has information that an INT3-only export would not contain.

The permitted executable sequence is:

```text
int3 -> int6 -> int9 -> int27
```

INT81 and later levels remain storage and policy contracts. FP64 exposes 52
significand bits, so an INT81 QAT calculation on FP32/FP64 hardware would not
represent 81 bits of learned numerical detail.

## Windows CUDA environment

Keep this environment separate from `research/.venv`, which is the strict CPU
resume-test environment. With `uv` installed:

```powershell
$env:UV_CACHE_DIR = "$PWD/research/.uv-cache"
uv venv --python 3.11 research/.venv-cuda
uv pip install `
  --python research/.venv-cuda/Scripts/python.exe `
  torch `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install `
  --python research/.venv-cuda/Scripts/python.exe `
  -e "research[training,test]"
```

The tested official wheel uses CUDA 12.8, the first CUDA generation with
Blackwell support. Verify that the installed wheel contains `sm_120` and
executes a kernel:

```powershell
research/.venv-cuda/Scripts/python.exe -c `
  "import torch; x=torch.ones(16,device='cuda'); print(torch.__version__, torch.cuda.get_arch_list(), (x*x).sum().item())"
```

## Run MOIRA

```powershell
$env:PYTHONPATH = "$PWD/research/src"
research/.venv-cuda/Scripts/python.exe -m moira elastic qat-smoke `
  --config research/configs/m0-smoke.toml `
  --initial-bits int3 `
  --promote-to int6 `
  --steps-per-stage 1 `
  --device cuda `
  --checkpoint research/runs/int3-to-int6.pt
```

The JSON result must report finite losses, `master_weights_retained: true`, a
precision history of `int3, int6`, and
`reconstructed_from_low_bit_artifact: false`. The checkpoint is an
experimental trusted-local continuation artifact loaded with PyTorch's
`weights_only` mode.

BF16 is available on the RTX 5060, but this command currently keeps FP32
compute because the Pulse emotional state enforces parameter/state dtype
equality. BF16 activation work requires a separate Pulse dtype migration and
tests.

## Why an 8 GB card cannot full-train 3B here

Packed INT3 weights for 3B parameters have a theoretical lower bound of 1.125
GB. Full training also needs roughly 12 GB of FP32 masters, 12 GB of FP32
gradients, and 24 GB for two FP32 Adam moments, before activations, temporary
fake-quantized views, scales, and allocator overhead. That already exceeds 49
GB. The current RTX 5060 can run the proof model and later PEFT experiments,
but it cannot perform from-scratch full-parameter 3B training.

Use the milestones honestly: validate QAT at M0/M1, measure quality against an
FP32 baseline, then scale with sharding, activation checkpointing, offload, and
distributed hardware. QLoRA/PEFT can reduce fine-tuning memory for an existing
model; they do not turn full from-scratch 3B pretraining into an 8 GB workload.
