# Free compute routes for MOIRA

Checked on August 26, 2026. Free quotas change. Confirm the price shown by the
provider before starting a job, set a spending limit when the provider supports
one, and keep a local copy of every checkpoint.

## Recommended order

| Route | Verified free access | Best MOIRA use | Main limit |
|---|---:|---|---|
| Kaggle | Free GPU notebooks after account verification | M1 training, bounded M2/M3 probes, saved outputs | GPU type, weekly quota, and runtime availability can change |
| Lightning AI | 15 monthly credits; its current pricing page describes up to 80 free GPU hours depending on GPU price | Longer experiments with persistent storage | Free Studios restart every four hours |
| Modal Starter | $30 compute credit each month | Short, scripted GPU jobs and evaluations | CPU, RAM, storage, and GPU time all consume credit |
| Beam Developer | $30 credit each month | Checkpointed short GPU jobs | CPU and RAM also consume the credit |
| Google Colab | Free GPU or TPU access | Interactive debugging and notebook reproduction | No fixed GPU type or published stable quota; free runtimes last at most 12 hours |
| Hugging Face ZeroGPU | Five GPU-minutes per day on free accounts; 48 GB or 96 GB allocation | Inference, demos, and short evaluation functions | Gradio-only and too short for pretraining |

The private Kaggle kernel pins PyTorch 2.7.1 with CUDA 12.6. Kaggle assigned a
Tesla P100 to the first run, while its preinstalled CUDA 12.8 wheel omitted the
P100's `sm_60` kernels. CUDA 12.6 is the last official PyTorch wheel line that
supports this Pascal GPU. The kernel asserts `sm_60` support before allocating
the model so an incompatible runtime fails with a direct diagnostic.

Primary sources:

- [Kaggle CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)
- [PyTorch 2.7 CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126/torch/)
- [PyTorch CUDA 12.8 architecture change](https://github.com/pytorch/pytorch/releases/tag/v2.8.0)
- [Lightning AI pricing](https://lightning.ai/pricing)
- [Modal pricing](https://modal.com/pricing)
- [Beam pricing](https://www.beam.cloud/pricing)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/main/spaces-zerogpu)

## Grants worth applying for

Lambda currently offers qualifying researchers up to $5,000 in cloud credits.
The application is live at [Lambda Research](https://lambda.ai/research). The
page does not promise an award or a review date, so do not plan a release around
it.

AWS accepts Cloud Credit for Research applications throughout the year. It is
limited to faculty, research staff, and graduate or postgraduate students at
accredited institutions. Student awards are capped at $5,000, and AWS lists a
90-to-120-day review cycle. See the [official eligibility
page](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/).

AIKosh and C-DAC AIRAWAT may offer India-specific compute, but their current
quota pages require an interactive account session. Treat the A100 sizes and
free-hour figures circulated in third-party summaries as unverified until they
appear in the signed-in portal.

## What each route can prove

A free GPU run can test a model surface, optimizer step, checkpoint, resume,
throughput, or evaluation. It cannot fund exact 3B pretraining from scratch.
Three billion FP32 parameters alone occupy 12 GB. Full FP32 AdamW also needs
gradients and two moment tensors before activations and temporary buffers.

MOIRA keeps two separate records:

- authoritative runs use FP32 parameters and FP32 AdamW state with a resumable
  checkpoint;
- capacity probes may use an 8-bit optimizer, BF16, QAT, or other approximations
  and must name the changed precision in their output.

Do not combine free accounts to evade a provider quota. Submit fewer,
checkpointed runs and resume them through each provider's supported workflow.
