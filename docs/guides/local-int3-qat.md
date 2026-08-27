# Superseded local QAT guide

MOIRA no longer uses the experimental INT3/INT6/INT9 precision ladder. The
supported integer weight views are INT2, INT4, and INT8. Whole-model arithmetic
uses BF16, FP16, or FP32; FP64 is restricted to bounded verifier work.

Use the current commands in the
[root-relative capacity and standard precision section](../../README.md#root-relative-capacity-and-standard-precision).
The exact local M1 profile is `research/configs/m1-local-int8.toml`.

This compatibility page remains at its old path so existing links lead to the
current guide instead of silently running invalid commands.
