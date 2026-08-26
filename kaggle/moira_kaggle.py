"""Private Kaggle GPU run for the bounded MOIRA M2 resume check."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = "https://github.com/anisayakmitra-in/FATE-O-SISTERS.git"
CHECKOUT = Path("/tmp/fate-o-sisters-source")
CHECKPOINT = Path("/kaggle/working/moira-artifacts/m2-text.pt")


def run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


run("git", "clone", "--depth", "1", "--branch", "main", REPOSITORY, str(CHECKOUT))
run("git", "rev-parse", "HEAD", cwd=CHECKOUT)
run(
    sys.executable,
    "-m",
    "pip",
    "install",
    "--index-url",
    "https://download.pytorch.org/whl/cu126",
    "torch==2.7.1+cu126",
)
run(sys.executable, "-m", "pip", "install", "-e", "research", cwd=CHECKOUT)
run(
    sys.executable,
    "-c",
    (
        "import torch; "
        "print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); "
        "assert 'sm_60' in torch.cuda.get_arch_list()"
    ),
)
run("nvidia-smi")

environment = os.environ.copy()
environment["PYTHONPATH"] = str(CHECKOUT / "research" / "src")


def moira(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "moira", *arguments],
        cwd=CHECKOUT,
        env=environment,
        check=True,
    )


CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
common = (
    "m2",
    "train-jsonl",
    "--catalog",
    "research/configs/m1-sources.toml",
    "--source",
    "zai-longcite-45k",
    "--path",
    "research/tests/fixtures/m1-text-sample.jsonl",
    "--batch-size",
    "1",
    "--device",
    "cuda",
    "--checkpoint",
    str(CHECKPOINT),
)
moira(*common, "--steps", "1")
moira(*common, "--steps", "2", "--resume")
