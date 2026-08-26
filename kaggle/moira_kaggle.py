"""Private Kaggle GPU run for the bounded MOIRA M1 and M3 checks."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = "https://github.com/anisayakmitra-in/FATE-O-SISTERS.git"
CHECKOUT = Path("/kaggle/working/fate-o-sisters-source")
CHECKPOINT = Path("/kaggle/working/moira-artifacts/m1-text.pt")


def run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


run("git", "clone", "--depth", "1", "--branch", "main", REPOSITORY, str(CHECKOUT))
run("git", "rev-parse", "HEAD", cwd=CHECKOUT)
run(sys.executable, "-m", "pip", "install", "-e", "research[training]", cwd=CHECKOUT)
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


moira(
    "m1",
    "scale-probe",
    "--milestone",
    "m3",
    "--optimizer",
    "adamw8bit",
    "--sequence-length",
    "64",
)

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
common = (
    "m1",
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
