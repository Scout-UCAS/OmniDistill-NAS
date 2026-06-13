from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


RUNTIME_ASSET_FILES = (
    "benchmarks/suites/toy_smoke.json",
    "configs/toy_experiment.json",
)


@contextmanager
def packaged_runtime_assets() -> Iterator[Path]:
    asset_root = resources.files("distill_nas_core.assets")
    with TemporaryDirectory(prefix="omnidistill-assets-") as tmp:
        materialized_root = Path(tmp)
        for relative in RUNTIME_ASSET_FILES:
            target = materialized_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = asset_root.joinpath(*relative.split("/"))
            target.write_bytes(source.read_bytes())
        yield materialized_root


@contextmanager
def packaged_benchmark_suite_path() -> Iterator[Path]:
    with packaged_runtime_assets() as asset_root:
        yield asset_root / "benchmarks" / "suites" / "toy_smoke.json"
