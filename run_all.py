"""End-to-end pipeline runner for the Building Products Demand Forecasting project.

Executes data generation, external fetch, feature engineering, forecasting,
segmentation, and inventory optimization in dependency order. Dashboard is
launched separately via `python -m src.dashboard.app`.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from typing import Callable

LOGGER = logging.getLogger("run_all")


PIPELINE: list[tuple[str, str]] = [
    ("synthetic data", "src.data.generate_synthetic"),
    ("external data", "src.data.fetch_external"),
    ("feature engineering", "src.features.build_features"),
    ("forecasting", "src.models.forecasting"),
    ("segmentation", "src.models.segmentation"),
    ("inventory optimization", "src.optimization.inventory"),
]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _run_step(label: str, module_name: str) -> float:
    LOGGER.info("=" * 70)
    LOGGER.info("STEP: %s -> %s", label, module_name)
    LOGGER.info("=" * 70)
    start = time.perf_counter()
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Module {module_name} has no main()")
    main_fn: Callable[[], None] = module.main
    main_fn()
    elapsed = time.perf_counter() - start
    LOGGER.info("Completed %s in %.1f sec", label, elapsed)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--skip", nargs="*", default=[], help="Step labels to skip")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    skip = {s.lower() for s in args.skip}

    total = 0.0
    for label, module_name in PIPELINE:
        if label.lower() in skip:
            LOGGER.info("SKIP: %s", label)
            continue
        total += _run_step(label, module_name)

    LOGGER.info("=" * 70)
    LOGGER.info("Pipeline complete in %.1f sec (%.1f min)", total, total / 60)
    LOGGER.info("=" * 70)
    LOGGER.info("Launch dashboard with: python -m src.dashboard.app")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        sys.exit(1)
