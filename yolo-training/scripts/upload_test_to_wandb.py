"""
Upload test evaluation results to wandb for starnet-s1 and baseline-yolo11m.

Results already computed via evaluate_test.py:
  - Starnet-S1:   mAP50=86.94, mAP50-95=56.61, P=87.04, R=81.86
  - Baseline:     mAP50=99.25, mAP50-95=81.75, P=96.23, R=96.92

Usage:
    uv run python scripts/upload_test_to_wandb.py
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "ship-detection"

EXPERIMENTS = {
    "baseline-yolo11m": {
        "test/mAP50": 99.25,
        "test/mAP50-95": 81.75,
        "test/Precision": 96.23,
        "test/Recall": 96.92,
    },
    "starnet-s1": {
        "test/mAP50": 86.94,
        "test/mAP50-95": 56.61,
        "test/Precision": 87.04,
        "test/Recall": 81.86,
    },
}


def main():
    logger.info("=" * 50)
    logger.info("Uploading test evaluation results to wandb")
    logger.info("=" * 50)

    import wandb

    for name, metrics in EXPERIMENTS.items():
        if wandb.run is not None:
            wandb.finish()

        run = wandb.init(project=PROJECT, name=f"test-eval-{name}")
        for key, value in metrics.items():
            run.summary[key] = value
        wandb.log(metrics)
        logger.info("Uploaded %s: mAP50=%.2f, mAP50-95=%.2f",
                    name, metrics["test/mAP50"], metrics["test/mAP50-95"])
        run.finish()

    logger.info("\nDone. Results uploaded to project: %s", PROJECT)


if __name__ == "__main__":
    main()
