"""Run the Liquid LFM2.5 online-SDFT notification benchmark."""

import argparse
from pathlib import Path

from online_sdft.config import MODEL_ID
from online_sdft.experiment import main
from online_sdft.reporting import replot_from_outputs


DEFAULT_OUTPUT_DIR = Path("outputs/bandit")
DEFAULT_FIGURE_DIR = Path("figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw figures from stored artifacts instead of running the model",
    )
    args = parser.parse_args()
    if args.figures_only:
        replot_from_outputs(
            args.output_dir or DEFAULT_OUTPUT_DIR,
            args.figure_dir or DEFAULT_FIGURE_DIR,
        )
    else:
        main(seeds=args.seeds, model_id=args.model_id, device=args.device,
             local_files_only=args.local_files_only, seed_start=args.seed_start,
             output_dir=args.output_dir, figure_dir=args.figure_dir)
