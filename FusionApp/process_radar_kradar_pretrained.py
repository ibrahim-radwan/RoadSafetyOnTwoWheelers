import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List
import torch

from kradar.sparse_radar_pc_inference import SparseRadarPCInference


@torch.no_grad()
def run_single_inference(
    cfg_path: str,
    checkpoint_path: str,
    npy_path: str,
    conf_thr: float,
    device: torch.device,
) -> Dict:
    runner = SparseRadarPCInference(cfg_path, checkpoint_path, device)
    return runner.run_on_file(npy_path, conf_thr)


def prepare_inference(
    cfg_path: str, checkpoint_path: str, device: torch.device
) -> SparseRadarPCInference:
    """Factory helper to create a reusable inference runner."""
    return SparseRadarPCInference(cfg_path, checkpoint_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference on a single sparse radar npy file."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config (e.g., cfg_RTNH_wide.yml)"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to pretrained checkpoint (.pt)"
    )
    parser.add_argument("--npy", required=True, help="Path to sparse radar numpy file")
    parser.add_argument(
        "--conf-thr", type=float, default=0.3, help="Confidence threshold"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run on (cuda or cpu)"
    )
    parser.add_argument("--output-json", help="Optional path to save detections")
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("Radar inference currently supports only CUDA execution.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is required by the radar model (pre-processor uses CUDA kernels)."
        )
    device = torch.device("cuda")
    result = run_single_inference(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        npy_path=args.npy,
        conf_thr=args.conf_thr,
        device=device,
    )

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"Saved detections to {args.output_json}")


if __name__ == "__main__":
    main()
