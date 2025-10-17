#!/usr/bin/env python3
"""Run inference on a single sparse radar npy file without the dataset loader."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models.skeletons import build_skeleton
from utils.util_config import cfg_from_yaml_file, cfg


class SparseRadarPCInference:
    """Reusable inference helper that loads the model once and runs on multiple npy files."""

    def __init__(self, cfg_path: str, checkpoint_path: str, device: torch.device):
        cfg_from_yaml_file(cfg_path, cfg)
        self.cfg = cfg
        self.device = device
        self.raw_input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM  # type: ignore[attr-defined]

        model = build_skeleton(cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model

        self.class_labels = self._extract_class_labels(cfg)

    @staticmethod
    def _extract_class_labels(config) -> Dict[int, str]:
        class_labels: Dict[int, str] = {}
        label_cfg = getattr(config.DATASET, "label", None)  # type: ignore[attr-defined]
        if isinstance(label_cfg, dict):
            for cls_name, values in label_cfg.items():
                if isinstance(values, list) and len(values) >= 2:
                    _, class_id, *_ = values
                    if isinstance(class_id, (int, float)) and class_id > 0:
                        class_labels[int(class_id)] = cls_name
        return class_labels

    def _prepare_batch(self, tensor_points: torch.Tensor, frame_id: str) -> Dict:
        return {
            "rdr_sparse": tensor_points,
            "batch_indices_rdr_sparse": torch.zeros(len(tensor_points), dtype=torch.long),
            "batch_size": 1,
            "gt_boxes": torch.zeros((1, 1, 8), dtype=torch.float32),
            "meta": [{"frame_id": frame_id}],
        }

    @torch.no_grad()
    def run_on_points(self, point_cloud: np.ndarray, frame_id: str, conf_thr: float) -> Dict:
        if point_cloud.shape[1] < self.raw_input_dim:
            raise ValueError(
                f"Input npy has {point_cloud.shape[1]} columns but model requires >= {self.raw_input_dim}."
            )
        point_cloud = point_cloud[:, : self.raw_input_dim]
        tensor_points = torch.from_numpy(point_cloud).float()

        batch_dict = self._prepare_batch(tensor_points, frame_id)
        output = self.model(batch_dict)

        pred_dict = output["pred_dicts"][0]
        boxes = pred_dict["pred_boxes"].detach().cpu()
        scores = pred_dict["pred_scores"].detach().cpu()
        labels = pred_dict["pred_labels"].detach().cpu()

        keep = scores > conf_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        total_preds = len(scores)

        if total_preds:
            mask_neg_x = boxes[:, 0] < 0
            dims = boxes[:, 3:6]
            mask_bad_dims = torch.logical_not(torch.all((dims > 0) & torch.isfinite(dims), dim=1))

            count_neg_x = int(mask_neg_x.sum().item())
            count_bad_dims = int(mask_bad_dims.sum().item())

            if count_neg_x:
                print(f"Filtered {count_neg_x}/{total_preds} preds with negative x")
            if count_bad_dims:
                print(f"Filtered {count_bad_dims}/{total_preds} preds with invalid dimensions")

            valid_mask = torch.logical_not(mask_neg_x | mask_bad_dims)
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]
        else:
            count_neg_x = 0
            count_bad_dims = 0

        class_names: List[str] = [self.class_labels.get(int(idx), "unknown") for idx in labels]

        return {
            "frame_id": frame_id,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "class_names": class_names,
            "total_preds_before_filter": total_preds,
            "filtered_neg_x": count_neg_x,
            "filtered_invalid_dims": count_bad_dims,
        }

    def run_on_file(self, npy_path: str, conf_thr: float) -> Dict:
        point_cloud = np.load(npy_path)
        frame_id = Path(npy_path).name
        return self.run_on_points(point_cloud, frame_id, conf_thr)

