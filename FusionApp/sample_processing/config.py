"""Pydantic-based configuration schema and loader for radar pipelines."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_NAME = "default.yaml"


_RANGE_KEY_PATTERN = re.compile(r"[\s,\-:]+")


def _parse_range_key(key: Any) -> Tuple[float, float]:
    """Parse a range expression (e.g., "0-10" or "(0, 10)") into floats."""

    if isinstance(key, (tuple, list)) and len(key) == 2:
        return float(key[0]), float(key[1])

    if not isinstance(key, str):
        raise ValueError(f"Unsupported range key type: {type(key)!r}")

    cleaned = key.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    parts = [part for part in _RANGE_KEY_PATTERN.split(cleaned) if part]
    if len(parts) != 2:
        raise ValueError(f"Could not parse range key '{key}'")
    start, end = map(float, parts)
    return start, end


def _ensure_configs_dir() -> Path:
    if not CONFIGS_DIR.exists():
        raise FileNotFoundError(f"Config directory not found: {CONFIGS_DIR}")
    return CONFIGS_DIR


class ROIConfig(BaseModel):
    """Axis-aligned region-of-interest bounds for sparse detections."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    x: Tuple[float, float] = (0.0, 100.0)
    y: Tuple[float, float] = (-6.2, 6.2)
    z: Tuple[float, float] = (-1.8, 5.8)


class PowerNormalizationConfig(BaseModel):
    """Simple power normalization via division by a fixed scalar."""

    model_config = ConfigDict(extra="ignore")

    divide_by: float = 1e13
    clip_input_max: Optional[float] = None
    range_based_divide_by: Optional[List[Tuple[float, float, float]]] = None
    range_based_default_divide_by: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_clip(cls, data: Any) -> Any:
        if isinstance(data, dict):
            clip_value = data.get("clip_input_max")
            if isinstance(clip_value, str):
                if clip_value.strip().lower() == "auto":
                    data["clip_input_max"] = None
                else:
                    data["clip_input_max"] = float(clip_value)
            elif isinstance(clip_value, (int, float)) and clip_value <= 0.0:
                data["clip_input_max"] = None

            range_value = data.get("range_based_divide_by")
            default_value: Optional[float] = None
            if isinstance(range_value, dict):
                segments: List[Tuple[float, float, float]] = []
                for key, value in range_value.items():
                    if isinstance(key, str) and key.strip().lower() in {
                        "default",
                        "else",
                        "fallback",
                    }:
                        default_value = float(value)
                        continue
                    start, end = _parse_range_key(key)
                    segments.append((start, end, float(value)))
                segments.sort(key=lambda item: item[0])
                data["range_based_divide_by"] = segments
                if default_value is not None:
                    data["range_based_default_divide_by"] = default_value
            elif isinstance(range_value, list):
                segments_list: List[Tuple[float, float, float]] = []
                for entry in range_value:
                    if isinstance(entry, dict) and {"range", "divide_by"} <= set(
                        entry.keys()
                    ):
                        rng = entry["range"]
                        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                            raise ValueError("range entry must be a 2-element sequence")
                        start, end = float(rng[0]), float(rng[1])
                        value = float(entry["divide_by"])
                        segments_list.append((start, end, value))
                        if "default" in entry:
                            default_value = float(entry["default"])
                    elif isinstance(entry, dict) and "default" in entry:
                        default_value = float(entry["default"])
                    elif isinstance(entry, (list, tuple)) and len(entry) == 3:
                        start, end, value = entry
                        segments_list.append((float(start), float(end), float(value)))
                    else:
                        raise ValueError(
                            "range_based_divide_by list entries must be dicts with 'range'/'divide_by' or 3-tuples"
                        )
                segments_list.sort(key=lambda item: item[0])
                data["range_based_divide_by"] = segments_list
                if default_value is not None:
                    data["range_based_default_divide_by"] = default_value
            elif isinstance(range_value, (int, float)):
                data["range_based_default_divide_by"] = float(range_value)
        return data

    @model_validator(mode="after")
    def _validate(self) -> "PowerNormalizationConfig":
        if self.divide_by <= 0.0:
            raise ValueError("power_normalization.divide_by must be positive")
        if (
            self.range_based_default_divide_by is not None
            and self.range_based_default_divide_by <= 0.0
        ):
            raise ValueError(
                "power_normalization.range_based_default_divide_by must be positive"
            )
        if self.range_based_divide_by:
            segments = list(self.range_based_divide_by)
            if not segments:
                self.range_based_divide_by = None
            else:
                prev_end = None
                normalized_segments: List[Tuple[float, float, float]] = []
                for start, end, value in segments:
                    start_f = float(start)
                    end_f = float(end)
                    value_f = float(value)
                    if value_f <= 0.0:
                        raise ValueError(
                            "power_normalization.range_based_divide_by divide_by values must be positive"
                        )
                    if end_f <= start_f:
                        raise ValueError(
                            "power_normalization.range_based_divide_by ranges must have end > start"
                        )
                    if prev_end is not None and start_f < prev_end:
                        raise ValueError(
                            "power_normalization.range_based_divide_by ranges must be non-overlapping and sorted"
                        )
                    prev_end = end_f
                    normalized_segments.append((start_f, end_f, value_f))
                object.__setattr__(self, "range_based_divide_by", normalized_segments)
        return self


class CartesianQuantileConfig(BaseModel):
    """Configuration for KRadar-style Cartesian quantile sparsification."""

    model_config = ConfigDict(extra="ignore")

    quantile_rate: float = 0.1
    normalization_value: Optional[float] = None
    add_half_grid_offset: bool = False
    offset_type: str = "minus"
    doppler_aggregation: str = "max"
    include_doppler: bool = False
    grid_size: Optional[float] = 0.4
    x_limits: Tuple[float, float] = (0.0, 99.6)
    y_limits: Tuple[float, float] = (-80.0, 79.6)
    z_limits: Tuple[float, float] = (-30.0, 29.6)


class PointCloudConfig(BaseModel):
    """Post-processing configuration for sparse point clouds."""

    model_config = ConfigDict(extra="ignore")

    generation_mode: str = "polar_quantile"
    range_scale: float = 0.65
    power_normalization: PowerNormalizationConfig = Field(
        default_factory=PowerNormalizationConfig
    )
    roi: ROIConfig = Field(default_factory=ROIConfig)
    cartesian_quantile: CartesianQuantileConfig = Field(
        default_factory=CartesianQuantileConfig
    )


class DCOffsetConfig(BaseModel):
    """DC offset removal settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    method: str = "per_channel"


class ChannelEqualizationConfig(BaseModel):
    """Channel equalisation settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    method: str = "rms"


class ClutterRemovalConfig(BaseModel):
    """Static clutter removal settings applied during Doppler processing."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True


class CalibrationConfig(BaseModel):
    """Container for calibration steps executed before range/Doppler FFTs."""

    model_config = ConfigDict(extra="ignore")

    dc_offset: DCOffsetConfig = Field(default_factory=DCOffsetConfig)
    channel_equalization: ChannelEqualizationConfig = Field(
        default_factory=ChannelEqualizationConfig
    )
    clutter_removal: ClutterRemovalConfig = Field(default_factory=ClutterRemovalConfig)


class SpatialWindowConfig(BaseModel):
    """Optional per-axis spatial window specification for angle FFTs."""

    model_config = ConfigDict(extra="ignore")

    azimuth: Optional[str] = None
    elevation: Optional[str] = None


class AngleProcessingConfig(BaseModel):
    """Angle processing parameters governing az/el FFT behaviour."""

    model_config = ConfigDict(extra="ignore")

    mode: str = "1d_fft"
    azimuth_range: Tuple[float, float] = (-53.0, 53.0)
    elevation_range: Tuple[float, float] = (-18.0, 18.0)
    azimuth_fft_size: int = 64
    elevation_fft_size: int = 32
    spatial_window: Optional[Union[str, SpatialWindowConfig]] = None


class PolarDetectionConfig(BaseModel):
    """Detection configuration applied on the polar (range/elev/az) cube."""

    model_config = ConfigDict(extra="ignore")

    method: str = "quantile"
    power_quantile: float = 0.985
    guard: Union[Tuple[int, int, int], Iterable[int], int] = (1, 1, 2)
    train: Union[Tuple[int, int, int], Iterable[int], int] = (4, 3, 6)
    pfa: float = 1e-4
    os_rank: Optional[Union[int, float]] = None
    os_alpha: Optional[float] = None
    doppler_guard_bins: int = 2


class FFTWindowConfig(BaseModel):
    """Range/Doppler windowing prior to FFT operations."""

    model_config = ConfigDict(extra="ignore")

    range: str = "hamming"
    doppler: str = "hamming"


class DetectionConfig(BaseModel):
    """Detection-related parameters for peak detection."""

    model_config = ConfigDict(extra="ignore")

    edge_mask_size: int = 4


class RadarPipelineConfig(BaseModel):
    """Top-level configuration for the radar processing pipeline."""

    model_config = ConfigDict(extra="ignore")

    windows: FFTWindowConfig = Field(default_factory=FFTWindowConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    angle: AngleProcessingConfig = Field(default_factory=AngleProcessingConfig)
    polar_detection: PolarDetectionConfig = Field(default_factory=PolarDetectionConfig)
    point_cloud: PointCloudConfig = Field(default_factory=PointCloudConfig)

    def overridden(self, overrides: Dict[str, Any]) -> "RadarPipelineConfig":
        """Return a copy of the config with dotted-key overrides applied."""

        base = self.model_dump()
        merged = _apply_overrides(base, overrides)
        return RadarPipelineConfig.model_validate(merged)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping (file: {path})")
    return data


def _apply_overrides(
    base: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not overrides:
        return base

    merged = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        keys = [segment.strip() for segment in dotted_key.split(".") if segment.strip()]
        if not keys:
            raise ValueError(f"Invalid override key: '{dotted_key}'")
        cursor = merged
        for key in keys[:-1]:
            next_cursor = cursor.get(key)
            if next_cursor is None or not isinstance(next_cursor, dict):
                next_cursor = {}
                cursor[key] = next_cursor
            cursor = next_cursor
        cursor[keys[-1]] = value
    return merged


def parse_override_entries(entries: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Parse CLI --set entries formatted as key=value into a dict."""

    overrides: Dict[str, Any] = {}
    if not entries:
        return overrides

    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Override must be key=value, got '{raw}'")
        key, raw_value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty in '{raw}'")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(
                f"Failed to parse override value '{raw_value}': {exc}"
            ) from exc
        overrides[key] = value
    return overrides


def load_radar_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RadarPipelineConfig:
    """Load pipeline configuration from YAML, applying optional overrides."""

    configs_dir = _ensure_configs_dir()
    if config_path is None:
        path = configs_dir / DEFAULT_CONFIG_NAME
    else:
        path = Path(config_path)
        if not path.is_absolute():
            path = (configs_dir if path.parts[0] != ".." else Path.cwd()) / path
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {path}")

    data = _read_yaml(path)
    merged = _apply_overrides(data, overrides)
    return RadarPipelineConfig.model_validate(merged)


__all__ = [
    "AngleProcessingConfig",
    "CalibrationConfig",
    "CartesianQuantileConfig",
    "DetectionConfig",
    "FFTWindowConfig",
    "PowerNormalizationConfig",
    "PointCloudConfig",
    "PolarDetectionConfig",
    "RadarPipelineConfig",
    "ROIConfig",
    "load_radar_config",
    "parse_override_entries",
]
