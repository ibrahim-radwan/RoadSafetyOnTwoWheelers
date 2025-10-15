"""Pydantic-based configuration schema and loader for radar pipelines."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_NAME = "default.yaml"


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
        return data

    @model_validator(mode="after")
    def _validate(self) -> "PowerNormalizationConfig":
        if self.divide_by <= 0.0:
            raise ValueError("power_normalization.divide_by must be positive")
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


class RadarPipelineConfig(BaseModel):
    """Top-level configuration for the radar processing pipeline."""

    model_config = ConfigDict(extra="ignore")

    windows: FFTWindowConfig = Field(default_factory=FFTWindowConfig)
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
            raise ValueError(f"Failed to parse override value '{raw_value}': {exc}") from exc
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
    "FFTWindowConfig",
    "PowerNormalizationConfig",
    "PointCloudConfig",
    "PolarDetectionConfig",
    "RadarPipelineConfig",
    "ROIConfig",
    "load_radar_config",
    "parse_override_entries",
]
