"""
Radar Heatmap Analyser for radar_app.py
Returns dictionary with heatmap and point cloud data for visualization
Based on dca1000_analyser_awr2243_pd.py
"""

import multiprocessing
import numpy as np
from queue import Empty, Full
from radar.dca1000_awr2243 import DCA1000Frame
from typing import Dict, Any, Optional
import time
import logging
import os
from config_params import CFGS
from sample_processing.radar_params import ADCParams
from sample_processing.radar_proc import (
    openradar_pd_process_frame,
    process_2D_radar_frame,
    pyradar_process_frame,
    process_3D_radar_frame,
    custom_process_frame,
)
from utils import setup_logger, disable_shm_resource_tracker
from multiprocessing import shared_memory

from engine.interfaces import RadarAnalyser


class RadarHeatmapAnalyser(RadarAnalyser):
    def __init__(
        self,
        config_file: Optional[str] = None,
        *,
        prealloc_shm_meta: Optional[dict] = None,
        prealloc_res_shm_meta: Optional[dict] = None,
        output_dir: Optional[str] = None,
        full_analysis: bool = False,
        enable_tesseract: bool = True,
        enable_zyx_cube: bool = True,
    ):
        # Only store serializable configuration
        self.config_file = config_file
        # Initialize these in run() method
        self.logger: Optional[logging.Logger] = None
        self.adc_params: Optional[ADCParams] = None
        # Shared memory state (initialized in run)
        self._shm_blocks = []
        self._shm_nbytes: int = 0
        self._shm_dtype: Optional[str] = None
        self._shm_shape: Optional[tuple] = None
        self._last_shm_seq: Optional[int] = None
        self._prealloc_shm_meta = prealloc_shm_meta
        # Results SHM (range_doppler, range_azimuth)
        self._prealloc_res_shm_meta = prealloc_res_shm_meta
        self._rd_blocks = []
        self._rd_shape = None
        self._rd_dtype = None
        self._ra_blocks = []
        self._ra_shape = None
        self._ra_dtype = None
        self._res_seq = 0
        # Full analysis controls
        self._output_dir = output_dir
        self._full_analysis = bool(full_analysis)
        # Independent artefact toggles (effective only when _full_analysis True)
        self._enable_tesseract = bool(enable_tesseract)
        self._enable_zyx_cube = bool(enable_zyx_cube)

    # === Artefact saving helpers ===
    def _resolve_stem(self, frame) -> Optional[str]:
        try:
            import os as _os, glob as _glob

            src_fp = getattr(frame, "filepath", None)
            if isinstance(src_fp, str) and src_fp:
                return _os.path.splitext(src_fp)[0]
            # Derive from timestamp and configured output_dir
            if self._output_dir and hasattr(frame, "timestamp"):
                ts = float(getattr(frame, "timestamp", 0.0))
                ts_i = int(ts)
                ts_f = int((ts - ts_i) * 1e5)
                prefix = f"{ts_i:010d}_{ts_f:05d}_"
                matches = _glob.glob(_os.path.join(self._output_dir, prefix + "*.bin"))
                if matches:
                    return _os.path.splitext(matches[0])[0]
        except Exception:
            pass
        return None

    def _save_point_cloud_npy(
        self, stem: str, point_cloud: Optional[Dict[str, Any]]
    ) -> None:
        if not self._full_analysis:
            return
        try:
            import numpy as _np

            if not point_cloud:
                return
            x = _np.asarray(point_cloud.get("x", []), dtype=float)
            y = _np.asarray(point_cloud.get("y", []), dtype=float)
            z_raw = point_cloud.get("z")
            z = (
                _np.asarray(z_raw, dtype=float)
                if z_raw is not None
                else _np.zeros_like(x)
            )
            inten_raw = point_cloud.get("intensity", point_cloud.get("snr"))
            inten = (
                _np.asarray(inten_raw, dtype=float)
                if inten_raw is not None
                else _np.zeros_like(x)
            )
            n = min(x.shape[0], y.shape[0], z.shape[0], inten.shape[0])
            p = inten[:n]
            arr = _np.stack((y[:n], x[:n], z[:n], p), axis=1)
            _np.save(stem + ".npy", arr)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Point cloud .npy save failed: {e}")

    def _save_tesseract_mat(self, stem: str, tesseract) -> float:
        """Save 4D tesseract to MAT file; return elapsed seconds."""
        if not self._full_analysis or not self._enable_tesseract:
            return 0.0
        import time as _time

        t0 = _time.perf_counter()
        try:
            import numpy as _np
            from scipy.io import savemat as _savemat

            if tesseract is None:
                tesseract = _np.zeros((0, 0, 0, 0), dtype=_np.float32)
            _savemat(stem + "_tesseract.mat", {"arrDREA": tesseract})
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tesseract .mat save failed: {e}")
        return _time.perf_counter() - t0

    def _save_zyx_cube_mat(self, stem: str, tesseract, az_grid, el_grid) -> float:
        """Generate & save K-Radar style arr_zyx cube derived from tesseract; return elapsed seconds.

        Previous implementation produced a cube named zyx_cube with shape (Z, Y, X) where:
            - Z axis index corresponded to height (z)
            - Y axis index corresponded to forward range (y)
            - X axis index corresponded to lateral (x)

        K-Radar convention (as referenced) expects an array named arr_zyx whose indexing
        order is (z, x, y). That is, after generation we transpose the middle and last axes
        from (Z, Y, X) -> (Z, X, Y).

        Only the arr_zyx array is stored in the MAT file (no auxiliary axes) per request.
        """
        if not self._full_analysis or not self._enable_zyx_cube:
            return 0.0
        import time as _time

        t0 = _time.perf_counter()
        try:
            import numpy as _np
            from scipy.io import savemat as _savemat

            if (
                tesseract is None
                or not isinstance(tesseract, _np.ndarray)
                or tesseract.size == 0
            ):
                _savemat(
                    stem + "_arr_zyx.mat",
                    {"arr_zyx": _np.zeros((0, 0, 0), dtype=_np.float32)},
                )
                return _time.perf_counter() - t0
            # Average across Doppler to get Range/Elevation/Azimuth cube
            rea = _np.mean(tesseract, axis=0)  # (R,E,A)
            range_res = float(getattr(self.adc_params, "range_resolution", 1.0))
            max_range = getattr(self.adc_params, "max_range", None)
            if max_range is None:
                max_range = range_res * float(
                    getattr(self.adc_params, "samples", rea.shape[0])
                )
            max_range = float(max_range)
            # Ensure az/el grids present
            if az_grid is None:
                az_grid = _np.linspace(-90.0, 90.0, rea.shape[2], dtype=_np.float32)
            if el_grid is None:
                el_grid = _np.linspace(-30.0, 30.0, rea.shape[1], dtype=_np.float32)
            az_r = _np.deg2rad(az_grid).astype(_np.float32)  # (A,)
            el_r = _np.deg2rad(el_grid).astype(_np.float32)  # (E,)
            r_bins = _np.arange(rea.shape[0], dtype=_np.float32) * range_res
            dr = range_res
            x_max = max_range
            y_max = max_range
            z_max = 0.3 * max_range
            xs = _np.arange(-x_max, x_max + 1e-9, dr, dtype=_np.float32)  # lateral
            ys = _np.arange(0.0, y_max + 1e-9, dr, dtype=_np.float32)  # forward
            zs = _np.arange(-z_max, z_max + 1e-9, dr, dtype=_np.float32)  # vertical

            # Internal build uses (Z, Y, X); initialize with -1 sentinel
            zyx_cube_internal = _np.full(
                (zs.size, ys.size, xs.size), -1.0, dtype=_np.float32
            )

            # Precompute 2D lateral/forward grids (Ny, Nx)
            y_grid, x_grid = _np.meshgrid(ys, xs, indexing="ij")  # forward, lateral
            x2 = x_grid * x_grid
            y2 = y_grid * y_grid
            horiz_sq = x2 + y2
            horiz = _np.sqrt(horiz_sq, dtype=_np.float32)

            # Helper to obtain interpolation indices & fractional part
            def _interval_indices(values: _np.ndarray, grid: _np.ndarray):
                # values, grid are float32 ascending
                idx = _np.searchsorted(grid, values, side="right") - 1
                valid = (idx >= 0) & (idx < grid.size - 1)
                # Clip to safe range for indexing (avoids negative/overflow)
                idx_clipped = _np.clip(idx, 0, grid.size - 2)
                g0 = grid[idx_clipped]
                g1 = grid[idx_clipped + 1]
                denom = g1 - g0
                with _np.errstate(divide="ignore", invalid="ignore"):
                    t = _np.where(
                        valid, (values - g0) / _np.where(denom == 0, 1.0, denom), 0.0
                    )
                # Ensure t in [0,1]
                _np.clip(t, 0.0, 1.0, out=t)
                return idx_clipped, t.astype(_np.float32), valid

            Ny, Nx = y_grid.shape
            # Iterate only along Z to keep memory bounded; vectorize X/Y plane
            report_every = max(1, zs.size // 10)
            for iz, z in enumerate(zs):
                z2 = float(z * z)
                # Range for this slice
                r = _np.sqrt(horiz_sq + z2, dtype=_np.float32)  # (Ny,Nx)
                # Masks for valid geometry
                in_range = r <= max_range
                # Elevation: arctan2(z, horiz)
                # Handle horiz == 0 -> +/- 90 deg
                with _np.errstate(divide="ignore", invalid="ignore"):
                    el = _np.arctan2(z, _np.where(horiz > 0.0, horiz, 1.0)).astype(
                        _np.float32
                    )
                # Replace where horiz ==0 explicitly
                if z > 0:
                    el = _np.where(horiz == 0.0, 0.5 * _np.pi, el)
                elif z < 0:
                    el = _np.where(horiz == 0.0, -0.5 * _np.pi, el)
                else:
                    el = _np.where(horiz == 0.0, 0.0, el)
                # Azimuth: arctan2(x, y)
                az = _np.arctan2(x_grid, y_grid).astype(_np.float32)

                ir, tr, valid_r = _interval_indices(r, r_bins)
                ie, te, valid_e = _interval_indices(el, el_r)
                ia, ta, valid_a = _interval_indices(az, az_r)
                valid = in_range & valid_r & valid_e & valid_a
                if not _np.any(valid):
                    continue

                # Prepare corner indices
                ir1 = ir + 1
                ie1 = ie + 1
                ia1 = ia + 1

                # Advanced indexing for corners (broadcast over Ny,Nx)
                c000 = rea[ir, ie, ia]
                c001 = rea[ir, ie, ia1]
                c010 = rea[ir, ie1, ia]
                c011 = rea[ir, ie1, ia1]
                c100 = rea[ir1, ie, ia]
                c101 = rea[ir1, ie, ia1]
                c110 = rea[ir1, ie1, ia]
                c111 = rea[ir1, ie1, ia1]

                # Interpolate along azimuth (ta)
                c00 = c000 * (1.0 - ta) + c001 * ta
                c01 = c010 * (1.0 - ta) + c011 * ta
                c10 = c100 * (1.0 - ta) + c101 * ta
                c11 = c110 * (1.0 - ta) + c111 * ta
                # Interpolate along elevation (te)
                c0 = c00 * (1.0 - te) + c01 * te
                c1 = c10 * (1.0 - te) + c11 * te
                # Interpolate along range (tr)
                vals = c0 * (1.0 - tr) + c1 * tr
                # Assign only valid voxels (others remain -1)
                slice_view = zyx_cube_internal[iz]
                slice_view[valid] = vals[valid]

                if self.logger and (iz % report_every == 0 or iz == zs.size - 1):
                    try:
                        self.logger.debug(
                            "arr_zyx build progress: %d/%d slices (%.1f%%)",
                            iz + 1,
                            zs.size,
                            100.0 * (iz + 1) / zs.size,
                        )
                    except Exception:
                        pass

            # Convert to K-Radar ordering (z, x, y) from internal (z, y, x)
            arr_zyx = _np.transpose(zyx_cube_internal, (0, 2, 1))
            _savemat(
                stem + "_arr_zyx.mat",
                {"arr_zyx": arr_zyx},
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"arr_zyx .mat save failed: {e}")
        return _time.perf_counter() - t0

    def _maybe_save_artefacts(self, frame, results: Dict[str, Any]) -> None:
        if not self._full_analysis:
            return
        stem = self._resolve_stem(frame)
        if not stem:
            return
        try:
            tesseract = results.pop("tesseract", None)
            az_grid = results.pop("tesseract_az_grid_deg", None)
            el_grid = results.pop("tesseract_el_grid_deg", None)
            tess_time_returned = results.pop("tesseract_time_s", None)
            # Save tesseract & cube
            t_time = 0.0
            z_time = 0.0
            if tesseract is not None:
                if self._enable_tesseract:
                    t_time = self._save_tesseract_mat(stem, tesseract)
                if self._enable_zyx_cube:
                    z_time = self._save_zyx_cube_mat(stem, tesseract, az_grid, el_grid)
            self._save_point_cloud_npy(stem, results.get("point_cloud"))
            if self.logger:
                self.logger.info(
                    "Full-analysis artefacts: enable_tess=%s enable_cube=%s | saved_tesseract=%.3fs (proc=%.3fs) saved_cube=%.3fs",
                    str(self._enable_tesseract),
                    str(self._enable_zyx_cube),
                    t_time,
                    (tess_time_returned if tess_time_returned else -1.0),
                    z_time,
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Artefact save error: {e}")
        finally:
            pass

    def _preprocess_frame_from_raw_data(self, dca_frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame data from raw DCA1000 data format to complex radar frame
        Based on preprocess_frame_from_awr2243 from sample_processing.radar_preproc but adapted for raw data input

        This method uses the same preprocessing logic as the existing preprocess_frame_from_awr2243
        function but works with raw data arrays instead of file paths.

        Args:
            dca_frame: Raw data array from DCA1000 format

        Returns:
            Complex-valued radar frame with shape (chirps, tx, rx, samples)
        """
        # The data is organized as:
        # [chirps, tx, adc_samples, IQ, rx]

        if self.adc_params is None:
            raise RuntimeError(
                "ADC parameters not initialized. Call run() method first."
            )

        frame = np.reshape(
            dca_frame,
            (
                self.adc_params.chirps,
                self.adc_params.tx,
                self.adc_params.samples,
                self.adc_params.IQ,
                self.adc_params.rx,
            ),
        )

        frame = np.transpose(frame, (0, 1, 4, 2, 3))
        # Shape is now (chirps, tx, rx, samples, IQ)

        complex_frame = (1j * frame[..., 1] + frame[..., 0]).astype(
            np.complex64
        )  # I first

        assert complex_frame.shape == (
            self.adc_params.chirps,
            self.adc_params.tx,
            self.adc_params.rx,
            self.adc_params.samples,
        ), f"complex_frame shape mismatch! Expected: {(self.adc_params.chirps, self.adc_params.tx, self.adc_params.rx, self.adc_params.samples)}, Actual: {complex_frame.shape}"

        return complex_frame

    def _analyse_frame(self, dca_frame: DCA1000Frame) -> Dict[str, Any]:
        """
        Analyse frame and return dictionary with heatmap and point cloud data

        Returns:
            Dictionary with keys:
            - 'range_doppler': 2D array for range-doppler heatmap
            - 'range_azimuth': 2D array for range-azimuth heatmap
            - 'point_cloud': dict with 'x', 'y', 'z', 'intensity' arrays
            - 'processing_time': float
        """
        start_time = time.perf_counter()

        # Preprocess frame using the unified preprocessing method
        frame = self._preprocess_frame_from_raw_data(dca_frame.data)

        # Use openradar_pd_process_frame for processing
        # IS_INDOOR=True is a reasonable default for most indoor radar applications

        if self.adc_params is None:
            raise RuntimeError(
                "ADC parameters not initialized. Call run() method first."
            )

        # result = openradar_pd_process_frame(frame, self.adc_params, IS_INDOOR=True)
        # Choose 2D vs 3D pipeline based on number of TX antennas
        if int(getattr(self.adc_params, "tx", 0)) == 2:
            result = process_2D_radar_frame(
                frame,
                self.adc_params,
                IS_INDOOR=True,
                tuning=getattr(self, "tuning", {}),
            )
        elif int(getattr(self.adc_params, "tx", 0)) == 3:
            # Use FFT-based KRadar pipeline for 3 TX configuration
            from sample_processing.radar_proc_kradar import (
                process_3d_radar_frame_kradar,
            )

            az_range = (-53, 53)
            el_range = (-18, 18)
            result = process_3d_radar_frame_kradar(
                frame,
                self.adc_params,
                tuning=getattr(self, "tuning", {}),
                az_range=az_range,
                el_range=el_range,
            )
        else:
            raise RuntimeError(
                f"Unsupported adc_params.tx={getattr(self.adc_params, 'tx', None)}; expected 2 or 3"
            )

        # frame = frame.reshape(frame.shape[0], frame.shape[1] * frame.shape[2], -1)
        # result = pyradar_process_frame(frame, self.adc_params, doa_method="MUSIC", IS_INDOOR=False)
        # result = custom_process_frame(frame, self.adc_params)

        # Extract results
        range_doppler_matrix = result[
            "range_doppler"
        ]  # This will be None for openradar method
        range_azimuth_matrix = result["range_azimuth"]
        x_pos = result["x_pos"]
        y_pos = result["y_pos"]
        z_pos = result["z_pos"]
        velocities = result["velocities"]
        snrs = result["snrs"]
        cluster_labels = result["cluster_labels"]

        # Create point cloud data from the results
        point_cloud_data = {
            "x": x_pos,
            "y": y_pos,
            "z": z_pos,
            "intensity": snrs,  # Use SNR as intensity
            # Also include per-point doppler (velocity)
            "doppler": velocities,
            # Provide max range/speed for downstream renderers
            "max_range": getattr(self.adc_params, "max_range", None),
            "max_speed": getattr(self.adc_params, "max_doppler", None),
        }

        processing_time = time.perf_counter() - start_time
        out = {
            "range_doppler": range_doppler_matrix,
            "range_azimuth": range_azimuth_matrix,
            "point_cloud": point_cloud_data,
            # Also include at top-level for consumers that don't dive into point_cloud
            "max_range": getattr(self.adc_params, "max_range", None),
            "max_speed": getattr(self.adc_params, "max_doppler", None),
            "processing_time": processing_time,
            "frame_timestamp": dca_frame.timestamp,
            # Propagate source filepath when available (replay mode)
            "src_filepath": getattr(dca_frame, "filepath", None),
        }
        # Keep internal artefacts (tesseract & grids) only for saving; do not expose externally
        if self._full_analysis and isinstance(result, dict):
            if result.get("tesseract") is not None:
                out["tesseract"] = result.get("tesseract")
                if result.get("tesseract_az_grid_deg") is not None:
                    out["tesseract_az_grid_deg"] = result.get("tesseract_az_grid_deg")
                if result.get("tesseract_el_grid_deg") is not None:
                    out["tesseract_el_grid_deg"] = result.get("tesseract_el_grid_deg")
                if result.get("tesseract_time_s") is not None:
                    out["tesseract_time_s"] = result.get("tesseract_time_s")

        return out

    def run(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event,
        control_queue: Optional[multiprocessing.Queue] = None,
        ack_queue: Optional[multiprocessing.Queue] = None,
    ):
        """Main processing loop"""
        # Initialize logger and ADC parameters in the target process
        self.logger = setup_logger("RadarHeatmapAnalyser")
        # Keep resource_tracker from touching SHM in this child
        try:
            disable_shm_resource_tracker(self.logger)
        except Exception:
            pass

        # Prevent hang on process exit if consumer stops reading our queue
        try:
            output_queue.cancel_join_thread()
        except Exception as e:
            if self.logger:
                self.logger.error("cancel_join_thread unavailable or failed: %s", e)

        # Initialize ADC parameters from provided config file or default
        config_to_use = (
            self.config_file if self.config_file else CFGS.AWR2243_CONFIG_FILE
        )
        self.adc_params = ADCParams(config_to_use)
        self.logger.info(
            f"ADC parameters initialized from config file: {config_to_use}"
        )

        self.logger.info("RadarHeatmapAnalyser starting...")

        # Check if ADC parameters are available
        if self.adc_params is None:
            self.logger.error("ADC parameters not initialized")
            return

        # Attach to preallocated radar SHM if provided by engine
        if self._prealloc_shm_meta:
            try:
                names = self._prealloc_shm_meta["names"]
                self._shm_blocks = [
                    shared_memory.SharedMemory(name=name) for name in names
                ]
                self._shm_nbytes = int(self._prealloc_shm_meta["nbytes"])
                self._shm_dtype = str(self._prealloc_shm_meta["dtype"])
                self._shm_shape = (
                    tuple(self._prealloc_shm_meta["shape"])
                    if isinstance(self._prealloc_shm_meta["shape"], (list, tuple))
                    else (int(self._prealloc_shm_meta["shape"]),)
                )
                self.logger.info(
                    f"Attached preallocated radar SHM: names={names}, nbytes={self._shm_nbytes}, dtype={self._shm_dtype}, shape={self._shm_shape}"
                )
            except Exception as e:
                self.logger.error(f"Failed to attach preallocated radar SHM: {e}")
                # If engine demanded SHM path, fail fast: we cannot function without raw SHM
                try:
                    stop_event.set()
                except Exception:
                    self.logger.error("Failed to set stop_event")
                return

        # Attach to preallocated results SHM (rd/ra)
        if self._prealloc_res_shm_meta:
            try:
                rd_meta = self._prealloc_res_shm_meta.get("rd")
                ra_meta = self._prealloc_res_shm_meta.get("ra")
                if rd_meta:
                    self._rd_blocks = [
                        shared_memory.SharedMemory(name=n) for n in rd_meta["names"]
                    ]
                    self._rd_shape = (
                        tuple(rd_meta["shape"])
                        if isinstance(rd_meta["shape"], (list, tuple))
                        else (int(rd_meta["shape"]),)
                    )
                    self._rd_dtype = str(rd_meta["dtype"])  # 'float32'
                if ra_meta:
                    self._ra_blocks = [
                        shared_memory.SharedMemory(name=n) for n in ra_meta["names"]
                    ]
                    self._ra_shape = (
                        tuple(ra_meta["shape"])
                        if isinstance(ra_meta["shape"], (list, tuple))
                        else (int(ra_meta["shape"]),)
                    )
                    self._ra_dtype = str(ra_meta["dtype"])  # 'float32'
                self.logger.info(
                    f"Attached preallocated results SHM: rd={rd_meta['names'] if rd_meta else None}, ra={ra_meta['names'] if ra_meta else None}"
                )
                # Send one-time init meta so GUI can attach without env vars
                try:
                    init_msg = {
                        "RADAR_RES_SHM_INIT": True,
                        "rd": (
                            {
                                "names": rd_meta["names"],
                                "shape": self._rd_shape,
                                "dtype": self._rd_dtype,
                            }
                            if rd_meta
                            else None
                        ),
                        "ra": (
                            {
                                "names": ra_meta["names"],
                                "shape": self._ra_shape,
                                "dtype": self._ra_dtype,
                            }
                            if ra_meta
                            else None
                        ),
                    }
                    # output_queue available later; stash for first loop send
                    self._pending_res_init = init_msg
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            "Failed building RADAR_RES_SHM_INIT meta: %s", e
                        )
                    self._pending_res_init = None
            except Exception as e:
                self.logger.error(f"Failed to attach preallocated results SHM: {e}")
                self._pending_res_init = None
        else:
            self._pending_res_init = None

        # Optionally consume ADC_PARAMS if present without losing first data item
        preloaded_item: Optional[Any] = None
        try:
            first_item = input_queue.get(timeout=10)
            if isinstance(first_item, dict) and "ADC_PARAMS" in first_item:
                self.logger.info(
                    "Skipping ADC_PARAMS from queue (using config file parameters)"
                )
            else:
                preloaded_item = first_item
        except Empty as e:
            self.logger.warning(f"No ADC_PARAMS received from queue: {e}")

        self.logger.info(
            f"Initialized with {self.adc_params.tx} TX, {self.adc_params.rx} RX antennas"
        )
        self.logger.info(f"Range Resolution: {self.adc_params.range_resolution:.4f} m")
        self.logger.info(
            f"Doppler Resolution: {self.adc_params.doppler_resolution:.4f} m/s"
        )

        # Runtime tuning container (updated via control messages if wired later)
        self.tuning = {}

        # Process frames
        self._total_dropped_frames = 0
        while not stop_event.is_set():
            try:
                wait_start_ns = time.perf_counter_ns()
                # If we have a pending SHM init for results, send it now
                if getattr(self, "_pending_res_init", None) is not None:
                    try:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            output_queue.put(self._pending_res_init)
                        else:
                            output_queue.put_nowait(self._pending_res_init)
                        self._pending_res_init = None
                    except Full:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            if self.logger:
                                self.logger.error(
                                    "Output queue full during SHM init meta send (full-analysis)"
                                )
                            stop_event.set()
                            break
                        else:
                            if self.logger:
                                self.logger.warning(
                                    "Output queue full during SHM init meta send"
                                )
                # Use preloaded first item if available; otherwise read from queue
                if preloaded_item is not None:
                    item = preloaded_item
                    preloaded_item = None
                else:
                    # Poll tuning updates first
                    if control_queue is not None:
                        try:
                            cmd = control_queue.get_nowait()
                            if isinstance(cmd, str) and cmd.startswith("TUNING:"):
                                import json as _json

                                self.tuning = _json.loads(cmd.split(":", 1)[1]) or {}
                                if self.logger:
                                    self.logger.info(
                                        "Updated tuning params in analyser"
                                    )
                        except Empty:
                            pass
                        except Exception:
                            pass
                    item = input_queue.get(timeout=1)
                # Support STOP sentinel for immediate shutdown
                if isinstance(item, dict) and item.get("STOP"):
                    break
                # SHM is engine-owned; SHM_INIT should not be received
                if isinstance(item, dict) and item.get("RADAR_SHM_INIT"):
                    self.logger.error(
                        "Unexpected RADAR_SHM_INIT received; engine should preallocate SHM"
                    )
                    continue
                # Handle SHM frame metadata
                if isinstance(item, dict) and item.get("RADAR_SHM_FRAME"):
                    if not self._shm_blocks:
                        self.logger.warning(
                            "Received SHM frame before SHM init, dropping."
                        )
                        continue

                    recv_ns = time.perf_counter_ns()
                    slot = int(item.get("slot", 0))
                    seq = int(item.get("seq", 0))
                    # Report missed sequence numbers as drops
                    if self._last_shm_seq is not None and seq != self._last_shm_seq + 1:
                        missed = seq - (self._last_shm_seq + 1)
                        if missed > 0:
                            self.logger.warning(
                                f"Radar frame drop detected: missed={missed} (last_seq={self._last_shm_seq}, seq={seq})"
                            )
                            self._total_dropped_frames += missed
                    self._last_shm_seq = seq

                    # Build numpy view into SHM and copy out
                    try:
                        np_dtype = (
                            np.dtype(self._shm_dtype)
                            if self._shm_dtype is not None
                            else np.int16
                        )
                        shm_view = np.ndarray(
                            self._shm_shape,
                            dtype=np_dtype,
                            buffer=self._shm_blocks[slot].buf,
                        )
                        dca_frame_data = shm_view.copy()
                    except Exception as e:
                        self.logger.warning(f"Failed to read SHM slot {slot}: {e}")
                        continue

                    # Construct frame object with propagated timestamps
                    latest_frame = DCA1000Frame(
                        timestamp=item.get("frame_timestamp", 0.0),
                        data=dca_frame_data,
                        capture_monotonic_ns=int(item.get("capture_monotonic_ns", 0)),
                        enqueue_monotonic_ns=int(item.get("enqueue_monotonic_ns", 0)),
                        filepath=None,  # no filepath in SHM path
                    )

                    # Analyse
                    results = self._analyse_frame(latest_frame)
                    # Save artefacts in full-analysis mode
                    self._maybe_save_artefacts(latest_frame, results)

                    end_ns = time.perf_counter_ns()

                    # Write results to SHM if available
                    shm_written = False
                    try:
                        slot_res = self._res_seq & 1
                        # Range-Doppler
                        if (
                            self._rd_blocks
                            and isinstance(results.get("range_doppler"), np.ndarray)
                            and results["range_doppler"].size > 0
                        ):
                            rd_out = np.asarray(
                                results["range_doppler"], dtype=np.float32
                            )
                            if self._rd_shape and tuple(rd_out.shape) != tuple(
                                self._rd_shape
                            ):
                                try:
                                    rd_out = rd_out.reshape(self._rd_shape)
                                except Exception as e:
                                    if self.logger:
                                        self.logger.error(
                                            "RD reshape to %s failed: %s",
                                            self._rd_shape,
                                            e,
                                        )
                            # Guard: ensure we don't overflow the SHM slot
                            try:
                                rd_expected_nbytes = (
                                    int(np.prod(self._rd_shape))
                                    * np.dtype(self._rd_dtype or "float32").itemsize
                                )
                            except Exception:
                                rd_expected_nbytes = rd_out.nbytes
                            if rd_out.nbytes != rd_expected_nbytes:
                                if self.logger:
                                    self.logger.error(
                                        "RD size mismatch (have %d bytes, expected %d); skipping RD SHM write",
                                        rd_out.nbytes,
                                        rd_expected_nbytes,
                                    )
                            else:
                                mv_rd = memoryview(self._rd_blocks[slot_res].buf)
                                mv_rd[: rd_out.nbytes] = rd_out.tobytes()
                                try:
                                    mv_rd.release()
                                except Exception as e:
                                    if self.logger:
                                        self.logger.error(
                                            "RD memoryview release failed: %s", e
                                        )
                                shm_written = True
                                results.pop("range_doppler", None)
                        # Range-Azimuth
                        if (
                            self._ra_blocks
                            and isinstance(results.get("range_azimuth"), np.ndarray)
                            and results["range_azimuth"].size > 0
                        ):
                            ra_out = np.asarray(
                                results["range_azimuth"], dtype=np.float32
                            )
                            if self._ra_shape and tuple(ra_out.shape) != tuple(
                                self._ra_shape
                            ):
                                try:
                                    ra_out = ra_out.reshape(self._ra_shape)
                                except Exception as e:
                                    if self.logger:
                                        self.logger.error(
                                            "RA reshape to %s failed: %s",
                                            self._ra_shape,
                                            e,
                                        )
                            # Guard: ensure we don't overflow the SHM slot
                            try:
                                ra_expected_nbytes = (
                                    int(np.prod(self._ra_shape))
                                    * np.dtype(self._ra_dtype or "float32").itemsize
                                )
                            except Exception:
                                ra_expected_nbytes = ra_out.nbytes
                            if ra_out.nbytes != ra_expected_nbytes:
                                if self.logger:
                                    self.logger.error(
                                        "RA size mismatch (have %d bytes, expected %d); skipping RA SHM write",
                                        ra_out.nbytes,
                                        ra_expected_nbytes,
                                    )
                            else:
                                mv_ra = memoryview(self._ra_blocks[slot_res].buf)
                                mv_ra[: ra_out.nbytes] = ra_out.tobytes()
                                try:
                                    mv_ra.release()
                                except Exception as e:
                                    if self.logger:
                                        self.logger.error(
                                            "RA memoryview release failed: %s", e
                                        )
                                shm_written = True
                                results.pop("range_azimuth", None)
                        if shm_written:
                            self._res_seq += 1
                    except Exception as e:
                        self.logger.error(f"Failed to write results to SHM: {e}")

                    # Best-effort queue size hint
                    try:
                        qsize_hint = input_queue.qsize()
                    except Exception:
                        qsize_hint = -1

                    results.update(
                        {
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": 0,
                            "drained_count": 0,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                        }
                    )
                    if shm_written:
                        meta = {
                            "RADAR_RES_SHM_FRAME": True,
                            "slot": (self._res_seq - 1) & 1,
                            "seq": self._res_seq - 1,
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": 0,
                            "drained_count": 0,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                            "frame_timestamp": latest_frame.timestamp,
                            # Keep small payloads (point cloud) in-band
                            "point_cloud": results.get("point_cloud"),
                            "src_filepath": getattr(latest_frame, "filepath", None),
                        }
                        try:
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                output_queue.put(meta)
                            else:
                                output_queue.put_nowait(meta)
                        except Full:
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                self.logger.error(
                                    "Output queue full (SHM meta) in full-analysis"
                                )
                                stop_event.set()
                                break
                            else:
                                self.logger.warning(
                                    "Output queue full, skipping frame (SHM meta)"
                                )
                        continue
                    else:
                        try:
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                output_queue.put(results)
                            else:
                                output_queue.put_nowait(results)
                        except Full:
                            if os.environ.get("FULL_ANALYSIS", "0") in (
                                "1",
                                "true",
                                "True",
                            ):
                                self.logger.error(
                                    "Output queue full (results) in full-analysis"
                                )
                                stop_event.set()
                                break
                            else:
                                self.logger.warning(
                                    "Output queue full, skipping frame (results)"
                                )
                        continue
                if isinstance(item, dict) and isinstance(
                    item.get("TUNING"), (str, bytes)
                ):
                    # Accept tuning updates injected through the queue (optional path)
                    try:
                        import json as _json

                        self.tuning = _json.loads(item.get("TUNING")) or {}
                        if self.logger:
                            self.logger.info("Updated tuning params in analyser")
                    except Exception:
                        pass
                    continue
                # Replay feed may wrap frame with its source filepath
                if isinstance(item, dict) and item.get("FRAME") is not None:
                    latest_frame = item.get("FRAME")
                    replay_frame_index = item.get("REPLAY_FRAME_INDEX", None)
                    try:
                        setattr(
                            latest_frame,
                            "filepath",
                            item.get(
                                "RADAR_REPLAY_FILEPATH",
                                getattr(latest_frame, "filepath", None),
                            ),
                        )
                    except Exception:
                        pass
                    drained_count = 0
                    drain_start_ns = time.perf_counter_ns()
                    drain_end_ns = drain_start_ns
                    recv_ns = time.perf_counter_ns()
                    proc_start = time.perf_counter()
                    if (
                        self.logger
                        and replay_frame_index is not None
                        and os.environ.get("FULL_ANALYSIS", "0")
                        in ("1", "true", "True")
                    ):
                        self.logger.info(
                            "REPLAY_PROC_START frame_index=%s", str(replay_frame_index)
                        )
                    results = self._analyse_frame(latest_frame)

                    self._maybe_save_artefacts(latest_frame, results)

                    end_ns = time.perf_counter_ns()
                    if (
                        self.logger
                        and replay_frame_index is not None
                        and os.environ.get("FULL_ANALYSIS", "0")
                        in ("1", "true", "True")
                    ):
                        self.logger.info(
                            "REPLAY_PROC_END   frame_index=%s proc_ms=%.2f",
                            str(replay_frame_index),
                            (time.perf_counter() - proc_start) * 1000.0,
                        )
                    # Attach timing/diag metadata
                    self._total_dropped_frames += drained_count
                    try:
                        qsize_hint = input_queue.qsize()
                    except Exception:
                        qsize_hint = -1
                    results.update(
                        {
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": drain_end_ns - drain_start_ns,
                            "drained_count": drained_count,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                        }
                    )
                    # Send results downstream
                    try:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            output_queue.put(results)
                        else:
                            output_queue.put_nowait(results)
                    except Full:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            self.logger.error(
                                "Output queue full (replay frame) in full-analysis"
                            )
                            stop_event.set()
                            break
                        else:
                            self.logger.warning(
                                "Output queue full, skipping frame (replay frame)"
                            )
                    # Emit ACK for replay pacing if requested
                    if ack_queue is not None and os.environ.get(
                        "FULL_ANALYSIS", "0"
                    ) in ("1", "true", "True"):
                        # Derive replay frame index from the queue item metadata
                        try:
                            replay_index = item.get("REPLAY_FRAME_INDEX", -1)
                            if self.logger:
                                self.logger.info(
                                    "REPLAY_ACK_SEND  frame_index=%s", str(replay_index)
                                )
                            ack_queue.put_nowait(
                                {"RADAR_FRAME_PROCESSED": replay_index}
                            )
                        except Exception:
                            pass
                    continue
                if isinstance(item, DCA1000Frame):
                    # Do not drop further in analyser; process the first frame we dequeued
                    latest_frame = item
                    drained_count = 0
                    drain_start_ns = time.perf_counter_ns()
                    drain_end_ns = drain_start_ns
                    recv_ns = time.perf_counter_ns()
                    # Process the most recent frame
                    # Pass tuning into processing functions via adc_params or kwargs if needed
                    results = self._analyse_frame(latest_frame)

                    self._maybe_save_artefacts(latest_frame, results)

                    end_ns = time.perf_counter_ns()
                    # Attach timing metadata (monotonic ns) and propagate capture times
                    self._total_dropped_frames += drained_count
                    # Best-effort queue size hint (may not be implemented on some platforms)
                    try:
                        qsize_hint = input_queue.qsize()
                    except Exception:
                        qsize_hint = -1
                    results.update(
                        {
                            "capture_monotonic_ns": getattr(
                                latest_frame, "capture_monotonic_ns", 0
                            ),
                            "capture_wall_ns": getattr(
                                latest_frame, "capture_wall_ns", 0
                            ),
                            "enqueue_monotonic_ns": getattr(
                                latest_frame, "enqueue_monotonic_ns", 0
                            ),
                            "analyser_receive_ns": recv_ns,
                            "analyser_end_ns": end_ns,
                            "first_dequeue_wait_ns": recv_ns - wait_start_ns,
                            "drain_ns": drain_end_ns - drain_start_ns,
                            "drained_count": drained_count,
                            "total_dropped_frames": self._total_dropped_frames,
                            "input_queue_size_hint": qsize_hint,
                        }
                    )

                    # Send results
                    try:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            output_queue.put(results)
                        else:
                            output_queue.put_nowait(results)
                    except Full:
                        if os.environ.get("FULL_ANALYSIS", "0") in (
                            "1",
                            "true",
                            "True",
                        ):
                            self.logger.error(
                                "Output queue full (raw DCA frame) in full-analysis"
                            )
                            stop_event.set()
                            break
                        else:
                            # Queue might be full, skip this frame
                            self.logger.warning(
                                "Output queue full, skipping frame (raw DCA frame)"
                            )
                        pass
                else:
                    # Ignore unrelated items to avoid log spam
                    self.logger.debug(f"Ignoring non-frame item: {type(item)}")
                    continue

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                import traceback

                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                stop_event.set()  # Stop on critical error

        self.logger.info("RadarHeatmapAnalyser stopped")

        # Detach from SHM blocks if attached
        if self._shm_blocks:
            for shm in self._shm_blocks:
                try:
                    shm.close()
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            "SHM close failed for raw block (%s): %s", shm.name, e
                        )
            self._shm_blocks = []

        # Detach from results SHM blocks if attached
        if self._rd_blocks:
            for shm in self._rd_blocks:
                try:
                    shm.close()
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            "SHM close failed for RD block (%s): %s", shm.name, e
                        )
            self._rd_blocks = []
        if self._ra_blocks:
            for shm in self._ra_blocks:
                try:
                    shm.close()
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            "SHM close failed for RA block (%s): %s", shm.name, e
                        )
            self._ra_blocks = []

        return
