import numpy as np
import logging
from typing import Optional
from pyapril.caCfar import CA_CFAR
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sample_processing.radar_params import ADCParams
from config_params import CFGS
from utils import setup_logger

# Set up logger for radar processing
logger = setup_logger("RadarProc")


def range_cube_fft(
    frame: np.ndarray, window_type: Optional[str] = None, **window_kwargs
) -> np.ndarray:
    """
    Performs windowing and FFT to calculate range cube from radar frame.

    Args:
        frame: Input radar frame with shape (chirps, tx*rx, samples)
        window_type: Type of window to apply. Options: 'kaiser', 'hanning', 'hamming',
                    'blackman', 'bartlett', None (no windowing)
        **window_kwargs: Additional keyword arguments for window function (e.g., beta for Kaiser)

    Returns:
        numpy.ndarray: Range cube with same shape as input frame
    """
    if window_type is None:
        # No windowing
        range_cube = np.fft.fft(frame, axis=-1)
    else:
        # Apply windowing
        num_samples = frame.shape[-1]

        if window_type.lower() == "kaiser":
            beta = window_kwargs.get("beta", 4)
            window = np.kaiser(num_samples, beta)
        elif window_type.lower() == "hanning":
            window = np.hanning(num_samples)
        elif window_type.lower() == "hamming":
            window = np.hamming(num_samples)
        elif window_type.lower() == "blackman":
            window = np.blackman(num_samples)
        elif window_type.lower() == "bartlett":
            window = np.bartlett(num_samples)
        else:
            raise ValueError(
                f"Unsupported window type: {window_type}. "
                f"Supported types: 'kaiser', 'hanning', 'hamming', 'blackman', 'bartlett', None"
            )

        # Apply window and compute FFT
        frame_windowed = frame * window
        range_cube = np.fft.fft(frame_windowed, axis=-1)

    return range_cube


def range_doppler_fft(
    range_cube: np.ndarray, window_type: Optional[str] = None, **window_kwargs
) -> np.ndarray:
    """
    Performs Doppler windowing and FFT to calculate range-doppler map from range cube.

    Args:
        range_cube: Input range cube with shape (chirps, tx*rx, samples)
        window_type: Type of window to apply along chirp dimension. Options: 'kaiser', 'hanning',
                    'hamming', 'blackman', 'bartlett', None (no windowing)
        **window_kwargs: Additional keyword arguments for window function (e.g., beta for Kaiser)

    Returns:
        numpy.ndarray: Range-doppler map with same shape as input, FFT-shifted along chirp axis
    """
    if window_type is None:
        # No windowing
        range_doppler = np.fft.fft(range_cube, axis=0)
    else:
        # Apply Doppler windowing along chirp dimension
        num_chirps = range_cube.shape[0]

        if window_type.lower() == "kaiser":
            beta = window_kwargs.get("beta", 2)
            window_doppler = np.kaiser(num_chirps, beta)
        elif window_type.lower() == "hanning":
            window_doppler = np.hanning(num_chirps)
        elif window_type.lower() == "hamming":
            window_doppler = np.hamming(num_chirps)
        elif window_type.lower() == "blackman":
            window_doppler = np.blackman(num_chirps)
        elif window_type.lower() == "bartlett":
            window_doppler = np.bartlett(num_chirps)
        else:
            raise ValueError(
                f"Unsupported window type: {window_type}. "
                f"Supported types: 'kaiser', 'hanning', 'hamming', 'blackman', 'bartlett', None"
            )

        # Apply window along chirp dimension and compute FFT
        range_cube_windowed = (range_cube.T * window_doppler).T
        range_doppler = np.fft.fft(range_cube_windowed, axis=0)

    # Apply FFT shift to center zero frequency
    range_doppler = np.fft.fftshift(range_doppler, axes=0)

    return range_doppler


def range_azimuth_fft(range_doppler: np.ndarray) -> np.ndarray:
    """
    Performs azimuth FFT to calculate range-azimuth map from range-doppler data.

    Args:
        range_doppler: Input range-doppler data with shape (chirps, tx*rx, samples)

    Returns:
        numpy.ndarray: Range-azimuth map with shape (chirps, tx*rx, samples)
    """
    azimuth_fft_size = range_doppler.shape[0]  # Use number of chirps as FFT size
    # Perform azimuth FFT with specified size
    range_azimuth = np.fft.fft(range_doppler, n=azimuth_fft_size, axis=0)

    # Apply FFT shift to center zero frequency
    range_azimuth = np.fft.fftshift(range_azimuth, axes=0)

    return range_azimuth


def pyapril_cfar(range_doppler, cfar_params=[10, 2, 1, 1], threshold=8):
    """
    Performs Constant False Alarm Rate (CFAR) detection on range-doppler data using PyApril's CA-CFAR implementation.
    This function processes range-doppler data by first computing the absolute sum across one axis to create
    a 2D map, then applies CA-CFAR detection to identify targets above the adaptive threshold.

    Args:
        range_doppler (numpy.ndarray): Input range-doppler data matrix
        cfar_params (list or tuple): CFAR window parameters in the following order:
            - [win_width, win_height, guard_width, guard_height]
        threshold (float, optional): Detection threshold multiplier. Defaults to 8.

    Returns:
        numpy.ndarray: 2D array of hit coordinates where each row contains [range_idx, doppler_idx]
                      of detected targets that exceed the CFAR threshold
    """
    rd_map = np.abs(range_doppler).sum(axis=1)

    # Use PyApril's fast CFAR implementation
    ca_cfar_obj = CA_CFAR(cfar_params, threshold, rd_map.shape)
    hit_matrix_fast_from_instance = ca_cfar_obj(rd_map)

    hit_rd = np.column_stack(np.where(hit_matrix_fast_from_instance))
    return hit_rd


def rd_hit_bin_to_value(hit_rd, adc_params):
    def _doppler_bins_to_velocity(doppler_bins, adc_params):
        velocities = []
        center_doppler_bin = adc_params.chirps // 2

        for doppler_bin in doppler_bins:
            velocity = (
                doppler_bin - center_doppler_bin
            ) * adc_params.doppler_resolution
            velocities.append(velocity)
        return velocities

    ranges = hit_rd[:, 1] * adc_params.range_resolution
    dopplers = _doppler_bins_to_velocity(hit_rd[:, 0], adc_params)

    return ranges, dopplers


def ranges_angles_to_xy(ranges, angles):
    x = ranges * -np.sin(np.deg2rad(angles))
    y = ranges * np.cos(np.deg2rad(angles))

    return x, y


# Convert all DOA lists to angles, filtering out NaN values
def pyapril_convert_to_angles(doa_list):
    def _doa_index_to_angle(doa_index, num_bins=180):
        if np.isnan(doa_index):
            return np.nan
        return (doa_index / num_bins) * 180 - 90

    angles = [_doa_index_to_angle(doa) for doa in doa_list if not np.isnan(doa)]
    return angles


def apply_radar_clustering(
    x_coords, y_coords, velocities, eps=0.5, min_samples=2, velocity_weight=1.0
):
    """
    Apply DBSCAN clustering to radar detections with spatial and velocity features.

    Args:
        x_coords: X coordinates (m)
        y_coords: Y coordinates (m)
        velocities: Doppler velocities (m/s)
        eps: Maximum distance between samples for clustering
        min_samples: Minimum samples in neighborhood for core point
        velocity_weight: Weight factor for velocity in distance metric

    Returns:
        cluster_labels: Array of cluster labels (-1 for noise, 0+ for clusters)
        n_clusters: Number of clusters found
        n_noise: Number of noise points
    """
    if len(x_coords) < min_samples:
        return np.full(len(x_coords), -1), 0, len(x_coords)

    try:
        # Create feature matrix: [X, Y, weighted_velocity]
        features = np.column_stack(
            [x_coords, y_coords, np.array(velocities) * velocity_weight]
        )

        # Standardize features for balanced clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)

        # Count clusters and noise
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        return cluster_labels, n_clusters, n_noise

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return np.full(len(x_coords), -1), 0, len(x_coords)


from mmwave.tracking import EKF

tracker = EKF()


def openradar_pd_process_frame(frame, adc_params: ADCParams, IS_INDOOR=True):
    """
    Process radar frame using OpenRadar methods with Capon beamforming and CFAR detection.

    This function performs range processing, static clutter removal, Capon beamforming for
    angle estimation, CFAR detection for target identification, and Doppler estimation.

    Args:
        frame (numpy.ndarray): Input radar frame data with shape (chirps, tx, rx, samples).
                              Expected to be complex-valued data from radar ADC.
        adc_params (ADCParams): ADC parameters object containing:
                               - tx: Number of transmit antennas
                               - rx: Number of receive antennas
                               - samples: Number of range samples
                               - chirps: Number of chirps per frame
                               - range_resolution: Range resolution in meters
                               - doppler_resolution: Doppler resolution in m/s
        IS_INDOOR (bool, optional): Flag indicating indoor vs outdoor environment.
                                   Affects CFAR parameters and processing. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "range_doppler": None (not computed in this method)
            - "range_azimuth" (numpy.ndarray): Range-azimuth heatmap with shape (angle_bins, samples)
            - "x_pos" (numpy.ndarray): X coordinates of detected targets in meters
            - "y_pos" (numpy.ndarray): Y coordinates of detected targets in meters
            - "z_pos" (numpy.ndarray): Z coordinates of detected targets in meters (zeros for ground-level targets)
            - "velocities" (numpy.ndarray): Doppler velocities of detected targets in m/s
            - "snrs" (numpy.ndarray): Signal-to-noise ratios of detected targets in dB
            - "cluster_labels" (numpy.ndarray): DBSCAN cluster labels for detected targets

    Note:
        This function uses the mmwave.dsp module for range processing and Capon beamforming.
        The frame is reshaped internally to (chirps*tx, rx, samples) for processing.
        Static clutter removal is performed by subtracting the mean across chirps.
    """
    import time
    from mmwave import dsp
    from config_params import CFGS

    function_start = time.perf_counter()

    # Frame counter for profile logging every 10 frames
    if not hasattr(openradar_pd_process_frame, "frame_count"):
        openradar_pd_process_frame.frame_count = 0
    openradar_pd_process_frame.frame_count += 1

    logger.debug(
        f"openradar_pd_process_frame: Processing frame with shape {frame.shape}, IS_INDOOR={IS_INDOOR}"
    )
    logger.debug(
        f"ADC Params - tx: {adc_params.tx}, rx: {adc_params.rx}, samples: {adc_params.samples}, chirps: {adc_params.chirps}, range_resolution: {adc_params.range_resolution}, doppler_resolution: {adc_params.doppler_resolution}"
    )

    ANGLE_RANGE = 90
    ANGLE_RES = 1
    ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1

    # Initialize timing
    step_start = time.perf_counter()

    range_azimuth = np.zeros((ANGLE_BINS, adc_params.samples))
    num_vec, steering_vec = dsp.gen_steering_vec(
        ANGLE_RANGE, ANGLE_RES, adc_params.tx * adc_params.rx
    )

    steering_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Steering vector generation: {steering_time:.4f}s"
    )

    # Reshape frame
    step_start = time.perf_counter()
    frame = frame.reshape(frame.shape[0] * frame.shape[1], frame.shape[2], -1)
    reshape_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Frame reshape: {reshape_time:.4f}s")

    # Range processing
    step_start = time.perf_counter()
    radar_cube = dsp.range_processing(frame, window_type_1d=dsp.utils.Window.HANNING)
    range_processing_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Range processing: {range_processing_time:.4f}s")

    # Static clutter removal
    step_start = time.perf_counter()
    mean = radar_cube.mean(0)
    radar_cube = radar_cube - mean
    clutter_removal_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Static clutter removal: {clutter_removal_time:.4f}s"
    )

    # --- capon beamforming
    # step_start = time.perf_counter()
    beamWeights = np.zeros(
        (adc_params.tx * adc_params.rx, adc_params.samples),
        dtype=np.complex64,
    )

    if adc_params.tx == 2:
        radar_cube = np.concatenate(
            (radar_cube[0::2, ...], radar_cube[1::2, ...]),
            axis=1,
        )
    elif adc_params.tx == 3:
        radar_cube = np.concatenate(
            (radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]),
            axis=1,
        )

    # Range-Doppler FFT
    # This is not used in this method, but kept for compatibility
    # with the original function signature.
    # e.g. for visualization purposes.
    # radar_cube[:, :, :5] = 0
    range_doppler = np.fft.fft(radar_cube, axis=0)
    range_doppler = np.fft.fftshift(range_doppler, axes=0)

    import scipy

    # Take log first (similar to what you do for CFAR)
    range_doppler_log = np.log(
        range_doppler + 1e-8
    )  # Add small epsilon to avoid log(0)
    # range_doppler_softmax = scipy.special.softmax(range_doppler_log.flatten()).reshape(
    #     range_doppler.shape
    # )
    # range_doppler[:, :, :2] = 1
    # zs_bin = range_doppler.shape[0]//2
    # range_doppler[zs_bin:zs_bin+1, :, :] = 1
    range_doppler = np.abs(range_doppler_log).sum(axis=1)

    # concat_time = time.perf_counter() - step_start
    # logger.debug(f"    [RADAR_PROFILE] Radar cube concatenation: {concat_time:.4f}s")

    # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
    # has doppler at the last dimension.
    capon_start = time.perf_counter()
    for i in range(adc_params.samples):
        range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(
            radar_cube[:, :, i].T, steering_vec, magnitude=True
        )

    # range_azimuth, beamWeights = dsp.aoa_capon_jitted(
    #     radar_cube, adc_params.tx, adc_params.rx, adc_params.samples, magnitude=True
    # )
    capon_time = time.perf_counter() - capon_start
    logger.debug(
        f"    [RADAR_PROFILE] Capon beamforming ({adc_params.samples} iterations): {capon_time:.4f}s"
    )
    logger.debug(
        f"    [RADAR_PROFILE] Average per Capon iteration: {capon_time/adc_params.samples:.6f}s"
    )

    """ 3 (Object Detection) """
    step_start = time.perf_counter()
    heatmap_log = np.log2(range_azimuth)
    heatmap_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Heatmap log computation: {heatmap_time:.4f}s")

    # logger.debug(f"Range-Doppler shape: {range_doppler.shape}")
    # logger.debug(f"Range-Azimuth shape: {range_azimuth.shape}")

    if IS_INDOOR:
        AZ_LBOUND = 1.5
        AZ_GUARD_LEN = 2
        AZ_NOISE_LEN = 10
        R_LBOUND = 3.0
        R_GUARD_LEN = 2
        R_NOISE_LEN = 10
    else:
        AZ_LBOUND = 1.0
        AZ_GUARD_LEN = 2
        AZ_NOISE_LEN = 10
        R_LBOUND = 1.5
        R_GUARD_LEN = 2
        R_NOISE_LEN = 10

    # --- cfar in azimuth direction
    step_start = time.perf_counter()
    first_pass, _ = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log,
        l_bound=AZ_LBOUND,
        guard_len=AZ_GUARD_LEN,
        noise_len=AZ_NOISE_LEN,
    )
    cfar_az_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR azimuth direction: {cfar_az_time:.4f}s")

    # --- cfar in range direction
    step_start = time.perf_counter()
    second_pass, noise_floor = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log.T,
        l_bound=R_LBOUND,
        guard_len=R_GUARD_LEN,
        noise_len=R_NOISE_LEN,
    )
    cfar_range_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR range direction: {cfar_range_time:.4f}s")

    # --- classify peaks and caclulate snrs
    step_start = time.perf_counter()
    noise_floor = noise_floor.T
    first_pass = heatmap_log > first_pass
    second_pass = heatmap_log > second_pass.T
    peaks = first_pass & second_pass
    peaks[: CFGS.RADAR_SKIP_SIZE, :] = 0
    peaks[-CFGS.RADAR_SKIP_SIZE :, :] = 0
    peaks[:, : CFGS.RADAR_SKIP_SIZE] = 0
    peaks[:, -CFGS.RADAR_SKIP_SIZE :] = 0
    pairs = np.argwhere(peaks)

    peak_classification_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Peak classification: {peak_classification_time:.4f}s"
    )
    logger.debug(f"[DEBUG] CFAR detected {len(pairs)} peaks")

    if len(pairs) == 0:
        logger.debug("[DEBUG] No peaks detected - returning empty results")

        # Profile logging every 10 frames
        if openradar_pd_process_frame.frame_count % 10 == 0:
            total_time = time.perf_counter() - function_start
            logger.info(
                f"    [RADAR_PROFILE] TOTAL openradar_pd_process_frame time: {total_time:.4f}s (frame {openradar_pd_process_frame.frame_count})"
            )
        return {
            "range_doppler": np.array([]),
            "range_azimuth": range_azimuth,
            "x_pos": np.array([]),
            "y_pos": np.array([]),
            "z_pos": np.array([]),
            "velocities": np.array([]),
            "snrs": np.array([]),
            "cluster_labels": np.array([]),
        }

    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:, 0], pairs[:, 1]] - noise_floor[pairs[:, 0], pairs[:, 1]]

    """ 4 (Doppler Estimation) """
    step_start = time.perf_counter()

    # --- get peak indices
    # beamWeights should be selected based on the range indices from CFAR.
    dopplerFFTInput = radar_cube[:, :, ranges]
    beamWeights = beamWeights[:, ranges]

    # --- estimate doppler values
    # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
    # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
    dopplerFFTInput = np.einsum("ijk,jk->ik", dopplerFFTInput, beamWeights)
    logger.debug(f"[DEBUG] dopplerFFTInput shape: {dopplerFFTInput.shape}")
    assert dopplerFFTInput.shape[-1], "Doppler FFT input should not be empty"

    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:] >= adc_params.chirps / 2] -= adc_params.chirps

    doppler_estimation_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Doppler estimation: {doppler_estimation_time:.4f}s"
    )

    # --- convert bins to units
    step_start = time.perf_counter()
    ranges = ranges * adc_params.range_resolution
    azimuths = azimuths - (ANGLE_BINS // 2)
    dopplers = dopplerEst * adc_params.doppler_resolution
    snrs = snrs

    logger.debug(
        f"[DEBUG] Detected {ranges.shape} targets with ranges, azimuths, dopplers, snrs"
    )

    x_pos, y_pos = ranges_angles_to_xy(ranges, azimuths)
    z_pos = np.zeros_like(x_pos)  # Assuming targets are at ground level
    velocities = dopplers
    conversion_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Unit conversion and coordinate transform: {conversion_time:.4f}s"
    )

    # cluster_labels, n_clusters, n_noise = apply_radar_clustering(
    #     x_pos, y_pos, dopplers, eps=0.2, min_samples=2, velocity_weight=0.0
    # )

    cluster_labels = np.array([])
    # tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    # targetDescr, tNum = tracker.step()

    # print(f"[DEBUG] EKF tracking step completed with {tNum} targets")

    # x_pos = np.array([])
    # y_pos = np.array([])
    # velocities = np.array([])

    # for t, tid in zip(targetDescr, range(int(tNum[0]))):
    #     x, y, x_vel, y_vel = t.S[:4]
    #     x = -x
    #     # z_pos = 0
    #     velocity = np.sqrt(x_vel**2 + y_vel**2)
    #     x_pos = np.append(x_pos, x)
    #     y_pos = np.append(y_pos, y)
    #     velocities = np.append(velocities, velocity)

    # Profile logging every 10 frames
    if openradar_pd_process_frame.frame_count % 10 == 0:
        total_time = time.perf_counter() - function_start
        logger.info(
            f"    [RADAR_PROFILE] TOTAL openradar_pd_process_frame time: {total_time:.4f}s (frame {openradar_pd_process_frame.frame_count})"
        )

    return {
        "range_doppler": range_doppler,
        "range_azimuth": range_azimuth,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "z_pos": z_pos,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": cluster_labels,
    }


def openradar_pd_process_frame_optimised(frame, adc_params: ADCParams, IS_INDOOR=True):
    """
    Process radar frame using OpenRadar methods with Capon beamforming and CFAR detection.

    This function performs range processing, static clutter removal, Capon beamforming for
    angle estimation, CFAR detection for target identification, and Doppler estimation.

    Args:
        frame (numpy.ndarray): Input radar frame data with shape (chirps, tx, rx, samples).
                              Expected to be complex-valued data from radar ADC.
        adc_params (ADCParams): ADC parameters object containing:
                               - tx: Number of transmit antennas
                               - rx: Number of receive antennas
                               - samples: Number of range samples
                               - chirps: Number of chirps per frame
                               - range_resolution: Range resolution in meters
                               - doppler_resolution: Doppler resolution in m/s
        IS_INDOOR (bool, optional): Flag indicating indoor vs outdoor environment.
                                   Affects CFAR parameters and processing. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "range_doppler": None (not computed in this method)
            - "range_azimuth" (numpy.ndarray): Range-azimuth heatmap with shape (angle_bins, samples)
            - "x_pos" (numpy.ndarray): X coordinates of detected targets in meters
            - "y_pos" (numpy.ndarray): Y coordinates of detected targets in meters
            - "z_pos" (numpy.ndarray): Z coordinates of detected targets in meters (zeros for ground-level targets)
            - "velocities" (numpy.ndarray): Doppler velocities of detected targets in m/s
            - "snrs" (numpy.ndarray): Signal-to-noise ratios of detected targets in dB
            - "cluster_labels" (numpy.ndarray): DBSCAN cluster labels for detected targets

    Note:
        This function uses the mmwave.dsp module for range processing and Capon beamforming.
        The frame is reshaped internally to (chirps*tx, rx, samples) for processing.
        Static clutter removal is performed by subtracting the mean across chirps.
    """
    import time
    from mmwave import dsp
    from config_params import CFGS

    function_start = time.perf_counter()

    # Frame counter for profile logging every 10 frames
    if not hasattr(openradar_pd_process_frame, "frame_count"):
        openradar_pd_process_frame.frame_count = 0
    openradar_pd_process_frame.frame_count += 1

    logger.debug(
        f"openradar_pd_process_frame: Processing frame with shape {frame.shape}, IS_INDOOR={IS_INDOOR}"
    )
    logger.debug(
        f"ADC Params - tx: {adc_params.tx}, rx: {adc_params.rx}, samples: {adc_params.samples}, chirps: {adc_params.chirps}, range_resolution: {adc_params.range_resolution}, doppler_resolution: {adc_params.doppler_resolution}"
    )

    ANGLE_RANGE = 90
    ANGLE_RES = 1
    ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1

    # Initialize timing
    step_start = time.perf_counter()

    # range_azimuth = np.zeros((ANGLE_BINS, adc_params.samples))
    # num_vec, steering_vec = dsp.gen_steering_vec(
    #     ANGLE_RANGE, ANGLE_RES, adc_params.tx * adc_params.rx
    # )

    steering_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Steering vector generation: {steering_time:.4f}s"
    )

    # Reshape frame
    step_start = time.perf_counter()
    frame = frame.reshape(frame.shape[0] * frame.shape[1], frame.shape[2], -1)
    reshape_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Frame reshape: {reshape_time:.4f}s")

    # Range processing
    step_start = time.perf_counter()
    radar_cube = dsp.range_processing(frame, window_type_1d=dsp.utils.Window.HANNING)
    range_processing_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Range processing: {range_processing_time:.4f}s")

    # Static clutter removal
    step_start = time.perf_counter()
    mean = radar_cube.mean(0)
    radar_cube = radar_cube - mean
    clutter_removal_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Static clutter removal: {clutter_removal_time:.4f}s"
    )

    # --- capon beamforming
    # step_start = time.perf_counter()
    # beamWeights = np.zeros(
    #     (adc_params.tx * adc_params.rx, adc_params.samples),
    #     dtype=np.complex64,
    # )

    if adc_params.tx == 2:
        radar_cube = np.concatenate(
            (radar_cube[0::2, ...], radar_cube[1::2, ...]),
            axis=1,
        )
    elif adc_params.tx == 3:
        radar_cube = np.concatenate(
            (radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]),
            axis=1,
        )

    # Range-Doppler FFT
    # This is not used in this method, but kept for compatibility
    # with the original function signature.
    # e.g. for visualization purposes.
    # radar_cube[:, :, :5] = 0
    range_doppler = np.fft.fft(radar_cube, axis=0)
    range_doppler = np.fft.fftshift(range_doppler, axes=0)

    import scipy

    # Take log first (similar to what you do for CFAR)
    range_doppler_log = np.log(
        range_doppler + 1e-8
    )  # Add small epsilon to avoid log(0)
    # range_doppler_softmax = scipy.special.softmax(range_doppler_log.flatten()).reshape(
    #     range_doppler.shape
    # )
    # range_doppler[:, :, :2] = 1
    # zs_bin = range_doppler.shape[0]//2
    # range_doppler[zs_bin:zs_bin+1, :, :] = 1
    # range_doppler = np.abs(range_doppler_softmax).sum(axis=1)
    range_doppler = np.abs(range_doppler_log).sum(axis=1)

    # concat_time = time.perf_counter() - step_start
    # logger.debug(f"    [RADAR_PROFILE] Radar cube concatenation: {concat_time:.4f}s")

    # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
    # has doppler at the last dimension.
    capon_start = time.perf_counter()
    # for i in range(adc_params.samples):
    #     range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(
    #         radar_cube[:, :, i].T, steering_vec, magnitude=True
    #     )

    range_azimuth, beamWeights = dsp.aoa_capon_jitted(
        radar_cube, adc_params.tx, adc_params.rx, adc_params.samples
    )

    capon_time = time.perf_counter() - capon_start
    logger.debug(
        f"    [RADAR_PROFILE] Capon beamforming ({adc_params.samples} iterations): {capon_time:.4f}s"
    )
    logger.debug(
        f"    [RADAR_PROFILE] Average per Capon iteration: {capon_time/adc_params.samples:.6f}s"
    )

    """ 3 (Object Detection) """
    step_start = time.perf_counter()
    heatmap_log = np.log2(range_azimuth)
    heatmap_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Heatmap log computation: {heatmap_time:.4f}s")

    # logger.debug(f"Range-Doppler shape: {range_doppler.shape}")
    # logger.debug(f"Range-Azimuth shape: {range_azimuth.shape}")

    if IS_INDOOR:
        AZ_LBOUND = 1.5
        AZ_GUARD_LEN = 2
        AZ_NOISE_LEN = 10
        R_LBOUND = 3.0
        R_GUARD_LEN = 2
        R_NOISE_LEN = 10
    else:
        AZ_LBOUND = 1.0
        AZ_GUARD_LEN = 2
        AZ_NOISE_LEN = 10
        R_LBOUND = 1.5
        R_GUARD_LEN = 2
        R_NOISE_LEN = 10

    # --- cfar in azimuth direction
    step_start = time.perf_counter()
    first_pass, _ = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log,
        l_bound=AZ_LBOUND,
        guard_len=AZ_GUARD_LEN,
        noise_len=AZ_NOISE_LEN,
    )
    cfar_az_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR azimuth direction: {cfar_az_time:.4f}s")

    # --- cfar in range direction
    step_start = time.perf_counter()
    second_pass, noise_floor = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log.T,
        l_bound=R_LBOUND,
        guard_len=R_GUARD_LEN,
        noise_len=R_NOISE_LEN,
    )
    cfar_range_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR range direction: {cfar_range_time:.4f}s")

    # --- classify peaks and caclulate snrs
    step_start = time.perf_counter()
    noise_floor = noise_floor.T
    first_pass = heatmap_log > first_pass
    second_pass = heatmap_log > second_pass.T
    peaks = first_pass & second_pass
    peaks[: CFGS.RADAR_SKIP_SIZE, :] = 0
    peaks[-CFGS.RADAR_SKIP_SIZE :, :] = 0
    peaks[:, : CFGS.RADAR_SKIP_SIZE] = 0
    peaks[:, -CFGS.RADAR_SKIP_SIZE :] = 0
    pairs = np.argwhere(peaks)

    peak_classification_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Peak classification: {peak_classification_time:.4f}s"
    )
    logger.debug(f"[DEBUG] CFAR detected {len(pairs)} peaks")

    if len(pairs) == 0:
        logger.debug("[DEBUG] No peaks detected - returning empty results")

        # Profile logging every 10 frames
        if openradar_pd_process_frame.frame_count % 10 == 0:
            total_time = time.perf_counter() - function_start
            logger.info(
                f"    [RADAR_PROFILE] TOTAL openradar_pd_process_frame time: {total_time:.4f}s (frame {openradar_pd_process_frame.frame_count})"
            )
        return {
            "range_doppler": np.array([]),
            "range_azimuth": range_azimuth,
            "x_pos": np.array([]),
            "y_pos": np.array([]),
            "z_pos": np.array([]),
            "velocities": np.array([]),
            "snrs": np.array([]),
            "cluster_labels": np.array([]),
        }

    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:, 0], pairs[:, 1]] - noise_floor[pairs[:, 0], pairs[:, 1]]

    """ 4 (Doppler Estimation) """
    step_start = time.perf_counter()

    # --- get peak indices
    # beamWeights should be selected based on the range indices from CFAR.
    dopplerFFTInput = radar_cube[:, :, ranges]
    beamWeights = beamWeights[:, ranges]

    # --- estimate doppler values
    # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
    # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
    dopplerFFTInput = np.einsum("ijk,jk->ik", dopplerFFTInput, beamWeights)
    logger.debug(f"[DEBUG] dopplerFFTInput shape: {dopplerFFTInput.shape}")
    assert dopplerFFTInput.shape[-1], "Doppler FFT input should not be empty"

    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:] >= adc_params.chirps / 2] -= adc_params.chirps

    doppler_estimation_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Doppler estimation: {doppler_estimation_time:.4f}s"
    )

    # --- convert bins to units
    step_start = time.perf_counter()
    ranges = ranges * adc_params.range_resolution
    azimuths = azimuths - (ANGLE_BINS // 2)
    dopplers = dopplerEst * adc_params.doppler_resolution
    snrs = snrs

    logger.debug(
        f"[DEBUG] Detected {ranges.shape} targets with ranges, azimuths, dopplers, snrs"
    )

    x_pos, y_pos = ranges_angles_to_xy(ranges, azimuths)
    z_pos = np.zeros_like(x_pos)  # Assuming targets are at ground level
    velocities = dopplers
    conversion_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Unit conversion and coordinate transform: {conversion_time:.4f}s"
    )

    # cluster_labels, n_clusters, n_noise = apply_radar_clustering(
    #     x_pos, y_pos, dopplers, eps=0.2, min_samples=2, velocity_weight=0.0
    # )

    cluster_labels = np.array([])
    # tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    # targetDescr, tNum = tracker.step()

    # print(f"[DEBUG] EKF tracking step completed with {tNum} targets")

    # x_pos = np.array([])
    # y_pos = np.array([])
    # velocities = np.array([])

    # for t, tid in zip(targetDescr, range(int(tNum[0]))):
    #     x, y, x_vel, y_vel = t.S[:4]
    #     x = -x
    #     # z_pos = 0
    #     velocity = np.sqrt(x_vel**2 + y_vel**2)
    #     x_pos = np.append(x_pos, x)
    #     y_pos = np.append(y_pos, y)
    #     velocities = np.append(velocities, velocity)

    # Profile logging every 10 frames
    if openradar_pd_process_frame.frame_count % 10 == 0:
        total_time = time.perf_counter() - function_start
        logger.info(
            f"    [RADAR_PROFILE] TOTAL openradar_pd_process_frame time: {total_time:.4f}s (frame {openradar_pd_process_frame.frame_count})"
        )
    else:
        total_time = time.perf_counter() - function_start
        logger.debug(f"[RADAR_PROFILE] TOTAL time: {total_time:.4f}s")

    return {
        "range_doppler": range_doppler,
        "range_azimuth": range_azimuth,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "z_pos": z_pos,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": cluster_labels,
    }


def openradar_rt_process_frame(frame, adc_params):
    from mmwave import dsp
    import time

    function_start = time.perf_counter()

    # Frame counter for profile logging every 10 frames
    if not hasattr(openradar_rt_process_frame, "frame_count"):
        openradar_rt_process_frame.frame_count = 0
    openradar_rt_process_frame.frame_count += 1

    logger.debug(
        f"openradar_rt_process_frame: Processing frame with shape {frame.shape}"
    )
    logger.debug(
        f"ADC Params - tx: {adc_params.tx}, rx: {adc_params.rx}, samples: {adc_params.samples}, chirps: {adc_params.chirps}, range_resolution: {adc_params.range_resolution}, doppler_resolution: {adc_params.doppler_resolution}"
    )

    # Extract actual dimensions from frame shape
    # Frame shape should be (chirps, actual_tx, rx, samples)
    if len(frame.shape) == 4:
        actual_chirps, actual_tx, actual_rx, actual_samples = frame.shape
        logger.debug(
            f"Frame dimensions: chirps={actual_chirps}, tx={actual_tx}, rx={actual_rx}, samples={actual_samples}"
        )
    else:
        logger.error(f"Unexpected frame shape: {frame.shape}")
        raise ValueError(
            f"Expected 4D frame shape, got {len(frame.shape)}D: {frame.shape}"
        )

    # Frame reshape
    step_start = time.perf_counter()
    frame = frame.reshape(
        adc_params.chirps * adc_params.tx, adc_params.rx, adc_params.samples
    )
    reshape_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Frame reshape: {reshape_time:.4f}s")

    # Range processing
    step_start = time.perf_counter()
    radar_cube = dsp.range_processing(frame, window_type_1d=dsp.utils.Window.HAMMING)
    range_processing_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Range processing: {range_processing_time:.4f}s")
    expected_shape = (
        adc_params.chirps * adc_params.tx,
        adc_params.rx,
        adc_params.samples,
    )

    assert (
        radar_cube.shape == expected_shape
    ), f"[ERROR] Radar cube shape mismatch! Expected: {expected_shape}, Actual: {radar_cube.shape}"

    logger.debug(f"Radar cube shape: {radar_cube.shape}")

    # Reshape radar cube from [loops*tx, rx, samples] to [loops, tx*rx, samples]
    step_start = time.perf_counter()
    radar_cube = radar_cube.reshape(
        adc_params.chirps, adc_params.tx * adc_params.rx, adc_params.samples
    )
    radar_cube_reshape_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Radar cube reshape: {radar_cube_reshape_time:.4f}s"
    )

    # Doppler Processing - creates Range-Doppler matrix
    step_start = time.perf_counter()
    # interleaved=True means that the data is interleaved across the transmit antennas
    det_matrix, aoa_input = dsp.doppler_processing(
        radar_cube,
        num_tx_antennas=adc_params.tx,
        clutter_removal_enabled=True,
        interleaved=False,
        window_type_2d=dsp.utils.Window.HAMMING,
    )
    doppler_processing_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Doppler processing: {doppler_processing_time:.4f}s"
    )

    logger.debug(f"Range-Doppler matrix shape: {det_matrix.shape}")
    logger.debug(f"AOA input shape: {aoa_input.shape}")

    # Add FFT shift to center zero velocity
    step_start = time.perf_counter()
    det_matrix = np.fft.fftshift(det_matrix, axes=1)  # Shift along Doppler axis
    aoa_input = np.fft.fftshift(aoa_input, axes=2)  # Shift along Doppler axis
    fft_shift_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] FFT shift: {fft_shift_time:.4f}s")

    logger.debug(f"Range-Doppler matrix shape: {det_matrix.shape}")
    logger.debug(f"AOA input shape: {aoa_input.shape}")

    # (4) Object Detection
    step_start = time.perf_counter()
    # --- CFAR, SNR is calculated as well.
    fft2d_sum = det_matrix.astype(np.int64)
    thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=fft2d_sum.T,
        l_bound=1.5,
        guard_len=4,
        noise_len=16,
    )

    thresholdRange, noiseFloorRange = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=fft2d_sum,
        l_bound=2.5,
        guard_len=4,
        noise_len=16,
    )

    thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T

    # Debug: Check threshold statistics
    logger.debug(
        f"Threshold Doppler - Min: {np.min(thresholdDoppler):.2f}, Max: {np.max(thresholdDoppler):.2f}"
    )
    logger.debug(
        f"Threshold Range - Min: {np.min(thresholdRange):.2f}, Max: {np.max(thresholdRange):.2f}"
    )
    logger.debug(
        f"Noise Floor Doppler - Min: {np.min(noiseFloorDoppler):.2f}, Max: {np.max(noiseFloorDoppler):.2f}"
    )
    logger.debug(
        f"Noise Floor Range - Min: {np.min(noiseFloorRange):.2f}, Max: {np.max(noiseFloorRange):.2f}"
    )
    det_doppler_mask = det_matrix > thresholdDoppler
    det_range_mask = det_matrix > thresholdRange

    # Get indices of detected peaks
    full_mask = det_doppler_mask & det_range_mask
    det_peaks_indices = np.argwhere(full_mask == True)

    # peakVals and SNR calculation
    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    cfar_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] CFAR detection and peak extraction: {cfar_time:.4f}s"
    )

    # Data structure creation
    step_start = time.perf_counter()

    dtype_location = "(" + str(adc_params.tx) + ",)<f4"
    dtype_detObj2D = np.dtype(
        {
            "names": ["rangeIdx", "dopplerIdx", "peakVal", "location", "SNR"],
            "formats": ["<i4", "<i4", "<f4", dtype_location, "<f4"],
        }
    )
    detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
    detObj2DRaw["rangeIdx"] = det_peaks_indices[:, 0].squeeze()
    detObj2DRaw["dopplerIdx"] = det_peaks_indices[:, 1].squeeze()
    detObj2DRaw["peakVal"] = peakVals.flatten()
    detObj2DRaw["SNR"] = snr.flatten()
    data_structure_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Data structure creation: {data_structure_time:.4f}s"
    )

    logger.debug(f"detObj2DRaw['rangeIdx'] = {len(detObj2DRaw['rangeIdx'])}")
    logger.debug(f"detObj2DRaw['peakVal'] = {len(detObj2DRaw['peakVal'])}")

    # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
    step_start = time.perf_counter()
    detObj2D = dsp.prune_to_peaks(
        detObj2DRaw,
        det_matrix,
        adc_params.chirps,
        reserve_neighbor=True,
    )
    prune_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Peak pruning: {prune_time:.4f}s")

    logger.debug(
        f"detObj2D['rangeIdx'] after peak pruning = {len(detObj2D['rangeIdx'])}"
    )

    # --- Peak Grouping
    step_start = time.perf_counter()
    detObj2D = dsp.peak_grouping_along_doppler(detObj2D, det_matrix, adc_params.chirps)
    peak_grouping_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Peak grouping: {peak_grouping_time:.4f}s")

    logger.debug(
        f"detObj2D['rangeIdx'] after peak grouping = {len(detObj2D['rangeIdx'])}"
    )

    SNRThresholds2 = np.array([[2, 30], [10, 20], [35, 16.0]])
    # SNRThresholds2 = np.array([[2, 0.5], [10, 6.5], [35, 16.0]])
    # peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
    peakValThresholds2 = np.array([[4, 100], [1, 400], [500, 0]])

    step_start = time.perf_counter()
    detObj2D = dsp.range_based_pruning(
        detObj2D,
        SNRThresholds2,
        peakValThresholds2,
        adc_params.samples,
        # 40,
        0.5,
        adc_params.range_resolution,
    )
    range_based_pruning_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Range-based pruning: {range_based_pruning_time:.4f}s"
    )

    logger.debug(f"aoa_input shape: {aoa_input.shape}")
    logger.debug(f"detObj2D rangeIdx: {detObj2D['rangeIdx']}")
    logger.debug(f"detObj2D dopplerIdx: {detObj2D['dopplerIdx']}")
    logger.debug(f"detObj2D peakVal: {detObj2D['peakVal']}")

    step_start = time.perf_counter()
    azimuthInput = aoa_input[detObj2D["rangeIdx"], :, detObj2D["dopplerIdx"]]
    azimuth_input_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Azimuth input preparation: {azimuth_input_time:.4f}s"
    )

    logger.debug(f"Azimuth input shape: {azimuthInput.shape}")

    step_start = time.perf_counter()
    x, y, z = dsp.naive_xyz(
        azimuthInput.T,
        num_tx=adc_params.tx,
        num_rx=adc_params.rx,
    )
    naive_xyz_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Naive XYZ calculation: {naive_xyz_time:.4f}s")

    logger.debug(f"naive_xyz output shapes: x={x.shape}, y={y.shape}, z={z.shape}")

    step_start = time.perf_counter()
    xyzVecN = np.zeros((3, x.shape[0]))

    xyzVecN[0] = x * adc_params.range_resolution * detObj2D["rangeIdx"]
    xyzVecN[1] = y * adc_params.range_resolution * detObj2D["rangeIdx"]
    xyzVecN[2] = z * adc_params.range_resolution * detObj2D["rangeIdx"]

    # print(f"xyzVecN2 (x, y, z): {xyzVecN}")

    Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(
        azimuthInput,
        detObj2D["rangeIdx"],
        adc_params.range_resolution,
        method="Bartlett",
    )

    cluster_labels = np.array([])
    snrs = detObj2D["SNR"]  # Use the SNR values from the detected objects
    velocities = (
        detObj2D["dopplerIdx"] * adc_params.doppler_resolution
    )  # Convert Doppler bins to velocities
    final_processing_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Final coordinate processing and beamforming: {final_processing_time:.4f}s"
    )

    # Profile logging every 10 frames
    if openradar_rt_process_frame.frame_count % 10 == 0:
        total_time = time.perf_counter() - function_start
        logger.info(
            f"    [RADAR_PROFILE] TOTAL openradar_rt_process_frame time: {total_time:.4f}s (frame {openradar_rt_process_frame.frame_count})"
        )

    return {
        "range_doppler": det_matrix.T,
        "range_azimuth": np.array([]),  # Not computed in this function
        "x_pos": xyzVec[0],
        "y_pos": xyzVec[1],
        "z_pos": xyzVec[2],
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": cluster_labels,
    }


def pyradar_process_frame(frame, adc_params, doa_method="MUSIC", IS_INDOOR=True):
    """
    Process radar frame using PyRadar methods with FFT processing, CFAR detection, and DOA estimation.

    This function performs range FFT, Doppler FFT, azimuth FFT processing, followed by CFAR detection
    for target identification and Direction of Arrival (DOA) estimation using various beamforming methods.

    Args:
        frame (numpy.ndarray): Input radar frame data with shape (chirps, tx, rx, samples).
                              Expected to be complex-valued data from radar ADC.
        adc_params (ADCParams): ADC parameters object containing:
                               - tx: Number of transmit antennas
                               - rx: Number of receive antennas
                               - samples: Number of range samples
                               - chirps: Number of chirps per frame
                               - range_resolution: Range resolution in meters
                               - doppler_resolution: Doppler resolution in m/s
        doa_method (str, optional): DOA estimation method to use. Tested options are:
                                   - "Fourier": Classical Fourier-based beamforming
                                   - "Capon": Minimum variance distortionless response (MVDR)
                                   - "MUSIC": Multiple signal classification algorithm
                                   Defaults to "MUSIC".
        IS_INDOOR (bool, optional): Flag indicating indoor vs outdoor environment.
                                   Affects windowing parameters, clutter removal, and CFAR thresholds.
                                   Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "range_doppler" (numpy.ndarray): Range-doppler map with shape (chirps, tx*rx, samples)
            - "range_azimuth" (numpy.ndarray): Range-azimuth map with shape (chirps, tx*rx, samples)
            - "x_pos" (numpy.ndarray): X coordinates of detected targets in meters
            - "y_pos" (numpy.ndarray): Y coordinates of detected targets in meters
            - "z_pos" (numpy.ndarray): Z coordinates of detected targets in meters (zeros for ground-level targets)
            - "velocities" (numpy.ndarray): Doppler velocities of detected targets in m/s
            - "snrs" (numpy.ndarray): Signal-to-noise ratios of detected targets (default values)
            - "cluster_labels" (numpy.ndarray): DBSCAN cluster labels for detected targets

    Note:
        The function reshapes the input frame internally from (chirps, tx, rx, samples) to
        (chirps, tx*rx, samples) for processing. For outdoor scenarios, static clutter removal
        is performed by subtracting the mean across chirps.
    """
    from pyapril.hitProcessor import target_DOA_estimation
    import time

    function_start = time.perf_counter()
    logger.debug(
        f"pyradar_process_frame: Processing frame with shape {frame.shape}, IS_INDOOR={IS_INDOOR}, doa_method={doa_method}"
    )
    logger.debug(
        f"ADC Params - tx: {adc_params.tx}, rx: {adc_params.rx}, samples: {adc_params.samples}, chirps: {adc_params.chirps}, range_resolution: {adc_params.range_resolution}, doppler_resolution: {adc_params.doppler_resolution}"
    )

    # Range FFT processing
    step_start = time.perf_counter()
    if IS_INDOOR:
        range_cube = range_cube_fft(frame, window_type="kaiser", beta=4)
    else:
        range_cube = range_cube_fft(frame, window_type="kaiser", beta=4)
        mean = range_cube.mean(0)
        range_cube = range_cube - mean
    range_fft_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Range FFT processing: {range_fft_time:.4f}s")

    # Doppler FFT processing
    step_start = time.perf_counter()
    if IS_INDOOR:
        range_doppler = range_doppler_fft(range_cube, window_type="kaiser")
    else:
        range_doppler = range_doppler_fft(range_cube, window_type="kaiser", beta=8)
    doppler_fft_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Doppler FFT processing: {doppler_fft_time:.4f}s")

    # Azimuth FFT processing
    step_start = time.perf_counter()
    range_azimuth = range_azimuth_fft(range_doppler)
    azimuth_fft_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Azimuth FFT processing: {azimuth_fft_time:.4f}s")

    logger.debug(f"Range-Doppler shape: {range_doppler.shape}")
    logger.debug(f"Range-Azimuth shape: {range_azimuth.shape}")

    # CFAR detection
    step_start = time.perf_counter()
    if IS_INDOOR:
        CFAR_PARAMS = [4, 2, 1, 1]
        THRESHOLD = 9
    else:
        CFAR_PARAMS = [4, 2, 1, 1]
        THRESHOLD = 5

    hit_rd = pyapril_cfar(range_doppler, cfar_params=CFAR_PARAMS, threshold=THRESHOLD)
    cfar_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR detection: {cfar_time:.4f}s")

    # Prepare data for DOA estimation
    step_start = time.perf_counter()
    range_doppler_transposed = np.transpose(range_doppler, (1, 0, 2))
    data_prep_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Data preparation for DOA: {data_prep_time:.4f}s")

    logger.debug(f"Hit RD shape: {hit_rd.shape}")
    logger.debug(f"Range-Doppler Transposed shape: {range_doppler_transposed.shape}")

    # DOA estimation
    step_start = time.perf_counter()
    d = 0.5
    Rx = 8
    array_alignment = np.arange(0, Rx, 1) * d

    logger.debug(f"Processing {doa_method} DOA estimation...")
    estimated_angle_bins = target_DOA_estimation(
        range_doppler_transposed, hit_rd, doa_method, array_alignment
    )
    doa_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] {doa_method} DOA estimation: {doa_time:.4f}s")
    logger.debug(
        f"{doa_method} completed successfully. Found {len(estimated_angle_bins)} angle estimates."
    )

    # Angle conversion and coordinate transformation
    step_start = time.perf_counter()
    angles = pyapril_convert_to_angles(estimated_angle_bins)
    logger.debug(f"{doa_method} angles: {angles}")

    ranges, velocities = rd_hit_bin_to_value(hit_rd, adc_params)
    x_pos, y_pos = ranges_angles_to_xy(ranges, angles)
    z_pos = np.zeros_like(x_pos)  # Assuming targets are at ground level
    snrs = np.ones_like(x_pos)  # Default SNR values for pyradar method
    coordinate_transform_time = time.perf_counter() - step_start
    logger.debug(
        f"    [RADAR_PROFILE] Coordinate transformation: {coordinate_transform_time:.4f}s"
    )

    # Clustering (currently disabled)
    step_start = time.perf_counter()
    if IS_INDOOR:
        EPS = 0.2
        SAMPLES = 2
        VELOCITY_WEIGHT = 0.0
    else:
        EPS = 0.05
        SAMPLES = 2
        VELOCITY_WEIGHT = 0.0

    # cluster_labels, n_clusters, n_noise = apply_radar_clustering(
    #     x_pos,
    #     y_pos,
    #     velocities,
    #     eps=EPS,
    #     min_samples=SAMPLES,
    #     velocity_weight=VELOCITY_WEIGHT,
    # )
    cluster_labels = np.array([])
    clustering_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Clustering (disabled): {clustering_time:.4f}s")

    total_time = time.perf_counter() - function_start
    logger.info(
        f"    [RADAR_PROFILE] TOTAL pyradar_process_frame time: {total_time:.4f}s"
    )

    return {
        "range_doppler": range_doppler,
        "range_azimuth": range_azimuth,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "z_pos": z_pos,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": cluster_labels,
    }


def custom_process_frame(frame, adc_params):
    """
    Process radar frame using PyRadar methods with FFT processing, CFAR detection, and DOA estimation.

    This function performs range FFT, Doppler FFT, azimuth FFT processing, followed by CFAR detection
    for target identification and Direction of Arrival (DOA) estimation using various beamforming methods.

    Args:
        frame (numpy.ndarray): Input radar frame data with shape (chirps, tx, rx, samples).
                              Expected to be complex-valued data from radar ADC.
        adc_params (ADCParams): ADC parameters object containing:
                               - tx: Number of transmit antennas
                               - rx: Number of receive antennas
                               - samples: Number of range samples
                               - chirps: Number of chirps per frame
                               - range_resolution: Range resolution in meters
                               - doppler_resolution: Doppler resolution in m/s
        doa_method (str, optional): DOA estimation method to use. Tested options are:
                                   - "Fourier": Classical Fourier-based beamforming
                                   - "Capon": Minimum variance distortionless response (MVDR)
                                   - "MUSIC": Multiple signal classification algorithm
                                   Defaults to "MUSIC".
        IS_INDOOR (bool, optional): Flag indicating indoor vs outdoor environment.
                                   Affects windowing parameters, clutter removal, and CFAR thresholds.
                                   Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "range_doppler" (numpy.ndarray): Range-doppler map with shape (chirps, tx*rx, samples)
            - "range_azimuth" (numpy.ndarray): Range-azimuth map with shape (chirps, tx*rx, samples)
            - "x_pos" (numpy.ndarray): X coordinates of detected targets in meters
            - "y_pos" (numpy.ndarray): Y coordinates of detected targets in meters
            - "z_pos" (numpy.ndarray): Z coordinates of detected targets in meters (zeros for ground-level targets)
            - "velocities" (numpy.ndarray): Doppler velocities of detected targets in m/s
            - "snrs" (numpy.ndarray): Signal-to-noise ratios of detected targets (default values)
            - "cluster_labels" (numpy.ndarray): DBSCAN cluster labels for detected targets

    Note:
        The function reshapes the input frame internally from (chirps, tx, rx, samples) to
        (chirps, tx*rx, samples) for processing. For outdoor scenarios, static clutter removal
        is performed by subtracting the mean across chirps.
    """
    from pyapril.hitProcessor import target_DOA_estimation
    import time

    function_start = time.perf_counter()
    logger.debug(f"pyradar_process_frame: Processing frame with shape {frame.shape}")

    logger.debug(
        f"ADC Params - tx: {adc_params.tx}, rx: {adc_params.rx}, samples: {adc_params.samples}, chirps: {adc_params.chirps}, range_resolution: {adc_params.range_resolution}, doppler_resolution: {adc_params.doppler_resolution}"
    )

    # Range FFT processing
    step_start = time.perf_counter()

    # range_cube = range_cube_fft(frame, window_type="kaiser", beta=4)
    range_cube = range_cube_fft(frame)
    mean = range_cube.mean(0)
    range_cube = range_cube - mean

    range_fft_time = time.perf_counter() - step_start

    logger.debug(f"    [RADAR_PROFILE] Range FFT processing: {range_fft_time:.4f}s")

    # Doppler FFT processing
    step_start = time.perf_counter()

    # range_doppler = range_doppler_fft(range_cube, window_type="kaiser", beta=8)
    range_doppler = range_doppler_fft(range_cube)

    doppler_fft_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Doppler FFT processing: {doppler_fft_time:.4f}s")

    # Azimuth FFT processing
    step_start = time.perf_counter()

    range_azimuth = range_azimuth_fft(range_doppler)

    azimuth_fft_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Azimuth FFT processing: {azimuth_fft_time:.4f}s")

    logger.debug(f"Range-Doppler shape: {range_doppler.shape}")
    logger.debug(f"Range-Azimuth shape: {range_azimuth.shape}")

    # CFAR detection
    step_start = time.perf_counter()

    CFAR_PARAMS = [4, 2, 1, 1]
    THRESHOLD = 5

    hit_rd = pyapril_cfar(range_doppler, cfar_params=CFAR_PARAMS, threshold=THRESHOLD)
    cfar_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] CFAR detection: {cfar_time:.4f}s")

    # Prepare data for DOA estimation
    step_start = time.perf_counter()
    range_doppler_transposed = np.transpose(range_doppler, (1, 0, 2))
    data_prep_time = time.perf_counter() - step_start
    logger.debug(f"    [RADAR_PROFILE] Data preparation for DOA: {data_prep_time:.4f}s")

    logger.debug(f"Hit RD shape: {hit_rd.shape}")
    logger.debug(f"Range-Doppler Transposed shape: {range_doppler_transposed.shape}")

    # DOA estimation
    # step_start = time.perf_counter()
    # d = 0.5
    # Rx = 8
    # array_alignment = np.arange(0, Rx, 1) * d

    # logger.debug(f"Processing {doa_method} DOA estimation...")
    # estimated_angle_bins = target_DOA_estimation(
    #     range_doppler_transposed, hit_rd, doa_method, array_alignment
    # )
    # doa_time = time.perf_counter() - step_start
    # logger.debug(f"    [RADAR_PROFILE] {doa_method} DOA estimation: {doa_time:.4f}s")
    # logger.debug(
    #     f"{doa_method} completed successfully. Found {len(estimated_angle_bins)} angle estimates."
    # )

    # # Angle conversion and coordinate transformation
    # step_start = time.perf_counter()
    # angles = pyapril_convert_to_angles(estimated_angle_bins)
    # logger.debug(f"{doa_method} angles: {angles}")

    # ranges, velocities = rd_hit_bin_to_value(hit_rd, adc_params)
    # x_pos, y_pos = ranges_angles_to_xy(ranges, angles)
    # z_pos = np.zeros_like(x_pos)  # Assuming targets are at ground level
    # snrs = np.ones_like(x_pos)  # Default SNR values for pyradar method
    # coordinate_transform_time = time.perf_counter() - step_start
    # logger.debug(
    #     f"    [RADAR_PROFILE] Coordinate transformation: {coordinate_transform_time:.4f}s"
    # )

    # Clustering (currently disabled)
    # step_start = time.perf_counter()
    # if IS_INDOOR:
    #     EPS = 0.2
    #     SAMPLES = 2
    #     VELOCITY_WEIGHT = 0.0
    # else:
    #     EPS = 0.05
    #     SAMPLES = 2
    #     VELOCITY_WEIGHT = 0.0

    # cluster_labels, n_clusters, n_noise = apply_radar_clustering(
    #     x_pos,
    #     y_pos,
    #     velocities,
    #     eps=EPS,
    #     min_samples=SAMPLES,
    #     velocity_weight=VELOCITY_WEIGHT,
    # )
    cluster_labels = np.array([])
    x_pos = np.array([])
    y_pos = np.array([])
    z_pos = np.array([])
    velocities = np.array([])
    snrs = np.array([])  # Use the SNR values from the detected objects
    # clustering_time = time.perf_counter() - step_start
    # logger.debug(f"    [RADAR_PROFILE] Clustering (disabled): {clustering_time:.4f}s")

    # total_time = time.perf_counter() - function_start
    # logger.info(
    #     f"    [RADAR_PROFILE] TOTAL pyradar_process_frame time: {total_time:.4f}s"
    # )

    return {
        "range_doppler": range_doppler,
        "range_azimuth": range_azimuth,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "z_pos": z_pos,
        "velocities": velocities,
        "snrs": snrs,
        "cluster_labels": cluster_labels,
    }
