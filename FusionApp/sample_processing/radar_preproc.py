import numpy as np
from sample_processing.radar_params import ADCParams


def preprocess_frame_from_awr2243(sample_path, adc_params: ADCParams) -> np.ndarray:
    """
    Preprocesses raw radar frame data from AWR2243 sensor.

    Args:
        sample_path: Path to binary file containing raw radar data
        adc_params: Object with attributes (chirps, tx, samples, IQ, rx) defining frame dimensions

    Returns:
        numpy.ndarray: Complex-valued radar frame with shape (chirps, tx, rx, samples)
    """
    dca_frame = np.fromfile(sample_path, dtype=np.int16)
    # The data is organized as:
    # [chirps, tx, adc_samples, IQ, rx]

    # Type assertions to ensure attributes are initialized
    frame = np.reshape(
        dca_frame,
        (
            adc_params.chirps,
            adc_params.tx,
            adc_params.samples,
            adc_params.IQ,
            adc_params.rx,
        ),
    )

    frame = np.transpose(frame, (0, 1, 4, 2, 3))
    print(f"Step 1 frame shape: {frame.shape}")

    complex_frame = (1j * frame[..., 1] + frame[..., 0]).astype(np.complex64)  # I first

    print(f"Step 2 complex_frame shape: {complex_frame.shape}")

    assert complex_frame.shape == (
        adc_params.chirps,
        adc_params.tx,
        adc_params.rx,
        adc_params.samples,
    ), f"complex_frame shape mismatch! Expected: {(adc_params.chirps, adc_params.tx, adc_params.rx, adc_params.samples)}, Actual: {complex_frame.shape}"

    print(f"Step 5 frames_data shape: {complex_frame.shape}")

    return complex_frame
