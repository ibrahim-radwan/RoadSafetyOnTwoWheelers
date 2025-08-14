#include "radardata.hpp"
#include <sstream>
#include "exceptions.hpp"

namespace radar
{

// RadarFrame constructor for raw data
RadarFrame::RadarFrame(double timestamp, const RawDataVector& raw_data,
                       size_t frame_number,
                       std::shared_ptr<AdcParams> adc_params)
    : _raw_data(raw_data),
      _timestamp(timestamp),
      _frame_number(frame_number),
      _adc_params(adc_params)
{
    if (!_adc_params)
    {
        throw ConfigException("ADC parameters cannot be null");
    }

    calculateDimensions();
    validateDimensions();
}

// RadarFrame constructor for complex data
RadarFrame::RadarFrame(double timestamp, const ComplexMatrix& complex_data,
                       size_t frame_number)
    : _complex_data(complex_data),
      _timestamp(timestamp),
      _frame_number(frame_number),
      _is_processed(true)
{
    if (complex_data.empty())
    {
        throw InvalidFrameException("Complex data cannot be empty");
    }

    _chirps = complex_data.size();
    _samples = complex_data.empty() ? 0 : complex_data[0].size();
}

const ComplexMatrix& RadarFrame::getComplexData()
{
    if (!_is_processed)
    {
        processRawData();
    }
    return _complex_data;
}

ComplexMatrix RadarFrame::getComplexDataCopy() const
{
    if (!_is_processed)
    {
        return convertRawToComplex();
    }
    return _complex_data;
}

bool RadarFrame::isValid() const
{
    if (!_adc_params && !_is_processed)
    {
        return false;
    }

    if (_is_processed)
    {
        return !_complex_data.empty() && !_complex_data[0].empty();
    }

    return _raw_data.size() == getExpectedDataSize();
}

size_t RadarFrame::getExpectedDataSize() const
{
    if (!_adc_params)
    {
        return 0;
    }

    return _chirps * _tx_antennas * _rx_antennas * _samples;
}

std::string RadarFrame::toString() const
{
    std::ostringstream oss;
    oss << "RadarFrame{";
    oss << "timestamp=" << _timestamp << ", ";
    oss << "frame_number=" << _frame_number << ", ";
    oss << "chirps=" << _chirps << ", ";
    oss << "tx_antennas=" << _tx_antennas << ", ";
    oss << "rx_antennas=" << _rx_antennas << ", ";
    oss << "samples=" << _samples << ", ";
    oss << "raw_data_size=" << _raw_data.size() << ", ";
    oss << "is_processed=" << (_is_processed ? "true" : "false");
    oss << "}";
    return oss.str();
}

void RadarFrame::processRawData()
{
    if (_is_processed)
    {
        return;
    }

    _complex_data = convertRawToComplex();
    _is_processed = true;
}

ComplexMatrix RadarFrame::convertRawToComplex() const
{
    if (_raw_data.empty())
    {
        throw InvalidFrameException("Raw data is empty");
    }

    // Convert raw int16 data to complex format
    // Assuming IQ interleaved format: I1, Q1, I2, Q2, ...
    ComplexMatrix result(_chirps * _tx_antennas);

    for (size_t chirp_tx = 0; chirp_tx < _chirps * _tx_antennas; ++chirp_tx)
    {
        result[chirp_tx].reserve(_rx_antennas * _samples);

        for (size_t rx = 0; rx < _rx_antennas; ++rx)
        {
            for (size_t sample = 0; sample < _samples; ++sample)
            {
                // Calculate index in raw data
                size_t base_idx =
                    (chirp_tx * _rx_antennas + rx) * _samples * 2 + sample * 2;

                if (base_idx + 1 >= _raw_data.size())
                {
                    throw InvalidFrameException(
                        "Insufficient raw data for conversion");
                }

                // Create complex number from I and Q components
                double i_val = static_cast<double>(_raw_data[base_idx]);
                double q_val = static_cast<double>(_raw_data[base_idx + 1]);
                result[chirp_tx].emplace_back(i_val, q_val);
            }
        }
    }

    return result;
}

void RadarFrame::validateDimensions() const
{
    if (_chirps == 0 || _tx_antennas == 0 || _rx_antennas == 0 || _samples == 0)
    {
        throw InvalidFrameException("Frame dimensions cannot be zero");
    }

    if (!_is_processed && _raw_data.size() != getExpectedDataSize())
    {
        std::ostringstream oss;
        oss << "Raw data size mismatch: expected " << getExpectedDataSize()
            << ", got " << _raw_data.size();
        throw InvalidFrameException(oss.str());
    }
}

void RadarFrame::calculateDimensions()
{
    if (!_adc_params)
    {
        return;
    }

    _chirps = _adc_params->chirps;
    _tx_antennas = _adc_params->tx;
    _rx_antennas = _adc_params->rx;
    _samples = _adc_params->samples;
}

// Basic placeholder implementations for radar processing
ComplexMatrix RadarFrame::rangeFFT() const
{
    // TODO: Implement proper FFT when Eigen/FFTW is available
    // For now, return copy of complex data
    return getComplexDataCopy();
}

ComplexMatrix RadarFrame::dopplerFFT() const
{
    // TODO: Implement proper Doppler FFT
    return getComplexDataCopy();
}

RealMatrix RadarFrame::rangeDopplerMap() const
{
    // TODO: Implement range-doppler map generation
    RealMatrix result(_chirps);
    for (size_t i = 0; i < _chirps; ++i)
    {
        result[i].resize(_samples, 0.0);
    }
    return result;
}

RealMatrix RadarFrame::rangeAzimuthMap() const
{
    // TODO: Implement range-azimuth map generation
    RealMatrix result(_rx_antennas);
    for (size_t i = 0; i < _rx_antennas; ++i)
    {
        result[i].resize(_samples, 0.0);
    }
    return result;
}

ComplexMatrix RadarFrame::reshapeForProcessing() const
{
    // Reshape to (chirps*tx, rx*samples) for processing
    return getComplexDataCopy();
}

ComplexMatrix RadarFrame::reshapeToChirpTxRxSamples() const
{
    // Reshape back to original format
    return getComplexDataCopy();
}

// RadarFrameFactory implementations
std::unique_ptr<RadarFrame> RadarFrameFactory::createFromFileData(
    double timestamp, const std::vector<int16_t>& file_data,
    size_t frame_number, std::shared_ptr<AdcParams> adc_params)
{
    RawDataVector raw_data(file_data.begin(), file_data.end());
    return std::make_unique<RadarFrame>(timestamp, raw_data, frame_number,
                                        adc_params);
}

std::unique_ptr<RadarFrame> RadarFrameFactory::createFromRawData(
    double timestamp, const RawDataVector& raw_data, size_t frame_number,
    std::shared_ptr<AdcParams> adc_params)
{
    return std::make_unique<RadarFrame>(timestamp, raw_data, frame_number,
                                        adc_params);
}

}  // namespace radar
