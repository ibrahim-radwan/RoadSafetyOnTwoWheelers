#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include "config.hpp"

namespace radar
{

// Type aliases for radar signal processing
using Complex = std::complex<double>;
using ComplexFloat = std::complex<float>;

// Matrix types using std::vector for now
// Format: [rows][cols] = row-major storage
template <typename T>
using Matrix2D = std::vector<std::vector<T>>;

using ComplexMatrix = Matrix2D<Complex>;
using ComplexMatrixF = Matrix2D<ComplexFloat>;
using RealMatrix = Matrix2D<double>;
using RealMatrixF = Matrix2D<float>;

// Vector types
using ComplexVector = std::vector<Complex>;
using ComplexVectorF = std::vector<ComplexFloat>;
using RealVector = std::vector<double>;
using RealVectorF = std::vector<float>;

// Raw data types (for file I/O)
using RawDataVector = std::vector<int16_t>;

/**
 * Radar frame data structure optimized for radar signal processing
 * Uses standard C++ containers for maximum compatibility
 */
class RadarFrame
{
  private:
    RawDataVector _raw_data;      // Raw int16 data from file
    ComplexMatrix _complex_data;  // Processed complex data
    double _timestamp;
    size_t _frame_number;
    bool _is_processed = false;

    // Dimensions
    size_t _chirps = 0;
    size_t _tx_antennas = 0;
    size_t _rx_antennas = 0;
    size_t _samples = 0;

    // ADC parameters for processing
    std::shared_ptr<AdcParams> _adc_params;

  public:
    /**
     * Constructor for raw frame data
     * @param timestamp Frame timestamp
     * @param raw_data Raw int16 data from radar
     * @param frame_number Sequential frame number
     * @param adc_params ADC configuration parameters
     */
    RadarFrame(double timestamp, const RawDataVector& raw_data,
               size_t frame_number, std::shared_ptr<AdcParams> adc_params);

    /**
     * Constructor for already processed complex data
     * @param timestamp Frame timestamp
     * @param complex_data Processed complex radar data
     * @param frame_number Sequential frame number
     */
    RadarFrame(double timestamp, const ComplexMatrix& complex_data,
               size_t frame_number);

    // Getters
    double getTimestamp() const
    {
        return _timestamp;
    }
    size_t getFrameNumber() const
    {
        return _frame_number;
    }
    const RawDataVector& getRawData() const
    {
        return _raw_data;
    }

    // Data access with lazy processing
    const ComplexMatrix& getComplexData();
    ComplexMatrix getComplexDataCopy() const;

    // Frame dimensions
    size_t getChirps() const
    {
        return _chirps;
    }
    size_t getTxAntennas() const
    {
        return _tx_antennas;
    }
    size_t getRxAntennas() const
    {
        return _rx_antennas;
    }
    size_t getSamples() const
    {
        return _samples;
    }

    // Radar processing operations (basic implementations)
    ComplexMatrix rangeFFT() const;
    ComplexMatrix dopplerFFT() const;
    RealMatrix rangeDopplerMap() const;
    RealMatrix rangeAzimuthMap() const;

    // Reshape operations for radar processing
    ComplexMatrix reshapeForProcessing() const;
    ComplexMatrix reshapeToChirpTxRxSamples() const;

    // Validation
    bool isValid() const;
    size_t getExpectedDataSize() const;

    // String representation
    std::string toString() const;

  private:
    // Internal processing methods
    void processRawData();
    ComplexMatrix convertRawToComplex() const;
    void validateDimensions() const;
    void calculateDimensions();
};

/**
 * Factory class for creating RadarFrame objects
 */
class RadarFrameFactory
{
  public:
    /**
     * Create frame from file data
     * @param timestamp Frame timestamp
     * @param file_data Raw data read from file
     * @param frame_number Sequential frame number
     * @param adc_params ADC configuration
     * @return Unique pointer to RadarFrame
     */
    static std::unique_ptr<RadarFrame> createFromFileData(
        double timestamp, const std::vector<int16_t>& file_data,
        size_t frame_number, std::shared_ptr<AdcParams> adc_params);

    /**
     * Create frame from raw data vector
     * @param timestamp Frame timestamp
     * @param raw_data Raw data as vector
     * @param frame_number Sequential frame number
     * @param adc_params ADC configuration
     * @return Unique pointer to RadarFrame
     */
    static std::unique_ptr<RadarFrame> createFromRawData(
        double timestamp, const RawDataVector& raw_data, size_t frame_number,
        std::shared_ptr<AdcParams> adc_params);
};

}  // namespace radar
