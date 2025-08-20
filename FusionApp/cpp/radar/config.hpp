#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace radar
{

class AdcParams
{
  public:
    // Raw parameters (direct from config)
    double startFreq = 0.0;              // GHz
    double idleTime = 0.0;               // microseconds
    double adc_valid_start_time = 0.0;   // microseconds
    double rampEndTime = 0.0;            // microseconds
    double freq_slope = 0.0;             // MHz/us
    double txStartTime = 0.0;            // microseconds
    int samples = 0;
    int sample_rate = 0;                 // ksps
    int rx = 0;
    int tx = 0;
    int IQ = 2;
    int bytes = 2;
    int chirps = 0;
    double frame_periodicity = 0.0;      // seconds

    // Derived parameters (computed)
    double range_resolution = 0.0;       // meters
    double chirp_bandwidth = 0.0;        // Hz
    double doppler_resolution = 0.0;     // m/s
    int max_azimuth = 60;                // degrees
    double max_range = 0.0;              // meters
    double max_doppler = 0.0;            // m/s
    std::vector<double> angle_bins;      // degrees
    std::vector<double> range_bins;      // meters
    std::vector<double> range_doppler_extents; // [minDoppler, maxDoppler, minRange, maxRange]
    std::vector<double> range_azimuth_extents; // [minAz, maxAz, minRange, maxRange]

    AdcParams() = default;
    explicit AdcParams(const std::string& config_file_path);
    AdcParams(const AdcParams& other);

    // Compute all derived parameters from the raw fields
    void finalize();

    std::string toString() const;
};

struct CfgParams
{
    int txAntMask = 0;
    int rxAntMask = 0;
    int numTxChan = 0;
    int numRxChan = 0;
    int lvdsBW = 0;
    int numlaneEn = 0;

    std::string toString() const;
};

class RadarConfig
{
  public:
    explicit RadarConfig(const std::string& config_file_path);

    // Property accessors
    const AdcParams& getAdcParams() const
    {
        return _adc_params;
    }
    const CfgParams& getCfgParams() const
    {
        return _cfg_params;
    }
    int getLVDSDataSizePerChirp() const
    {
        return _lvds_data_size_per_chirp;
    }
    double getMaxSendBytesPerChirp() const
    {
        return _max_send_bytes_per_chirp;
    }

    std::string toString() const;

  private:
    std::string _config_file_path;
    AdcParams _adc_params;
    CfgParams _cfg_params;
    int _lvds_data_size_per_chirp = 0;
    double _max_send_bytes_per_chirp = 0.0;

    // Chirp configuration buffers
    std::vector<int> _chirp_start_idx_buf;
    std::vector<int> _chirp_end_idx_buf;
    std::vector<int> _profile_id_cpcfg_buf;
    std::vector<int> _tx_enable_buf;

    // Private parsing methods
    void parseConfigFile();
    void parseChannelConfig(const std::string& key, const std::string& value);
    void parseProfileConfig(const std::string& key, const std::string& value,
                            int& cur_profile_id);
    void parseChirpConfig(const std::string& key, const std::string& value);
    void parseFrameConfig(const std::string& key, const std::string& value,
                          int& chirp_start_idx_fcf, int& chirp_end_idx_fcf,
                          int& loop_count);

    // Helper methods
    int countBitsSet(int mask, int max_bits) const;
    int mapDataRateToMbps(int data_rate) const;
    void calculateDerivedParams();
    void validateConfig() const;

    // Utility methods
    std::string trim(const std::string& str) const;
    std::pair<std::string, std::string> parseLine(
        const std::string& line) const;
};

}  // namespace radar
