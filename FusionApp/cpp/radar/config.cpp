#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace radar
{

// --- AdcParams: derived computations and copy constructor ---
AdcParams::AdcParams(const AdcParams& other) = default;

AdcParams::AdcParams(const std::string& config_file_path)
{
    // Build from a temporary RadarConfig to reuse parsing
    RadarConfig cfg(config_file_path);
    *this = cfg.getAdcParams();
    this->finalize();
}

void AdcParams::finalize()
{
    // Compute derived parameters analogous to Python ADCParams
    // range_resolution and chirp_bandwidth
    if (samples > 0 && sample_rate > 0)
    {
        // Python uses dsp.range_resolution(samples, sample_rate, freq_slope)
        // Here we approximate bandwidth from sample_rate and freq_slope
        // chirp_bandwidth (Hz) = freq_slope(MHz/us) * 1e6 * rampEndTime(us) * 1e-6 = freq_slope * rampEndTime (MHz)
        // Convert to Hz
        chirp_bandwidth = freq_slope * 1e6 * (rampEndTime * 1e-6);

        // Simple approximation similar to mmwave.dsp: range_resolution ~ c / (2*BW)
        constexpr double c = 299792458.0;
        if (chirp_bandwidth > 0)
        {
            range_resolution = c / (2.0 * chirp_bandwidth);
        }
    }

    // doppler_resolution approximation based on Python signature
    // dsp.doppler_resolution(chirp_bandwidth, startFreq, rampEndTime, idleTime, chirps, tx)
    if (chirps > 0 && tx > 0 && (rampEndTime + idleTime) > 0)
    {
        // PRI = rampEndTime + idleTime (us) -> seconds
        const double pri_s = (rampEndTime + idleTime) * 1e-6;
        const double lambda = (startFreq > 0) ? (299792458.0 / (startFreq * 1e9)) : 0.0;
        if (lambda > 0)
        {
            // v_res ~ lambda / (2 * T_obs) where T_obs = PRI * chirps / tx
            const double tobs = pri_s * static_cast<double>(chirps) / static_cast<double>(tx);
            if (tobs > 0)
            {
                doppler_resolution = lambda / (2.0 * tobs);
            }
        }
    }

    // angle_bins and range_bins extents
    angle_bins.clear();
    if (chirps > 0)
    {
        angle_bins.reserve(static_cast<size_t>(chirps));
        const int bins = chirps;
        for (int i = 0; i < bins; ++i)
        {
            double angle = -90.0 + 180.0 * (static_cast<double>(i) / (bins - 1));
            angle_bins.push_back(angle);
        }
    }

    range_bins.clear();
    if (samples > 0)
    {
        range_bins.reserve(static_cast<size_t>(samples));
        // Python used 0.0485 multiplier as a placeholder; compute from range_resolution if available
        const double step = (range_resolution > 0.0) ? range_resolution : 0.0485;
        for (int i = 0; i < samples; ++i)
        {
            range_bins.push_back(i * step);
        }
    }

    max_range = samples * ((range_resolution > 0.0) ? range_resolution : 0.0);
    max_doppler = (chirps * ((doppler_resolution > 0.0) ? doppler_resolution : 0.0)) / 2.0;

    range_doppler_extents = { -max_doppler, max_doppler, 0.0, max_range };
    range_azimuth_extents = { -static_cast<double>(max_azimuth), static_cast<double>(max_azimuth), 0.0, max_range };
}

RadarConfig::RadarConfig(const std::string& config_file_path)
    : _config_file_path(config_file_path)
{
    // Parse config file immediately in constructor
    parseConfigFile();
    calculateDerivedParams();
    _adc_params.finalize();
    validateConfig();
}

void RadarConfig::parseConfigFile()
{
    std::ifstream config_file(_config_file_path);
    if (!config_file.is_open())
    {
        throw std::runtime_error("Cannot open config file: " +
                                 _config_file_path);
    }

    std::string line;
    int cur_profile_id = 0;
    int chirp_start_idx_fcf = 0;
    int chirp_end_idx_fcf = 0;
    int loop_count = 0;

    while (std::getline(config_file, line))
    {
        line = trim(line);

        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        auto [key, value] = parseLine(line);
        if (key.empty())
        {
            continue;
        }

        // Parse different configuration sections
        parseChannelConfig(key, value);
        parseProfileConfig(key, value, cur_profile_id);
        parseChirpConfig(key, value);
        parseFrameConfig(key, value, chirp_start_idx_fcf, chirp_end_idx_fcf,
                         loop_count);
    }

    config_file.close();
}

void RadarConfig::parseChannelConfig(const std::string& key,
                                     const std::string& value)
{
    if (key == "channelTx")
    {
        _cfg_params.txAntMask = std::stoi(value);
        _cfg_params.numTxChan = countBitsSet(_cfg_params.txAntMask, 3);
    }
    else if (key == "channelRx")
    {
        _cfg_params.rxAntMask = std::stoi(value);
        _cfg_params.numRxChan = countBitsSet(_cfg_params.rxAntMask, 4);
        _adc_params.rx = _cfg_params.numRxChan;
    }
    else if (key == "adcBitsD")
    {
        _adc_params.bytes = 2;
    }
    else if (key == "adcFmt")
    {
        _adc_params.IQ = 2;
    }
    else if (key == "dataRate")
    {
        int data_rate = std::stoi(value);
        _cfg_params.lvdsBW = mapDataRateToMbps(data_rate);
    }
    else if (key == "laneEn")
    {
        int lane_mask = std::stoi(value);
        _cfg_params.numlaneEn = countBitsSet(lane_mask, 4);
    }
}

void RadarConfig::parseProfileConfig(const std::string& key,
                                     const std::string& value,
                                     int& cur_profile_id)
{
    if (key == "profileId")
    {
        cur_profile_id = std::stoi(value);
    }
    else if (cur_profile_id == 0)
    {  // Only take first profile
        if (key == "startFreqConst")
        {
            _adc_params.startFreq = std::stoi(value) * (3.6 / (1 << 26));
        }
        else if (key == "idleTimeConst")
        {
            _adc_params.idleTime = std::stoi(value) / 100.0;
        }
        else if (key == "adcStartTimeConst")
        {
            _adc_params.adc_valid_start_time = std::stoi(value) / 100.0;
        }
        else if (key == "rampEndTime")
        {
            _adc_params.rampEndTime = std::stoi(value) / 100.0;
        }
        else if (key == "freqSlopeConst")
        {
            _adc_params.freq_slope =
                std::stoi(value) * (3.6e3 * 900 / (1 << 26));
        }
        else if (key == "txStartTime")
        {
            _adc_params.txStartTime = std::stoi(value) / 100.0;
        }
        else if (key == "numAdcSamples")
        {
            _adc_params.samples = std::stoi(value);
        }
        else if (key == "digOutSampleRate")
        {
            _adc_params.sample_rate = std::stoi(value);
        }
    }

    if (key == "rxGain")
    {
        cur_profile_id++;  // Avoid frameCfg numAdcSamples being treated as
                           // profile
    }
}

void RadarConfig::parseChirpConfig(const std::string& key,
                                   const std::string& value)
{
    if (key == "chirpStartIdx")
    {
        _chirp_start_idx_buf.push_back(std::stoi(value));
    }
    else if (key == "chirpEndIdx")
    {
        _chirp_end_idx_buf.push_back(std::stoi(value));
    }
    else if (key == "profileIdCPCFG")
    {
        _profile_id_cpcfg_buf.push_back(std::stoi(value));
    }
    else if (key == "txEnable")
    {
        _tx_enable_buf.push_back(std::stoi(value));
    }
}

void RadarConfig::parseFrameConfig(const std::string& key,
                                   const std::string& value,
                                   int& chirp_start_idx_fcf,
                                   int& chirp_end_idx_fcf, int& loop_count)
{
    if (key == "chirpStartIdxFCF")
    {
        chirp_start_idx_fcf = std::stoi(value);
    }
    else if (key == "chirpEndIdxFCF")
    {
        chirp_end_idx_fcf = std::stoi(value);
    }
    else if (key == "loopCount")
    {
        loop_count = std::stoi(value);
    }
    else if (key == "periodicity")
    {
        _adc_params.frame_periodicity = std::stoi(value) / 200000.0;
    }

    // Calculate chirps when we have all frame config data
    if (loop_count > 0 && !_chirp_end_idx_buf.empty())
    {
        int tmp_chirp_num = _chirp_end_idx_buf[0] - _chirp_start_idx_buf[0] + 1;
        _adc_params.chirps = loop_count * tmp_chirp_num;
    }
}

int RadarConfig::countBitsSet(int mask, int max_bits) const
{
    int count = 0;
    for (int i = 0; i < max_bits; ++i)
    {
        if ((mask >> i) & 1)
        {
            count++;
        }
    }
    return count;
}

int RadarConfig::mapDataRateToMbps(int data_rate) const
{
    switch (data_rate)
    {
        case 0b0001:
            return 600;  // 600 Mbps (DDR only)
        case 0b0010:
            return 450;  // 450 Mbps (SDR, DDR)
        case 0b0011:
            return 400;  // 400 Mbps (DDR only)
        case 0b0100:
            return 300;  // 300 Mbps (SDR, DDR)
        case 0b0101:
            return 225;  // 225 Mbps (DDR only)
        case 0b0110:
            return 150;  // 150 Mbps (DDR only)
        default:
            return 0;
    }
}

void RadarConfig::calculateDerivedParams()
{
    _adc_params.tx = static_cast<int>(_tx_enable_buf.size());

    // Calculate LVDS data size per chirp
    _lvds_data_size_per_chirp = _adc_params.samples * _adc_params.rx *
                                    _adc_params.IQ * _adc_params.bytes +
                                52;
    _lvds_data_size_per_chirp =
        static_cast<int>(std::ceil(_lvds_data_size_per_chirp / 256.0) * 256);

    // Calculate max send bytes per chirp
    _max_send_bytes_per_chirp =
        (_adc_params.idleTime + _adc_params.rampEndTime) *
        _cfg_params.numlaneEn * _cfg_params.lvdsBW / 8.0;
}

void RadarConfig::validateConfig() const
{
    if (_adc_params.tx > _cfg_params.numTxChan)
    {
        throw std::runtime_error("exceed max tx num, check channelTx(" +
                                 std::to_string(_cfg_params.numTxChan) +
                                 ") and chirp cfg(" +
                                 std::to_string(_adc_params.tx) + ").");
    }

    if (_chirp_start_idx_buf.empty() || _chirp_end_idx_buf.empty())
    {
        throw std::runtime_error("Missing chirp configuration data");
    }

    int tmp_chirp_num = _chirp_end_idx_buf[0] - _chirp_start_idx_buf[0] + 1;
    int all_chirp_num = tmp_chirp_num;

    for (size_t i = 1; i < _chirp_end_idx_buf.size(); ++i)
    {
        int current_chirp_num =
            _chirp_end_idx_buf[i] - _chirp_start_idx_buf[i] + 1;
        all_chirp_num += current_chirp_num;

        if (tmp_chirp_num != current_chirp_num)
        {
            throw std::runtime_error(
                "AWR2243_read_config does not support different chirp number "
                "in different tx ant "
                "yet.");
        }
    }

    // Additional validation can be added here based on frame config
}

std::string RadarConfig::trim(const std::string& str) const
{
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos)
        return "";

    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::pair<std::string, std::string> RadarConfig::parseLine(
    const std::string& line) const
{
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos)
    {
        return {"", ""};
    }

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);

    // Remove spaces
    key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    // Extract value before semicolon
    size_t semicolon_pos = value.find(';');
    if (semicolon_pos != std::string::npos)
    {
        value = value.substr(0, semicolon_pos);
    }

    return {key, value};
}

// --- toString implementations ---
std::string radar::AdcParams::toString() const
{
    return "AdcParams{" + std::string("startFreq=") +
           std::to_string(startFreq) + ", " +
           "idleTime=" + std::to_string(idleTime) + ", " +
           "adc_valid_start_time=" + std::to_string(adc_valid_start_time) +
           ", " + "rampEndTime=" + std::to_string(rampEndTime) + ", " +
           "freq_slope=" + std::to_string(freq_slope) + ", " +
           "txStartTime=" + std::to_string(txStartTime) + ", " +
           "samples=" + std::to_string(samples) + ", " +
           "sample_rate=" + std::to_string(sample_rate) + ", " +
           "rx=" + std::to_string(rx) + ", " + "tx=" + std::to_string(tx) +
           ", " + "IQ=" + std::to_string(IQ) + ", " +
           "bytes=" + std::to_string(bytes) + ", " +
           "chirps=" + std::to_string(chirps) + ", " +
           "frame_periodicity=" + std::to_string(frame_periodicity) + ", " +
           "range_resolution=" + std::to_string(range_resolution) + ", " +
           "chirp_bandwidth=" + std::to_string(chirp_bandwidth) + ", " +
           "doppler_resolution=" + std::to_string(doppler_resolution) + ", " +
           "max_range=" + std::to_string(max_range) + ", " +
           "max_doppler=" + std::to_string(max_doppler) + "}";
}

std::string radar::CfgParams::toString() const
{
    return "CfgParams{" + std::string("txAntMask=") +
           std::to_string(txAntMask) + ", " +
           "rxAntMask=" + std::to_string(rxAntMask) + ", " +
           "numTxChan=" + std::to_string(numTxChan) + ", " +
           "numRxChan=" + std::to_string(numRxChan) + ", " +
           "lvdsBW=" + std::to_string(lvdsBW) + ", " +
           "numlaneEn=" + std::to_string(numlaneEn) + "}";
}

std::string radar::RadarConfig::toString() const
{
    return std::string("RadarConfig{\n  ") + _adc_params.toString() + ",\n  " +
           _cfg_params.toString() + ",\n  " +
           "LVDSDataSizePerChirp=" + std::to_string(_lvds_data_size_per_chirp) +
           ", " +
           "MaxSendBytesPerChirp=" + std::to_string(_max_send_bytes_per_chirp) +
           "\n}";
}
}  // namespace radar
