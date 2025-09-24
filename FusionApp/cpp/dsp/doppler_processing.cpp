#include "doppler_processing.hpp"
#include <stdexcept>
#include "utils.hpp"

namespace radar
{
namespace dsp
{

af::array doppler_processing(const af::array& radar_cube, int chirps, int vrx,
                             int samples, Window win)
{
    af::dim4 d = radar_cube.dims();
    if (d[0] != (dim_t)chirps || d[1] != (dim_t)vrx || d[2] != (dim_t)samples)
    {
        throw std::runtime_error("doppler_processing: unexpected input shape");
    }
    // Window and FFT along chirp axis (axis 0)
    af::array windowed = windowing(radar_cube, win, 0);
    af::array fft_out = af::fft(windowed, chirps);  // 1D FFT along dim0
    return fft_out;                                 // (chirps, vrx, samples)
}

}  // namespace dsp
}  // namespace radar
