#pragma once
#include <arrayfire.h>
#include "utils.hpp"

namespace radar
{
namespace dsp
{

// Doppler processing:
// Input: radar_cube (chirps, vrx, samples)
// Output: doppler_fft (chirps, vrx, samples) after window + FFT along chirp
// dimension
af::array doppler_processing(const af::array& radar_cube, int chirps, int vrx,
                             int samples, Window win = Window::HAMMING);

}  // namespace dsp
}  // namespace radar
