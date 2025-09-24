#pragma once
#include <arrayfire.h>
#include "utils.hpp"

namespace radar
{
namespace dsp
{

// Range processing (window + FFT along samples axis) akin to Python
// mmwave.dsp.range_processing. Input: complex frame (chirps, tx, rx, samples)
// Output: radar_cube (chirps, tx*rx, samples)
af::array range_processing(const af::array& frame, int chirps, int tx, int rx,
                           int samples, Window win = Window::HAMMING);

}  // namespace dsp
}  // namespace radar
