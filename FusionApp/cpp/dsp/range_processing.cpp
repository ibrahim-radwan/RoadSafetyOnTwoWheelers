#include "range_processing.hpp"
#include <stdexcept>
#include "utils.hpp"

namespace radar
{
namespace dsp
{

af::array range_processing(const af::array& frame, int chirps, int tx, int rx,
                           int samples, Window win)
{
    // Expect frame dims: (chirps, tx, rx, samples)
    af::dim4 d = frame.dims();
    if (d[0] != (dim_t)chirps || d[1] != (dim_t)tx || d[2] != (dim_t)rx ||
        d[3] != (dim_t)samples)
    {
        throw std::runtime_error("range_processing: unexpected frame shape");
    }
    // Collapse tx,rx -> virtual rx dimension: (chirps, vrx, samples)
    af::array collapsed =
        af::moddims(frame, (dim_t)chirps, (dim_t)(tx * rx), (dim_t)samples);
    // Reorder so samples axis becomes dim 0 for both window + FFT: (samples,
    // chirps, vrx)
    af::array samples_front = af::reorder(collapsed, 2, 0, 1);
    // Window along axis 0 (now samples)
    af::array windowed = windowing(samples_front, win, 0);
    // FFT along axis 0
    af::array ffted = af::fft(windowed, samples);
    // Reorder back to (chirps, vrx, samples)
    af::array back = af::reorder(ffted, 1, 2, 0);
    return back;
}

}  // namespace dsp
}  // namespace radar
