#pragma once
#include <arrayfire.h>

namespace radar
{
namespace dsp
{

enum class Window
{
    NONE = 0,
    HAMMING = 1
};

// Generate a 1D Hamming window of length N (float32 ArrayFire array)
af::array hamming(int N);

// Generic windowing similar to Python mmwave.dsp.utils.windowing.
// data: input array; win_type: window enum; axis: dimension to apply window.
// Returns new array (data * window) or original data if NONE.
af::array windowing(const af::array& data, Window win_type, int axis);

// 1D fftshift along a given dimension (supports up to 4D). Matches
// numpy.fft.fftshift behavior.
af::array fftshift_dim(const af::array& arr, int dim);

}  // namespace dsp
}  // namespace radar
