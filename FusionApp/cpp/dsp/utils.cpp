#include "utils.hpp"
#include <stdexcept>
#include <utility>  // for std::swap

namespace radar
{
namespace dsp
{

af::array hamming(int N)
{
    if (N <= 1)
        return af::constant(1.0f, N, f32);
    af::array n = af::range(af::dim4(N), 0, f32);
    return 0.54f - 0.46f * af::cos(2.0f * af::Pi * n / (N - 1.0f));
}

af::array windowing(const af::array& data, Window win_type, int axis)
{
    if (win_type == Window::NONE)
        return data;
    if (axis < 0 || axis >= (int)data.numdims())
    {
        throw std::runtime_error("windowing: axis out of range");
    }
    af::dim4 dims = data.dims();
    int N = dims[axis];
    af::array win;
    switch (win_type)
    {
        case Window::HAMMING:
            win = hamming(N);
            break;
        default:
            return data;  // Unknown window -> no-op
    }
    // Fast path: if axis already front (0), just tile & multiply
    if (axis == 0)
    {
        af::array win_b = af::tile(win, 1, dims[1], dims[2], dims[3]);
        return data * win_b;
    }
    // Build permutation to bring axis to front
    int order[4] = {0, 1, 2, 3};
    std::swap(order[0], order[axis]);
    af::array front = af::reorder(data, order[0], order[1], order[2], order[3]);
    af::dim4 fd = front.dims();
    af::array win_b = af::tile(win, 1, fd[1], fd[2], fd[3]);
    af::array applied = front * win_b;
    // Invert permutation
    int inv[4];
    for (int i = 0; i < 4; ++i)
        inv[order[i]] = i;
    return af::reorder(applied, inv[0], inv[1], inv[2], inv[3]);
}

af::array fftshift_dim(const af::array& arr, int dim)
{
    if (dim < 0 || dim >= (int)arr.numdims())
    {
        throw std::runtime_error("fftshift_dim: dim out of range");
    }
    af::dim4 d = arr.dims();
    int N = d[dim];
    if (N <= 1)
        return arr;
    int half = N / 2;
    // Build seqs for slices
    af::seq first(0, half - 1);
    af::seq second(half, N - 1);
    // We need to index along chosen dim; easiest: reorder target dim to front,
    // split, join, reorder back
    int order[4] = {0, 1, 2, 3};
    std::swap(order[0], order[dim]);
    af::array front =
        af::reorder(arr, order[0], order[1], order[2], order[3]);  // dim now 0
    af::array left = front(af::seq(0, half - 1), af::span, af::span, af::span);
    af::array right = front(af::seq(half, N - 1), af::span, af::span, af::span);
    af::array shifted = af::join(0, right, left);
    int inv[4];
    for (int i = 0; i < 4; ++i)
        inv[order[i]] = i;
    return af::reorder(shifted, inv[0], inv[1], inv[2], inv[3]);
}

}  // namespace dsp
}  // namespace radar
