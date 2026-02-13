/**
 * @file bin_sensitivity.cu
 * @brief Compute per-bin SAY likelihood sensitivities (adjoint method step 2).
 *
 * For each histogram bin, computes dSAY/d(w_sum) and dSAY/d(w2_sum) using
 * GPUDual<2> automatic differentiation through the SAY formula. These
 * "bin adjoints" are then used by the weight gradient kernel to propagate
 * gradients back to the fit parameters.
 */

#include "cuda/GPUCommon.h"
#include <PhysTools/cuda/GPUAutodiff.h>
#include <cfloat>

namespace gollumfit {
namespace gpu {

using phys_tools::autodiff::gpu::GPUDual;

/**
 * @brief Compute SAY likelihood for a single bin using dual numbers
 *
 * Same formula as computeSAYBinLikelihood in likelihood.cu, but
 * with GPUDual<2> to get derivatives w.r.t. w_sum and w2_sum.
 */
__device__ GPUDual<2> computeSAYBinDual(
    double k,
    const GPUDual<2>& w_sum,
    const GPUDual<2>& w2_sum
) {
    using namespace phys_tools::autodiff::gpu;
    using Dual2 = GPUDual<2>;

    // Handle edge cases (return constant zero — no gradient contribution)
    if (w_sum.value() <= 0.0) {
        return Dual2(0.0);
    }

    if (w2_sum.value() <= 0.0) {
        // Poisson fallback: k*log(w_sum) - w_sum - lgamma(k+1)
        return Dual2(k) * log(w_sum) - w_sum - Dual2(::lgamma(k + 1.0));
    }

    // SAY parameters
    Dual2 alpha = w_sum * w_sum / w2_sum + Dual2(1.0);
    Dual2 beta = w_sum / w2_sum;

    // SAY likelihood
    Dual2 llh = alpha * log(beta);
    llh += lgamma(Dual2(k) + alpha);
    llh -= lgamma(alpha);
    llh -= Dual2(::lgamma(k + 1.0));
    llh -= (Dual2(k) + alpha) * log1p(beta);

    return llh;
}

/**
 * @brief Kernel: compute bin sensitivities (adjoints)
 *
 * Each thread processes one histogram bin. Outputs the negated derivatives
 * since we minimize -logL: adjoint = -dSAY/d(variable).
 */
__global__ void computeBinSensitivitiesKernel(
    const double* __restrict__ dataCount,
    const double* __restrict__ wSum,
    const double* __restrict__ w2Sum,
    double* __restrict__ adjoint_wsum,
    double* __restrict__ adjoint_w2sum,
    int numBins
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= numBins) return;

    double k = dataCount[bid];

    // Create dual numbers: variable 0 = w_sum, variable 1 = w2_sum
    GPUDual<2> ws(wSum[bid], 0);
    GPUDual<2> w2s(w2Sum[bid], 1);

    GPUDual<2> llh = computeSAYBinDual(k, ws, w2s);

    // Negate because we minimize -logL
    adjoint_wsum[bid] = -llh.derivative(0);
    adjoint_w2sum[bid] = -llh.derivative(1);
}

//==============================================================================
// Wrapper function
//==============================================================================

void launchBinSensitivities(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    double* d_adjoint_wsum,
    double* d_adjoint_w2sum,
    int numBins,
    cudaStream_t stream
) {
    if (numBins == 0) return;

    int blockSize = 256;
    int gridSize = (numBins + blockSize - 1) / blockSize;

    computeBinSensitivitiesKernel<<<gridSize, blockSize, 0, stream>>>(
        d_dataCount, d_wSum, d_w2Sum,
        d_adjoint_wsum, d_adjoint_w2sum,
        numBins
    );

    CUDA_CHECK_KERNEL();
}

} // namespace gpu
} // namespace gollumfit
