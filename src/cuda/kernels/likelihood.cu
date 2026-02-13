/**
 * @file likelihood.cu
 * @brief CUDA kernel for SAY likelihood evaluation.
 *
 * This file implements the GPU version of the SAY (Sum of Asymmetric Yields)
 * likelihood function used in GollumFit. The SAY likelihood accounts for
 * Monte Carlo statistical uncertainty through the sum of squared weights.
 *
 * The SAY likelihood formula (per bin):
 *   If w_sum = 0 and k = 0: llh = 0
 *   If w_sum = 0 and k > 0: llh = -inf (impossible)
 *   Otherwise:
 *     alpha = w_sum^2 / w2_sum + 1
 *     beta = w_sum / w2_sum
 *     llh = alpha * log(beta) + lgamma(k + alpha) - lgamma(alpha)
 *           - lgamma(k + 1) - (k + alpha) * log(1 + beta)
 *
 * Where:
 *   k = observed count in bin
 *   w_sum = sum of MC weights in bin
 *   w2_sum = sum of squared MC weights in bin
 */

#include "cuda/GPUCommon.h"
#include <cmath>
#include <cfloat>

namespace gollumfit {
namespace gpu {

//==============================================================================
// Device Helper Functions
//==============================================================================

/**
 * @brief GPU-compatible lgamma function using Stirling approximation
 * More accurate than CUDA's built-in for large arguments
 */
__device__ double lgamma_approx(double x) {
    if (x <= 0.0) return DBL_MAX;
    if (x < 7.0) {
        // Use recurrence relation to shift argument
        double result = 0.0;
        while (x < 7.0) {
            result -= log(x);
            x += 1.0;
        }
        return result + lgamma_approx(x);
    }
    // Stirling's approximation for x >= 7
    // lgamma(x) ~ (x-0.5)*ln(x) - x + 0.5*ln(2*pi) + 1/(12x) - 1/(360x^3) + ...
    double x2 = x * x;
    double x3 = x2 * x;
    double x5 = x3 * x2;
    return (x - 0.5) * log(x) - x + 0.9189385332046727  // 0.5*ln(2*pi)
           + 1.0 / (12.0 * x) - 1.0 / (360.0 * x3) + 1.0 / (1260.0 * x5);
}

/**
 * @brief Compute SAY likelihood for a single bin
 *
 * @param k Observed count
 * @param w_sum Sum of MC weights
 * @param w2_sum Sum of squared MC weights
 * @return Log-likelihood contribution from this bin
 */
__device__ double computeSAYBinLikelihood(
    double k,
    double w_sum,
    double w2_sum
) {
    // Handle edge cases
    if (w_sum <= 0.0) {
        if (k == 0.0) {
            return 0.0;  // No expectation, no observation - contributes nothing
        } else {
            return -DBL_MAX;  // Observation without expectation - impossible
        }
    }

    if (w2_sum <= 0.0) {
        // Fall back to simple Poisson if no MC uncertainty
        // Poisson: k*log(lambda) - lambda - lgamma(k+1)
        return k * log(w_sum) - w_sum - lgamma(k + 1.0);
    }

    // SAY parameters
    double alpha = w_sum * w_sum / w2_sum + 1.0;
    double beta = w_sum / w2_sum;

    // SAY likelihood
    // log P(k | alpha, beta) = alpha*log(beta) + lgamma(k+alpha) - lgamma(alpha)
    //                         - lgamma(k+1) - (k+alpha)*log(1+beta)
    double llh = alpha * log(beta);
    llh += lgamma(k + alpha);
    llh -= lgamma(alpha);
    llh -= lgamma(k + 1.0);
    llh -= (k + alpha) * log1p(beta);  // log1p for numerical stability (matching CPU)

    return llh;
}

//==============================================================================
// Likelihood Computation Kernel
//==============================================================================

/**
 * @brief Compute per-bin SAY likelihood contributions
 *
 * Each thread processes one histogram bin, computing the log-likelihood
 * contribution from that bin.
 *
 * @param dataCount Observed counts per bin [numBins]
 * @param w_sum Sum of MC weights per bin [numBins]
 * @param w2_sum Sum of squared MC weights per bin [numBins]
 * @param binLLH Output: per-bin likelihood contributions [numBins]
 * @param numBins Number of histogram bins
 */
__global__ void computeSAYLikelihoodKernel(
    const double* __restrict__ dataCount,
    const double* __restrict__ w_sum,
    const double* __restrict__ w2_sum,
    double* __restrict__ binLLH,
    const int numBins
) {
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= numBins) return;

    const double k = dataCount[bid];
    const double ws = w_sum[bid];
    const double w2s = w2_sum[bid];

    binLLH[bid] = computeSAYBinLikelihood(k, ws, w2s);
}

/**
 * @brief Compute total likelihood via parallel reduction
 *
 * This kernel performs a block-level parallel reduction to sum
 * the per-bin likelihood contributions into a total.
 *
 * @param binLLH Per-bin likelihood contributions [numBins]
 * @param totalLLH Output: total log-likelihood (single value)
 * @param numBins Number of bins
 */
__global__ void reduceLikelihoodKernel(
    const double* __restrict__ binLLH,
    double* __restrict__ totalLLH,
    const int numBins
) {
    extern __shared__ double s_partial[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate
    double sum = 0.0;
    while (i < numBins) {
        sum += binLLH[i];
        i += blockDim.x * gridDim.x;
    }
    s_partial[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAddDouble(totalLLH, s_partial[0]);
    }
}

//==============================================================================
// Combined Likelihood Kernel (Optimized)
//==============================================================================

/**
 * @brief Compute and reduce likelihood in a single kernel
 *
 * This fused kernel computes per-bin likelihood and performs block-level
 * reduction, minimizing memory traffic for the intermediate per-bin results.
 *
 * @param dataCount Observed counts per bin [numBins]
 * @param w_sum Sum of MC weights per bin [numBins]
 * @param w2_sum Sum of squared MC weights per bin [numBins]
 * @param totalLLH Output: total log-likelihood
 * @param numBins Number of bins
 */
__global__ void computeAndReduceLikelihoodKernel(
    const double* __restrict__ dataCount,
    const double* __restrict__ w_sum,
    const double* __restrict__ w2_sum,
    double* __restrict__ totalLLH,
    const int numBins
) {
    extern __shared__ double s_partial[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Compute and accumulate local sum
    double sum = 0.0;
    while (i < numBins) {
        double k = dataCount[i];
        double ws = w_sum[i];
        double w2s = w2_sum[i];
        sum += computeSAYBinLikelihood(k, ws, w2s);
        i += stride;
    }
    s_partial[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    // Atomic add block result to global
    if (tid == 0) {
        atomicAddDouble(totalLLH, s_partial[0]);
    }
}

//==============================================================================
// Wrapper Functions
//==============================================================================

void launchSAYLikelihoodKernel(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    double* d_binLLH,
    int numBins,
    cudaStream_t stream
) {
    if (numBins == 0) return;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numBins + blockSize - 1) / blockSize;

    computeSAYLikelihoodKernel<<<gridSize, blockSize, 0, stream>>>(
        d_dataCount, d_wSum, d_w2Sum, d_binLLH, numBins
    );

    CUDA_CHECK_KERNEL();
}

double computeTotalSAYLikelihood(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    int numBins,
    cudaStream_t stream
) {
    if (numBins == 0) return 0.0;

    // Allocate device memory for result
    double* d_totalLLH;
    CUDA_CHECK(cudaMalloc(&d_totalLLH, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_totalLLH, 0, sizeof(double), stream));

    int blockSize = 256;
    int gridSize = min(256, (numBins + blockSize - 1) / blockSize);
    size_t sharedMemSize = blockSize * sizeof(double);

    computeAndReduceLikelihoodKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_dataCount, d_wSum, d_w2Sum, d_totalLLH, numBins
    );

    CUDA_CHECK_KERNEL();

    // Copy result back
    double h_totalLLH;
    CUDA_CHECK(cudaMemcpyAsync(&h_totalLLH, d_totalLLH, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_totalLLH);

    return h_totalLLH;
}

//==============================================================================
// Prior Evaluation Kernel
//==============================================================================

/**
 * @brief Compute Gaussian prior contribution
 *
 * Matches PhysTools GaussianPrior: log(norm) - z*z/2
 * where norm = 1/(sigma*sqrt(2*pi)), z = (x - mu)/sigma.
 * For infinite sigma (flat prior): returns 0.0.
 *
 * @param params Current parameter values [numParams]
 * @param priorMeans Prior means [numParams]
 * @param priorSigmas Prior widths [numParams]
 * @param priorFlags Which parameters have priors (1 = has prior)
 * @param priorLLH Output: prior contribution for each parameter [numParams]
 * @param numParams Number of parameters
 */
__global__ void computeGaussianPriorsKernel(
    const double* __restrict__ params,
    const double* __restrict__ priorMeans,
    const double* __restrict__ priorSigmas,
    const int* __restrict__ priorFlags,
    double* __restrict__ priorLLH,
    const int numParams
) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParams) return;

    if (priorFlags[pid] == 0) {
        // No prior for this parameter
        priorLLH[pid] = 0.0;
        return;
    }

    double x = params[pid];
    double mu = priorMeans[pid];
    double sigma = priorSigmas[pid];

    // Match PhysTools GaussianPrior: if sigma is inf/nan, return 0 (flat prior)
    if (isinf(sigma) || isnan(sigma)) {
        priorLLH[pid] = 0.0;
        return;
    }

    // norm = 1/(sigma * sqrt(2*pi))
    // log(norm) = -log(sigma) - 0.5*log(2*pi)
    double z = (x - mu) / sigma;
    priorLLH[pid] = -log(sigma) - 0.9189385332046727 - 0.5 * z * z;
}

/**
 * @brief Sum prior contributions
 */
double computeTotalPrior(
    const double* d_params,
    const double* d_priorMeans,
    const double* d_priorSigmas,
    const int* d_priorFlags,
    int numParams,
    cudaStream_t stream
) {
    if (numParams == 0) return 0.0;

    // Allocate temporary storage
    double* d_priorLLH;
    double* d_totalPrior;
    CUDA_CHECK(cudaMalloc(&d_priorLLH, numParams * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_totalPrior, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_totalPrior, 0, sizeof(double), stream));

    // Compute individual priors
    int blockSize = 64;  // Small block size for few parameters
    int gridSize = (numParams + blockSize - 1) / blockSize;

    computeGaussianPriorsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_params, d_priorMeans, d_priorSigmas, d_priorFlags, d_priorLLH, numParams
    );

    // Reduce (simple for 38 parameters)
    reduceLikelihoodKernel<<<1, 64, 64 * sizeof(double), stream>>>(
        d_priorLLH, d_totalPrior, numParams
    );

    double h_totalPrior;
    CUDA_CHECK(cudaMemcpyAsync(&h_totalPrior, d_totalPrior, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_priorLLH);
    cudaFree(d_totalPrior);

    return h_totalPrior;
}

//==============================================================================
// Full Likelihood + Prior Computation
//==============================================================================

/**
 * @brief Compute negative log-likelihood (for minimization)
 *
 * Returns -1 * (data_llh + prior_llh) so that minimization finds the
 * maximum likelihood estimate.
 *
 * @param d_dataCount Observed counts [numBins]
 * @param d_wSum MC weight sums [numBins]
 * @param d_w2Sum MC squared weight sums [numBins]
 * @param numBins Number of histogram bins
 * @param d_params Current parameters [numParams]
 * @param d_priorMeans Prior means [numParams]
 * @param d_priorSigmas Prior sigmas [numParams]
 * @param d_priorFlags Prior flags [numParams]
 * @param numParams Number of parameters
 * @param includePrior Whether to include prior term
 * @param stream CUDA stream
 * @return Negative log-likelihood value
 */
double computeNegLogLikelihood(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    int numBins,
    const double* d_params,
    const double* d_priorMeans,
    const double* d_priorSigmas,
    const int* d_priorFlags,
    int numParams,
    bool includePrior,
    cudaStream_t stream
) {
    // Compute data likelihood
    double dataLLH = computeTotalSAYLikelihood(
        d_dataCount, d_wSum, d_w2Sum, numBins, stream
    );

    // Compute prior if requested
    double priorLLH = 0.0;
    if (includePrior && d_priorMeans != nullptr) {
        priorLLH = computeTotalPrior(
            d_params, d_priorMeans, d_priorSigmas, d_priorFlags,
            numParams, stream
        );
    }

    // Return negative log-likelihood for minimization
    return -(dataLLH + priorLLH);
}

} // namespace gpu
} // namespace gollumfit
