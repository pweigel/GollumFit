/**
 * @file histogram_reduction.cu
 * @brief CUDA kernels for histogram accumulation with FP64 atomics.
 *
 * This file implements GPU kernels for accumulating weighted events into
 * histogram bins. The kernels support:
 * - Sum of weights (w_sum) for expectation calculation
 * - Sum of squared weights (w2_sum) for SAY likelihood uncertainty
 * - Native FP64 atomics (efficient on Ampere/Hopper GPUs)
 */

#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"

namespace gollumfit {
namespace gpu {

//==============================================================================
// Histogram Accumulation Kernel
//==============================================================================

/**
 * @brief Accumulate weighted events into histogram bins
 *
 * Each thread processes one event, atomically adding its weight to the
 * appropriate bin. This leverages native FP64 atomics on Ampere+ GPUs.
 *
 * @param weights Event weights [numEvents]
 * @param binIndices Precomputed bin indices [numEvents] (-1 for out-of-bounds)
 * @param binSums Output: sum of weights per bin [numBins]
 * @param numEvents Number of events
 * @param numBins Total number of histogram bins
 */
__global__ void accumulateHistogramKernel(
    const double* __restrict__ weights,
    const int32_t* __restrict__ binIndices,
    double* __restrict__ binSums,
    const int numEvents,
    const int numBins
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    const int binIdx = binIndices[tid];

    // Skip events outside histogram bounds
    if (binIdx < 0 || binIdx >= numBins) return;

    const double w = weights[tid];

    // Atomic add to bin (uses native FP64 atomic on SM 6.0+)
    atomicAddDouble(&binSums[binIdx], w);
}

/**
 * @brief Accumulate both weights and squared weights into histogram bins
 *
 * This combined kernel reduces memory traffic by reading weights once
 * and computing both sums needed for SAY likelihood.
 *
 * @param weights Event weights [numEvents]
 * @param weightsSquared Squared weights (w^2/num_events) [numEvents]
 * @param binIndices Precomputed bin indices [numEvents]
 * @param binSums Output: sum of weights per bin [numBins]
 * @param binSqSums Output: sum of squared weights per bin [numBins]
 * @param numEvents Number of events
 * @param numBins Total number of histogram bins
 */
__global__ void accumulateHistogramWithSquaresKernel(
    const double* __restrict__ weights,
    const double* __restrict__ weightsSquared,
    const int32_t* __restrict__ binIndices,
    double* __restrict__ binSums,
    double* __restrict__ binSqSums,
    const int numEvents,
    const int numBins
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    const int binIdx = binIndices[tid];

    // Skip events outside histogram bounds
    if (binIdx < 0 || binIdx >= numBins) return;

    const double w = weights[tid];
    const double w2 = weightsSquared[tid];

    // Atomic adds to bins
    atomicAddDouble(&binSums[binIdx], w);
    atomicAddDouble(&binSqSums[binIdx], w2);
}

//==============================================================================
// Optimized Histogram Accumulation with Shared Memory
//==============================================================================

/**
 * @brief Block-level histogram accumulation with shared memory
 *
 * This optimized kernel uses shared memory to accumulate within each block,
 * reducing the number of global atomic operations. Each block handles a
 * subset of events and maintains a local histogram copy.
 *
 * Note: This kernel is beneficial when many events fall into the same bins
 * (high collision rate). For sparse histograms, the simple kernel may be faster.
 *
 * @param weights Event weights [numEvents]
 * @param binIndices Bin indices [numEvents]
 * @param binSums Output histogram [numBins]
 * @param numEvents Number of events
 * @param numBins Total number of bins
 */
__global__ void accumulateHistogramSharedKernel(
    const double* __restrict__ weights,
    const int32_t* __restrict__ binIndices,
    double* __restrict__ binSums,
    const int numEvents,
    const int numBins
) {
    // Dynamic shared memory for local histogram
    extern __shared__ double s_localHist[];

    // Initialize shared memory histogram to zero
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_localHist[i] = 0.0;
    }
    __syncthreads();

    // Each thread processes multiple events
    const int stride = blockDim.x * gridDim.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < numEvents;
         tid += stride)
    {
        const int binIdx = binIndices[tid];

        if (binIdx >= 0 && binIdx < numBins) {
            const double w = weights[tid];
            // Atomic add to shared memory (less contention than global)
            atomicAddDouble(&s_localHist[binIdx], w);
        }
    }
    __syncthreads();

    // Merge block-local histogram to global
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        if (s_localHist[i] != 0.0) {
            atomicAddDouble(&binSums[i], s_localHist[i]);
        }
    }
}

//==============================================================================
// Wrapper Functions
//==============================================================================

void launchHistogramAccumulation(
    const double* d_weights,
    const int32_t* d_binIndices,
    double* d_binSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    // Zero the output histogram
    CUDA_CHECK(cudaMemsetAsync(d_binSums, 0, numBins * sizeof(double), stream));

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    accumulateHistogramKernel<<<gridSize, blockSize, 0, stream>>>(
        d_weights, d_binIndices, d_binSums, numEvents, numBins
    );

    CUDA_CHECK_KERNEL();
}

void launchHistogramAccumulationWithSquares(
    const double* d_weights,
    const double* d_weightsSquared,
    const int32_t* d_binIndices,
    double* d_binSums,
    double* d_binSqSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    // Zero the output histograms
    CUDA_CHECK(cudaMemsetAsync(d_binSums, 0, numBins * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_binSqSums, 0, numBins * sizeof(double), stream));

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    accumulateHistogramWithSquaresKernel<<<gridSize, blockSize, 0, stream>>>(
        d_weights, d_weightsSquared, d_binIndices,
        d_binSums, d_binSqSums, numEvents, numBins
    );

    CUDA_CHECK_KERNEL();
}

void launchHistogramAccumulationShared(
    const double* d_weights,
    const int32_t* d_binIndices,
    double* d_binSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
) {
    if (numEvents == 0) return;

    // Zero the output histogram
    CUDA_CHECK(cudaMemsetAsync(d_binSums, 0, numBins * sizeof(double), stream));

    // Use fewer blocks but more threads per block for shared memory approach
    int blockSize = 256;
    int gridSize = min(256, (numEvents + blockSize - 1) / blockSize);
    size_t sharedMemSize = numBins * sizeof(double);

    // Check shared memory requirements
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (sharedMemSize > prop.sharedMemPerBlock) {
        // Fall back to simple kernel if histogram doesn't fit in shared memory
        launchHistogramAccumulation(d_weights, d_binIndices, d_binSums,
                                    numEvents, numBins, stream);
        return;
    }

    accumulateHistogramSharedKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_weights, d_binIndices, d_binSums, numEvents, numBins
    );

    CUDA_CHECK_KERNEL();
}

//==============================================================================
// Utility Kernels
//==============================================================================

/**
 * @brief Initialize histogram bins to zero
 */
__global__ void zeroHistogramKernel(double* histogram, int numBins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numBins) {
        histogram[tid] = 0.0;
    }
}

/**
 * @brief Copy histogram from device to host
 */
void downloadHistogram(
    const double* d_histogram,
    double* h_histogram,
    int numBins,
    cudaStream_t stream
) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(h_histogram, d_histogram,
                                   numBins * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram,
                              numBins * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
}

/**
 * @brief Get sum of all histogram bins (parallel reduction)
 */
__global__ void sumHistogramKernel(
    const double* __restrict__ histogram,
    double* __restrict__ result,
    int numBins
) {
    extern __shared__ double s_partial[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate
    double sum = 0.0;
    while (i < numBins) {
        sum += histogram[i];
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
        atomicAddDouble(result, s_partial[0]);
    }
}

double getTotalHistogramSum(
    const double* d_histogram,
    int numBins,
    cudaStream_t stream
) {
    // Allocate device memory for result
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(double), stream));

    int blockSize = 256;
    int gridSize = min(256, (numBins + blockSize - 1) / blockSize);
    size_t sharedMemSize = blockSize * sizeof(double);

    sumHistogramKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_histogram, d_result, numBins
    );

    double h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_result);
    return h_result;
}

} // namespace gpu
} // namespace gollumfit
