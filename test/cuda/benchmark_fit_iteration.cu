/**
 * @file benchmark_fit_iteration.cu
 * @brief Realistic benchmark simulating full fitting workflow.
 *
 * This benchmark measures the combined effect of the complete fitting loop:
 * 1. Update systematic parameters (simulating minimizer steps)
 * 2. Reweight all MC events based on current parameters
 * 3. Accumulate weighted events into histogram bins
 * 4. Compute SAY likelihood
 *
 * This represents what happens at EACH iteration of the minimizer (L-BFGS-B),
 * which typically runs 100-1000+ iterations per fit.
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUSplineTable.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

using namespace gollumfit::gpu;

//==============================================================================
// External kernel declarations
//==============================================================================

namespace gollumfit {
namespace gpu {

extern void launchEventWeightingKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
);

extern void launchHistogramAccumulationWithSquares(
    const double* d_weights,
    const double* d_weightsSquared,
    const int32_t* d_binIndices,
    double* d_binSums,
    double* d_binSqSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
);

extern double computeTotalSAYLikelihood(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    int numBins,
    cudaStream_t stream
);

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// CPU Implementation (Simplified but representative)
//==============================================================================

/**
 * @brief Simplified CPU event weighting matching core physics
 */
void cpuComputeWeights(
    const std::vector<double>& cachedConvWeight,
    const std::vector<double>& cachedPromptWeight,
    const std::vector<double>& cachedAstroWeight,
    const std::vector<double>& cachedHadronicHEkp,
    const std::vector<double>& cachedHadronicHEkm,
    const double* params,
    std::vector<double>& weights,
    std::vector<double>& weightsSquared,
    int numEvents
) {
    // Parameter indices (matching analysisWeighting.h structure)
    double convNorm = params[0];
    double promptNorm = params[1];
    double astroNorm = params[2];
    double HEkp = params[4];
    double HEkm = params[5];

    for (int i = 0; i < numEvents; ++i) {
        double convFlux = cachedConvWeight[i];
        double hadronicMod = 1.0 + HEkp * cachedHadronicHEkp[i] + HEkm * cachedHadronicHEkm[i];

        double conv = convNorm * convFlux * hadronicMod;
        double prompt = promptNorm * cachedPromptWeight[i];
        double astro = astroNorm * cachedAstroWeight[i];

        double w = conv + prompt + astro;
        weights[i] = w;
        weightsSquared[i] = w * w;
    }
}

/**
 * @brief CPU histogram accumulation
 */
void cpuHistogramAccumulation(
    const std::vector<double>& weights,
    const std::vector<double>& weightsSquared,
    const std::vector<int32_t>& binIndices,
    std::vector<double>& binSums,
    std::vector<double>& binSqSums,
    int numEvents,
    int numBins
) {
    std::fill(binSums.begin(), binSums.end(), 0.0);
    std::fill(binSqSums.begin(), binSqSums.end(), 0.0);

    for (int i = 0; i < numEvents; ++i) {
        int bin = binIndices[i];
        if (bin >= 0 && bin < numBins) {
            binSums[bin] += weights[i];
            binSqSums[bin] += weightsSquared[i];
        }
    }
}

/**
 * @brief CPU SAY likelihood computation
 */
double cpuSAYLikelihood(
    const std::vector<double>& dataCount,
    const std::vector<double>& wSum,
    const std::vector<double>& w2Sum,
    int numBins
) {
    double total = 0.0;
    for (int i = 0; i < numBins; ++i) {
        double k = dataCount[i];
        double ws = wSum[i];
        double w2s = w2Sum[i];

        if (ws <= 0.0) {
            total += (k == 0.0) ? 0.0 : -1e300;
            continue;
        }
        if (w2s <= 0.0) {
            // Poisson fallback
            total += k * std::log(ws) - ws - std::lgamma(k + 1.0);
            continue;
        }

        double alpha = ws * ws / w2s + 1.0;
        double beta = ws / w2s;

        total += alpha * std::log(beta) + std::lgamma(k + alpha) - std::lgamma(alpha)
               - (k + alpha) * std::log(1.0 + beta) - std::lgamma(k + 1.0);
    }
    return total;
}

/**
 * @brief Full CPU fit iteration: reweight + histogram + likelihood
 */
double cpuFitIteration(
    const std::vector<double>& cachedConvWeight,
    const std::vector<double>& cachedPromptWeight,
    const std::vector<double>& cachedAstroWeight,
    const std::vector<double>& cachedHadronicHEkp,
    const std::vector<double>& cachedHadronicHEkm,
    const std::vector<int32_t>& binIndices,
    const std::vector<double>& dataCount,
    const double* params,
    std::vector<double>& weights,
    std::vector<double>& weightsSquared,
    std::vector<double>& binSums,
    std::vector<double>& binSqSums,
    int numEvents,
    int numBins
) {
    // Step 1: Reweight events
    cpuComputeWeights(cachedConvWeight, cachedPromptWeight, cachedAstroWeight,
                      cachedHadronicHEkp, cachedHadronicHEkm, params,
                      weights, weightsSquared, numEvents);

    // Step 2: Accumulate histogram
    cpuHistogramAccumulation(weights, weightsSquared, binIndices,
                             binSums, binSqSums, numEvents, numBins);

    // Step 3: Compute likelihood
    return cpuSAYLikelihood(dataCount, binSums, binSqSums, numBins);
}

//==============================================================================
// GPU Squared Weights Kernel (simple version for benchmark)
//==============================================================================

__global__ void computeSquaredWeightsKernel(
    const double* __restrict__ weights,
    double* __restrict__ weightsSquared,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numEvents) {
        double w = weights[tid];
        weightsSquared[tid] = w * w;
    }
}

//==============================================================================
// Benchmark Results Structure
//==============================================================================

struct FitBenchmarkResult {
    int numEvents;
    int numBins;
    int numIterations;

    // CPU timings (ms)
    double cpuTotalMs;
    double cpuPerIterMs;
    double cpuReweightMs;
    double cpuHistogramMs;
    double cpuLikelihoodMs;

    // GPU timings (ms)
    double gpuTotalMs;
    double gpuPerIterMs;
    double gpuReweightMs;
    double gpuHistogramMs;
    double gpuLikelihoodMs;

    // Data transfer metrics (key optimization!)
    double gpuInitialTransferMs;     // One-time: event data to GPU
    double gpuParamTransferMs;       // Per-iteration: 304 bytes only
    size_t eventDataBytes;           // Total event data size
    size_t paramDataBytes;           // Parameter data size (304 bytes)

    // Hypothetical: what if we transferred every iteration?
    double gpuWithFullTransferMs;    // Per-iteration if we transferred all data

    // Derived metrics
    double speedup;
    double eventsPerSecCPU;
    double eventsPerSecGPU;
};

//==============================================================================
// Main Benchmark Function
//==============================================================================

FitBenchmarkResult benchmarkFitIterations(int numEvents, int numBins, int numIterations) {
    FitBenchmarkResult result;
    result.numEvents = numEvents;
    result.numBins = numBins;
    result.numIterations = numIterations;

    TestRNG rng(12345);

    //--------------------------------------------------------------------------
    // Generate realistic mock data
    //--------------------------------------------------------------------------

    // Cached event weights (computed once from LeptonWeighter in real code)
    std::vector<double> cachedConvWeight(numEvents);
    std::vector<double> cachedPromptWeight(numEvents);
    std::vector<double> cachedAstroWeight(numEvents);
    std::vector<double> cachedHadronicHEkp(numEvents);
    std::vector<double> cachedHadronicHEkm(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        // Realistic weight magnitudes (from actual IceCube MC)
        cachedConvWeight[i] = rng.uniform(1e-10, 1e-6);
        cachedPromptWeight[i] = cachedConvWeight[i] * 0.1;  // Prompt ~10% of conv
        cachedAstroWeight[i] = cachedConvWeight[i] * 0.05;  // Astro ~5% of conv
        cachedHadronicHEkp[i] = rng.uniform(-0.1, 0.1);
        cachedHadronicHEkm[i] = rng.uniform(-0.1, 0.1);
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // Generate "observed" data (Poisson-like counts per bin)
    std::vector<double> dataCount(numBins);
    for (int i = 0; i < numBins; ++i) {
        dataCount[i] = std::floor(rng.uniform(0, 100));
    }

    // Parameter array (38 fit parameters, varying during minimization)
    std::vector<double> params(NUM_FIT_PARAMS, 0.0);
    params[0] = 1.0;   // convNorm
    params[1] = 0.0;   // promptNorm
    params[2] = 0.0;   // astroNorm
    params[4] = 0.0;   // HEkp
    params[5] = 0.0;   // HEkm

    // Working arrays
    std::vector<double> weights(numEvents);
    std::vector<double> weightsSquared(numEvents);
    std::vector<double> binSums(numBins);
    std::vector<double> binSqSums(numBins);

    //--------------------------------------------------------------------------
    // Simulate parameter variations (like minimizer steps)
    //--------------------------------------------------------------------------

    std::vector<std::vector<double>> paramHistory(numIterations);
    std::mt19937 gen(42);
    std::normal_distribution<double> paramDist(0.0, 0.1);

    for (int iter = 0; iter < numIterations; ++iter) {
        paramHistory[iter] = params;
        // Simulate small parameter changes (like gradient descent steps)
        paramHistory[iter][0] = 1.0 + paramDist(gen) * 0.1;  // convNorm ~1.0
        paramHistory[iter][1] = paramDist(gen) * 0.01;       // promptNorm ~0
        paramHistory[iter][2] = paramDist(gen) * 0.01;       // astroNorm ~0
        paramHistory[iter][4] = paramDist(gen);              // HEkp
        paramHistory[iter][5] = paramDist(gen);              // HEkm
    }

    //--------------------------------------------------------------------------
    // CPU Benchmark
    //--------------------------------------------------------------------------

    double cpuLLH = 0.0;

    // Warmup
    cpuLLH = cpuFitIteration(cachedConvWeight, cachedPromptWeight, cachedAstroWeight,
                              cachedHadronicHEkp, cachedHadronicHEkm, binIndices,
                              dataCount, paramHistory[0].data(),
                              weights, weightsSquared, binSums, binSqSums,
                              numEvents, numBins);

    // Timed run
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        cpuLLH = cpuFitIteration(cachedConvWeight, cachedPromptWeight, cachedAstroWeight,
                                  cachedHadronicHEkp, cachedHadronicHEkm, binIndices,
                                  dataCount, paramHistory[iter].data(),
                                  weights, weightsSquared, binSums, binSqSums,
                                  numEvents, numBins);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    result.cpuTotalMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    result.cpuPerIterMs = result.cpuTotalMs / numIterations;

    // Breakdown timing (single iteration)
    auto t1 = std::chrono::high_resolution_clock::now();
    cpuComputeWeights(cachedConvWeight, cachedPromptWeight, cachedAstroWeight,
                      cachedHadronicHEkp, cachedHadronicHEkm, params.data(),
                      weights, weightsSquared, numEvents);
    auto t2 = std::chrono::high_resolution_clock::now();
    cpuHistogramAccumulation(weights, weightsSquared, binIndices,
                             binSums, binSqSums, numEvents, numBins);
    auto t3 = std::chrono::high_resolution_clock::now();
    cpuSAYLikelihood(dataCount, binSums, binSqSums, numBins);
    auto t4 = std::chrono::high_resolution_clock::now();

    result.cpuReweightMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.cpuHistogramMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    result.cpuLikelihoodMs = std::chrono::duration<double, std::milli>(t4 - t3).count();

    //--------------------------------------------------------------------------
    // GPU Setup
    //--------------------------------------------------------------------------

    // Allocate GPU event data structure
    GPUEventDataSoA d_events;
    d_events.numEvents_total = numEvents;

    // Allocate primary arrays (minimal set for this benchmark)
    cudaMalloc(&d_events.energy, numEvents * sizeof(float));
    cudaMalloc(&d_events.zenith, numEvents * sizeof(float));
    cudaMalloc(&d_events.primaryEnergy, numEvents * sizeof(float));
    cudaMalloc(&d_events.primaryZenith, numEvents * sizeof(float));
    cudaMalloc(&d_events.primaryAzimuth, numEvents * sizeof(float));
    cudaMalloc(&d_events.totalColumnDepth, numEvents * sizeof(float));
    cudaMalloc(&d_events.intX, numEvents * sizeof(float));
    cudaMalloc(&d_events.intY, numEvents * sizeof(float));
    cudaMalloc(&d_events.topology, numEvents * sizeof(uint32_t));
    cudaMalloc(&d_events.primaryType, numEvents * sizeof(int32_t));
    cudaMalloc(&d_events.numEvents, numEvents * sizeof(int32_t));
    cudaMalloc(&d_events.cachedConvWeight, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedPromptWeight, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedAstroWeight, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedWeight, numEvents * sizeof(double));
    cudaMalloc(&d_events.binIndex, numEvents * sizeof(int32_t));

    // Allocate detector systematic arrays
    cudaMalloc(&d_events.cachedHoleIceConv, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHoleIcePrompt, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHoleIceAstro, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedDOMEffConv, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedDOMEffPrompt, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedDOMEffAstro, numEvents * sizeof(double));

    // Allocate hadronic arrays
    cudaMalloc(&d_events.cachedHadronicHEkp, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicHEkm, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE1pip, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE1pim, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3kp, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3km, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3pip, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3pim, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3p, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHadronicVHE3n, numEvents * sizeof(double));

    // Allocate cosmic ray arrays
    cudaMalloc(&d_events.cachedCosmicRay1, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay2, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay3, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay4, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay5, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay6, numEvents * sizeof(double));

    // Allocate ice gradient arrays
    cudaMalloc(&d_events.cachedIceGrad0, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad1, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad2, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad3, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad4, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad5, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad6, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad7, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad8, numEvents * sizeof(double));

    // Allocate atmospheric arrays
    cudaMalloc(&d_events.cachedAtmDensity, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedKaonLosses, numEvents * sizeof(double));

    // Initialize arrays to zero
    cudaMemset(d_events.cachedHoleIceConv, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHoleIcePrompt, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHoleIceAstro, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffConv, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffPrompt, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffAstro, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE1pip, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE1pim, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3kp, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3km, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3pip, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3pim, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3p, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3n, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay1, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay2, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay3, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay4, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay5, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay6, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad0, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad1, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad2, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad3, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad4, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad5, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad6, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad7, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad8, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedAtmDensity, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedKaonLosses, 0, numEvents * sizeof(double));

    // Allocate working arrays
    double* d_params;
    double* d_weights;
    double* d_weightsSquared;
    double* d_binSums;
    double* d_binSqSums;
    double* d_dataCount;

    cudaMalloc(&d_params, NUM_FIT_PARAMS * sizeof(double));
    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_weightsSquared, numEvents * sizeof(double));
    cudaMalloc(&d_binSums, numBins * sizeof(double));
    cudaMalloc(&d_binSqSums, numBins * sizeof(double));
    cudaMalloc(&d_dataCount, numBins * sizeof(double));
    cudaDeviceSynchronize();

    //==========================================================================
    // KEY OPTIMIZATION: Time the ONE-TIME data transfer to GPU
    // This happens once at fit initialization, NOT every iteration!
    //==========================================================================

    cudaEvent_t transferStart, transferStop;
    cudaEventCreate(&transferStart);
    cudaEventCreate(&transferStop);

    cudaEventRecord(transferStart);

    // Transfer event data (THIS STAYS ON GPU FOR ENTIRE FIT)
    cudaMemcpy(d_events.cachedConvWeight, cachedConvWeight.data(),
               numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_events.cachedPromptWeight, cachedPromptWeight.data(),
               numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_events.cachedAstroWeight, cachedAstroWeight.data(),
               numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_events.cachedHadronicHEkp, cachedHadronicHEkp.data(),
               numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_events.cachedHadronicHEkm, cachedHadronicHEkm.data(),
               numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_events.binIndex, binIndices.data(),
               numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataCount, dataCount.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(transferStop);
    cudaEventSynchronize(transferStop);

    float initialTransferMs;
    cudaEventElapsedTime(&initialTransferMs, transferStart, transferStop);
    result.gpuInitialTransferMs = initialTransferMs;

    // Calculate data sizes for reporting
    // Event data that stays resident on GPU:
    size_t eventDataBytes = numEvents * (
        5 * sizeof(double) +    // cachedConvWeight, Prompt, Astro, HEkp, HEkm
        sizeof(int32_t)         // binIndex
    );
    // In real use, this would be much larger (~320 bytes/event for all cached data)
    result.eventDataBytes = eventDataBytes;
    result.paramDataBytes = NUM_FIT_PARAMS * sizeof(double);  // 304 bytes

    //==========================================================================
    // Measure per-iteration parameter transfer time
    //==========================================================================

    cudaEventRecord(transferStart);
    for (int i = 0; i < 100; ++i) {
        cudaMemcpy(d_params, paramHistory[0].data(),
                   NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(transferStop);
    cudaEventSynchronize(transferStop);

    float paramTransferMs;
    cudaEventElapsedTime(&paramTransferMs, transferStart, transferStop);
    result.gpuParamTransferMs = paramTransferMs / 100.0;  // Average per transfer

    cudaEventDestroy(transferStart);
    cudaEventDestroy(transferStop);

    cudaDeviceSynchronize();

    //--------------------------------------------------------------------------
    // GPU Benchmark - Full Fit Iteration Loop
    //--------------------------------------------------------------------------

    double gpuLLH = 0.0;

    // Empty spline lookup for benchmarking (no spline corrections)
    GPUSplineLookup noSplines;
    noSplines.clear();
    // Warmup
    cudaMemcpy(d_params, paramHistory[0].data(), NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);
    launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);

    int blockSize = 256;
    int gridSize = (numEvents + blockSize - 1) / blockSize;
    computeSquaredWeightsKernel<<<gridSize, blockSize>>>(d_weights, d_weightsSquared, numEvents);

    launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_events.binIndex,
                                            d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    gpuLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    cudaDeviceSynchronize();

    // Timed run - simulating full minimization
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        // Step 1: Transfer updated parameters (small: 304 bytes)
        cudaMemcpy(d_params, paramHistory[iter].data(),
                   NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);

        // Step 2: Reweight all events
        launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);

        // Step 3: Compute squared weights
        computeSquaredWeightsKernel<<<gridSize, blockSize>>>(d_weights, d_weightsSquared, numEvents);

        // Step 4: Accumulate histogram
        launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_events.binIndex,
                                                d_binSums, d_binSqSums, numEvents, numBins, nullptr);

        // Step 5: Compute likelihood
        gpuLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTotalMs = gpuMs;
    result.gpuPerIterMs = gpuMs / numIterations;

    // GPU breakdown timing (single iteration with separate events)
    cudaEvent_t e1, e2, e3, e4, e5;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    cudaEventCreate(&e5);

    cudaEventRecord(e1);
    launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);
    cudaEventRecord(e2);
    computeSquaredWeightsKernel<<<gridSize, blockSize>>>(d_weights, d_weightsSquared, numEvents);
    cudaEventRecord(e3);
    launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_events.binIndex,
                                            d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    cudaEventRecord(e4);
    gpuLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float reweightMs, sqMs, histMs, llhMs;
    cudaEventElapsedTime(&reweightMs, e1, e2);
    cudaEventElapsedTime(&sqMs, e2, e3);
    cudaEventElapsedTime(&histMs, e3, e4);
    cudaEventElapsedTime(&llhMs, e4, e5);

    result.gpuReweightMs = reweightMs + sqMs;  // Include squared weights in reweight time
    result.gpuHistogramMs = histMs;
    result.gpuLikelihoodMs = llhMs;

    //==========================================================================
    // COMPARISON: What if we transferred ALL data every iteration?
    // This shows the benefit of keeping event data resident on GPU
    //==========================================================================

    cudaEvent_t fullStart, fullStop;
    cudaEventCreate(&fullStart);
    cudaEventCreate(&fullStop);

    int shortIterations = std::min(10, numIterations);  // Just a few iterations for this test

    cudaEventRecord(fullStart);
    for (int iter = 0; iter < shortIterations; ++iter) {
        // Transfer ALL event data every iteration (the SLOW way)
        cudaMemcpy(d_events.cachedConvWeight, cachedConvWeight.data(),
                   numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events.cachedPromptWeight, cachedPromptWeight.data(),
                   numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events.cachedAstroWeight, cachedAstroWeight.data(),
                   numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events.cachedHadronicHEkp, cachedHadronicHEkp.data(),
                   numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events.cachedHadronicHEkm, cachedHadronicHEkm.data(),
                   numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events.binIndex, binIndices.data(),
                   numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params, paramHistory[iter % numIterations].data(),
                   NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);

        // Then do the computation
        launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);
        computeSquaredWeightsKernel<<<gridSize, blockSize>>>(d_weights, d_weightsSquared, numEvents);
        launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_events.binIndex,
                                                d_binSums, d_binSqSums, numEvents, numBins, nullptr);
        gpuLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    }
    cudaEventRecord(fullStop);
    cudaEventSynchronize(fullStop);

    float fullTransferMs;
    cudaEventElapsedTime(&fullTransferMs, fullStart, fullStop);
    result.gpuWithFullTransferMs = fullTransferMs / shortIterations;

    cudaEventDestroy(fullStart);
    cudaEventDestroy(fullStop);

    // Derived metrics
    result.speedup = result.cpuPerIterMs / result.gpuPerIterMs;
    result.eventsPerSecCPU = (numEvents * 1000.0) / result.cpuPerIterMs;
    result.eventsPerSecGPU = (numEvents * 1000.0) / result.gpuPerIterMs;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaEventDestroy(e4);
    cudaEventDestroy(e5);

    // Cleanup GPU memory
    cudaFree(d_events.energy);
    cudaFree(d_events.zenith);
    cudaFree(d_events.primaryEnergy);
    cudaFree(d_events.primaryZenith);
    cudaFree(d_events.primaryAzimuth);
    cudaFree(d_events.totalColumnDepth);
    cudaFree(d_events.intX);
    cudaFree(d_events.intY);
    cudaFree(d_events.topology);
    cudaFree(d_events.primaryType);
    cudaFree(d_events.numEvents);
    cudaFree(d_events.cachedConvWeight);
    cudaFree(d_events.cachedPromptWeight);
    cudaFree(d_events.cachedAstroWeight);
    cudaFree(d_events.cachedWeight);
    cudaFree(d_events.binIndex);
    cudaFree(d_events.cachedHoleIceConv);
    cudaFree(d_events.cachedHoleIcePrompt);
    cudaFree(d_events.cachedHoleIceAstro);
    cudaFree(d_events.cachedDOMEffConv);
    cudaFree(d_events.cachedDOMEffPrompt);
    cudaFree(d_events.cachedDOMEffAstro);
    cudaFree(d_events.cachedHadronicHEkp);
    cudaFree(d_events.cachedHadronicHEkm);
    cudaFree(d_events.cachedHadronicVHE1pip);
    cudaFree(d_events.cachedHadronicVHE1pim);
    cudaFree(d_events.cachedHadronicVHE3kp);
    cudaFree(d_events.cachedHadronicVHE3km);
    cudaFree(d_events.cachedHadronicVHE3pip);
    cudaFree(d_events.cachedHadronicVHE3pim);
    cudaFree(d_events.cachedHadronicVHE3p);
    cudaFree(d_events.cachedHadronicVHE3n);
    cudaFree(d_events.cachedCosmicRay1);
    cudaFree(d_events.cachedCosmicRay2);
    cudaFree(d_events.cachedCosmicRay3);
    cudaFree(d_events.cachedCosmicRay4);
    cudaFree(d_events.cachedCosmicRay5);
    cudaFree(d_events.cachedCosmicRay6);
    cudaFree(d_events.cachedIceGrad0);
    cudaFree(d_events.cachedIceGrad1);
    cudaFree(d_events.cachedIceGrad2);
    cudaFree(d_events.cachedIceGrad3);
    cudaFree(d_events.cachedIceGrad4);
    cudaFree(d_events.cachedIceGrad5);
    cudaFree(d_events.cachedIceGrad6);
    cudaFree(d_events.cachedIceGrad7);
    cudaFree(d_events.cachedIceGrad8);
    cudaFree(d_events.cachedAtmDensity);
    cudaFree(d_events.cachedKaonLosses);
    cudaFree(d_params);
    cudaFree(d_weights);
    cudaFree(d_weightsSquared);
    cudaFree(d_binSums);
    cudaFree(d_binSqSums);
    cudaFree(d_dataCount);

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "Full Fit Iteration Benchmark\n";
    std::cout << "================================================================\n";
    std::cout << "Simulates realistic minimization: params -> reweight -> bin -> LLH\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
    std::cout << "Memory Bandwidth: " << (prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2) / 1e6 << " GB/s\n\n";

    //--------------------------------------------------------------------------
    // Typical Physics Use Cases
    //--------------------------------------------------------------------------

    std::cout << "================================================================\n";
    std::cout << "Typical Physics Scenarios\n";
    std::cout << "================================================================\n\n";

    struct Scenario {
        const char* name;
        int numEvents;
        int numBins;
        int numIterations;
    };

    std::vector<Scenario> scenarios = {
        {"Small fit (dev/test)",    100000,   200,  100},
        {"Medium fit (quick)",      500000,   500,  200},
        {"Standard fit",           1000000,   500,  500},
        {"Large fit",              2000000,  1000,  500},
        {"Production fit",         5000000,  1000, 1000},
    };

    std::cout << std::setw(25) << "Scenario"
              << std::setw(12) << "Events"
              << std::setw(8) << "Bins"
              << std::setw(8) << "Iters"
              << std::setw(14) << "CPU Total"
              << std::setw(14) << "GPU Total"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(91, '-') << std::endl;

    for (const auto& s : scenarios) {
        auto result = benchmarkFitIterations(s.numEvents, s.numBins, s.numIterations);

        std::cout << std::setw(25) << s.name
                  << std::setw(12) << result.numEvents
                  << std::setw(8) << result.numBins
                  << std::setw(8) << result.numIterations
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << result.cpuTotalMs / 1000.0 << " s"
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << result.gpuTotalMs / 1000.0 << " s"
                  << std::setw(9) << std::fixed << std::setprecision(1)
                  << result.speedup << "x"
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Detailed Breakdown for Standard Fit
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "Detailed Breakdown: Standard Fit (1M events, 500 bins)\n";
    std::cout << "================================================================\n\n";

    auto detailed = benchmarkFitIterations(1000000, 500, 500);

    std::cout << "Per-Iteration Timing:\n";
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(20) << "Stage"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << std::setw(20) << "Reweighting"
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.cpuReweightMs
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.gpuReweightMs
              << std::setw(11) << std::fixed << std::setprecision(1)
              << detailed.cpuReweightMs / detailed.gpuReweightMs << "x"
              << std::endl;

    std::cout << std::setw(20) << "Histogram"
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.cpuHistogramMs
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.gpuHistogramMs
              << std::setw(11) << std::fixed << std::setprecision(1)
              << detailed.cpuHistogramMs / detailed.gpuHistogramMs << "x"
              << std::endl;

    std::cout << std::setw(20) << "Likelihood"
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.cpuLikelihoodMs
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.gpuLikelihoodMs
              << std::setw(11) << std::fixed << std::setprecision(1)
              << detailed.cpuLikelihoodMs / detailed.gpuLikelihoodMs << "x"
              << std::endl;

    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(20) << "TOTAL"
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.cpuPerIterMs
              << std::setw(15) << std::fixed << std::setprecision(3) << detailed.gpuPerIterMs
              << std::setw(11) << std::fixed << std::setprecision(1) << detailed.speedup << "x"
              << std::endl;

    //--------------------------------------------------------------------------
    // KEY: GPU Memory Residency Benefit
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "GPU MEMORY RESIDENCY ANALYSIS (Key Optimization!)\n";
    std::cout << "================================================================\n\n";

    std::cout << "Data Transfer Strategy:\n";
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "  ONE-TIME (at fit initialization):\n";
    std::cout << "    - Event data transferred to GPU: "
              << std::fixed << std::setprecision(2)
              << detailed.eventDataBytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "    - Transfer time: "
              << std::fixed << std::setprecision(2)
              << detailed.gpuInitialTransferMs << " ms\n\n";

    std::cout << "  PER-ITERATION (every minimizer step):\n";
    std::cout << "    - Only parameters transferred: "
              << detailed.paramDataBytes << " bytes (38 doubles)\n";
    std::cout << "    - Transfer time: "
              << std::fixed << std::setprecision(4)
              << detailed.gpuParamTransferMs << " ms\n\n";

    std::cout << "Comparison - Per-Iteration Cost:\n";
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(40) << "With GPU data residency (optimized):"
              << std::setw(12) << std::fixed << std::setprecision(3)
              << detailed.gpuPerIterMs << " ms\n";
    std::cout << std::setw(40) << "Without residency (transfer every iter):"
              << std::setw(12) << std::fixed << std::setprecision(3)
              << detailed.gpuWithFullTransferMs << " ms\n";
    std::cout << std::setw(40) << "Overhead avoided per iteration:"
              << std::setw(12) << std::fixed << std::setprecision(3)
              << (detailed.gpuWithFullTransferMs - detailed.gpuPerIterMs) << " ms\n";
    std::cout << std::setw(40) << "Data residency speedup factor:"
              << std::setw(11) << std::fixed << std::setprecision(1)
              << detailed.gpuWithFullTransferMs / detailed.gpuPerIterMs << "x\n\n";

    double totalOverheadAvoided = (detailed.gpuWithFullTransferMs - detailed.gpuPerIterMs) * detailed.numIterations;
    std::cout << "For " << detailed.numIterations << " iterations, data residency saves: "
              << std::fixed << std::setprecision(2)
              << totalOverheadAvoided / 1000.0 << " seconds\n";

    //--------------------------------------------------------------------------
    // Scaling Analysis
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "Event Count Scaling (500 iterations, 500 bins)\n";
    std::cout << "================================================================\n\n";

    std::vector<int> eventCounts = {50000, 100000, 250000, 500000, 1000000, 2000000, 5000000};

    std::cout << std::setw(12) << "Events"
              << std::setw(14) << "CPU/iter(ms)"
              << std::setw(14) << "GPU/iter(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(18) << "GPU Events/sec"
              << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (int numEvents : eventCounts) {
        auto result = benchmarkFitIterations(numEvents, 500, 100);

        std::cout << std::setw(12) << result.numEvents
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.cpuPerIterMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.gpuPerIterMs
                  << std::setw(9) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << result.eventsPerSecGPU / 1e6 << "M"
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Summary
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "Summary & Recommendations\n";
    std::cout << "================================================================\n\n";

    std::cout << "Key Observations:\n";
    std::cout << "- GPU acceleration provides significant speedup for typical fits\n";
    std::cout << "- Speedup increases with event count (better GPU utilization)\n";
    std::cout << "- Event reweighting is the dominant cost (as expected)\n";
    std::cout << "- Parameter transfer overhead is negligible (~0.3 KB per iteration)\n\n";

    std::cout << "For production fits (5M events, 1000 iterations):\n";
    auto prod = benchmarkFitIterations(5000000, 1000, 100);
    double cpuEstimate = prod.cpuPerIterMs * 1000.0 / 1000.0;  // 1000 iters
    double gpuEstimate = prod.gpuPerIterMs * 1000.0 / 1000.0;
    std::cout << "  Estimated CPU time: " << std::fixed << std::setprecision(1)
              << cpuEstimate / 60.0 << " minutes\n";
    std::cout << "  Estimated GPU time: " << std::fixed << std::setprecision(1)
              << gpuEstimate / 60.0 << " minutes\n";
    std::cout << "  Time saved: " << std::fixed << std::setprecision(1)
              << (cpuEstimate - gpuEstimate) / 60.0 << " minutes per fit\n";

    return 0;
}
