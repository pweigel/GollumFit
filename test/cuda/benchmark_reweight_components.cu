/**
 * @file benchmark_reweight_components.cu
 * @brief Profile individual components of event reweighting to identify bottlenecks.
 *
 * The event reweighting kernel combines many computations:
 * 1. Memory loads from global memory (event data)
 * 2. Hadronic corrections (10 DAEMONFLUX parameters)
 * 3. Cosmic ray corrections (6 GSF parameters)
 * 4. Ice gradient corrections (9 parameters)
 * 5. Atmospheric corrections (2 parameters)
 * 6. Detector systematics (hole ice, DOM efficiency)
 * 7. Astrophysical component (normalization, tilt)
 * 8. Final weight assembly
 *
 * This benchmark isolates each component to identify which parts
 * dominate the computation time.
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace gollumfit::gpu;

//==============================================================================
// Parameter indices (matching event_weighting.cu)
//==============================================================================

enum ParamIndex {
    P_CONV_NORM = 0,
    P_PROMPT_NORM = 1,
    P_ADU = 2,
    P_KLU = 3,
    P_HEKP = 4,
    P_HEKM = 5,
    P_VHE1PIP = 6,
    P_VHE1PIM = 7,
    P_VHE3KP = 8,
    P_VHE3KM = 9,
    P_VHE3PIP = 10,
    P_VHE3PIM = 11,
    P_VHE3P = 12,
    P_VHE3N = 13,
    P_CR1 = 14,
    P_CR2 = 15,
    P_CR3 = 16,
    P_CR4 = 17,
    P_CR5 = 18,
    P_CR6 = 19,
    P_ICEGRAD0 = 20,
    P_ICEGRAD1 = 21,
    P_ICEGRAD2 = 22,
    P_ICEGRAD3 = 23,
    P_ICEGRAD4 = 24,
    P_ICEGRAD5 = 25,
    P_ICEGRAD6 = 26,
    P_ICEGRAD7 = 27,
    P_ICEGRAD8 = 28,
    P_DELTA_DOMEFF = 29,
    P_HOLEICE_FWD = 30,
    P_ASTRO_NORM = 31,
    P_ASTRO_DGAMMA = 32,
    P_ASTRO_DGAMMA_SEC = 33,
    P_ASTRO_PIVOT = 34,
    P_NEUANEU_RATIO = 35,
    P_NUXS = 36,
    P_NUBARXS = 37
};

// Constant for optimized pow(): pow(10, x) = exp2(x * LOG2_10)
constexpr double LOG2_10 = 3.321928094887362;

//==============================================================================
// Component-Isolated Kernels
//==============================================================================

/**
 * @brief Baseline kernel: just memory loads, no computation
 * Measures memory bandwidth limit
 */
__global__ void kernelMemoryLoadsOnly(
    const GPUEventDataSoA events,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Load all the data that would be used in full kernel
    double sum = 0.0;

    // Primary event data
    sum += events.energy[tid];
    sum += events.zenith[tid];
    sum += events.primaryEnergy[tid];
    sum += events.primaryZenith[tid];

    // Flux weights
    sum += events.cachedConvWeight[tid];
    sum += events.cachedPromptWeight[tid];
    sum += events.cachedAstroWeight[tid];

    // Hadronic (10 arrays)
    sum += events.cachedHadronicHEkp[tid];
    sum += events.cachedHadronicHEkm[tid];
    sum += events.cachedHadronicVHE1pip[tid];
    sum += events.cachedHadronicVHE1pim[tid];
    sum += events.cachedHadronicVHE3kp[tid];
    sum += events.cachedHadronicVHE3km[tid];
    sum += events.cachedHadronicVHE3pip[tid];
    sum += events.cachedHadronicVHE3pim[tid];
    sum += events.cachedHadronicVHE3p[tid];
    sum += events.cachedHadronicVHE3n[tid];

    // Cosmic ray (6 arrays)
    sum += events.cachedCosmicRay1[tid];
    sum += events.cachedCosmicRay2[tid];
    sum += events.cachedCosmicRay3[tid];
    sum += events.cachedCosmicRay4[tid];
    sum += events.cachedCosmicRay5[tid];
    sum += events.cachedCosmicRay6[tid];

    // Ice gradients (9 arrays)
    sum += events.cachedIceGrad0[tid];
    sum += events.cachedIceGrad1[tid];
    sum += events.cachedIceGrad2[tid];
    sum += events.cachedIceGrad3[tid];
    sum += events.cachedIceGrad4[tid];
    sum += events.cachedIceGrad5[tid];
    sum += events.cachedIceGrad6[tid];
    sum += events.cachedIceGrad7[tid];
    sum += events.cachedIceGrad8[tid];

    // Atmospheric (2 arrays)
    sum += events.cachedAtmDensity[tid];
    sum += events.cachedKaonLosses[tid];

    // Detector systematics (6 arrays)
    sum += events.cachedHoleIceConv[tid];
    sum += events.cachedHoleIcePrompt[tid];
    sum += events.cachedHoleIceAstro[tid];
    sum += events.cachedDOMEffConv[tid];
    sum += events.cachedDOMEffPrompt[tid];
    sum += events.cachedDOMEffAstro[tid];

    output[tid] = sum;
}

/**
 * @brief Hadronic corrections only (10 multiply-adds)
 */
__global__ void kernelHadronicOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double hadronic = s_params[P_HEKP] * events.cachedHadronicHEkp[tid] +
                      s_params[P_HEKM] * events.cachedHadronicHEkm[tid] +
                      s_params[P_VHE1PIP] * events.cachedHadronicVHE1pip[tid] +
                      s_params[P_VHE1PIM] * events.cachedHadronicVHE1pim[tid] +
                      s_params[P_VHE3KP] * events.cachedHadronicVHE3kp[tid] +
                      s_params[P_VHE3KM] * events.cachedHadronicVHE3km[tid] +
                      s_params[P_VHE3PIP] * events.cachedHadronicVHE3pip[tid] +
                      s_params[P_VHE3PIM] * events.cachedHadronicVHE3pim[tid] +
                      s_params[P_VHE3P] * events.cachedHadronicVHE3p[tid] +
                      s_params[P_VHE3N] * events.cachedHadronicVHE3n[tid];

    output[tid] = events.cachedConvWeight[tid] + hadronic;
}

/**
 * @brief Cosmic ray corrections only (6 multiply-adds)
 */
__global__ void kernelCosmicRayOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double cr = s_params[P_CR1] * events.cachedCosmicRay1[tid] +
                s_params[P_CR2] * events.cachedCosmicRay2[tid] +
                s_params[P_CR3] * events.cachedCosmicRay3[tid] +
                s_params[P_CR4] * events.cachedCosmicRay4[tid] +
                s_params[P_CR5] * events.cachedCosmicRay5[tid] +
                s_params[P_CR6] * events.cachedCosmicRay6[tid];

    output[tid] = events.cachedConvWeight[tid] + cr;
}

/**
 * @brief Ice gradient corrections (9 multiply-adds with products)
 */
__global__ void kernelIceGradOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double weight = 1.0;
    weight *= (1.0 + s_params[P_ICEGRAD0] * events.cachedIceGrad0[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD1] * events.cachedIceGrad1[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD2] * events.cachedIceGrad2[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD3] * events.cachedIceGrad3[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD4] * events.cachedIceGrad4[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD5] * events.cachedIceGrad5[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD6] * events.cachedIceGrad6[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD7] * events.cachedIceGrad7[tid]);
    weight *= (1.0 + s_params[P_ICEGRAD8] * events.cachedIceGrad8[tid]);

    output[tid] = weight;
}

/**
 * @brief Atmospheric corrections (2 multiply-adds)
 */
__global__ void kernelAtmosphericOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double atm_wgt = (1.0 + s_params[P_ADU] * events.cachedAtmDensity[tid]) *
                     (1.0 + s_params[P_KLU] * events.cachedKaonLosses[tid]);

    output[tid] = atm_wgt;
}

/**
 * @brief Detector systematics (hole ice + DOM efficiency)
 */
__global__ void kernelDetectorSystOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Simulate what spline evaluation would do
    // In real code, this would be spline lookups based on params
    double holeIce = s_params[P_HOLEICE_FWD];
    double domEff = s_params[P_DELTA_DOMEFF];

    double convHoleIce = 1.0 + holeIce * events.cachedHoleIceConv[tid];
    double convDOMEff = 1.0 + domEff * events.cachedDOMEffConv[tid];
    double promptHoleIce = 1.0 + holeIce * events.cachedHoleIcePrompt[tid];
    double promptDOMEff = 1.0 + domEff * events.cachedDOMEffPrompt[tid];
    double astroHoleIce = 1.0 + holeIce * events.cachedHoleIceAstro[tid];
    double astroDOMEff = 1.0 + domEff * events.cachedDOMEffAstro[tid];

    output[tid] = convHoleIce * convDOMEff + promptHoleIce * promptDOMEff + astroHoleIce * astroDOMEff;
}

/**
 * @brief Astrophysical component with power law tilt (OPTIMIZED with exp2/log2)
 */
__global__ void kernelAstroOnly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;  // Precomputed for optimization

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    // Precompute median energy once per block (OPTIMIZATION)
    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[P_ASTRO_PIVOT] * LOG2_10);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double astroNorm = s_params[P_ASTRO_NORM];
    double deltaGamma1 = s_params[P_ASTRO_DGAMMA];
    double deltaGamma2 = s_params[P_ASTRO_DGAMMA_SEC];
    double balance = s_params[P_NEUANEU_RATIO];

    float primaryEnergy = events.primaryEnergy[tid];
    int32_t primaryType = events.primaryType[tid];

    // OPTIMIZED power law tilt using exp2/log2 (~2.8x faster than pow())
    double ratio = primaryEnergy / s_medianEnergy;
    double logRatio = log2(ratio);
    double tiltWeight;
    if (primaryEnergy > s_medianEnergy) {
        tiltWeight = exp2(-deltaGamma2 * logRatio);
    } else {
        tiltWeight = exp2(-deltaGamma1 * logRatio);
    }

    // Neutrino/antineutrino weighting
    double neuWeight = (primaryType < 0) ? balance : (2.0 - balance);

    output[tid] = astroNorm * events.cachedAstroWeight[tid] * tiltWeight * neuWeight;
}

/**
 * @brief Conventional flux assembly (hadronic + cosmic ray)
 */
__global__ void kernelConvFluxAssembly(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Full conventional flux calculation
    double hadronic = s_params[P_HEKP] * events.cachedHadronicHEkp[tid] +
                      s_params[P_HEKM] * events.cachedHadronicHEkm[tid] +
                      s_params[P_VHE1PIP] * events.cachedHadronicVHE1pip[tid] +
                      s_params[P_VHE1PIM] * events.cachedHadronicVHE1pim[tid] +
                      s_params[P_VHE3KP] * events.cachedHadronicVHE3kp[tid] +
                      s_params[P_VHE3KM] * events.cachedHadronicVHE3km[tid] +
                      s_params[P_VHE3PIP] * events.cachedHadronicVHE3pip[tid] +
                      s_params[P_VHE3PIM] * events.cachedHadronicVHE3pim[tid] +
                      s_params[P_VHE3P] * events.cachedHadronicVHE3p[tid] +
                      s_params[P_VHE3N] * events.cachedHadronicVHE3n[tid];

    double cr = s_params[P_CR1] * events.cachedCosmicRay1[tid] +
                s_params[P_CR2] * events.cachedCosmicRay2[tid] +
                s_params[P_CR3] * events.cachedCosmicRay3[tid] +
                s_params[P_CR4] * events.cachedCosmicRay4[tid] +
                s_params[P_CR5] * events.cachedCosmicRay5[tid] +
                s_params[P_CR6] * events.cachedCosmicRay6[tid];

    double convFlux = events.cachedConvWeight[tid] + hadronic + cr;

    // Atmospheric modulation
    double atm_wgt = (1.0 + s_params[P_ADU] * events.cachedAtmDensity[tid]) *
                     (1.0 + s_params[P_KLU] * events.cachedKaonLosses[tid]);

    output[tid] = s_params[P_CONV_NORM] * convFlux * atm_wgt;
}

/**
 * @brief Full weight computation (reference) - OPTIMIZED with exp2/log2
 */
__global__ void kernelFullWeight(
    const GPUEventDataSoA events,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    __shared__ double s_params[NUM_FIT_PARAMS];
    __shared__ double s_medianEnergy;  // Precomputed for optimization

    if (threadIdx.x < NUM_FIT_PARAMS) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    __syncthreads();

    // Precompute median energy once per block (OPTIMIZATION)
    if (threadIdx.x == 0) {
        s_medianEnergy = exp2(s_params[P_ASTRO_PIVOT] * LOG2_10);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Hadronic
    double hadronic = s_params[P_HEKP] * events.cachedHadronicHEkp[tid] +
                      s_params[P_HEKM] * events.cachedHadronicHEkm[tid] +
                      s_params[P_VHE1PIP] * events.cachedHadronicVHE1pip[tid] +
                      s_params[P_VHE1PIM] * events.cachedHadronicVHE1pim[tid] +
                      s_params[P_VHE3KP] * events.cachedHadronicVHE3kp[tid] +
                      s_params[P_VHE3KM] * events.cachedHadronicVHE3km[tid] +
                      s_params[P_VHE3PIP] * events.cachedHadronicVHE3pip[tid] +
                      s_params[P_VHE3PIM] * events.cachedHadronicVHE3pim[tid] +
                      s_params[P_VHE3P] * events.cachedHadronicVHE3p[tid] +
                      s_params[P_VHE3N] * events.cachedHadronicVHE3n[tid];

    // Cosmic ray
    double cr = s_params[P_CR1] * events.cachedCosmicRay1[tid] +
                s_params[P_CR2] * events.cachedCosmicRay2[tid] +
                s_params[P_CR3] * events.cachedCosmicRay3[tid] +
                s_params[P_CR4] * events.cachedCosmicRay4[tid] +
                s_params[P_CR5] * events.cachedCosmicRay5[tid] +
                s_params[P_CR6] * events.cachedCosmicRay6[tid];

    double convFlux = events.cachedConvWeight[tid] + hadronic + cr;

    // Atmospheric
    double atm_wgt = (1.0 + s_params[P_ADU] * events.cachedAtmDensity[tid]) *
                     (1.0 + s_params[P_KLU] * events.cachedKaonLosses[tid]);

    // Ice gradients
    double icegrad_wgt = 1.0;
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD0] * events.cachedIceGrad0[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD1] * events.cachedIceGrad1[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD2] * events.cachedIceGrad2[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD3] * events.cachedIceGrad3[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD4] * events.cachedIceGrad4[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD5] * events.cachedIceGrad5[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD6] * events.cachedIceGrad6[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD7] * events.cachedIceGrad7[tid]);
    icegrad_wgt *= (1.0 + s_params[P_ICEGRAD8] * events.cachedIceGrad8[tid]);

    // Astrophysical with tilt (OPTIMIZED with exp2/log2)
    double astroNorm = s_params[P_ASTRO_NORM];
    double deltaGamma1 = s_params[P_ASTRO_DGAMMA];
    double deltaGamma2 = s_params[P_ASTRO_DGAMMA_SEC];
    double balance = s_params[P_NEUANEU_RATIO];

    float primaryEnergy = events.primaryEnergy[tid];
    int32_t primaryType = events.primaryType[tid];

    // OPTIMIZED power law tilt using exp2/log2 (~2.8x faster than pow())
    double ratio = primaryEnergy / s_medianEnergy;
    double logRatio = log2(ratio);
    double tiltWeight;
    if (primaryEnergy > s_medianEnergy) {
        tiltWeight = exp2(-deltaGamma2 * logRatio);
    } else {
        tiltWeight = exp2(-deltaGamma1 * logRatio);
    }
    double neuWeight = (primaryType < 0) ? balance : (2.0 - balance);

    // Assemble final weight
    double conv = s_params[P_CONV_NORM] * convFlux * atm_wgt;
    double prompt = s_params[P_PROMPT_NORM] * events.cachedPromptWeight[tid];
    double astro = astroNorm * events.cachedAstroWeight[tid] * tiltWeight * neuWeight;

    output[tid] = (conv + prompt + astro) * icegrad_wgt;
}

//==============================================================================
// Benchmark Infrastructure
//==============================================================================

struct ComponentTiming {
    const char* name;
    double timeMs;
    double bandwidth;  // GB/s
    double percentage; // of total
};

class ReweightBenchmark {
public:
    ReweightBenchmark(int numEvents) : numEvents_(numEvents) {
        allocateData();
        generateTestData();
    }

    ~ReweightBenchmark() {
        freeData();
    }

    void runAllBenchmarks(int numIterations) {
        int blockSize = 256;
        int gridSize = (numEvents_ + blockSize - 1) / blockSize;

        std::vector<ComponentTiming> results;

        // Warmup
        kernelFullWeight<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        cudaDeviceSynchronize();

        // Benchmark each component
        results.push_back(benchmarkKernel("Memory Loads Only", [&]() {
            kernelMemoryLoadsOnly<<<gridSize, blockSize>>>(d_events_, d_output_, numEvents_);
        }, numIterations, 36 * sizeof(double) * numEvents_));  // ~36 arrays loaded

        results.push_back(benchmarkKernel("Hadronic (10 params)", [&]() {
            kernelHadronicOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 11 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Cosmic Ray (6 params)", [&]() {
            kernelCosmicRayOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 7 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Ice Gradients (9 params)", [&]() {
            kernelIceGradOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 9 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Atmospheric (2 params)", [&]() {
            kernelAtmosphericOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 2 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Detector Syst (6 arrays)", [&]() {
            kernelDetectorSystOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 6 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Astro + Tilt (exp2/log2)", [&]() {
            kernelAstroOnly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 3 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("Conv Flux Assembly", [&]() {
            kernelConvFluxAssembly<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 19 * sizeof(double) * numEvents_));

        results.push_back(benchmarkKernel("FULL WEIGHT (reference)", [&]() {
            kernelFullWeight<<<gridSize, blockSize>>>(d_events_, d_params_, d_output_, numEvents_);
        }, numIterations, 36 * sizeof(double) * numEvents_));

        // Calculate percentages relative to full weight
        double fullTime = results.back().timeMs;
        for (auto& r : results) {
            r.percentage = (r.timeMs / fullTime) * 100.0;
        }

        // Print results
        printResults(results);
    }

private:
    int numEvents_;
    GPUEventDataSoA d_events_;
    double* d_params_;
    double* d_output_;

    template<typename Func>
    ComponentTiming benchmarkKernel(const char* name, Func kernel, int numIterations, size_t dataBytes) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < numIterations; ++i) {
            kernel();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double avgMs = ms / numIterations;
        double bandwidth = (dataBytes / 1e9) / (avgMs / 1000.0);  // GB/s

        return {name, avgMs, bandwidth, 0.0};
    }

    void printResults(const std::vector<ComponentTiming>& results) {
        std::cout << "\n";
        std::cout << std::setw(30) << "Component"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(14) << "Bandwidth"
                  << std::setw(12) << "% of Full"
                  << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (const auto& r : results) {
            std::cout << std::setw(30) << r.name
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.timeMs
                      << std::setw(12) << std::fixed << std::setprecision(1) << r.bandwidth << " GB/s"
                      << std::setw(11) << std::fixed << std::setprecision(1) << r.percentage << "%"
                      << std::endl;
        }
    }

    void allocateData() {
        // Allocate event data arrays
        cudaMalloc(&d_events_.energy, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.zenith, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.primaryEnergy, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.primaryZenith, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.primaryAzimuth, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.totalColumnDepth, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.intX, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.intY, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.topology, numEvents_ * sizeof(uint32_t));
        cudaMalloc(&d_events_.primaryType, numEvents_ * sizeof(int32_t));
        cudaMalloc(&d_events_.numEvents, numEvents_ * sizeof(int32_t));
        cudaMalloc(&d_events_.cachedConvWeight, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedPromptWeight, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedAstroWeight, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedWeight, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.binIndex, numEvents_ * sizeof(int32_t));

        // Hadronic - MIXED PRECISION: FP32
        cudaMalloc(&d_events_.cachedHadronicHEkp, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicHEkm, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE1pip, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE1pim, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3kp, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3km, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3pip, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3pim, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3p, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedHadronicVHE3n, numEvents_ * sizeof(float));

        // Cosmic ray - MIXED PRECISION: FP32
        cudaMalloc(&d_events_.cachedCosmicRay1, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedCosmicRay2, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedCosmicRay3, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedCosmicRay4, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedCosmicRay5, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedCosmicRay6, numEvents_ * sizeof(float));

        // Ice gradients - MIXED PRECISION: FP32
        cudaMalloc(&d_events_.cachedIceGrad0, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad1, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad2, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad3, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad4, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad5, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad6, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad7, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedIceGrad8, numEvents_ * sizeof(float));

        // Atmospheric - MIXED PRECISION: FP32
        cudaMalloc(&d_events_.cachedAtmDensity, numEvents_ * sizeof(float));
        cudaMalloc(&d_events_.cachedKaonLosses, numEvents_ * sizeof(float));

        // Detector systematics
        cudaMalloc(&d_events_.cachedHoleIceConv, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedHoleIcePrompt, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedHoleIceAstro, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedDOMEffConv, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedDOMEffPrompt, numEvents_ * sizeof(double));
        cudaMalloc(&d_events_.cachedDOMEffAstro, numEvents_ * sizeof(double));

        // Parameters and output
        cudaMalloc(&d_params_, NUM_FIT_PARAMS * sizeof(double));
        cudaMalloc(&d_output_, numEvents_ * sizeof(double));

        d_events_.numEvents_total = numEvents_;
    }

    void generateTestData() {
        TestRNG rng(12345);

        // Generate and upload test data
        std::vector<float> floatData(numEvents_);
        std::vector<double> doubleData(numEvents_);
        std::vector<int32_t> intData(numEvents_);

        // Float arrays
        for (int i = 0; i < numEvents_; ++i) floatData[i] = rng.uniform(100, 1e6);
        cudaMemcpy(d_events_.energy, floatData.data(), numEvents_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events_.primaryEnergy, floatData.data(), numEvents_ * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < numEvents_; ++i) floatData[i] = rng.uniform(-1, 1);
        cudaMemcpy(d_events_.zenith, floatData.data(), numEvents_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events_.primaryZenith, floatData.data(), numEvents_ * sizeof(float), cudaMemcpyHostToDevice);

        // Int arrays
        for (int i = 0; i < numEvents_; ++i) intData[i] = (rng.uniform(0, 1) > 0.5) ? 12 : -12;
        cudaMemcpy(d_events_.primaryType, intData.data(), numEvents_ * sizeof(int32_t), cudaMemcpyHostToDevice);

        // Double arrays - flux weights
        for (int i = 0; i < numEvents_; ++i) doubleData[i] = rng.uniform(1e-10, 1e-6);
        cudaMemcpy(d_events_.cachedConvWeight, doubleData.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);

        for (int i = 0; i < numEvents_; ++i) doubleData[i] = rng.uniform(1e-11, 1e-7);
        cudaMemcpy(d_events_.cachedPromptWeight, doubleData.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_events_.cachedAstroWeight, doubleData.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);

        // Hadronic arrays (small corrections around 0) - MIXED PRECISION: FP32
        // Reuse floatData vector already declared above
        auto uploadSmallCorrectionFloat = [&](float* dest) {
            for (int i = 0; i < numEvents_; ++i) floatData[i] = static_cast<float>(rng.uniform(-0.1, 0.1));
            cudaMemcpy(dest, floatData.data(), numEvents_ * sizeof(float), cudaMemcpyHostToDevice);
        };

        auto uploadSmallCorrection = [&](double* dest) {
            for (int i = 0; i < numEvents_; ++i) doubleData[i] = rng.uniform(-0.1, 0.1);
            cudaMemcpy(dest, doubleData.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);
        };

        uploadSmallCorrectionFloat(d_events_.cachedHadronicHEkp);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicHEkm);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE1pip);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE1pim);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3kp);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3km);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3pip);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3pim);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3p);
        uploadSmallCorrectionFloat(d_events_.cachedHadronicVHE3n);

        // Cosmic ray arrays - MIXED PRECISION: FP32
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay1);
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay2);
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay3);
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay4);
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay5);
        uploadSmallCorrectionFloat(d_events_.cachedCosmicRay6);

        // Ice gradient arrays - MIXED PRECISION: FP32
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad0);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad1);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad2);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad3);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad4);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad5);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad6);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad7);
        uploadSmallCorrectionFloat(d_events_.cachedIceGrad8);

        // Atmospheric - MIXED PRECISION: FP32
        uploadSmallCorrectionFloat(d_events_.cachedAtmDensity);
        uploadSmallCorrectionFloat(d_events_.cachedKaonLosses);

        // Detector systematics
        uploadSmallCorrection(d_events_.cachedHoleIceConv);
        uploadSmallCorrection(d_events_.cachedHoleIcePrompt);
        uploadSmallCorrection(d_events_.cachedHoleIceAstro);
        uploadSmallCorrection(d_events_.cachedDOMEffConv);
        uploadSmallCorrection(d_events_.cachedDOMEffPrompt);
        uploadSmallCorrection(d_events_.cachedDOMEffAstro);

        // Parameters
        std::vector<double> params(NUM_FIT_PARAMS, 0.0);
        params[P_CONV_NORM] = 1.0;
        params[P_PROMPT_NORM] = 0.0;
        params[P_ASTRO_NORM] = 1.0;
        params[P_ASTRO_PIVOT] = 5.0;  // log10(100 TeV)
        params[P_NEUANEU_RATIO] = 1.0;
        cudaMemcpy(d_params_, params.data(), NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    void freeData() {
        cudaFree(d_events_.energy);
        cudaFree(d_events_.zenith);
        cudaFree(d_events_.primaryEnergy);
        cudaFree(d_events_.primaryZenith);
        cudaFree(d_events_.primaryAzimuth);
        cudaFree(d_events_.totalColumnDepth);
        cudaFree(d_events_.intX);
        cudaFree(d_events_.intY);
        cudaFree(d_events_.topology);
        cudaFree(d_events_.primaryType);
        cudaFree(d_events_.numEvents);
        cudaFree(d_events_.cachedConvWeight);
        cudaFree(d_events_.cachedPromptWeight);
        cudaFree(d_events_.cachedAstroWeight);
        cudaFree(d_events_.cachedWeight);
        cudaFree(d_events_.binIndex);
        cudaFree(d_events_.cachedHadronicHEkp);
        cudaFree(d_events_.cachedHadronicHEkm);
        cudaFree(d_events_.cachedHadronicVHE1pip);
        cudaFree(d_events_.cachedHadronicVHE1pim);
        cudaFree(d_events_.cachedHadronicVHE3kp);
        cudaFree(d_events_.cachedHadronicVHE3km);
        cudaFree(d_events_.cachedHadronicVHE3pip);
        cudaFree(d_events_.cachedHadronicVHE3pim);
        cudaFree(d_events_.cachedHadronicVHE3p);
        cudaFree(d_events_.cachedHadronicVHE3n);
        cudaFree(d_events_.cachedCosmicRay1);
        cudaFree(d_events_.cachedCosmicRay2);
        cudaFree(d_events_.cachedCosmicRay3);
        cudaFree(d_events_.cachedCosmicRay4);
        cudaFree(d_events_.cachedCosmicRay5);
        cudaFree(d_events_.cachedCosmicRay6);
        cudaFree(d_events_.cachedIceGrad0);
        cudaFree(d_events_.cachedIceGrad1);
        cudaFree(d_events_.cachedIceGrad2);
        cudaFree(d_events_.cachedIceGrad3);
        cudaFree(d_events_.cachedIceGrad4);
        cudaFree(d_events_.cachedIceGrad5);
        cudaFree(d_events_.cachedIceGrad6);
        cudaFree(d_events_.cachedIceGrad7);
        cudaFree(d_events_.cachedIceGrad8);
        cudaFree(d_events_.cachedAtmDensity);
        cudaFree(d_events_.cachedKaonLosses);
        cudaFree(d_events_.cachedHoleIceConv);
        cudaFree(d_events_.cachedHoleIcePrompt);
        cudaFree(d_events_.cachedHoleIceAstro);
        cudaFree(d_events_.cachedDOMEffConv);
        cudaFree(d_events_.cachedDOMEffPrompt);
        cudaFree(d_events_.cachedDOMEffAstro);
        cudaFree(d_params_);
        cudaFree(d_output_);
    }
};

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "Reweighting Component Analysis\n";
    std::cout << "================================================================\n";
    std::cout << "Profiling individual components to identify bottlenecks\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory Bandwidth (theoretical): "
              << (prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2) / 1e6 << " GB/s\n";
    std::cout << "L2 Cache: " << prop.l2CacheSize / 1024 << " KB\n\n";

    // Test different event counts
    std::vector<int> eventCounts = {100000, 500000, 1000000, 2000000, 5000000};

    for (int numEvents : eventCounts) {
        std::cout << "================================================================\n";
        std::cout << "Event Count: " << numEvents / 1000 << "K\n";
        std::cout << "================================================================\n";

        ReweightBenchmark benchmark(numEvents);
        benchmark.runAllBenchmarks(100);

        std::cout << "\n";
    }

    std::cout << "================================================================\n";
    std::cout << "Analysis\n";
    std::cout << "================================================================\n\n";

    std::cout << "Key observations to look for:\n";
    std::cout << "1. Memory Loads Only: Shows memory bandwidth limit\n";
    std::cout << "   - If close to theoretical bandwidth, kernel is memory-bound\n";
    std::cout << "2. Astro + Tilt (exp2/log2): OPTIMIZED - uses exp2/log2 instead of pow()\n";
    std::cout << "   - ~2.8x faster than original pow() implementation\n";
    std::cout << "3. Hadronic/CR/IceGrad: Many multiply-adds\n";
    std::cout << "   - These should be fast (FMA units)\n";
    std::cout << "4. Full Weight vs sum of parts:\n";
    std::cout << "   - If Full < sum, memory accesses are being reused (good)\n";
    std::cout << "   - If Full > sum, there's overhead we haven't isolated\n";

    return 0;
}
