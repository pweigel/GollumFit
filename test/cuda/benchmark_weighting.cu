/**
 * @file benchmark_weighting.cu
 * @brief Benchmark event weight computation: CPU vs GPU performance.
 *
 * Event weighting is the main computational bottleneck in fitting.
 * This benchmark measures:
 * 1. Weight computation for varying event counts
 * 2. Impact of different systematic parameter configurations
 * 3. Comparison with and without gradient computation
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

using namespace gollumfit::gpu;

// External kernel declarations
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

extern void launchEventWeightingWithSquaresKernel(
    const GPUEventDataSoA& events,
    const double* d_params,
    const GPUSplineLookup& splines,
    double* d_weights,
    double* d_weightsSquared,
    int numEvents,
    bool enableTotalNorm,
    cudaStream_t stream
);

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// Simple CPU Event Weighting (Approximation)
//==============================================================================

// Simplified weight computation matching GPU kernel structure
void cpuComputeWeights(
    const std::vector<double>& cachedConvWeight,
    const std::vector<double>& cachedPromptWeight,
    const std::vector<double>& cachedAstroWeight,
    const std::vector<double>& cachedHadronicHEkp,
    const std::vector<double>& cachedHadronicHEkm,
    const double* params,
    std::vector<double>& weights,
    int numEvents
) {
    // Parameter indices (simplified)
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

        weights[i] = conv + prompt + astro;
    }
}

//==============================================================================
// Benchmark Event Weight Computation
//==============================================================================

struct WeightingBenchmarkResult {
    int numEvents;
    int numIterations;
    double cpuTimeMs;
    double gpuTimeMs;
    double gpuWithSquaresMs;
    double speedup;
    double speedupWithSquares;
};

WeightingBenchmarkResult benchmarkWeighting(int numEvents, int numIterations) {
    WeightingBenchmarkResult result;
    result.numEvents = numEvents;
    result.numIterations = numIterations;

    TestRNG rng(54321);

    // Generate mock event data
    std::vector<double> cachedConvWeight(numEvents);
    std::vector<double> cachedPromptWeight(numEvents);
    std::vector<double> cachedAstroWeight(numEvents);
    std::vector<double> cachedHadronicHEkp(numEvents);
    std::vector<double> cachedHadronicHEkm(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        cachedConvWeight[i] = rng.uniform(1e-10, 1e-6);
        cachedPromptWeight[i] = cachedConvWeight[i] * 0.1;
        cachedAstroWeight[i] = cachedConvWeight[i] * 0.01;
        cachedHadronicHEkp[i] = rng.uniform(-0.1, 0.1);
        cachedHadronicHEkm[i] = rng.uniform(-0.1, 0.1);
    }

    // Parameters (38 total, but we only use a few in simplified CPU version)
    std::vector<double> params(NUM_FIT_PARAMS, 0.0);
    params[0] = 1.0;   // convNorm
    params[1] = 0.0;   // promptNorm
    params[2] = 0.0;   // astroNorm
    params[4] = 0.0;   // HEkp
    params[5] = 0.0;   // HEkm

    std::vector<double> cpuWeights(numEvents);

    // CPU Benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        cpuComputeWeights(cachedConvWeight, cachedPromptWeight, cachedAstroWeight,
                          cachedHadronicHEkp, cachedHadronicHEkm, params.data(),
                          cpuWeights, numEvents);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    result.cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / numIterations;

    // GPU Setup - Create mock SoA event data
    GPUEventDataSoA d_events;
    d_events.numEvents_total = numEvents;

    // Allocate and populate GPU arrays
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

    // Allocate hole ice arrays
    cudaMalloc(&d_events.cachedHoleIceConv, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHoleIcePrompt, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedHoleIceAstro, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHoleIceConv, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHoleIcePrompt, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHoleIceAstro, 0, numEvents * sizeof(double));

    // Allocate DOM efficiency arrays
    cudaMalloc(&d_events.cachedDOMEffConv, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedDOMEffPrompt, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedDOMEffAstro, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffConv, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffPrompt, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedDOMEffAstro, 0, numEvents * sizeof(double));

    // Allocate hadronic arrays (10 DAEMONFLUX parameters)
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
    cudaMemset(d_events.cachedHadronicHEkp, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicHEkm, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE1pip, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE1pim, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3kp, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3km, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3pip, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3pim, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3p, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedHadronicVHE3n, 0, numEvents * sizeof(double));

    // Allocate cosmic ray arrays (6 GSF parameters)
    cudaMalloc(&d_events.cachedCosmicRay1, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay2, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay3, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay4, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay5, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedCosmicRay6, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay1, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay2, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay3, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay4, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay5, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedCosmicRay6, 0, numEvents * sizeof(double));

    // Allocate ice gradient arrays (9 parameters)
    cudaMalloc(&d_events.cachedIceGrad0, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad1, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad2, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad3, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad4, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad5, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad6, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad7, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedIceGrad8, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad0, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad1, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad2, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad3, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad4, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad5, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad6, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad7, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedIceGrad8, 0, numEvents * sizeof(double));

    // Allocate atmospheric arrays
    cudaMalloc(&d_events.cachedAtmDensity, numEvents * sizeof(double));
    cudaMalloc(&d_events.cachedKaonLosses, numEvents * sizeof(double));
    cudaMalloc(&d_events.binIndex, numEvents * sizeof(int32_t));

    cudaMemset(d_events.cachedAtmDensity, 0, numEvents * sizeof(double));
    cudaMemset(d_events.cachedKaonLosses, 0, numEvents * sizeof(double));

    // Copy the data we actually use
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

    double* d_params;
    double* d_weights;
    double* d_weightsSquared;
    cudaMalloc(&d_params, NUM_FIT_PARAMS * sizeof(double));
    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_weightsSquared, numEvents * sizeof(double));

    cudaMemcpy(d_params, params.data(), NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Empty spline lookup for benchmarking (no spline corrections)
    GPUSplineLookup noSplines;
    noSplines.clear();
    // Warm up
    launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);
    cudaDeviceSynchronize();

    // GPU Benchmark - weight only
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        launchEventWeightingKernel(d_events, d_params, noSplines, d_weights, numEvents, true, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTimeMs = gpuMs / numIterations;
    result.speedup = result.cpuTimeMs / result.gpuTimeMs;

    // GPU Benchmark - weight with squares
    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        launchEventWeightingWithSquaresKernel(d_events, d_params, noSplines, d_weights,
                                               d_weightsSquared, numEvents, true, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuWithSquaresMs = gpuMs / numIterations;
    result.speedupWithSquares = result.cpuTimeMs / result.gpuWithSquaresMs;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

    // Free hole ice arrays
    cudaFree(d_events.cachedHoleIceConv);
    cudaFree(d_events.cachedHoleIcePrompt);
    cudaFree(d_events.cachedHoleIceAstro);

    // Free DOM efficiency arrays
    cudaFree(d_events.cachedDOMEffConv);
    cudaFree(d_events.cachedDOMEffPrompt);
    cudaFree(d_events.cachedDOMEffAstro);

    // Free hadronic arrays
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

    // Free cosmic ray arrays
    cudaFree(d_events.cachedCosmicRay1);
    cudaFree(d_events.cachedCosmicRay2);
    cudaFree(d_events.cachedCosmicRay3);
    cudaFree(d_events.cachedCosmicRay4);
    cudaFree(d_events.cachedCosmicRay5);
    cudaFree(d_events.cachedCosmicRay6);

    // Free ice gradient arrays
    cudaFree(d_events.cachedIceGrad0);
    cudaFree(d_events.cachedIceGrad1);
    cudaFree(d_events.cachedIceGrad2);
    cudaFree(d_events.cachedIceGrad3);
    cudaFree(d_events.cachedIceGrad4);
    cudaFree(d_events.cachedIceGrad5);
    cudaFree(d_events.cachedIceGrad6);
    cudaFree(d_events.cachedIceGrad7);
    cudaFree(d_events.cachedIceGrad8);

    // Free atmospheric arrays
    cudaFree(d_events.cachedAtmDensity);
    cudaFree(d_events.cachedKaonLosses);
    cudaFree(d_events.binIndex);
    cudaFree(d_params);
    cudaFree(d_weights);
    cudaFree(d_weightsSquared);

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Event Weighting Benchmark\n";
    std::cout << "========================================\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n\n";

    std::vector<int> eventCounts = {10000, 50000, 100000, 500000, 1000000, 2000000, 5000000};
    int numIterations = 50;

    std::cout << std::setw(12) << "Events"
              << std::setw(14) << "CPU (ms)"
              << std::setw(14) << "GPU (ms)"
              << std::setw(16) << "GPU+Sq (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Speedup+Sq"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (int numEvents : eventCounts) {
        auto result = benchmarkWeighting(numEvents, numIterations);

        std::cout << std::setw(12) << result.numEvents
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.cpuTimeMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.gpuTimeMs
                  << std::setw(16) << std::fixed << std::setprecision(3) << result.gpuWithSquaresMs
                  << std::setw(10) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(11) << std::fixed << std::setprecision(1) << result.speedupWithSquares << "x"
                  << std::endl;
    }

    std::cout << "\n========================================\n";
    std::cout << "Notes\n";
    std::cout << "========================================\n";
    std::cout << "- CPU implementation is simplified (fewer systematics)\n";
    std::cout << "- GPU kernel includes all 38 parameters and systematics\n";
    std::cout << "- Real speedup for full physics is likely higher\n";
    std::cout << "- 'GPU+Sq' computes both weights and squared weights\n";

    return 0;
}
