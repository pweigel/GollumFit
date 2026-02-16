/**
 * @file benchmark_likelihood.cu
 * @brief Benchmark SAY likelihood evaluation: CPU vs GPU performance comparison.
 *
 * This benchmark measures the full likelihood evaluation pipeline:
 * 1. Weight computation (not measured separately here)
 * 2. Histogram accumulation
 * 3. SAY likelihood computation
 *
 * Tests various configurations of bins and repeated evaluations
 * (simulating fitting iterations).
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace gollumfit::gpu;

// External kernel declarations
namespace gollumfit {
namespace gpu {

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
// CPU Implementation of SAY Likelihood
//==============================================================================

double cpuSAYBinLikelihood(double k, double w_sum, double w2_sum) {
    if (w_sum <= 0.0) {
        return (k == 0.0) ? 0.0 : -1e300;
    }
    if (w2_sum <= 0.0) {
        // Poisson fallback
        return k * std::log(w_sum) - w_sum - std::lgamma(k + 1.0);
    }

    double alpha = w_sum * w_sum / w2_sum + 1.0;
    double beta = w_sum / w2_sum;

    return alpha * std::log(beta) + std::lgamma(k + alpha) - std::lgamma(alpha)
           - (k + alpha) * std::log(1.0 + beta) - std::lgamma(k + 1.0);
}

double cpuTotalSAYLikelihood(const double* dataCount, const double* wSum,
                              const double* w2Sum, int numBins) {
    double total = 0.0;
    for (int i = 0; i < numBins; ++i) {
        total += cpuSAYBinLikelihood(dataCount[i], wSum[i], w2Sum[i]);
    }
    return total;
}

void cpuHistogramAccumulationWithSquares(
    const double* weights,
    const double* weightsSquared,
    const int32_t* binIndices,
    double* binSums,
    double* binSqSums,
    int numEvents,
    int numBins
) {
    std::fill(binSums, binSums + numBins, 0.0);
    std::fill(binSqSums, binSqSums + numBins, 0.0);
    for (int i = 0; i < numEvents; ++i) {
        int bin = binIndices[i];
        if (bin >= 0 && bin < numBins) {
            binSums[bin] += weights[i];
            binSqSums[bin] += weightsSquared[i];
        }
    }
}

//==============================================================================
// Benchmark: Full Likelihood Evaluation
//==============================================================================

struct LikelihoodBenchmarkResult {
    int numEvents;
    int numBins;
    int numIterations;
    double cpuTotalTimeMs;
    double cpuPerIterMs;
    double gpuTotalTimeMs;
    double gpuPerIterMs;
    double speedup;
};

LikelihoodBenchmarkResult benchmarkLikelihood(int numEvents, int numBins,
                                                int numIterations) {
    LikelihoodBenchmarkResult result;
    result.numEvents = numEvents;
    result.numBins = numBins;
    result.numIterations = numIterations;

    TestRNG rng(12345);

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<double> weightsSquared(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    std::vector<double> dataCount(numBins);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.001, 1.0);
        weightsSquared[i] = weights[i] * weights[i] / 1000.0;
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // Generate realistic data counts (Poisson-like)
    for (int i = 0; i < numBins; ++i) {
        dataCount[i] = std::floor(rng.uniform(0, 100));
    }

    std::vector<double> binSums(numBins);
    std::vector<double> binSqSums(numBins);

    // CPU Benchmark: Full pipeline (histogram + likelihood) per iteration
    auto cpuStart = std::chrono::high_resolution_clock::now();
    double cpuLLH = 0.0;
    for (int iter = 0; iter < numIterations; ++iter) {
        // In a real fit, weights would change each iteration
        // Here we just re-accumulate to simulate the work
        cpuHistogramAccumulationWithSquares(weights.data(), weightsSquared.data(),
                                             binIndices.data(), binSums.data(),
                                             binSqSums.data(), numEvents, numBins);
        cpuLLH = cpuTotalSAYLikelihood(dataCount.data(), binSums.data(),
                                        binSqSums.data(), numBins);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    result.cpuTotalTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    result.cpuPerIterMs = result.cpuTotalTimeMs / numIterations;

    // GPU Setup
    double* d_weights;
    double* d_weightsSquared;
    int32_t* d_binIndices;
    double* d_binSums;
    double* d_binSqSums;
    double* d_dataCount;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_weightsSquared, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));
    cudaMalloc(&d_binSqSums, numBins * sizeof(double));
    cudaMalloc(&d_dataCount, numBins * sizeof(double));

    // Transfer data to GPU (done once in real fitting scenario)
    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsSquared, weightsSquared.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataCount, dataCount.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Warm up
    launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_binIndices,
                                            d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    double warmupLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    cudaDeviceSynchronize();
    (void)warmupLLH;

    // GPU Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    double gpuLLH = 0.0;
    for (int iter = 0; iter < numIterations; ++iter) {
        launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_binIndices,
                                                d_binSums, d_binSqSums, numEvents, numBins, nullptr);
        gpuLLH = computeTotalSAYLikelihood(d_dataCount, d_binSums, d_binSqSums, numBins, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTotalTimeMs = gpuMs;
    result.gpuPerIterMs = gpuMs / numIterations;
    result.speedup = result.cpuPerIterMs / result.gpuPerIterMs;

    // Verify results match
    double relError = std::abs(gpuLLH - cpuLLH) / (std::abs(cpuLLH) + 1e-15);
    if (relError > 1e-6) {
        std::cout << "  WARNING: LLH mismatch! CPU=" << cpuLLH << " GPU=" << gpuLLH
                  << " relError=" << relError << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_weights);
    cudaFree(d_weightsSquared);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);
    cudaFree(d_binSqSums);
    cudaFree(d_dataCount);

    return result;
}

//==============================================================================
// Benchmark: Simulated Fit (Many Iterations)
//==============================================================================

void benchmarkSimulatedFit(int numEvents, int numBins) {
    std::cout << "\n========================================\n";
    std::cout << "Simulated Fit: " << numEvents << " events, " << numBins << " bins\n";
    std::cout << "========================================\n\n";

    // Typical fit might have 100-1000 likelihood evaluations
    std::vector<int> iterationCounts = {10, 50, 100, 500, 1000};

    std::cout << std::setw(12) << "Iterations"
              << std::setw(14) << "CPU Total(s)"
              << std::setw(14) << "GPU Total(s)"
              << std::setw(14) << "CPU/iter(ms)"
              << std::setw(14) << "GPU/iter(ms)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (int numIter : iterationCounts) {
        auto result = benchmarkLikelihood(numEvents, numBins, numIter);

        std::cout << std::setw(12) << result.numIterations
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << result.cpuTotalTimeMs / 1000.0
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << result.gpuTotalTimeMs / 1000.0
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << result.cpuPerIterMs
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << result.gpuPerIterMs
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << result.speedup << "x"
                  << std::endl;
    }
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SAY Likelihood Evaluation Benchmark\n";
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory Bandwidth: " << prop.memoryBusWidth << " bit @ "
              << prop.memoryClockRate / 1e6 << " GHz\n\n";

    // Quick comparison across different sizes
    std::cout << "========================================\n";
    std::cout << "Size Scaling (100 iterations each)\n";
    std::cout << "========================================\n\n";

    std::cout << std::setw(12) << "Events"
              << std::setw(10) << "Bins"
              << std::setw(14) << "CPU/iter(ms)"
              << std::setw(14) << "GPU/iter(ms)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::vector<std::pair<int, int>> configs = {
        {10000, 100},
        {50000, 200},
        {100000, 500},
        {500000, 500},
        {1000000, 1000},
        {2000000, 1000},
        {5000000, 1000},
    };

    for (auto& cfg : configs) {
        auto result = benchmarkLikelihood(cfg.first, cfg.second, 100);
        std::cout << std::setw(12) << result.numEvents
                  << std::setw(10) << result.numBins
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.cpuPerIterMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.gpuPerIterMs
                  << std::setw(10) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::endl;
    }

    // Detailed simulation for typical physics use case
    benchmarkSimulatedFit(1000000, 500);  // 1M events, 500 bins
    benchmarkSimulatedFit(5000000, 1000); // 5M events, 1000 bins

    std::cout << "\n========================================\n";
    std::cout << "Interpretation\n";
    std::cout << "========================================\n";
    std::cout << "- CPU time is dominated by event loop (O(N) events)\n";
    std::cout << "- GPU parallelizes across all events simultaneously\n";
    std::cout << "- Speedup increases with event count (better parallelism)\n";
    std::cout << "- For typical fits (1M events, 500 iterations):\n";
    std::cout << "    CPU: ~X minutes, GPU: ~Y seconds (estimate from above)\n";

    return 0;
}
