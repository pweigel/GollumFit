/**
 * @file benchmark_histogram.cu
 * @brief Benchmark histogram accumulation: CPU vs GPU performance comparison.
 *
 * Measures performance across:
 * - Different event counts (10K to 10M)
 * - Different bin counts (100 to 10000)
 * - With and without squared weight accumulation
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace gollumfit::gpu;

// External kernel declarations
namespace gollumfit {
namespace gpu {

extern void launchHistogramAccumulation(
    const double* d_weights,
    const int32_t* d_binIndices,
    double* d_binSums,
    int numEvents,
    int numBins,
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

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// CPU Implementation
//==============================================================================

void cpuHistogramAccumulation(
    const double* weights,
    const int32_t* binIndices,
    double* binSums,
    int numEvents,
    int numBins
) {
    std::fill(binSums, binSums + numBins, 0.0);
    for (int i = 0; i < numEvents; ++i) {
        int bin = binIndices[i];
        if (bin >= 0 && bin < numBins) {
            binSums[bin] += weights[i];
        }
    }
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
// Benchmark Functions
//==============================================================================

struct BenchmarkResult {
    int numEvents;
    int numBins;
    double cpuTimeMs;
    double gpuTimeMs;
    double gpuTimeWithTransferMs;
    double speedup;
    double speedupWithTransfer;
};

BenchmarkResult benchmarkHistogram(int numEvents, int numBins, int numIterations = 10) {
    BenchmarkResult result;
    result.numEvents = numEvents;
    result.numBins = numBins;

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(42);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    std::vector<double> cpuBinSums(numBins);
    std::vector<double> gpuBinSums(numBins);

    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        cpuHistogramAccumulation(weights.data(), binIndices.data(),
                                  cpuBinSums.data(), numEvents, numBins);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    result.cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / numIterations;

    // GPU setup
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));

    // GPU benchmark (kernel only, data already on GPU)
    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Warm up
    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTimeMs = gpuMs / numIterations;

    // GPU benchmark (including transfer)
    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
        launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
        cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTimeWithTransferMs = gpuMs / numIterations;

    result.speedup = result.cpuTimeMs / result.gpuTimeMs;
    result.speedupWithTransfer = result.cpuTimeMs / result.gpuTimeWithTransferMs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);

    return result;
}

BenchmarkResult benchmarkHistogramWithSquares(int numEvents, int numBins, int numIterations = 10) {
    BenchmarkResult result;
    result.numEvents = numEvents;
    result.numBins = numBins;

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<double> weightsSquared(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(42);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        weightsSquared[i] = weights[i] * weights[i] / 100.0;
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    std::vector<double> cpuBinSums(numBins);
    std::vector<double> cpuBinSqSums(numBins);

    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        cpuHistogramAccumulationWithSquares(weights.data(), weightsSquared.data(),
                                             binIndices.data(), cpuBinSums.data(),
                                             cpuBinSqSums.data(), numEvents, numBins);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    result.cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count() / numIterations;

    // GPU setup
    double* d_weights;
    double* d_weightsSquared;
    int32_t* d_binIndices;
    double* d_binSums;
    double* d_binSqSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_weightsSquared, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));
    cudaMalloc(&d_binSqSums, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightsSquared, weightsSquared.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Warm up
    launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_binIndices,
                                            d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int iter = 0; iter < numIterations; ++iter) {
        launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_binIndices,
                                                d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs;
    cudaEventElapsedTime(&gpuMs, start, stop);
    result.gpuTimeMs = gpuMs / numIterations;

    result.speedup = result.cpuTimeMs / result.gpuTimeMs;
    result.gpuTimeWithTransferMs = 0;  // Not measured for this variant
    result.speedupWithTransfer = 0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_weights);
    cudaFree(d_weightsSquared);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);
    cudaFree(d_binSqSums);

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Histogram Accumulation Benchmark\n";
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
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";

    // Test configurations
    std::vector<int> eventCounts = {10000, 50000, 100000, 500000, 1000000, 5000000, 10000000};
    std::vector<int> binCounts = {100, 500, 1000, 5000};

    std::cout << "========================================\n";
    std::cout << "Basic Histogram (sum only)\n";
    std::cout << "========================================\n\n";

    std::cout << std::setw(12) << "Events"
              << std::setw(10) << "Bins"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "GPU (ms)"
              << std::setw(14) << "GPU+Xfer(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "w/Transfer"
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    for (int numBins : binCounts) {
        for (int numEvents : eventCounts) {
            BenchmarkResult result = benchmarkHistogram(numEvents, numBins);
            std::cout << std::setw(12) << result.numEvents
                      << std::setw(10) << result.numBins
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.cpuTimeMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.gpuTimeMs
                      << std::setw(14) << std::fixed << std::setprecision(3) << result.gpuTimeWithTransferMs
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.speedup << "x"
                      << std::setw(11) << std::fixed << std::setprecision(1) << result.speedupWithTransfer << "x"
                      << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "\n========================================\n";
    std::cout << "Histogram with Squared Weights (SAY)\n";
    std::cout << "========================================\n\n";

    std::cout << std::setw(12) << "Events"
              << std::setw(10) << "Bins"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "GPU (ms)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    for (int numBins : {500, 1000}) {
        for (int numEvents : eventCounts) {
            BenchmarkResult result = benchmarkHistogramWithSquares(numEvents, numBins);
            std::cout << std::setw(12) << result.numEvents
                      << std::setw(10) << result.numBins
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.cpuTimeMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.gpuTimeMs
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.speedup << "x"
                      << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "GPU acceleration is most effective for:\n";
    std::cout << "  - Large event counts (>100K events)\n";
    std::cout << "  - When data is already on GPU (no transfer overhead)\n";
    std::cout << "  - Multiple iterations (amortize setup cost)\n";

    return 0;
}
