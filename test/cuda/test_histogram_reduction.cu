/**
 * @file test_histogram_reduction.cu
 * @brief Tests for GPU histogram accumulation kernels.
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

using namespace gollumfit::gpu;

// External kernel declarations (in gollumfit::gpu namespace)
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

extern void launchHistogramAccumulationShared(
    const double* d_weights,
    const int32_t* d_binIndices,
    double* d_binSums,
    int numEvents,
    int numBins,
    cudaStream_t stream
);

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// CPU Reference Implementation
//==============================================================================

void cpuHistogramAccumulation(
    const std::vector<double>& weights,
    const std::vector<int32_t>& binIndices,
    std::vector<double>& binSums,
    int numBins
) {
    std::fill(binSums.begin(), binSums.end(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        int bin = binIndices[i];
        if (bin >= 0 && bin < numBins) {
            binSums[bin] += weights[i];
        }
    }
}

void cpuHistogramAccumulationWithSquares(
    const std::vector<double>& weights,
    const std::vector<double>& weightsSquared,
    const std::vector<int32_t>& binIndices,
    std::vector<double>& binSums,
    std::vector<double>& binSqSums,
    int numBins
) {
    std::fill(binSums.begin(), binSums.end(), 0.0);
    std::fill(binSqSums.begin(), binSqSums.end(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        int bin = binIndices[i];
        if (bin >= 0 && bin < numBins) {
            binSums[bin] += weights[i];
            binSqSums[bin] += weightsSquared[i];
        }
    }
}

//==============================================================================
// Test Functions
//==============================================================================

TestResult testBasicHistogramAccumulation() {
    TestResult result;
    result.name = "Histogram Accumulation - Basic";

    TestRNG rng(42);
    const int numEvents = 100000;
    const int numBins = 500;

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // CPU computation
    std::vector<double> cpuBinSums(numBins);
    Timer timer;
    timer.start();
    cpuHistogramAccumulation(weights, binIndices, cpuBinSums, numBins);
    result.cpuTime = timer.stop();

    // GPU computation
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    timer.start();
    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();
    result.gpuTime = timer.stop();

    std::vector<double> gpuBinSums(numBins);
    cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);

    // Compare results
    const double tol = 1e-10;
    result.passed = compareArrays(cpuBinSums.data(), gpuBinSums.data(), numBins, tol, "binSums");

    if (!result.passed) {
        result.message = "Histogram sums do not match";
    }

    std::cout << "  CPU time: " << result.cpuTime << " ms, GPU time: " << result.gpuTime << " ms\n";

    return result;
}

TestResult testHistogramWithSquares() {
    TestResult result;
    result.name = "Histogram Accumulation - With Squared Weights";

    TestRNG rng(123);
    const int numEvents = 100000;
    const int numBins = 500;

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<double> weightsSquared(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        weightsSquared[i] = weights[i] * weights[i] / 100.0;  // Simulated w^2/n
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // CPU computation
    std::vector<double> cpuBinSums(numBins);
    std::vector<double> cpuBinSqSums(numBins);
    Timer timer;
    timer.start();
    cpuHistogramAccumulationWithSquares(weights, weightsSquared, binIndices,
                                        cpuBinSums, cpuBinSqSums, numBins);
    result.cpuTime = timer.stop();

    // GPU computation
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

    timer.start();
    launchHistogramAccumulationWithSquares(d_weights, d_weightsSquared, d_binIndices,
                                           d_binSums, d_binSqSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();
    result.gpuTime = timer.stop();

    std::vector<double> gpuBinSums(numBins);
    std::vector<double> gpuBinSqSums(numBins);
    cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuBinSqSums.data(), d_binSqSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_weightsSquared);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);
    cudaFree(d_binSqSums);

    // Compare results
    const double tol = 1e-10;
    result.passed = true;

    if (!compareArrays(cpuBinSums.data(), gpuBinSums.data(), numBins, tol, "binSums")) {
        result.passed = false;
    }
    if (!compareArrays(cpuBinSqSums.data(), gpuBinSqSums.data(), numBins, tol, "binSqSums")) {
        result.passed = false;
    }

    if (!result.passed) {
        result.message = "Histogram sums do not match";
    }

    std::cout << "  CPU time: " << result.cpuTime << " ms, GPU time: " << result.gpuTime << " ms\n";

    return result;
}

TestResult testSharedMemoryHistogram() {
    TestResult result;
    result.name = "Histogram Accumulation - Shared Memory Optimization";

    TestRNG rng(456);
    const int numEvents = 100000;
    const int numBins = 100;  // Small enough for shared memory

    // Generate test data with clustered bin indices (high collision rate)
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        // Cluster events in a few bins to stress atomic operations
        binIndices[i] = rng.uniformInt(0, numBins / 2) * 2;  // Only even bins
    }

    // CPU computation
    std::vector<double> cpuBinSums(numBins);
    Timer timer;
    timer.start();
    cpuHistogramAccumulation(weights, binIndices, cpuBinSums, numBins);
    result.cpuTime = timer.stop();

    // GPU computation with shared memory kernel
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    timer.start();
    launchHistogramAccumulationShared(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();
    result.gpuTime = timer.stop();

    std::vector<double> gpuBinSums(numBins);
    cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);

    // Compare results
    const double tol = 1e-10;
    result.passed = compareArrays(cpuBinSums.data(), gpuBinSums.data(), numBins, tol, "binSums");

    if (!result.passed) {
        result.message = "Shared memory histogram sums do not match";
    }

    std::cout << "  CPU time: " << result.cpuTime << " ms, GPU time: " << result.gpuTime << " ms\n";

    return result;
}

TestResult testOutOfBoundsHandling() {
    TestResult result;
    result.name = "Histogram Accumulation - Out of Bounds Handling";

    TestRNG rng(789);
    const int numEvents = 10000;
    const int numBins = 100;

    // Generate test data with some out-of-bounds indices
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        // 20% of events are out of bounds
        if (rng.uniform() < 0.2) {
            if (rng.uniform() < 0.5) {
                binIndices[i] = -1;  // Below range
            } else {
                binIndices[i] = numBins + rng.uniformInt(0, 10);  // Above range
            }
        } else {
            binIndices[i] = rng.uniformInt(0, numBins);
        }
    }

    // CPU computation
    std::vector<double> cpuBinSums(numBins);
    cpuHistogramAccumulation(weights, binIndices, cpuBinSums, numBins);

    // GPU computation
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> gpuBinSums(numBins);
    cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);

    // Compare results
    const double tol = 1e-10;
    result.passed = compareArrays(cpuBinSums.data(), gpuBinSums.data(), numBins, tol, "binSums");

    if (!result.passed) {
        result.message = "Out of bounds handling differs between CPU and GPU";
    }

    return result;
}

TestResult testLargeScale() {
    TestResult result;
    result.name = "Histogram Accumulation - Large Scale (1M events)";

    TestRNG rng(999);
    const int numEvents = 1000000;
    const int numBins = 1000;

    // Generate test data
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.01, 100.0);
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // CPU computation
    std::vector<double> cpuBinSums(numBins);
    Timer timer;
    timer.start();
    cpuHistogramAccumulation(weights, binIndices, cpuBinSums, numBins);
    result.cpuTime = timer.stop();

    // GPU computation
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Warm up
    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    timer.start();
    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();
    result.gpuTime = timer.stop();

    std::vector<double> gpuBinSums(numBins);
    cudaMemcpy(gpuBinSums.data(), d_binSums, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums);

    // Compare results
    const double tol = 1e-9;  // Slightly relaxed for accumulated sums
    result.passed = compareArrays(cpuBinSums.data(), gpuBinSums.data(), numBins, tol, "binSums");

    if (!result.passed) {
        result.message = "Large scale histogram sums do not match";
    }

    std::cout << "  CPU time: " << result.cpuTime << " ms, GPU time: " << result.gpuTime << " ms";
    std::cout << " (Speedup: " << result.cpuTime / result.gpuTime << "x)\n";

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Histogram Reduction Tests\n";
    std::cout << "========================================\n\n";

    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "Native FP64 Atomics: " << (prop.major >= 8 ? "Yes" : "No") << "\n\n";

    TestSuite suite;

    suite.addResult(testBasicHistogramAccumulation());
    suite.addResult(testHistogramWithSquares());
    suite.addResult(testSharedMemoryHistogram());
    suite.addResult(testOutOfBoundsHandling());
    suite.addResult(testLargeScale());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
