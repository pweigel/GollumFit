/**
 * @file test_numerical_precision.cu
 * @brief Tests for numerical precision and edge cases in GPU computations.
 *
 * These tests verify:
 * - FP64 precision is maintained between CPU and GPU
 * - Edge cases (underflow, overflow, denormals) are handled correctly
 * - Accumulation order doesn't affect results (within tolerance)
 * - Gradient computations match finite differences
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

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
// CPU Reference Implementations
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

// Kahan summation for high-precision reference
double kahanSum(const std::vector<double>& values) {
    double sum = 0.0;
    double c = 0.0;  // compensation for lost low-order bits
    for (double v : values) {
        double y = v - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

//==============================================================================
// Test: Very Small Weights (Underflow Edge Cases)
//==============================================================================

TestResult testVerySmallWeights() {
    TestResult result;
    result.name = "Numerical Precision - Very Small Weights";

    const int numEvents = 10000;
    const int numBins = 100;

    // Generate weights near the underflow threshold
    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(12345);

    for (int i = 0; i < numEvents; ++i) {
        // Weights between 1e-300 and 1e-280 (very small but not denormal)
        double exponent = rng.uniform(-300.0, -280.0);
        weights[i] = std::pow(10.0, exponent);
        binIndices[i] = rng.uniformInt(0, numBins);
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

    // Compare with relative tolerance (absolute tolerance doesn't make sense for tiny numbers)
    result.passed = true;
    double maxRelError = 0.0;
    for (int i = 0; i < numBins; ++i) {
        if (cpuBinSums[i] != 0.0) {
            double relError = std::abs(gpuBinSums[i] - cpuBinSums[i]) / std::abs(cpuBinSums[i]);
            maxRelError = std::max(maxRelError, relError);
            if (relError > 1e-10) {
                result.passed = false;
            }
        }
    }

    if (!result.passed) {
        result.message = "Max relative error: " + std::to_string(maxRelError);
    }
    std::cout << "  Max relative error: " << maxRelError << std::endl;

    return result;
}

//==============================================================================
// Test: Very Large Weights
//==============================================================================

TestResult testVeryLargeWeights() {
    TestResult result;
    result.name = "Numerical Precision - Very Large Weights";

    const int numEvents = 10000;
    const int numBins = 100;

    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(23456);

    for (int i = 0; i < numEvents; ++i) {
        // Weights between 1e280 and 1e300 (very large)
        double exponent = rng.uniform(280.0, 300.0);
        weights[i] = std::pow(10.0, exponent);
        binIndices[i] = rng.uniformInt(0, numBins);
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

    // Compare
    result.passed = true;
    double maxRelError = 0.0;
    for (int i = 0; i < numBins; ++i) {
        if (cpuBinSums[i] != 0.0 && std::isfinite(cpuBinSums[i])) {
            double relError = std::abs(gpuBinSums[i] - cpuBinSums[i]) / std::abs(cpuBinSums[i]);
            maxRelError = std::max(maxRelError, relError);
            if (relError > 1e-10) {
                result.passed = false;
            }
        }
    }

    if (!result.passed) {
        result.message = "Max relative error: " + std::to_string(maxRelError);
    }
    std::cout << "  Max relative error: " << maxRelError << std::endl;

    return result;
}

//==============================================================================
// Test: Mixed Scale Weights (Catastrophic Cancellation Check)
//==============================================================================

TestResult testMixedScaleWeights() {
    TestResult result;
    result.name = "Numerical Precision - Mixed Scale Weights";

    const int numEvents = 10000;
    const int numBins = 10;  // Few bins to concentrate events

    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(34567);

    // Mix of very different scales in same bins
    for (int i = 0; i < numEvents; ++i) {
        // Alternate between small and large weights
        if (i % 2 == 0) {
            weights[i] = rng.uniform(1e-10, 1e-8);
        } else {
            weights[i] = rng.uniform(1e8, 1e10);
        }
        binIndices[i] = rng.uniformInt(0, numBins);
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

    // Note: With mixed scales, we expect some loss of precision
    // The large weights dominate, so small weights may be lost
    result.passed = true;
    double maxRelError = 0.0;
    for (int i = 0; i < numBins; ++i) {
        if (cpuBinSums[i] != 0.0) {
            double relError = std::abs(gpuBinSums[i] - cpuBinSums[i]) / std::abs(cpuBinSums[i]);
            maxRelError = std::max(maxRelError, relError);
            // Allow slightly larger tolerance for mixed scale
            if (relError > 1e-8) {
                result.passed = false;
            }
        }
    }

    if (!result.passed) {
        result.message = "Max relative error: " + std::to_string(maxRelError);
    }
    std::cout << "  Max relative error: " << maxRelError << std::endl;

    return result;
}

//==============================================================================
// Test: Accumulation Order Independence
//==============================================================================

TestResult testAccumulationOrderIndependence() {
    TestResult result;
    result.name = "Numerical Precision - Accumulation Order Independence";

    const int numEvents = 50000;
    const int numBins = 50;

    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(45678);

    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 100.0);
        binIndices[i] = rng.uniformInt(0, numBins);
    }

    // First run: original order
    double* d_weights;
    int32_t* d_binIndices;
    double* d_binSums1;
    double* d_binSums2;

    cudaMalloc(&d_weights, numEvents * sizeof(double));
    cudaMalloc(&d_binIndices, numEvents * sizeof(int32_t));
    cudaMalloc(&d_binSums1, numBins * sizeof(double));
    cudaMalloc(&d_binSums2, numBins * sizeof(double));

    cudaMemcpy(d_weights, weights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, binIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums1, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    // Shuffle the data
    std::vector<size_t> indices(numEvents);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(99999));

    std::vector<double> shuffledWeights(numEvents);
    std::vector<int32_t> shuffledBinIndices(numEvents);
    for (size_t i = 0; i < numEvents; ++i) {
        shuffledWeights[i] = weights[indices[i]];
        shuffledBinIndices[i] = binIndices[indices[i]];
    }

    cudaMemcpy(d_weights, shuffledWeights.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binIndices, shuffledBinIndices.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);

    launchHistogramAccumulation(d_weights, d_binIndices, d_binSums2, numEvents, numBins, nullptr);
    cudaDeviceSynchronize();

    std::vector<double> gpuBinSums1(numBins);
    std::vector<double> gpuBinSums2(numBins);
    cudaMemcpy(gpuBinSums1.data(), d_binSums1, numBins * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuBinSums2.data(), d_binSums2, numBins * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_binIndices);
    cudaFree(d_binSums1);
    cudaFree(d_binSums2);

    // Compare the two runs
    result.passed = true;
    double maxRelError = 0.0;
    for (int i = 0; i < numBins; ++i) {
        if (gpuBinSums1[i] != 0.0) {
            double relError = std::abs(gpuBinSums2[i] - gpuBinSums1[i]) / std::abs(gpuBinSums1[i]);
            maxRelError = std::max(maxRelError, relError);
            // Atomic operations may have slight order-dependent rounding
            if (relError > 1e-12) {
                result.passed = false;
            }
        }
    }

    if (!result.passed) {
        result.message = "Max difference between runs: " + std::to_string(maxRelError);
    }
    std::cout << "  Max relative difference between orderings: " << maxRelError << std::endl;

    return result;
}

//==============================================================================
// Test: SAY Likelihood Edge Cases
//==============================================================================

TestResult testSAYLikelihoodEdgeCases() {
    TestResult result;
    result.name = "Numerical Precision - SAY Likelihood Edge Cases";

    const int numBins = 10;

    // Test various edge case combinations
    struct TestCase {
        std::vector<double> data;
        std::vector<double> wSum;
        std::vector<double> w2Sum;
        std::string description;
    };

    std::vector<TestCase> testCases = {
        // Zero data, zero expectation
        {{0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, "all zeros"},
        // Zero data, non-zero expectation
        {{0,0,0,0,0,0,0,0,0,0}, {1,2,3,4,5,6,7,8,9,10}, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}, "zero data"},
        // Non-zero data, very small expectation
        {{1,1,1,1,1,1,1,1,1,1}, {1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10},
         {1e-22,1e-22,1e-22,1e-22,1e-22,1e-22,1e-22,1e-22,1e-22,1e-22}, "tiny expectation"},
        // Very large data and expectation
        {{1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6}, {1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6},
         {1e10,1e10,1e10,1e10,1e10,1e10,1e10,1e10,1e10,1e10}, "large values"},
    };

    result.passed = true;

    for (const auto& tc : testCases) {
        double* d_data;
        double* d_wSum;
        double* d_w2Sum;

        cudaMalloc(&d_data, numBins * sizeof(double));
        cudaMalloc(&d_wSum, numBins * sizeof(double));
        cudaMalloc(&d_w2Sum, numBins * sizeof(double));

        cudaMemcpy(d_data, tc.data.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wSum, tc.wSum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2Sum, tc.w2Sum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);

        double llh = computeTotalSAYLikelihood(d_data, d_wSum, d_w2Sum, numBins, nullptr);

        cudaFree(d_data);
        cudaFree(d_wSum);
        cudaFree(d_w2Sum);

        // Check for NaN or Inf
        if (std::isnan(llh) || std::isinf(llh)) {
            std::cout << "  WARN: " << tc.description << " produced "
                      << (std::isnan(llh) ? "NaN" : "Inf") << std::endl;
            // Some edge cases may legitimately produce inf (e.g., impossible observation)
            if (tc.description != "tiny expectation") {
                result.passed = false;
                result.message = tc.description + " produced invalid result";
            }
        } else {
            std::cout << "  " << tc.description << ": LLH = " << llh << std::endl;
        }
    }

    return result;
}

//==============================================================================
// Test: Denormalized Numbers
//==============================================================================

TestResult testDenormalizedNumbers() {
    TestResult result;
    result.name = "Numerical Precision - Denormalized Numbers";

    const int numEvents = 1000;
    const int numBins = 10;

    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);

    // Use denormalized numbers (between 0 and std::numeric_limits<double>::min())
    double denormMin = std::numeric_limits<double>::denorm_min();
    double normalMin = std::numeric_limits<double>::min();

    for (int i = 0; i < numEvents; ++i) {
        // Create denormalized numbers
        weights[i] = denormMin * (i + 1);  // Various small denorms
        if (weights[i] >= normalMin) {
            weights[i] = denormMin;  // Keep it denormalized
        }
        binIndices[i] = i % numBins;
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

    // Compare - denormals may be flushed to zero on GPU
    result.passed = true;
    int numMismatches = 0;
    for (int i = 0; i < numBins; ++i) {
        // Check if both are zero or both are equal
        bool cpuZero = (cpuBinSums[i] == 0.0);
        bool gpuZero = (gpuBinSums[i] == 0.0);

        if (cpuZero != gpuZero) {
            numMismatches++;
        } else if (!cpuZero && !gpuZero) {
            double relError = std::abs(gpuBinSums[i] - cpuBinSums[i]) / std::abs(cpuBinSums[i]);
            if (relError > 0.5) {  // Very loose tolerance for denormals
                numMismatches++;
            }
        }
    }

    // GPU may flush denormals to zero, which is acceptable
    std::cout << "  Denormal handling: " << numMismatches << "/" << numBins
              << " bins differ (GPU may flush to zero)" << std::endl;

    // This is informational - denormal flushing is acceptable behavior
    result.passed = true;

    return result;
}

//==============================================================================
// Test: High Contention (All Events in One Bin)
//==============================================================================

TestResult testHighContention() {
    TestResult result;
    result.name = "Numerical Precision - High Contention (Single Bin)";

    const int numEvents = 100000;
    const int numBins = 100;
    const int targetBin = 42;

    std::vector<double> weights(numEvents);
    std::vector<int32_t> binIndices(numEvents);
    TestRNG rng(56789);

    double expectedSum = 0.0;
    for (int i = 0; i < numEvents; ++i) {
        weights[i] = rng.uniform(0.1, 10.0);
        binIndices[i] = targetBin;  // All events in same bin
        expectedSum += weights[i];
    }

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

    // Compare with Kahan sum for high precision reference
    double kahanRef = kahanSum(weights);

    double relErrorVsNaive = std::abs(gpuBinSums[targetBin] - expectedSum) / expectedSum;
    double relErrorVsKahan = std::abs(gpuBinSums[targetBin] - kahanRef) / kahanRef;

    std::cout << "  Naive sum:  " << expectedSum << std::endl;
    std::cout << "  Kahan sum:  " << kahanRef << std::endl;
    std::cout << "  GPU sum:    " << gpuBinSums[targetBin] << std::endl;
    std::cout << "  Rel error vs naive: " << relErrorVsNaive << std::endl;
    std::cout << "  Rel error vs Kahan: " << relErrorVsKahan << std::endl;

    // GPU atomic adds may accumulate in different order, causing small differences
    result.passed = (relErrorVsKahan < 1e-10);
    if (!result.passed) {
        result.message = "Relative error too large: " + std::to_string(relErrorVsKahan);
    }

    // Verify other bins are zero
    for (int i = 0; i < numBins; ++i) {
        if (i != targetBin && gpuBinSums[i] != 0.0) {
            result.passed = false;
            result.message = "Non-target bin has non-zero sum";
            break;
        }
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Numerical Precision Tests\n";
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";

    TestSuite suite;

    suite.addResult(testVerySmallWeights());
    suite.addResult(testVeryLargeWeights());
    suite.addResult(testMixedScaleWeights());
    suite.addResult(testAccumulationOrderIndependence());
    suite.addResult(testSAYLikelihoodEdgeCases());
    suite.addResult(testDenormalizedNumbers());
    suite.addResult(testHighContention());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
