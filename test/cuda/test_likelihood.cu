/**
 * @file test_likelihood.cu
 * @brief Tests for GPU SAY likelihood evaluation.
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gollumfit::gpu;

// External kernel declarations (in gollumfit::gpu namespace)
namespace gollumfit {
namespace gpu {

extern double computeTotalSAYLikelihood(
    const double* d_dataCount,
    const double* d_wSum,
    const double* d_w2Sum,
    int numBins,
    cudaStream_t stream
);

extern double computeTotalPrior(
    const double* d_params,
    const double* d_priorMeans,
    const double* d_priorSigmas,
    const int* d_priorFlags,
    int numParams,
    cudaStream_t stream
);

} // namespace gpu
} // namespace gollumfit

//==============================================================================
// CPU Reference Implementation
//==============================================================================

/**
 * @brief CPU implementation of SAY likelihood for a single bin.
 * Matches the GPU implementation in likelihood.cu.
 */
double cpuSAYBinLikelihood(double k, double w_sum, double w2_sum) {
    // Handle edge cases
    if (w_sum <= 0.0) {
        if (k == 0.0) {
            return 0.0;  // No expectation, no observation
        } else {
            return -std::numeric_limits<double>::max();  // Impossible
        }
    }

    if (w2_sum <= 0.0) {
        // Fall back to Poisson
        return k * std::log(w_sum) - w_sum - std::lgamma(k + 1.0);
    }

    // SAY parameters
    double alpha = w_sum * w_sum / w2_sum + 1.0;
    double beta = w_sum / w2_sum;

    // SAY likelihood
    double llh = alpha * std::log(beta);
    llh += std::lgamma(k + alpha);
    llh -= std::lgamma(alpha);
    llh -= std::lgamma(k + 1.0);
    llh -= (k + alpha) * std::log(1.0 + beta);

    return llh;
}

double cpuTotalSAYLikelihood(
    const std::vector<double>& dataCount,
    const std::vector<double>& wSum,
    const std::vector<double>& w2Sum
) {
    double total = 0.0;
    for (size_t i = 0; i < dataCount.size(); ++i) {
        total += cpuSAYBinLikelihood(dataCount[i], wSum[i], w2Sum[i]);
    }
    return total;
}

double cpuGaussianPrior(
    const std::vector<double>& params,
    const std::vector<double>& means,
    const std::vector<double>& sigmas,
    const std::vector<int>& flags
) {
    double total = 0.0;
    for (size_t i = 0; i < params.size(); ++i) {
        if (flags[i]) {
            double sigma = sigmas[i];
            // Match PhysTools GaussianPrior: log(norm) - z*z/2
            // where norm = 1/(sigma*sqrt(2*pi))
            if (std::isinf(sigma) || std::isnan(sigma)) {
                continue;  // flat prior contributes 0
            }
            double z = (params[i] - means[i]) / sigma;
            total += -std::log(sigma) - 0.9189385332046727 - 0.5 * z * z;
        }
    }
    return total;
}

//==============================================================================
// Test Functions
//==============================================================================

TestResult testSAYLikelihoodSingleBin() {
    TestResult result;
    result.name = "SAY Likelihood - Single Bin Cases";

    // Test cases: (k, w_sum, w2_sum)
    struct TestCase {
        double k, w_sum, w2_sum;
        const char* description;
    };

    std::vector<TestCase> cases = {
        {10.0, 10.0, 1.0, "typical case"},
        {0.0, 5.0, 0.5, "zero observation"},
        {100.0, 95.0, 10.0, "high count"},
        {1.0, 2.0, 0.5, "low count"},
        {50.0, 50.0, 50.0, "high MC uncertainty"},
        {0.0, 0.0, 0.0, "empty bin"},
    };

    result.passed = true;
    const double tol = 1e-10;

    for (const auto& tc : cases) {
        double cpu_llh = cpuSAYBinLikelihood(tc.k, tc.w_sum, tc.w2_sum);

        // For single-bin GPU test, create small arrays
        std::vector<double> dataCount = {tc.k};
        std::vector<double> wSum = {tc.w_sum};
        std::vector<double> w2Sum = {tc.w2_sum};

        double* d_dataCount;
        double* d_wSum;
        double* d_w2Sum;

        cudaMalloc(&d_dataCount, sizeof(double));
        cudaMalloc(&d_wSum, sizeof(double));
        cudaMalloc(&d_w2Sum, sizeof(double));

        cudaMemcpy(d_dataCount, dataCount.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wSum, wSum.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2Sum, w2Sum.data(), sizeof(double), cudaMemcpyHostToDevice);

        double gpu_llh = computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, 1, nullptr);

        cudaFree(d_dataCount);
        cudaFree(d_wSum);
        cudaFree(d_w2Sum);

        if (!compareValues(cpu_llh, gpu_llh, tol, tc.description)) {
            result.passed = false;
            result.message = std::string("Mismatch for case: ") + tc.description;
            std::cerr << "  Case '" << tc.description << "': CPU=" << cpu_llh
                      << ", GPU=" << gpu_llh << std::endl;
        }
    }

    return result;
}

TestResult testSAYLikelihoodMultiBin() {
    TestResult result;
    result.name = "SAY Likelihood - Multi-Bin Summation";

    TestRNG rng(42);
    const int numBins = 500;

    // Generate realistic test data
    std::vector<double> dataCount(numBins);
    std::vector<double> wSum(numBins);
    std::vector<double> w2Sum(numBins);

    for (int i = 0; i < numBins; ++i) {
        double expected = rng.uniform(1.0, 100.0);
        wSum[i] = expected;
        w2Sum[i] = expected * rng.uniform(0.05, 0.2);  // 5-20% MC uncertainty
        // Poisson-like fluctuation
        dataCount[i] = std::max(0.0, expected + rng.uniform(-2, 2) * std::sqrt(expected));
    }

    // CPU computation
    Timer timer;
    timer.start();
    double cpu_llh = cpuTotalSAYLikelihood(dataCount, wSum, w2Sum);
    result.cpuTime = timer.stop();

    // GPU computation
    double* d_dataCount;
    double* d_wSum;
    double* d_w2Sum;

    cudaMalloc(&d_dataCount, numBins * sizeof(double));
    cudaMalloc(&d_wSum, numBins * sizeof(double));
    cudaMalloc(&d_w2Sum, numBins * sizeof(double));

    cudaMemcpy(d_dataCount, dataCount.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wSum, wSum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2Sum, w2Sum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);

    // Warm up
    computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, numBins, nullptr);

    timer.start();
    double gpu_llh = computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, numBins, nullptr);
    result.gpuTime = timer.stop();

    cudaFree(d_dataCount);
    cudaFree(d_wSum);
    cudaFree(d_w2Sum);

    // Compare results
    const double tol = 1e-8;  // Relaxed due to summation order differences
    result.passed = compareValues(cpu_llh, gpu_llh, tol, "total SAY LLH");

    if (!result.passed) {
        result.message = "Total likelihood mismatch";
    }

    std::cout << "  CPU LLH: " << cpu_llh << ", GPU LLH: " << gpu_llh << std::endl;
    std::cout << "  CPU time: " << result.cpuTime << " ms, GPU time: " << result.gpuTime << " ms\n";

    return result;
}

TestResult testGaussianPrior() {
    TestResult result;
    result.name = "Gaussian Prior Evaluation";

    const int numParams = 38;
    TestRNG rng(123);

    // Generate test data
    std::vector<double> params(numParams);
    std::vector<double> means(numParams);
    std::vector<double> sigmas(numParams);
    std::vector<int> flags(numParams);

    for (int i = 0; i < numParams; ++i) {
        params[i] = rng.uniform(-2.0, 2.0);
        means[i] = 0.0;
        sigmas[i] = 1.0;
        flags[i] = (rng.uniform() < 0.7) ? 1 : 0;  // 70% have priors
    }

    // CPU computation
    double cpu_prior = cpuGaussianPrior(params, means, sigmas, flags);

    // GPU computation
    double* d_params;
    double* d_means;
    double* d_sigmas;
    int* d_flags;

    cudaMalloc(&d_params, numParams * sizeof(double));
    cudaMalloc(&d_means, numParams * sizeof(double));
    cudaMalloc(&d_sigmas, numParams * sizeof(double));
    cudaMalloc(&d_flags, numParams * sizeof(int));

    cudaMemcpy(d_params, params.data(), numParams * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means.data(), numParams * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas, sigmas.data(), numParams * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, flags.data(), numParams * sizeof(int), cudaMemcpyHostToDevice);

    double gpu_prior = computeTotalPrior(d_params, d_means, d_sigmas, d_flags, numParams, nullptr);

    cudaFree(d_params);
    cudaFree(d_means);
    cudaFree(d_sigmas);
    cudaFree(d_flags);

    // Compare results
    const double tol = 1e-12;
    result.passed = compareValues(cpu_prior, gpu_prior, tol, "prior LLH");

    if (!result.passed) {
        result.message = "Prior likelihood mismatch";
    }

    std::cout << "  CPU prior: " << cpu_prior << ", GPU prior: " << gpu_prior << std::endl;

    return result;
}

TestResult testEdgeCases() {
    TestResult result;
    result.name = "SAY Likelihood - Edge Cases";

    result.passed = true;
    const double tol = 1e-10;

    // Test case 1: Observation without expectation (should be -inf)
    {
        std::vector<double> dataCount = {5.0};
        std::vector<double> wSum = {0.0};
        std::vector<double> w2Sum = {0.0};

        double* d_dataCount;
        double* d_wSum;
        double* d_w2Sum;

        cudaMalloc(&d_dataCount, sizeof(double));
        cudaMalloc(&d_wSum, sizeof(double));
        cudaMalloc(&d_w2Sum, sizeof(double));

        cudaMemcpy(d_dataCount, dataCount.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wSum, wSum.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2Sum, w2Sum.data(), sizeof(double), cudaMemcpyHostToDevice);

        double gpu_llh = computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, 1, nullptr);

        cudaFree(d_dataCount);
        cudaFree(d_wSum);
        cudaFree(d_w2Sum);

        // Should be very negative
        if (gpu_llh > -1e10) {
            result.passed = false;
            result.message = "Impossible case (k>0, w=0) should return -inf";
            std::cerr << "  Edge case 1 failed: GPU returned " << gpu_llh << std::endl;
        }
    }

    // Test case 2: Zero w2_sum should fall back to Poisson
    {
        double k = 10.0;
        double w = 10.0;
        std::vector<double> dataCount = {k};
        std::vector<double> wSum = {w};
        std::vector<double> w2Sum = {0.0};

        double* d_dataCount;
        double* d_wSum;
        double* d_w2Sum;

        cudaMalloc(&d_dataCount, sizeof(double));
        cudaMalloc(&d_wSum, sizeof(double));
        cudaMalloc(&d_w2Sum, sizeof(double));

        cudaMemcpy(d_dataCount, dataCount.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wSum, wSum.data(), sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2Sum, w2Sum.data(), sizeof(double), cudaMemcpyHostToDevice);

        double gpu_llh = computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, 1, nullptr);

        cudaFree(d_dataCount);
        cudaFree(d_wSum);
        cudaFree(d_w2Sum);

        // Expected Poisson: k*log(lambda) - lambda - lgamma(k+1)
        double expected_poisson = k * std::log(w) - w - std::lgamma(k + 1.0);

        if (!compareValues(expected_poisson, gpu_llh, tol, "Poisson fallback")) {
            result.passed = false;
            result.message = "Poisson fallback test failed";
        }
    }

    return result;
}

TestResult testLargeScaleLikelihood() {
    TestResult result;
    result.name = "SAY Likelihood - Large Scale (10K bins)";

    TestRNG rng(456);
    const int numBins = 10000;

    // Generate realistic test data
    std::vector<double> dataCount(numBins);
    std::vector<double> wSum(numBins);
    std::vector<double> w2Sum(numBins);

    for (int i = 0; i < numBins; ++i) {
        double expected = rng.uniform(0.1, 500.0);
        wSum[i] = expected;
        w2Sum[i] = expected * rng.uniform(0.01, 0.3);
        dataCount[i] = std::max(0.0, expected + rng.uniform(-3, 3) * std::sqrt(expected));
    }

    // CPU computation
    Timer timer;
    timer.start();
    double cpu_llh = cpuTotalSAYLikelihood(dataCount, wSum, w2Sum);
    result.cpuTime = timer.stop();

    // GPU computation
    double* d_dataCount;
    double* d_wSum;
    double* d_w2Sum;

    cudaMalloc(&d_dataCount, numBins * sizeof(double));
    cudaMalloc(&d_wSum, numBins * sizeof(double));
    cudaMalloc(&d_w2Sum, numBins * sizeof(double));

    cudaMemcpy(d_dataCount, dataCount.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wSum, wSum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2Sum, w2Sum.data(), numBins * sizeof(double), cudaMemcpyHostToDevice);

    // Warm up
    computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, numBins, nullptr);

    timer.start();
    double gpu_llh = computeTotalSAYLikelihood(d_dataCount, d_wSum, d_w2Sum, numBins, nullptr);
    result.gpuTime = timer.stop();

    cudaFree(d_dataCount);
    cudaFree(d_wSum);
    cudaFree(d_w2Sum);

    // Compare results (relaxed tolerance for large sums)
    const double tol = 1e-7;
    result.passed = compareValues(cpu_llh, gpu_llh, tol, "total SAY LLH");

    if (!result.passed) {
        result.message = "Large scale likelihood mismatch";
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
    std::cout << "SAY Likelihood Tests\n";
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
    std::cout << "Using GPU: " << prop.name << "\n\n";

    TestSuite suite;

    suite.addResult(testSAYLikelihoodSingleBin());
    suite.addResult(testSAYLikelihoodMultiBin());
    suite.addResult(testGaussianPrior());
    suite.addResult(testEdgeCases());
    suite.addResult(testLargeScaleLikelihood());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
