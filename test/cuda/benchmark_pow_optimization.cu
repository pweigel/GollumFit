/**
 * @file benchmark_pow_optimization.cu
 * @brief Benchmark different pow() optimization strategies for astrophysical tilt.
 *
 * The standard pow(x, y) function is expensive on GPU. This benchmark compares:
 * 1. Standard pow() - baseline
 * 2. exp2/log2 approach: pow(x,y) = exp2(y * log2(x))
 * 3. Fast math intrinsics: __expf/__logf (single precision)
 * 4. Precomputed exp/log with FMA
 * 5. Lookup table approximation
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

//==============================================================================
// Different pow() implementations
//==============================================================================

/**
 * @brief Original implementation using standard pow()
 */
__device__ __forceinline__ double powOriginal(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex
) {
    double medianEnergy = pow(10.0, medianLog10Energy);
    double ratio = primaryEnergy / medianEnergy;
    return pow(ratio, -deltaIndex);
}

/**
 * @brief Optimized using exp2/log2
 * pow(x, y) = 2^(y * log2(x))
 * This is faster because exp2 and log2 have hardware support
 */
__device__ __forceinline__ double powExp2Log2(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex
) {
    // pow(10, x) = exp2(x * log2(10)) = exp2(x * 3.321928...)
    constexpr double LOG2_10 = 3.321928094887362;
    double medianEnergy = exp2(medianLog10Energy * LOG2_10);
    double ratio = primaryEnergy / medianEnergy;

    // pow(ratio, -deltaIndex) = exp2(-deltaIndex * log2(ratio))
    return exp2(-deltaIndex * log2(ratio));
}

/**
 * @brief Fast math version using single precision intrinsics
 * Trades precision for speed - good when deltaIndex is small
 */
__device__ __forceinline__ double powFastMath(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex
) {
    constexpr float LOG2_10f = 3.321928094887362f;
    float medianEnergy = exp2f(static_cast<float>(medianLog10Energy) * LOG2_10f);
    float ratio = primaryEnergy / medianEnergy;

    // Use fast intrinsics
    float result = __expf(-static_cast<float>(deltaIndex) * __log2f(ratio));
    return static_cast<double>(result);
}

/**
 * @brief Hybrid: precompute median once per block, use exp2/log2 for ratio
 */
__device__ __forceinline__ double powHybrid(
    float primaryEnergy,
    double precomputedMedianEnergy,  // Already computed: pow(10, medianLog10Energy)
    double deltaIndex
) {
    double ratio = primaryEnergy / precomputedMedianEnergy;
    return exp2(-deltaIndex * log2(ratio));
}

/**
 * @brief Using natural log/exp (sometimes faster than log2/exp2)
 */
__device__ __forceinline__ double powExpLog(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex
) {
    constexpr double LN_10 = 2.302585092994046;
    double medianEnergy = exp(medianLog10Energy * LN_10);
    double ratio = primaryEnergy / medianEnergy;
    return exp(-deltaIndex * log(ratio));
}

/**
 * @brief Polynomial approximation for small exponents
 * Uses Taylor expansion: x^a ≈ 1 + a*ln(x) + 0.5*a²*ln²(x) for |a| < 0.5
 * Good when deltaIndex is small (typical case in physics)
 */
__device__ __forceinline__ double powTaylorApprox(
    float primaryEnergy,
    double medianLog10Energy,
    double deltaIndex
) {
    constexpr double LN_10 = 2.302585092994046;
    double medianEnergy = exp(medianLog10Energy * LN_10);
    double ratio = primaryEnergy / medianEnergy;

    double lnRatio = log(ratio);
    double a = -deltaIndex;

    // Taylor expansion to 3rd order: exp(a*ln(x)) ≈ 1 + a*ln(x) + 0.5*a²*ln²(x) + ...
    // More accurate: use exp() on small argument
    double exponent = a * lnRatio;

    // For small exponents, use Taylor; otherwise fall back to exp
    if (fabs(exponent) < 0.1) {
        double e2 = exponent * exponent;
        return 1.0 + exponent + 0.5 * e2 + (1.0/6.0) * e2 * exponent;
    }
    return exp(exponent);
}

//==============================================================================
// Benchmark Kernels
//==============================================================================

__global__ void kernelPowOriginal(
    const float* __restrict__ energies,
    const double* __restrict__ params,  // [medianLog10Energy, deltaIndex1, deltaIndex2]
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double medianLog10Energy = params[0];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    double medianEnergy = pow(10.0, medianLog10Energy);
    double result;
    if (energy > medianEnergy) {
        result = powOriginal(energy, medianLog10Energy, deltaIndex2);
    } else {
        result = powOriginal(energy, medianLog10Energy, deltaIndex1);
    }
    output[tid] = result;
}

__global__ void kernelPowExp2Log2(
    const float* __restrict__ energies,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double medianLog10Energy = params[0];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    constexpr double LOG2_10 = 3.321928094887362;
    double medianEnergy = exp2(medianLog10Energy * LOG2_10);

    double result;
    if (energy > medianEnergy) {
        result = powExp2Log2(energy, medianLog10Energy, deltaIndex2);
    } else {
        result = powExp2Log2(energy, medianLog10Energy, deltaIndex1);
    }
    output[tid] = result;
}

__global__ void kernelPowFastMath(
    const float* __restrict__ energies,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double medianLog10Energy = params[0];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    constexpr float LOG2_10f = 3.321928094887362f;
    float medianEnergy = exp2f(static_cast<float>(medianLog10Energy) * LOG2_10f);

    double result;
    if (energy > medianEnergy) {
        result = powFastMath(energy, medianLog10Energy, deltaIndex2);
    } else {
        result = powFastMath(energy, medianLog10Energy, deltaIndex1);
    }
    output[tid] = result;
}

__global__ void kernelPowHybrid(
    const float* __restrict__ energies,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    // Precompute median energy once per block using shared memory
    __shared__ double s_medianEnergy;

    if (threadIdx.x == 0) {
        constexpr double LOG2_10 = 3.321928094887362;
        s_medianEnergy = exp2(params[0] * LOG2_10);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    double result;
    if (energy > s_medianEnergy) {
        result = powHybrid(energy, s_medianEnergy, deltaIndex2);
    } else {
        result = powHybrid(energy, s_medianEnergy, deltaIndex1);
    }
    output[tid] = result;
}

__global__ void kernelPowExpLog(
    const float* __restrict__ energies,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double medianLog10Energy = params[0];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    constexpr double LN_10 = 2.302585092994046;
    double medianEnergy = exp(medianLog10Energy * LN_10);

    double result;
    if (energy > medianEnergy) {
        result = powExpLog(energy, medianLog10Energy, deltaIndex2);
    } else {
        result = powExpLog(energy, medianLog10Energy, deltaIndex1);
    }
    output[tid] = result;
}

__global__ void kernelPowTaylor(
    const float* __restrict__ energies,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    float energy = energies[tid];
    double medianLog10Energy = params[0];
    double deltaIndex1 = params[1];
    double deltaIndex2 = params[2];

    constexpr double LN_10 = 2.302585092994046;
    double medianEnergy = exp(medianLog10Energy * LN_10);

    double result;
    if (energy > medianEnergy) {
        result = powTaylorApprox(energy, medianLog10Energy, deltaIndex2);
    } else {
        result = powTaylorApprox(energy, medianLog10Energy, deltaIndex1);
    }
    output[tid] = result;
}

//==============================================================================
// Full Astro Weight Kernels (to see impact on full computation)
//==============================================================================

__global__ void kernelFullAstroOriginal(
    const float* __restrict__ energies,
    const int32_t* __restrict__ primaryTypes,
    const double* __restrict__ cachedAstroWeights,
    const double* __restrict__ params,  // [astroNorm, medianLog10, delta1, delta2, balance]
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double astroNorm = params[0];
    double medianLog10Energy = params[1];
    double deltaIndex1 = params[2];
    double deltaIndex2 = params[3];
    double balance = params[4];

    float energy = energies[tid];
    int32_t primaryType = primaryTypes[tid];
    double cachedAstro = cachedAstroWeights[tid];

    // Original pow() implementation
    double medianEnergy = pow(10.0, medianLog10Energy);
    double ratio = energy / medianEnergy;
    double tiltWeight;
    if (energy > medianEnergy) {
        tiltWeight = pow(ratio, -deltaIndex2);
    } else {
        tiltWeight = pow(ratio, -deltaIndex1);
    }

    double neuWeight = (primaryType < 0) ? balance : (2.0 - balance);
    output[tid] = astroNorm * cachedAstro * tiltWeight * neuWeight;
}

__global__ void kernelFullAstroOptimized(
    const float* __restrict__ energies,
    const int32_t* __restrict__ primaryTypes,
    const double* __restrict__ cachedAstroWeights,
    const double* __restrict__ params,
    double* __restrict__ output,
    int numEvents
) {
    // Precompute median energy once per block
    __shared__ double s_medianEnergy;
    __shared__ double s_params[5];

    if (threadIdx.x < 5) {
        s_params[threadIdx.x] = params[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        constexpr double LOG2_10 = 3.321928094887362;
        s_medianEnergy = exp2(s_params[1] * LOG2_10);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    double astroNorm = s_params[0];
    double deltaIndex1 = s_params[2];
    double deltaIndex2 = s_params[3];
    double balance = s_params[4];

    float energy = energies[tid];
    int32_t primaryType = primaryTypes[tid];
    double cachedAstro = cachedAstroWeights[tid];

    // Optimized: use exp2/log2 with precomputed median
    double ratio = energy / s_medianEnergy;
    double logRatio = log2(ratio);
    double tiltWeight;
    if (energy > s_medianEnergy) {
        tiltWeight = exp2(-deltaIndex2 * logRatio);
    } else {
        tiltWeight = exp2(-deltaIndex1 * logRatio);
    }

    double neuWeight = (primaryType < 0) ? balance : (2.0 - balance);
    output[tid] = astroNorm * cachedAstro * tiltWeight * neuWeight;
}

//==============================================================================
// Benchmark Infrastructure
//==============================================================================

struct BenchmarkResult {
    const char* name;
    double timeMs;
    double speedup;
    double maxRelError;
};

enum KernelType {
    KERNEL_ORIGINAL,
    KERNEL_EXP2LOG2,
    KERNEL_FASTMATH,
    KERNEL_HYBRID,
    KERNEL_EXPLOG,
    KERNEL_TAYLOR
};

void launchPowKernel(KernelType type, int gridSize, int blockSize,
                      float* d_energies, double* d_params, double* d_output, int numEvents) {
    switch (type) {
        case KERNEL_ORIGINAL:
            kernelPowOriginal<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
        case KERNEL_EXP2LOG2:
            kernelPowExp2Log2<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
        case KERNEL_FASTMATH:
            kernelPowFastMath<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
        case KERNEL_HYBRID:
            kernelPowHybrid<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
        case KERNEL_EXPLOG:
            kernelPowExpLog<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
        case KERNEL_TAYLOR:
            kernelPowTaylor<<<gridSize, blockSize>>>(d_energies, d_params, d_output, numEvents);
            break;
    }
}

BenchmarkResult runBenchmark(
    const char* name,
    KernelType kernelType,
    float* d_energies,
    double* d_params,
    double* d_output,
    double* d_reference,
    int numEvents,
    int numIterations,
    double baselineMs = 0.0
) {
    int blockSize = 256;
    int gridSize = (numEvents + blockSize - 1) / blockSize;

    // Warmup
    launchPowKernel(kernelType, gridSize, blockSize, d_energies, d_params, d_output, numEvents);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        launchPowKernel(kernelType, gridSize, blockSize, d_energies, d_params, d_output, numEvents);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double avgMs = ms / numIterations;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute error vs reference
    std::vector<double> output(numEvents);
    std::vector<double> reference(numEvents);
    cudaMemcpy(output.data(), d_output, numEvents * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(reference.data(), d_reference, numEvents * sizeof(double), cudaMemcpyDeviceToHost);

    double maxRelError = 0.0;
    for (int i = 0; i < numEvents; ++i) {
        if (reference[i] != 0.0) {
            double relErr = std::abs((output[i] - reference[i]) / reference[i]);
            maxRelError = std::max(maxRelError, relErr);
        }
    }

    return {name, avgMs, (baselineMs > 0) ? baselineMs / avgMs : 1.0, maxRelError};
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "pow() Optimization Benchmark for Astrophysical Tilt\n";
    std::cout << "================================================================\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n\n";

    const int numEvents = 5000000;
    const int numIterations = 100;

    // Allocate memory
    float* d_energies;
    int32_t* d_primaryTypes;
    double* d_cachedAstro;
    double* d_params;
    double* d_output;
    double* d_reference;

    cudaMalloc(&d_energies, numEvents * sizeof(float));
    cudaMalloc(&d_primaryTypes, numEvents * sizeof(int32_t));
    cudaMalloc(&d_cachedAstro, numEvents * sizeof(double));
    cudaMalloc(&d_params, 5 * sizeof(double));
    cudaMalloc(&d_output, numEvents * sizeof(double));
    cudaMalloc(&d_reference, numEvents * sizeof(double));

    // Generate test data
    TestRNG rng(12345);
    std::vector<float> energies(numEvents);
    std::vector<int32_t> primaryTypes(numEvents);
    std::vector<double> cachedAstro(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        energies[i] = std::pow(10.0f, rng.uniform(2.0f, 7.0f));  // 100 GeV to 10 PeV
        primaryTypes[i] = (rng.uniform(0, 1) > 0.5) ? 12 : -12;
        cachedAstro[i] = rng.uniform(1e-12, 1e-8);
    }

    // Typical physics parameters
    std::vector<double> params = {
        1.0,    // astroNorm
        5.0,    // medianLog10Energy (100 TeV)
        0.1,    // deltaIndex1 (below median)
        0.2,    // deltaIndex2 (above median)
        1.0     // balance (neutrino/antineutrino)
    };

    cudaMemcpy(d_energies, energies.data(), numEvents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primaryTypes, primaryTypes.data(), numEvents * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cachedAstro, cachedAstro.data(), numEvents * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params.data(), 5 * sizeof(double), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // Benchmark pow() implementations
    //--------------------------------------------------------------------------

    std::cout << "================================================================\n";
    std::cout << "Isolated pow() Implementations (5M events)\n";
    std::cout << "================================================================\n\n";

    // Generate reference using original
    int blockSize = 256;
    int gridSize = (numEvents + blockSize - 1) / blockSize;
    kernelPowOriginal<<<gridSize, blockSize>>>(d_energies, d_params, d_reference, numEvents);
    cudaDeviceSynchronize();

    std::vector<BenchmarkResult> results;

    // Benchmark original (baseline)
    auto baseline = runBenchmark("Original pow()", KERNEL_ORIGINAL,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations);
    baseline.speedup = 1.0;
    results.push_back(baseline);

    // Benchmark exp2/log2
    results.push_back(runBenchmark("exp2/log2", KERNEL_EXP2LOG2,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations, baseline.timeMs));

    // Benchmark fast math
    results.push_back(runBenchmark("Fast math (__expf)", KERNEL_FASTMATH,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations, baseline.timeMs));

    // Benchmark hybrid (precomputed median)
    results.push_back(runBenchmark("Hybrid (precomputed)", KERNEL_HYBRID,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations, baseline.timeMs));

    // Benchmark exp/log
    results.push_back(runBenchmark("exp/log (natural)", KERNEL_EXPLOG,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations, baseline.timeMs));

    // Benchmark Taylor approximation
    results.push_back(runBenchmark("Taylor approx", KERNEL_TAYLOR,
        d_energies, d_params, d_output, d_reference, numEvents, numIterations, baseline.timeMs));

    // Print results
    std::cout << std::setw(25) << "Method"
              << std::setw(14) << "Time (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(16) << "Max Rel Error"
              << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(25) << r.name
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.timeMs
                  << std::setw(9) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(16) << std::scientific << std::setprecision(2) << r.maxRelError
                  << std::endl;
    }

    //--------------------------------------------------------------------------
    // Benchmark full astrophysical weight computation
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "Full Astrophysical Weight (with memory loads)\n";
    std::cout << "================================================================\n\n";

    // Reference for full astro
    kernelFullAstroOriginal<<<gridSize, blockSize>>>(
        d_energies, d_primaryTypes, d_cachedAstro, d_params, d_reference, numEvents);
    cudaDeviceSynchronize();

    // Benchmark original full astro
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        kernelFullAstroOriginal<<<gridSize, blockSize>>>(
            d_energies, d_primaryTypes, d_cachedAstro, d_params, d_output, numEvents);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float origMs;
    cudaEventElapsedTime(&origMs, start, stop);
    double origAvg = origMs / numIterations;

    // Benchmark optimized full astro
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        kernelFullAstroOptimized<<<gridSize, blockSize>>>(
            d_energies, d_primaryTypes, d_cachedAstro, d_params, d_output, numEvents);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float optMs;
    cudaEventElapsedTime(&optMs, start, stop);
    double optAvg = optMs / numIterations;

    // Compute error
    std::vector<double> output(numEvents);
    std::vector<double> reference(numEvents);
    cudaMemcpy(output.data(), d_output, numEvents * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(reference.data(), d_reference, numEvents * sizeof(double), cudaMemcpyDeviceToHost);

    double maxErr = 0.0;
    for (int i = 0; i < numEvents; ++i) {
        if (reference[i] != 0.0) {
            double relErr = std::abs((output[i] - reference[i]) / reference[i]);
            maxErr = std::max(maxErr, relErr);
        }
    }

    std::cout << std::setw(25) << "Method"
              << std::setw(14) << "Time (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(16) << "Max Rel Error"
              << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    std::cout << std::setw(25) << "Original (pow)"
              << std::setw(14) << std::fixed << std::setprecision(4) << origAvg
              << std::setw(9) << "1.00x"
              << std::setw(16) << "0.00e+00"
              << std::endl;
    std::cout << std::setw(25) << "Optimized (exp2/log2)"
              << std::setw(14) << std::fixed << std::setprecision(4) << optAvg
              << std::setw(9) << std::fixed << std::setprecision(2) << origAvg / optAvg << "x"
              << std::setw(16) << std::scientific << std::setprecision(2) << maxErr
              << std::endl;

    //--------------------------------------------------------------------------
    // Summary
    //--------------------------------------------------------------------------

    std::cout << "\n================================================================\n";
    std::cout << "Recommendations\n";
    std::cout << "================================================================\n\n";

    // Find best method
    double bestSpeedup = 0;
    const char* bestMethod = "Original";
    double bestError = 0;
    for (const auto& r : results) {
        if (r.speedup > bestSpeedup && r.maxRelError < 1e-10) {
            bestSpeedup = r.speedup;
            bestMethod = r.name;
            bestError = r.maxRelError;
        }
    }

    std::cout << "Best high-precision method: " << bestMethod << "\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << bestSpeedup << "x\n";
    std::cout << "  Max relative error: " << std::scientific << bestError << "\n\n";

    // Find fastest overall
    double fastestTime = results[0].timeMs;
    const char* fastestMethod = results[0].name;
    for (const auto& r : results) {
        if (r.timeMs < fastestTime) {
            fastestTime = r.timeMs;
            fastestMethod = r.name;
        }
    }

    std::cout << "Fastest method (any precision): " << fastestMethod << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << fastestTime << " ms\n";

    std::cout << "\nFor physics fitting, recommend:\n";
    std::cout << "  - Use 'Hybrid (precomputed)' or 'exp2/log2' for best speed with FP64 precision\n";
    std::cout << "  - Use 'Fast math' only if single precision is acceptable\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_energies);
    cudaFree(d_primaryTypes);
    cudaFree(d_cachedAstro);
    cudaFree(d_params);
    cudaFree(d_output);
    cudaFree(d_reference);

    return 0;
}
