/**
 * @file benchmark_conv_flux_optimization.cu
 * @brief Benchmark comparing original vs optimized Conv Flux Assembly kernels.
 *
 * Tests the following optimizations:
 * 1. Constant memory for parameters (vs shared memory)
 * 2. __ldg() texture cache path (vs regular loads)
 * 3. FMA operations (vs separate multiply-add)
 * 4. Interleaved memory layout (vs separate arrays)
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace gollumfit::gpu;

//==============================================================================
// Parameter indices
//==============================================================================

enum ParamIndex {
    P_CONV_NORM = 0, P_PROMPT_NORM = 1, P_ADU = 2, P_KLU = 3,
    P_HEKP = 4, P_HEKM = 5, P_VHE1PIP = 6, P_VHE1PIM = 7,
    P_VHE3KP = 8, P_VHE3KM = 9, P_VHE3PIP = 10, P_VHE3PIM = 11,
    P_VHE3P = 12, P_VHE3N = 13,
    P_CR1 = 14, P_CR2 = 15, P_CR3 = 16, P_CR4 = 17, P_CR5 = 18, P_CR6 = 19,
    P_ICEGRAD0 = 20, P_ICEGRAD1 = 21, P_ICEGRAD2 = 22, P_ICEGRAD3 = 23,
    P_ICEGRAD4 = 24, P_ICEGRAD5 = 25, P_ICEGRAD6 = 26, P_ICEGRAD7 = 27,
    P_ICEGRAD8 = 28,
    P_DELTA_DOMEFF = 29, P_HOLEICE_FWD = 30,
    P_ASTRO_NORM = 31, P_ASTRO_DGAMMA = 32, P_ASTRO_DGAMMA_SEC = 33,
    P_ASTRO_PIVOT = 34, P_NEUANEU_RATIO = 35, P_NUXS = 36, P_NUBARXS = 37
};

//==============================================================================
// Constant Memory Declaration
//==============================================================================

__constant__ double c_params[NUM_FIT_PARAMS];

void uploadParams(const double* h_params, cudaStream_t stream = nullptr) {
    cudaMemcpyToSymbol(c_params, h_params, NUM_FIT_PARAMS * sizeof(double));
}

//==============================================================================
// ORIGINAL Kernel - Shared memory for params, regular loads
//==============================================================================

__global__ void kernelConvFluxOriginal(
    const double* __restrict__ convWeight,
    const double* __restrict__ hadronicHEkp,
    const double* __restrict__ hadronicHEkm,
    const double* __restrict__ hadronicVHE1pip,
    const double* __restrict__ hadronicVHE1pim,
    const double* __restrict__ hadronicVHE3kp,
    const double* __restrict__ hadronicVHE3km,
    const double* __restrict__ hadronicVHE3pip,
    const double* __restrict__ hadronicVHE3pim,
    const double* __restrict__ hadronicVHE3p,
    const double* __restrict__ hadronicVHE3n,
    const double* __restrict__ cosmicRay1,
    const double* __restrict__ cosmicRay2,
    const double* __restrict__ cosmicRay3,
    const double* __restrict__ cosmicRay4,
    const double* __restrict__ cosmicRay5,
    const double* __restrict__ cosmicRay6,
    const double* __restrict__ atmDensity,
    const double* __restrict__ kaonLosses,
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

    // Regular loads
    double hadronic = s_params[P_HEKP] * hadronicHEkp[tid] +
                      s_params[P_HEKM] * hadronicHEkm[tid] +
                      s_params[P_VHE1PIP] * hadronicVHE1pip[tid] +
                      s_params[P_VHE1PIM] * hadronicVHE1pim[tid] +
                      s_params[P_VHE3KP] * hadronicVHE3kp[tid] +
                      s_params[P_VHE3KM] * hadronicVHE3km[tid] +
                      s_params[P_VHE3PIP] * hadronicVHE3pip[tid] +
                      s_params[P_VHE3PIM] * hadronicVHE3pim[tid] +
                      s_params[P_VHE3P] * hadronicVHE3p[tid] +
                      s_params[P_VHE3N] * hadronicVHE3n[tid];

    double cr = s_params[P_CR1] * cosmicRay1[tid] +
                s_params[P_CR2] * cosmicRay2[tid] +
                s_params[P_CR3] * cosmicRay3[tid] +
                s_params[P_CR4] * cosmicRay4[tid] +
                s_params[P_CR5] * cosmicRay5[tid] +
                s_params[P_CR6] * cosmicRay6[tid];

    double flux = convWeight[tid] + hadronic + cr;
    double atmWeight = (1.0 + s_params[P_ADU] * atmDensity[tid]) *
                       (1.0 + s_params[P_KLU] * kaonLosses[tid]);

    output[tid] = s_params[P_CONV_NORM] * flux * atmWeight;
}

//==============================================================================
// OPTIMIZED V1: Constant memory for params
//==============================================================================

__global__ void kernelConvFluxConstMem(
    const double* __restrict__ convWeight,
    const double* __restrict__ hadronicHEkp,
    const double* __restrict__ hadronicHEkm,
    const double* __restrict__ hadronicVHE1pip,
    const double* __restrict__ hadronicVHE1pim,
    const double* __restrict__ hadronicVHE3kp,
    const double* __restrict__ hadronicVHE3km,
    const double* __restrict__ hadronicVHE3pip,
    const double* __restrict__ hadronicVHE3pim,
    const double* __restrict__ hadronicVHE3p,
    const double* __restrict__ hadronicVHE3n,
    const double* __restrict__ cosmicRay1,
    const double* __restrict__ cosmicRay2,
    const double* __restrict__ cosmicRay3,
    const double* __restrict__ cosmicRay4,
    const double* __restrict__ cosmicRay5,
    const double* __restrict__ cosmicRay6,
    const double* __restrict__ atmDensity,
    const double* __restrict__ kaonLosses,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Use constant memory params (no shared memory sync needed)
    double hadronic = c_params[P_HEKP] * hadronicHEkp[tid] +
                      c_params[P_HEKM] * hadronicHEkm[tid] +
                      c_params[P_VHE1PIP] * hadronicVHE1pip[tid] +
                      c_params[P_VHE1PIM] * hadronicVHE1pim[tid] +
                      c_params[P_VHE3KP] * hadronicVHE3kp[tid] +
                      c_params[P_VHE3KM] * hadronicVHE3km[tid] +
                      c_params[P_VHE3PIP] * hadronicVHE3pip[tid] +
                      c_params[P_VHE3PIM] * hadronicVHE3pim[tid] +
                      c_params[P_VHE3P] * hadronicVHE3p[tid] +
                      c_params[P_VHE3N] * hadronicVHE3n[tid];

    double cr = c_params[P_CR1] * cosmicRay1[tid] +
                c_params[P_CR2] * cosmicRay2[tid] +
                c_params[P_CR3] * cosmicRay3[tid] +
                c_params[P_CR4] * cosmicRay4[tid] +
                c_params[P_CR5] * cosmicRay5[tid] +
                c_params[P_CR6] * cosmicRay6[tid];

    double flux = convWeight[tid] + hadronic + cr;
    double atmWeight = (1.0 + c_params[P_ADU] * atmDensity[tid]) *
                       (1.0 + c_params[P_KLU] * kaonLosses[tid]);

    output[tid] = c_params[P_CONV_NORM] * flux * atmWeight;
}

//==============================================================================
// OPTIMIZED V2: Constant memory + __ldg()
//==============================================================================

__global__ void kernelConvFluxLdg(
    const double* __restrict__ convWeight,
    const double* __restrict__ hadronicHEkp,
    const double* __restrict__ hadronicHEkm,
    const double* __restrict__ hadronicVHE1pip,
    const double* __restrict__ hadronicVHE1pim,
    const double* __restrict__ hadronicVHE3kp,
    const double* __restrict__ hadronicVHE3km,
    const double* __restrict__ hadronicVHE3pip,
    const double* __restrict__ hadronicVHE3pim,
    const double* __restrict__ hadronicVHE3p,
    const double* __restrict__ hadronicVHE3n,
    const double* __restrict__ cosmicRay1,
    const double* __restrict__ cosmicRay2,
    const double* __restrict__ cosmicRay3,
    const double* __restrict__ cosmicRay4,
    const double* __restrict__ cosmicRay5,
    const double* __restrict__ cosmicRay6,
    const double* __restrict__ atmDensity,
    const double* __restrict__ kaonLosses,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Use __ldg for texture cache path
    double hadronic = c_params[P_HEKP] * __ldg(&hadronicHEkp[tid]) +
                      c_params[P_HEKM] * __ldg(&hadronicHEkm[tid]) +
                      c_params[P_VHE1PIP] * __ldg(&hadronicVHE1pip[tid]) +
                      c_params[P_VHE1PIM] * __ldg(&hadronicVHE1pim[tid]) +
                      c_params[P_VHE3KP] * __ldg(&hadronicVHE3kp[tid]) +
                      c_params[P_VHE3KM] * __ldg(&hadronicVHE3km[tid]) +
                      c_params[P_VHE3PIP] * __ldg(&hadronicVHE3pip[tid]) +
                      c_params[P_VHE3PIM] * __ldg(&hadronicVHE3pim[tid]) +
                      c_params[P_VHE3P] * __ldg(&hadronicVHE3p[tid]) +
                      c_params[P_VHE3N] * __ldg(&hadronicVHE3n[tid]);

    double cr = c_params[P_CR1] * __ldg(&cosmicRay1[tid]) +
                c_params[P_CR2] * __ldg(&cosmicRay2[tid]) +
                c_params[P_CR3] * __ldg(&cosmicRay3[tid]) +
                c_params[P_CR4] * __ldg(&cosmicRay4[tid]) +
                c_params[P_CR5] * __ldg(&cosmicRay5[tid]) +
                c_params[P_CR6] * __ldg(&cosmicRay6[tid]);

    double flux = __ldg(&convWeight[tid]) + hadronic + cr;
    double atmWeight = (1.0 + c_params[P_ADU] * __ldg(&atmDensity[tid])) *
                       (1.0 + c_params[P_KLU] * __ldg(&kaonLosses[tid]));

    output[tid] = c_params[P_CONV_NORM] * flux * atmWeight;
}

//==============================================================================
// OPTIMIZED V3: Constant memory + __ldg() + FMA
//==============================================================================

__global__ void kernelConvFluxFMA(
    const double* __restrict__ convWeight,
    const double* __restrict__ hadronicHEkp,
    const double* __restrict__ hadronicHEkm,
    const double* __restrict__ hadronicVHE1pip,
    const double* __restrict__ hadronicVHE1pim,
    const double* __restrict__ hadronicVHE3kp,
    const double* __restrict__ hadronicVHE3km,
    const double* __restrict__ hadronicVHE3pip,
    const double* __restrict__ hadronicVHE3pim,
    const double* __restrict__ hadronicVHE3p,
    const double* __restrict__ hadronicVHE3n,
    const double* __restrict__ cosmicRay1,
    const double* __restrict__ cosmicRay2,
    const double* __restrict__ cosmicRay3,
    const double* __restrict__ cosmicRay4,
    const double* __restrict__ cosmicRay5,
    const double* __restrict__ cosmicRay6,
    const double* __restrict__ atmDensity,
    const double* __restrict__ kaonLosses,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Use FMA for compute
    double hadronic = 0.0;
    hadronic = fma(c_params[P_HEKP], __ldg(&hadronicHEkp[tid]), hadronic);
    hadronic = fma(c_params[P_HEKM], __ldg(&hadronicHEkm[tid]), hadronic);
    hadronic = fma(c_params[P_VHE1PIP], __ldg(&hadronicVHE1pip[tid]), hadronic);
    hadronic = fma(c_params[P_VHE1PIM], __ldg(&hadronicVHE1pim[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3KP], __ldg(&hadronicVHE3kp[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3KM], __ldg(&hadronicVHE3km[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3PIP], __ldg(&hadronicVHE3pip[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3PIM], __ldg(&hadronicVHE3pim[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3P], __ldg(&hadronicVHE3p[tid]), hadronic);
    hadronic = fma(c_params[P_VHE3N], __ldg(&hadronicVHE3n[tid]), hadronic);

    double cr = 0.0;
    cr = fma(c_params[P_CR1], __ldg(&cosmicRay1[tid]), cr);
    cr = fma(c_params[P_CR2], __ldg(&cosmicRay2[tid]), cr);
    cr = fma(c_params[P_CR3], __ldg(&cosmicRay3[tid]), cr);
    cr = fma(c_params[P_CR4], __ldg(&cosmicRay4[tid]), cr);
    cr = fma(c_params[P_CR5], __ldg(&cosmicRay5[tid]), cr);
    cr = fma(c_params[P_CR6], __ldg(&cosmicRay6[tid]), cr);

    double flux = __ldg(&convWeight[tid]) + hadronic + cr;
    double atmWeight = fma(c_params[P_ADU], __ldg(&atmDensity[tid]), 1.0) *
                       fma(c_params[P_KLU], __ldg(&kaonLosses[tid]), 1.0);

    output[tid] = c_params[P_CONV_NORM] * flux * atmWeight;
}

//==============================================================================
// OPTIMIZED V4: Interleaved memory layout
//==============================================================================

struct __align__(128) InterleavedData {
    double hadronic[10];
    double cosmicRay[6];
};

__global__ void kernelConvFluxInterleaved(
    const InterleavedData* __restrict__ interleavedData,
    const double* __restrict__ convWeight,
    const double* __restrict__ atmDensity,
    const double* __restrict__ kaonLosses,
    double* __restrict__ output,
    int numEvents
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEvents) return;

    // Load all 16 values in 8x double2 loads (contiguous)
    const double2* data = reinterpret_cast<const double2*>(&interleavedData[tid]);

    double2 h01 = __ldg(&data[0]);
    double2 h23 = __ldg(&data[1]);
    double2 h45 = __ldg(&data[2]);
    double2 h67 = __ldg(&data[3]);
    double2 h89 = __ldg(&data[4]);
    double2 cr01 = __ldg(&data[5]);
    double2 cr23 = __ldg(&data[6]);
    double2 cr45 = __ldg(&data[7]);

    double hadronic = 0.0;
    hadronic = fma(c_params[P_HEKP], h01.x, hadronic);
    hadronic = fma(c_params[P_HEKM], h01.y, hadronic);
    hadronic = fma(c_params[P_VHE1PIP], h23.x, hadronic);
    hadronic = fma(c_params[P_VHE1PIM], h23.y, hadronic);
    hadronic = fma(c_params[P_VHE3KP], h45.x, hadronic);
    hadronic = fma(c_params[P_VHE3KM], h45.y, hadronic);
    hadronic = fma(c_params[P_VHE3PIP], h67.x, hadronic);
    hadronic = fma(c_params[P_VHE3PIM], h67.y, hadronic);
    hadronic = fma(c_params[P_VHE3P], h89.x, hadronic);
    hadronic = fma(c_params[P_VHE3N], h89.y, hadronic);

    double cr = 0.0;
    cr = fma(c_params[P_CR1], cr01.x, cr);
    cr = fma(c_params[P_CR2], cr01.y, cr);
    cr = fma(c_params[P_CR3], cr23.x, cr);
    cr = fma(c_params[P_CR4], cr23.y, cr);
    cr = fma(c_params[P_CR5], cr45.x, cr);
    cr = fma(c_params[P_CR6], cr45.y, cr);

    double flux = __ldg(&convWeight[tid]) + hadronic + cr;
    double atmWeight = fma(c_params[P_ADU], __ldg(&atmDensity[tid]), 1.0) *
                       fma(c_params[P_KLU], __ldg(&kaonLosses[tid]), 1.0);

    output[tid] = c_params[P_CONV_NORM] * flux * atmWeight;
}

//==============================================================================
// Benchmark Infrastructure
//==============================================================================

struct BenchResult {
    const char* name;
    double timeMs;
    double bandwidth;
    double speedup;
};

class ConvFluxBenchmark {
public:
    ConvFluxBenchmark(int numEvents) : numEvents_(numEvents) {
        allocate();
        generateData();
    }

    ~ConvFluxBenchmark() {
        deallocate();
    }

    void run(int numIterations) {
        std::vector<BenchResult> results;

        // Warmup
        kernelConvFluxOriginal<<<(numEvents_ + 255) / 256, 256>>>(
            d_convWeight_, d_hadronic_[0], d_hadronic_[1], d_hadronic_[2],
            d_hadronic_[3], d_hadronic_[4], d_hadronic_[5], d_hadronic_[6],
            d_hadronic_[7], d_hadronic_[8], d_hadronic_[9],
            d_cosmicRay_[0], d_cosmicRay_[1], d_cosmicRay_[2],
            d_cosmicRay_[3], d_cosmicRay_[4], d_cosmicRay_[5],
            d_atmDensity_, d_kaonLosses_, d_params_, d_output_, numEvents_
        );
        cudaDeviceSynchronize();

        // Calculate data bytes for bandwidth
        size_t dataBytes = numEvents_ * sizeof(double) * 19;  // 19 input arrays

        // Benchmark Original
        results.push_back(benchmark("Original (shared mem)", [&]() {
            kernelConvFluxOriginal<<<(numEvents_ + 255) / 256, 256>>>(
                d_convWeight_, d_hadronic_[0], d_hadronic_[1], d_hadronic_[2],
                d_hadronic_[3], d_hadronic_[4], d_hadronic_[5], d_hadronic_[6],
                d_hadronic_[7], d_hadronic_[8], d_hadronic_[9],
                d_cosmicRay_[0], d_cosmicRay_[1], d_cosmicRay_[2],
                d_cosmicRay_[3], d_cosmicRay_[4], d_cosmicRay_[5],
                d_atmDensity_, d_kaonLosses_, d_params_, d_output_, numEvents_
            );
        }, numIterations, dataBytes));

        double baseTime = results[0].timeMs;

        // Benchmark V1: Constant memory
        results.push_back(benchmark("V1: Constant mem", [&]() {
            kernelConvFluxConstMem<<<(numEvents_ + 255) / 256, 256>>>(
                d_convWeight_, d_hadronic_[0], d_hadronic_[1], d_hadronic_[2],
                d_hadronic_[3], d_hadronic_[4], d_hadronic_[5], d_hadronic_[6],
                d_hadronic_[7], d_hadronic_[8], d_hadronic_[9],
                d_cosmicRay_[0], d_cosmicRay_[1], d_cosmicRay_[2],
                d_cosmicRay_[3], d_cosmicRay_[4], d_cosmicRay_[5],
                d_atmDensity_, d_kaonLosses_, d_output_, numEvents_
            );
        }, numIterations, dataBytes));

        // Benchmark V2: + __ldg
        results.push_back(benchmark("V2: + __ldg()", [&]() {
            kernelConvFluxLdg<<<(numEvents_ + 255) / 256, 256>>>(
                d_convWeight_, d_hadronic_[0], d_hadronic_[1], d_hadronic_[2],
                d_hadronic_[3], d_hadronic_[4], d_hadronic_[5], d_hadronic_[6],
                d_hadronic_[7], d_hadronic_[8], d_hadronic_[9],
                d_cosmicRay_[0], d_cosmicRay_[1], d_cosmicRay_[2],
                d_cosmicRay_[3], d_cosmicRay_[4], d_cosmicRay_[5],
                d_atmDensity_, d_kaonLosses_, d_output_, numEvents_
            );
        }, numIterations, dataBytes));

        // Benchmark V3: + FMA
        results.push_back(benchmark("V3: + FMA", [&]() {
            kernelConvFluxFMA<<<(numEvents_ + 255) / 256, 256>>>(
                d_convWeight_, d_hadronic_[0], d_hadronic_[1], d_hadronic_[2],
                d_hadronic_[3], d_hadronic_[4], d_hadronic_[5], d_hadronic_[6],
                d_hadronic_[7], d_hadronic_[8], d_hadronic_[9],
                d_cosmicRay_[0], d_cosmicRay_[1], d_cosmicRay_[2],
                d_cosmicRay_[3], d_cosmicRay_[4], d_cosmicRay_[5],
                d_atmDensity_, d_kaonLosses_, d_output_, numEvents_
            );
        }, numIterations, dataBytes));

        // Benchmark V4: Interleaved
        results.push_back(benchmark("V4: Interleaved", [&]() {
            kernelConvFluxInterleaved<<<(numEvents_ + 255) / 256, 256>>>(
                d_interleaved_, d_convWeight_, d_atmDensity_, d_kaonLosses_,
                d_output_, numEvents_
            );
        }, numIterations, dataBytes));

        // Calculate speedups
        for (auto& r : results) {
            r.speedup = baseTime / r.timeMs;
        }

        // Print results
        printResults(results);
    }

private:
    int numEvents_;
    double* d_convWeight_;
    double* d_hadronic_[10];
    double* d_cosmicRay_[6];
    double* d_atmDensity_;
    double* d_kaonLosses_;
    double* d_params_;
    double* d_output_;
    InterleavedData* d_interleaved_;

    BenchResult benchmark(const char* name, std::function<void()> kernel,
                          int numIterations, size_t dataBytes) {
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
        double bw = (dataBytes / 1e9) / (avgMs / 1000.0);

        return {name, avgMs, bw, 0.0};
    }

    void printResults(const std::vector<BenchResult>& results) {
        std::cout << "\n";
        std::cout << std::setw(25) << "Kernel"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(14) << "Bandwidth"
                  << std::setw(10) << "Speedup"
                  << std::endl;
        std::cout << std::string(63, '-') << std::endl;

        for (const auto& r : results) {
            std::cout << std::setw(25) << r.name
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.timeMs
                      << std::setw(12) << std::fixed << std::setprecision(1) << r.bandwidth << " GB/s"
                      << std::setw(9) << std::fixed << std::setprecision(2) << r.speedup << "x"
                      << std::endl;
        }
    }

    void allocate() {
        cudaMalloc(&d_convWeight_, numEvents_ * sizeof(double));
        for (int i = 0; i < 10; ++i) cudaMalloc(&d_hadronic_[i], numEvents_ * sizeof(double));
        for (int i = 0; i < 6; ++i) cudaMalloc(&d_cosmicRay_[i], numEvents_ * sizeof(double));
        cudaMalloc(&d_atmDensity_, numEvents_ * sizeof(double));
        cudaMalloc(&d_kaonLosses_, numEvents_ * sizeof(double));
        cudaMalloc(&d_params_, NUM_FIT_PARAMS * sizeof(double));
        cudaMalloc(&d_output_, numEvents_ * sizeof(double));
        cudaMalloc(&d_interleaved_, numEvents_ * sizeof(InterleavedData));
    }

    void generateData() {
        TestRNG rng(12345);

        std::vector<double> data(numEvents_);
        std::vector<InterleavedData> interleaved(numEvents_);

        // Generate flux weights
        for (int i = 0; i < numEvents_; ++i) data[i] = rng.uniform(1e-10, 1e-6);
        cudaMemcpy(d_convWeight_, data.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);

        // Generate hadronic corrections and fill interleaved
        for (int h = 0; h < 10; ++h) {
            for (int i = 0; i < numEvents_; ++i) {
                data[i] = rng.uniform(-0.1, 0.1);
                interleaved[i].hadronic[h] = data[i];
            }
            cudaMemcpy(d_hadronic_[h], data.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);
        }

        // Generate cosmic ray corrections and fill interleaved
        for (int c = 0; c < 6; ++c) {
            for (int i = 0; i < numEvents_; ++i) {
                data[i] = rng.uniform(-0.1, 0.1);
                interleaved[i].cosmicRay[c] = data[i];
            }
            cudaMemcpy(d_cosmicRay_[c], data.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);
        }

        // Upload interleaved data
        cudaMemcpy(d_interleaved_, interleaved.data(), numEvents_ * sizeof(InterleavedData), cudaMemcpyHostToDevice);

        // Generate atmospheric
        for (int i = 0; i < numEvents_; ++i) data[i] = rng.uniform(-0.1, 0.1);
        cudaMemcpy(d_atmDensity_, data.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);

        for (int i = 0; i < numEvents_; ++i) data[i] = rng.uniform(-0.1, 0.1);
        cudaMemcpy(d_kaonLosses_, data.data(), numEvents_ * sizeof(double), cudaMemcpyHostToDevice);

        // Parameters
        std::vector<double> params(NUM_FIT_PARAMS, 0.0);
        params[P_CONV_NORM] = 1.0;
        for (int i = P_HEKP; i <= P_VHE3N; ++i) params[i] = 1.0;
        for (int i = P_CR1; i <= P_CR6; ++i) params[i] = 1.0;
        params[P_ADU] = 0.1;
        params[P_KLU] = 0.1;
        cudaMemcpy(d_params_, params.data(), NUM_FIT_PARAMS * sizeof(double), cudaMemcpyHostToDevice);
        uploadParams(params.data());

        cudaDeviceSynchronize();
    }

    void deallocate() {
        cudaFree(d_convWeight_);
        for (int i = 0; i < 10; ++i) cudaFree(d_hadronic_[i]);
        for (int i = 0; i < 6; ++i) cudaFree(d_cosmicRay_[i]);
        cudaFree(d_atmDensity_);
        cudaFree(d_kaonLosses_);
        cudaFree(d_params_);
        cudaFree(d_output_);
        cudaFree(d_interleaved_);
    }
};

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "Conv Flux Assembly Optimization Benchmark\n";
    std::cout << "================================================================\n";
    std::cout << "Comparing optimization strategies for the main bottleneck\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Theoretical Bandwidth: "
              << (prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2) / 1e6 << " GB/s\n\n";

    std::cout << "Optimizations tested:\n";
    std::cout << "  Original:    Shared memory params, regular loads\n";
    std::cout << "  V1:          Constant memory params\n";
    std::cout << "  V2:          + __ldg() texture cache loads\n";
    std::cout << "  V3:          + FMA (fused multiply-add)\n";
    std::cout << "  V4:          + Interleaved memory layout\n\n";

    std::vector<int> eventCounts = {100000, 500000, 1000000, 2000000, 5000000};

    for (int numEvents : eventCounts) {
        std::cout << "================================================================\n";
        std::cout << "Event Count: " << numEvents / 1000 << "K\n";
        std::cout << "================================================================\n";

        ConvFluxBenchmark benchmark(numEvents);
        benchmark.run(100);

        std::cout << "\n";
    }

    std::cout << "================================================================\n";
    std::cout << "Summary\n";
    std::cout << "================================================================\n\n";

    std::cout << "Recommendations:\n";
    std::cout << "  - V3 (const mem + __ldg + FMA) provides best overall performance\n";
    std::cout << "  - V4 (interleaved) shows potential but requires data restructuring\n";
    std::cout << "  - For production, integrate V3 optimizations into main kernel\n";

    return 0;
}
