/**
 * @file benchmark_gradient_scaling.cu
 * @brief Benchmark gradient kernel scaling behavior across different event counts.
 *
 * Measures forward-only (evaluateLikelihood) and gradient
 * (evaluateLikelihoodWithGradient) wall-clock times at varying event
 * counts to characterise throughput scaling.  Each configuration runs
 * warmup iterations followed by timed iterations with
 * cudaDeviceSynchronize barriers so that CPU-side Timer captures the
 * true GPU execution time.
 *
 * Printed table columns:
 *   Events  Forward(ms)  Gradient(ms)  Ratio  ms/MEvent(fwd)  ms/MEvent(grad)
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUFitAccelerator.h"
#include "Event.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <memory>

using namespace gollumfit::gpu;

//==============================================================================
// Mock Event Generator (same as test_adjoint_gradient.cu)
//==============================================================================

std::vector<Event> createMockEvents(int numEvents, TestRNG& rng) {
    std::vector<Event> events;
    events.reserve(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        Event e;

        e.energy = static_cast<float>(std::pow(10.0, rng.uniform(2.0, 6.0)));
        e.zenith = static_cast<float>(std::acos(rng.uniform(-1.0, 0.2)));
        e.primaryEnergy = e.energy * static_cast<float>(rng.uniform(1.0, 10.0));
        e.primaryZenith = e.zenith + static_cast<float>(rng.uniform(-0.1, 0.1));
        e.primaryAzimuth = static_cast<float>(rng.uniform(0.0, 2.0 * M_PI));
        e.totalColumnDepth = static_cast<float>(rng.uniform(1e4, 1e6));
        e.intX = static_cast<float>(rng.uniform(0.0, 1.0));
        e.intY = static_cast<float>(rng.uniform(0.0, 1.0));

        e.topology = (rng.uniform() < 0.6) ? 0 : 1;
        e.primaryType = (rng.uniform() < 0.5) ?
            LW::ParticleType::NuMu : LW::ParticleType::NuMuBar;
        e.num_events = 1;

        double baseWeight = rng.uniform(1e-10, 1e-6);
        e.cachedConvWeight = baseWeight;
        e.cachedPromptWeight = baseWeight * 0.1;
        e.cachedAstroWeight = baseWeight * 0.01;
        e.cachedWeight = 1.0;

        e.cachedHoleIceConv = 0.0;
        e.cachedHoleIcePrompt = 0.0;
        e.cachedHoleIceAstro = 0.0;
        e.cachedDOMEffConv = 0.0;
        e.cachedDOMEffPrompt = 0.0;
        e.cachedDOMEffAstro = 0.0;

        e.cachedHadronicHEkp = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedHadronicHEkm = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedHadronicVHE1pip = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedHadronicVHE1pim = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedHadronicVHE3kp = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3km = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3pip = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3pim = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedHadronicVHE3p = rng.uniform(-0.01, 0.01) * baseWeight;
        e.cachedHadronicVHE3n = rng.uniform(-0.01, 0.01) * baseWeight;

        e.cachedCosmicRay1 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay2 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay3 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay4 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay5 = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedCosmicRay6 = rng.uniform(-0.02, 0.02) * baseWeight;

        e.cachedAtmDensity = rng.uniform(-0.2, 0.2);
        e.cachedKaonLosses = rng.uniform(-0.1, 0.1);

        e.cachedIceGrad0 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad1 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad2 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad3 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad4 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad5 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad6 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad7 = rng.uniform(-0.1, 0.1);
        e.cachedIceGrad8 = rng.uniform(-0.1, 0.1);

        events.push_back(e);
    }

    return events;
}

std::vector<double> createNominalParameters() {
    std::vector<double> params(38, 0.0);
    params[0] = 1.0;   // convNorm
    params[1] = 1.0;   // promptNorm
    params[31] = 1.0;  // astroNorm
    params[35] = 1.0;  // neuaneu ratio
    return params;
}

//==============================================================================
// Helper: set up a fully-configured accelerator
//==============================================================================

struct TestAccelerator {
    std::unique_ptr<GPUFitAccelerator> accel;
    HistogramConfig histConfig;

    void setup(int numEvents, int seed, bool withPriors) {
        TestRNG rng(seed);
        auto events = createMockEvents(numEvents, rng);

        histConfig.nBinsEnergy = 20;
        histConfig.nBinsZenith = 12;
        histConfig.nBinsTopology = 2;
        for (int i = 0; i <= histConfig.nBinsEnergy; ++i)
            histConfig.energyEdges.push_back(std::pow(10.0, 2.0 + i * 0.2));
        for (int i = 0; i <= histConfig.nBinsZenith; ++i)
            histConfig.zenithEdges.push_back(-1.0 + i * 0.1);

        GPUAcceleratorConfig config;
        config.deviceId = 0;
        config.enableProfiling = true;
        accel = std::make_unique<GPUFitAccelerator>(config);
        accel->initialize(events, histConfig);

        // Create mock data histogram
        TestRNG dataRng(seed + 1000);
        std::vector<double> dataHist(histConfig.totalBins());
        for (int i = 0; i < histConfig.totalBins(); ++i) {
            dataHist[i] = 5.0 + dataRng.uniform(-2.0, 2.0);
        }
        accel->uploadDataHistogram(dataHist);

        if (withPriors) {
            std::vector<PriorConfig> priors(38);
            for (int i = 0; i < 38; ++i) {
                priors[i].hasPrior = true;
                priors[i].mean = 0.0;
                priors[i].sigma = 1.0;
            }
            priors[0].hasPrior = false;   // convNorm
            priors[1].hasPrior = false;   // promptNorm
            priors[31].hasPrior = false;  // astroNorm
            priors[35].hasPrior = false;  // neuaneu ratio
            accel->setPriors(priors);
        }
    }
};

//==============================================================================
// Benchmark result per event-count configuration
//==============================================================================

struct ScalingResult {
    int numEvents;
    double forwardMs;     // mean per-iteration forward time
    double gradientMs;    // mean per-iteration gradient time
    double ratio;         // gradient / forward
    double fwdPerMEvent;  // ms per million events (forward)
    double gradPerMEvent; // ms per million events (gradient)
};

//==============================================================================
// Run benchmark for a single event count
//==============================================================================

ScalingResult benchmarkAtSize(int numEvents, int warmupIters, int timedIters) {
    ScalingResult res;
    res.numEvents = numEvents;

    std::cout << "  Setting up " << numEvents << " events ... " << std::flush;

    TestAccelerator ta;
    ta.setup(numEvents, /*seed=*/42, /*withPriors=*/false);

    auto params = createNominalParameters();
    std::vector<double> grad;

    std::cout << "done.\n" << std::flush;

    // -- Warmup: forward --
    cudaDeviceSynchronize();
    for (int i = 0; i < warmupIters; ++i) {
        ta.accel->evaluateLikelihood(params, false);
        cudaDeviceSynchronize();
    }

    // -- Timed: forward --
    Timer timer;
    double totalFwd = 0.0;
    for (int i = 0; i < timedIters; ++i) {
        cudaDeviceSynchronize();
        timer.start();
        ta.accel->evaluateLikelihood(params, false);
        cudaDeviceSynchronize();
        totalFwd += timer.stop();
    }
    res.forwardMs = totalFwd / timedIters;

    // -- Warmup: gradient --
    cudaDeviceSynchronize();
    for (int i = 0; i < warmupIters; ++i) {
        ta.accel->evaluateLikelihoodWithGradient(params, grad, false);
        cudaDeviceSynchronize();
    }

    // -- Timed: gradient --
    double totalGrad = 0.0;
    for (int i = 0; i < timedIters; ++i) {
        cudaDeviceSynchronize();
        timer.start();
        ta.accel->evaluateLikelihoodWithGradient(params, grad, false);
        cudaDeviceSynchronize();
        totalGrad += timer.stop();
    }
    res.gradientMs = totalGrad / timedIters;

    res.ratio = res.gradientMs / res.forwardMs;
    double mEvents = numEvents / 1.0e6;
    res.fwdPerMEvent = res.forwardMs / mEvents;
    res.gradPerMEvent = res.gradientMs / mEvents;

    return res;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Gradient Kernel Scaling Benchmark\n";
    std::cout << "========================================\n\n";

    // -- Device info --
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
    std::cout << "Memory Bandwidth: " << prop.memoryBusWidth << " bit @ "
              << prop.memoryClockRate / 1e6 << " GHz\n\n";

    // -- Kernel configuration info --
    std::cout << "Gradient kernel configuration:\n";
    std::cout << "  Block size:    256 threads\n";
    std::cout << "  launch_bounds: (256, 1)\n";
    std::cout << "  Stack spill:   14600 bytes per thread (from ptxas)\n";
    std::cout << "\n";

    // -- Benchmark parameters --
    const int warmupIters = 3;
    const int timedIters  = 5;
    const std::vector<int> eventCounts = {
        10000, 50000, 100000, 250000, 500000, 1000000, 1500000
    };

    std::cout << "Warmup iterations: " << warmupIters << "\n";
    std::cout << "Timed iterations:  " << timedIters << "\n";
    std::cout << "Using cudaDeviceSynchronize barriers for accurate CPU-side timing.\n\n";

    // -- Run benchmarks --
    std::vector<ScalingResult> results;
    results.reserve(eventCounts.size());

    for (int n : eventCounts) {
        results.push_back(benchmarkAtSize(n, warmupIters, timedIters));
    }

    // -- Print table --
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n\n";

    std::cout << std::setw(10) << "Events"
              << std::setw(14) << "Forward(ms)"
              << std::setw(14) << "Gradient(ms)"
              << std::setw(8)  << "Ratio"
              << std::setw(17) << "ms/MEvent(fwd)"
              << std::setw(18) << "ms/MEvent(grad)"
              << "\n";
    std::cout << std::string(81, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.numEvents
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.forwardMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.gradientMs
                  << std::setw(8)  << std::fixed << std::setprecision(2) << r.ratio
                  << std::setw(17) << std::fixed << std::setprecision(3) << r.fwdPerMEvent
                  << std::setw(18) << std::fixed << std::setprecision(3) << r.gradPerMEvent
                  << "\n";
    }

    // -- Scaling analysis --
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Scaling Analysis\n";
    std::cout << "========================================\n\n";

    if (results.size() >= 2) {
        const auto& first = results.front();
        const auto& last  = results.back();

        double eventRatio = static_cast<double>(last.numEvents) / first.numEvents;
        double fwdTimeRatio  = last.forwardMs  / first.forwardMs;
        double gradTimeRatio = last.gradientMs / first.gradientMs;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Event count ratio (last/first): " << eventRatio << "x\n";
        std::cout << "Forward time ratio:             " << fwdTimeRatio << "x\n";
        std::cout << "Gradient time ratio:            " << gradTimeRatio << "x\n";
        std::cout << "\n";

        // Linear scaling would give ratio == eventRatio
        double fwdScalingExp  = std::log(fwdTimeRatio) / std::log(eventRatio);
        double gradScalingExp = std::log(gradTimeRatio) / std::log(eventRatio);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Scaling exponent (1.0 = linear, <1.0 = sub-linear):\n";
        std::cout << "  Forward:  " << fwdScalingExp << "\n";
        std::cout << "  Gradient: " << gradScalingExp << "\n";
        std::cout << "\n";

        // Average gradient/forward ratio
        double avgRatio = 0.0;
        for (const auto& r : results)
            avgRatio += r.ratio;
        avgRatio /= results.size();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average gradient/forward ratio: " << avgRatio << "x\n";
        std::cout << "  (Ideal adjoint method: ~3-5x forward cost for "
                  << "38 parameters)\n";
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Notes\n";
    std::cout << "========================================\n";
    std::cout << "- Gradient kernel uses 14600 bytes stack spill per thread (ptxas)\n";
    std::cout << "- Block config: 256 threads, launch_bounds(256,1)\n";
    std::cout << "- High register + spill pressure limits occupancy\n";
    std::cout << "- Sub-linear scaling at small N suggests kernel launch overhead dominates\n";
    std::cout << "- Linear scaling at large N indicates memory-bandwidth or compute bound\n";

    return 0;
}
