/**
 * @file test_integration.cu
 * @brief Integration tests for the full GPU acceleration pipeline.
 *
 * This test creates mock event data and verifies the complete workflow:
 * 1. Event data transfer
 * 2. Weight computation
 * 3. Histogram accumulation
 * 4. Likelihood evaluation
 */

#include "test_gpu_common.h"
#include "cuda/GPUCommon.h"
#include "cuda/GPUEventData.h"
#include "cuda/GPUFitAccelerator.h"
#include "Event.h"  // Event class is at global scope
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace gollumfit::gpu;

//==============================================================================
// Mock Event Generator
//==============================================================================

/**
 * @brief Create mock events for testing
 */
std::vector<Event> createMockEvents(int numEvents, TestRNG& rng) {
    std::vector<Event> events;
    events.reserve(numEvents);

    for (int i = 0; i < numEvents; ++i) {
        Event e;

        // Primary physics quantities
        e.energy = static_cast<float>(std::pow(10.0, rng.uniform(2.0, 6.0)));  // 100 GeV to 1 PeV
        e.zenith = static_cast<float>(std::acos(rng.uniform(-1.0, 0.2)));      // Up-going
        e.primaryEnergy = e.energy * static_cast<float>(rng.uniform(1.0, 10.0));
        e.primaryZenith = e.zenith + static_cast<float>(rng.uniform(-0.1, 0.1));
        e.primaryAzimuth = static_cast<float>(rng.uniform(0.0, 2.0 * M_PI));
        e.totalColumnDepth = static_cast<float>(rng.uniform(1e4, 1e6));
        e.intX = static_cast<float>(rng.uniform(0.0, 1.0));
        e.intY = static_cast<float>(rng.uniform(0.0, 1.0));

        // Discrete quantities
        e.topology = (rng.uniform() < 0.6) ? 0 : 1;  // 60% cascades, 40% tracks
        e.primaryType = (rng.uniform() < 0.5) ?
            LW::ParticleType::NuMu : LW::ParticleType::NuMuBar;
        e.num_events = 1;

        // Flux weights (mock values)
        double baseWeight = rng.uniform(1e-10, 1e-6);
        e.cachedConvWeight = baseWeight;
        e.cachedPromptWeight = baseWeight * 0.1;
        e.cachedAstroWeight = baseWeight * 0.01;
        e.cachedWeight = 1.0;

        // Detector systematics (unity for mock)
        e.cachedHoleIceConv = 0.0;
        e.cachedHoleIcePrompt = 0.0;
        e.cachedHoleIceAstro = 0.0;
        e.cachedDOMEffConv = 0.0;
        e.cachedDOMEffPrompt = 0.0;
        e.cachedDOMEffAstro = 0.0;

        // Hadronic parameters (small deviations)
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

        // Cosmic ray parameters
        e.cachedCosmicRay1 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay2 = rng.uniform(-0.1, 0.1) * baseWeight;
        e.cachedCosmicRay3 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay4 = rng.uniform(-0.05, 0.05) * baseWeight;
        e.cachedCosmicRay5 = rng.uniform(-0.02, 0.02) * baseWeight;
        e.cachedCosmicRay6 = rng.uniform(-0.02, 0.02) * baseWeight;

        // Atmospheric parameters
        e.cachedAtmDensity = rng.uniform(-0.2, 0.2);
        e.cachedKaonLosses = rng.uniform(-0.1, 0.1);

        // Ice gradient parameters
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

/**
 * @brief Create default fit parameters (at nominal values)
 */
std::vector<double> createNominalParameters() {
    std::vector<double> params(38, 0.0);

    params[0] = 1.0;   // convNorm
    params[1] = 1.0;   // promptNorm
    params[31] = 1.0;  // astroNorm
    params[35] = 1.0;  // neuaneu ratio

    // All other parameters at 0 (no systematic shifts)
    return params;
}

//==============================================================================
// Test Functions
//==============================================================================

TestResult testEventDataTransfer() {
    TestResult result;
    result.name = "Integration - Event Data Transfer";

    TestRNG rng(42);
    const int numEvents = 10000;

    // Create mock events
    auto events = createMockEvents(numEvents, rng);

    // Create event manager and upload
    GPUEventDataManager manager;

    Timer timer;
    timer.start();
    manager.uploadFromCPU(events, nullptr);
    result.gpuTime = timer.stop();

    // Verify
    result.passed = true;

    if (!manager.isInitialized()) {
        result.passed = false;
        result.message = "Manager not initialized after upload";
    }

    if (manager.getNumEvents() != static_cast<size_t>(numEvents)) {
        result.passed = false;
        result.message = "Event count mismatch";
    }

    const GPUEventDataSoA& data = manager.getDeviceData();
    if (!data.isValid()) {
        result.passed = false;
        result.message = "Device data not valid";
    }

    // Download and verify a few values
    std::vector<float> h_energy(numEvents);
    cudaMemcpy(h_energy.data(), data.energy, numEvents * sizeof(float), cudaMemcpyDeviceToHost);

    const double tol = 1e-5;
    for (int i = 0; i < std::min(100, numEvents); ++i) {
        if (std::abs(h_energy[i] - events[i].energy) > tol) {
            result.passed = false;
            result.message = "Energy values don't match after transfer";
            break;
        }
    }

    std::cout << "  Upload time: " << result.gpuTime << " ms for " << numEvents << " events\n";
    std::cout << "  Memory usage: " << manager.getMemoryUsage() / (1024.0 * 1024.0) << " MB\n";

    return result;
}

TestResult testBinIndexComputation() {
    TestResult result;
    result.name = "Integration - Bin Index Computation";

    TestRNG rng(123);
    const int numEvents = 5000;

    auto events = createMockEvents(numEvents, rng);

    GPUEventDataManager manager;
    manager.uploadFromCPU(events, nullptr);

    // Define histogram bins
    std::vector<double> energyEdges;
    for (double e = 2.0; e <= 6.0; e += 0.2) {
        energyEdges.push_back(std::pow(10.0, e));
    }

    std::vector<double> zenithEdges;
    for (double cz = -1.0; cz <= 0.2; cz += 0.1) {
        zenithEdges.push_back(cz);
    }

    int numTopologies = 2;

    // Compute bin indices
    manager.computeBinIndices(energyEdges, zenithEdges, numTopologies);

    // Download and verify
    const GPUEventDataSoA& data = manager.getDeviceData();
    std::vector<int32_t> h_binIndex(numEvents);
    cudaMemcpy(h_binIndex.data(), data.binIndex, numEvents * sizeof(int32_t), cudaMemcpyDeviceToHost);

    result.passed = true;
    int validBins = 0;
    int outOfBounds = 0;

    int nEBins = static_cast<int>(energyEdges.size()) - 1;
    int nZBins = static_cast<int>(zenithEdges.size()) - 1;
    int totalBins = nEBins * nZBins * numTopologies;

    for (int i = 0; i < numEvents; ++i) {
        if (h_binIndex[i] >= 0) {
            if (h_binIndex[i] >= totalBins) {
                result.passed = false;
                result.message = "Bin index exceeds total bins";
            }
            validBins++;
        } else {
            outOfBounds++;
        }
    }

    std::cout << "  Valid bins: " << validBins << "/" << numEvents << "\n";
    std::cout << "  Out of bounds: " << outOfBounds << "/" << numEvents << "\n";

    return result;
}

TestResult testFullAcceleratorWorkflow() {
    TestResult result;
    result.name = "Integration - Full Accelerator Workflow";

    TestRNG rng(456);
    const int numEvents = 50000;

    // Create mock events
    auto events = createMockEvents(numEvents, rng);

    // Set up histogram configuration
    HistogramConfig histConfig;
    histConfig.nBinsEnergy = 20;
    histConfig.nBinsZenith = 12;
    histConfig.nBinsTopology = 2;

    // Energy edges (log scale)
    for (int i = 0; i <= histConfig.nBinsEnergy; ++i) {
        histConfig.energyEdges.push_back(std::pow(10.0, 2.0 + i * 0.2));
    }

    // Zenith edges (cos(zenith))
    for (int i = 0; i <= histConfig.nBinsZenith; ++i) {
        histConfig.zenithEdges.push_back(-1.0 + i * 0.1);
    }

    // Create accelerator
    GPUAcceleratorConfig config;
    config.deviceId = 0;
    config.enableProfiling = true;

    try {
        GPUFitAccelerator accelerator(config);

        // Initialize with events
        Timer timer;
        timer.start();
        accelerator.initialize(events, histConfig);
        double initTime = timer.stop();

        // Create mock data histogram (Asimov-like)
        std::vector<double> dataHist(histConfig.totalBins(), 0.0);
        // For simplicity, use uniform expectation
        for (int i = 0; i < histConfig.totalBins(); ++i) {
            dataHist[i] = 10.0 + rng.uniform(-2, 2);  // ~10 events per bin
        }
        accelerator.uploadDataHistogram(dataHist);

        // Set up priors (all Gaussian with sigma=1)
        std::vector<PriorConfig> priors(38);
        for (int i = 0; i < 38; ++i) {
            priors[i].hasPrior = true;
            priors[i].mean = 0.0;
            priors[i].sigma = 1.0;
        }
        // Disable priors for norm parameters
        priors[0].hasPrior = false;
        priors[1].hasPrior = false;
        priors[31].hasPrior = false;
        accelerator.setPriors(priors);

        // Evaluate likelihood at nominal
        auto params = createNominalParameters();

        // Warm up
        accelerator.evaluateLikelihood(params, true);

        timer.start();
        double llh = accelerator.evaluateLikelihood(params, true);
        double evalTime = timer.stop();

        // Get timing breakdown
        auto timing = accelerator.getLastTimingStats();

        result.passed = true;

        // Basic sanity checks
        if (!std::isfinite(llh)) {
            result.passed = false;
            result.message = "Likelihood is not finite";
        }

        // Check that expectation histogram is reasonable
        std::vector<double> expectation;
        accelerator.getExpectationHistogram(expectation);

        double totalExpectation = 0.0;
        for (double e : expectation) {
            if (e < 0) {
                result.passed = false;
                result.message = "Negative expectation in histogram";
                break;
            }
            totalExpectation += e;
        }

        if (totalExpectation <= 0) {
            result.passed = false;
            result.message = "Total expectation is zero or negative";
        }

        std::cout << "  Initialization time: " << initTime << " ms\n";
        std::cout << "  Evaluation time: " << evalTime << " ms\n";
        std::cout << "    - Parameter transfer: " << timing.paramTransferMs << " ms\n";
        std::cout << "    - Weight computation: " << timing.weightComputeMs << " ms\n";
        std::cout << "    - Histogram: " << timing.histogramMs << " ms\n";
        std::cout << "    - Likelihood: " << timing.likelihoodMs << " ms\n";
        std::cout << "  Likelihood value: " << llh << "\n";
        std::cout << "  Total expectation: " << totalExpectation << "\n";
        std::cout << "  Events: " << accelerator.getNumEvents() << "\n";
        std::cout << "  Bins: " << accelerator.getNumBins() << "\n";

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

TestResult testParameterSensitivity() {
    TestResult result;
    result.name = "Integration - Parameter Sensitivity";

    TestRNG rng(789);
    const int numEvents = 20000;

    auto events = createMockEvents(numEvents, rng);

    HistogramConfig histConfig;
    histConfig.nBinsEnergy = 15;
    histConfig.nBinsZenith = 10;
    histConfig.nBinsTopology = 2;

    for (int i = 0; i <= histConfig.nBinsEnergy; ++i) {
        histConfig.energyEdges.push_back(std::pow(10.0, 2.0 + i * 0.25));
    }
    for (int i = 0; i <= histConfig.nBinsZenith; ++i) {
        histConfig.zenithEdges.push_back(-1.0 + i * 0.12);
    }

    try {
        GPUAcceleratorConfig config;
        GPUFitAccelerator accelerator(config);
        accelerator.initialize(events, histConfig);

        // Create data histogram
        std::vector<double> dataHist(histConfig.totalBins(), 5.0);
        accelerator.uploadDataHistogram(dataHist);

        // Evaluate at nominal
        auto params = createNominalParameters();
        double llh_nominal = accelerator.evaluateLikelihood(params, false);

        // Test that varying parameters changes likelihood
        result.passed = true;

        // Test convNorm sensitivity
        params[0] = 1.5;  // 50% increase
        double llh_conv = accelerator.evaluateLikelihood(params, false);
        params[0] = 1.0;

        if (llh_conv == llh_nominal) {
            result.passed = false;
            result.message = "Likelihood not sensitive to convNorm";
        }

        // Test ice gradient sensitivity
        params[20] = 0.5;  // Shift ice gradient
        double llh_ice = accelerator.evaluateLikelihood(params, false);
        params[20] = 0.0;

        if (llh_ice == llh_nominal) {
            result.passed = false;
            result.message = "Likelihood not sensitive to ice gradient";
        }

        std::cout << "  LLH at nominal: " << llh_nominal << "\n";
        std::cout << "  LLH with convNorm=1.5: " << llh_conv
                  << " (diff=" << llh_conv - llh_nominal << ")\n";
        std::cout << "  LLH with icegrad0=0.5: " << llh_ice
                  << " (diff=" << llh_ice - llh_nominal << ")\n";

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }

    return result;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "GPU Integration Tests\n";
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
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n\n";

    TestSuite suite;

    suite.addResult(testEventDataTransfer());
    suite.addResult(testBinIndexComputation());
    suite.addResult(testFullAcceleratorWorkflow());
    suite.addResult(testParameterSensitivity());

    suite.printSummary();

    return suite.allPassed() ? 0 : 1;
}
